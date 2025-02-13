# Copyright © 2023 United States Government as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All Rights Reserved.

import io
from functools import cached_property
from typing import Optional

import astropy.units as u  # type: ignore
import httpx
import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore
from astropy.io.votable import parse_single_table  # type: ignore
from astropy.time import Time  # type: ignore
from pydantic import Field

from ..base.common import ACROSSAPIBase, ceil_time, floor_time
from ..core.enums import VisibilityType
from ..ephemeris import Ephemeris
from .schema import ObservatoryConstraints, VisibilitySchema, VisWindow


class EphemVisibilityBase(VisibilitySchema):
    """
    Calculate visibility of a given object, based on a given spacecraft
    Ephemeris and constraints. Currently supported constraints: Sun, Moon,
    Earth Limb, SAA, Pole.

    Parameters
    ----------
    ra
        Right Ascension in decimal degrees
    dec
        Declination in decimal degrees
    begin
        Start time of visibility search
    end
        End time of visibility search
    hires
        Use high resolution ephemeris for calculations and include earth
        occultations
    constraints
        ObservatoryConstraints object to use for visibility calculations

    Attributes
    ----------
    ephem
        Ephem object to use for visibility calculations
    saa
        SAA object to use for visibility calculations
    entries
        list of visibility windows
    timestamp
        Array of timestamps
    calculated_constraints
        dictionary of calculated constraints
    inconstraint
        Array of booleans indicating if the spacecraft is in a constraint

    Methods
    -------
    get_ephem_vis
        Query visibility for given parameters
    constraint
        What kind of constraints are in place at a given time index
    make_windows
        Create a list of visibility windows from constraint arrays

    """

    # Constraint definitions
    constraints: ObservatoryConstraints | None = Field(None, exclude=True)

    # Computed values
    timestamp: np.typing.NDArray = Field(np.array([]), exclude=True)
    calculated_constraints: dict[str, np.typing.NDArray] = Field({}, exclude=True)
    inconstraint: np.typing.NDArray = Field(np.array([]), exclude=True)
    ephem: Optional[Ephemeris] = Field(None, exclude=True)

    def __getitem__(self, i):
        return self.entries[i]

    def __len__(self):
        return len(self.timestamp)

    @cached_property
    def skycoord(self):
        """
        Create array of RA/Dec and vector of these.

        Returns
        -------
        numpy.ndarray
            Array of RA/Dec coordinates.
        """
        return SkyCoord(self.ra * u.deg, self.dec * u.deg)

    def visible(self, t: Time) -> bool:
        """
        For a given time, is the target visible?

        Parameters
        ----------
        t
            Time to check

        Returns
        -------
            True if visible, False if not
        """
        return any(t >= win.begin and t <= win.end for win in self.entries)

    @cached_property
    def ephstart(self) -> Optional[int]:
        """
        Returns the ephemeris index of the beginning time.
        """
        if self.ephem is None:
            return None
        return self.ephem.ephindex(Time(self.begin))

    @cached_property
    def ephstop(self) -> Optional[int]:
        """
        Returns the ephemeris index of the stopping time.
        """
        if self.ephem is None:
            return None
        i = self.ephem.ephindex(Time(self.end))
        if i is None:
            return None
        return i + 1

    @property
    def step_size(self) -> int:
        """
        Returns the step size in seconds based on the hires attribute.
        If high resolution is enabled, returns 60 seconds, otherwise returns 3600 seconds.
        """
        if self.hires:
            return 60
        return 3600

    async def get_ephem_vis(self) -> bool:
        """
        Query visibility for given parameters.

        Returns
        -------
            True if successful, False otherwise.
        """
        # Reset windows
        self.entries = list()

        # Check if constraints are available
        if self.constraints is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Constraints not available.",
            )

        # Check everything is kosher, if just run calculation
        if (
            self.ephem is None
            or self.ephem.timestamp is None
            or self.ephstart is None
            or self.ephstop is None
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ephemeris not available.",
            )

        # Calculate the times to calculate the visibility
        self.timestamp = self.ephem.timestamp[self.ephstart : self.ephstop]

        # Calculate all the individual constraints
        self.calculated_constraints = {
            cons.short_name: cons(time=self.timestamp, ephem=self.ephem, skycoord=self.skycoord)
            for cons in self.constraints.constraints
        }

        # self.inconstraint is the logical or of all constraints
        self.inconstraint = np.logical_or.reduce([v for v in self.calculated_constraints.values()])

        # Calculate good windows from combined constraints
        self.entries = await self.make_windows(self.inconstraint.tolist())

        return True

    async def constraint(self, index: int) -> str:
        """
        What kind of constraints are in place at a given time index.

        Parameters
        ----------
        index
            Index of timestamp to check

        Returns
        -------
            String indicating what constraint is in place at given time index
        """
        # Sanity check
        assert self.ephstart is not None
        assert self.ephstop is not None

        # Check if index is out of bounds
        if index == self.ephstart - 1 or index >= self.ephstop - 1:
            return "Window"

        # Return what constraint is causing the window to open/close
        for k, v in self.calculated_constraints.items():
            if v[index]:
                return k

        return "Unknown"

    async def make_windows(self, inconstraint: np.typing.NDArray) -> list:
        """
        Record SAAEntry from array of booleans and timestamps

        Parameters
        ----------
        inconstraint : list
            list of booleans indicating if the spacecraft is in the SAA
        wintype : VisWindow
            Type of window to create (default: VisWindow)

        Returns
        -------
        list
            list of SAAEntry objects
        """

        # Find the start and end of the visibility windows
        buff: np.typing.NDArray = np.concatenate(([False], np.logical_not(inconstraint), [False]))
        begin = np.flatnonzero(~buff[:-1] & buff[1:])
        end = np.flatnonzero(buff[:-1] & ~buff[1:])
        indices = np.column_stack((begin, end - 1))

        # Return as list of VisWindows
        return [
            VisWindow(
                begin=self.timestamp[i[0]].datetime,
                end=self.timestamp[i[1]].datetime,
                visibility=int((self.timestamp[i[1]] - self.timestamp[i[0]]).to_value(u.s)),
                initial=await self.constraint(i[0] - 1),
                final=await self.constraint(i[1] + 1),
            )
            for i in indices
            if self.timestamp[i[0]].datetime != self.timestamp[i[1]].datetime
        ]


class ObjVisSAPVisibilityBase(VisibilitySchema, ACROSSAPIBase):
    """
    Calculate visibility of a given object using the VO ObjVisSAP protocol

    Parameters
    ----------
    ra
        Right Ascension in decimal degrees
    dec
        Declination in decimal degrees
    begin
        Start time of visibility search
    end
        End time of visibility search
    objvissap_url
        URL of the VO ObjVisSAP visibility tool
    """

    objvissap_url: str | None = Field(None, exclude=True)
    objvissap_default_params: dict = Field({}, exclude=True)
    inconstraint: np.typing.NDArray = Field(np.array([]), exclude=True)
    timestamp: np.typing.NDArray = Field(np.array([]), exclude=True)

    async def get_objvissap(self):
        # Construct the ObjVisSAP query parameters
        params = {
            "s_ra": self.ra,
            "s_dec": self.dec,
            "t_min": Time(self.begin).mjd,
            "t_max": Time(self.end).mjd,
            "min_vis": 1,
        }
        # Perform the query
        async with httpx.AsyncClient() as client:
            r = await client.get(self.objvissap_url, params=dict(params, **self.objvissap_default_params))

        # If successful, parse the results
        if r.status_code == 200:
            # Parse VOTABLE into VisWindow objects
            xmlvo = r.text
            # Fix broke non-conformant VO from XMM
            xmlvo = xmlvo.replace('datatype="timestamp"', 'datatype="float"')
            xmlvo = xmlvo.replace('name="t_end"', 'name="t_stop"')

            votablefile = io.BytesIO(xmlvo.encode())
            votable = parse_single_table(votablefile)

            for i in range(len(votable.array.data)):  # type: ignore
                vw = VisWindow(
                    begin=Time(votable.array.data["t_start"][i], format="mjd").datetime,
                    end=Time(votable.array.data["t_stop"][i], format="mjd").datetime,
                    visibility=votable.array.data["t_visibility"][i],
                    initial="Window",
                    final="Window",
                )
                self.entries.append(vw)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Visibility tool offline.",
            )


def envkey(*args, env={}, **kwargs):
    for k, v in kwargs.items():
        kwargs[k] = str(v)
    key = hashkey(*args, **kwargs)
    key += tuple(sorted(env.items()))
    return key


@cached(cache=TTLCache(maxsize=128, ttl=86400), key=envkey)
class ObservatoryVisibility(
    EphemVisibilityBase,
    ObjVisSAPVisibilityBase,
    ObjVisSAPBase,
):
    """
    Class to calculate Observatory visibility for a given observatory ID.

    Attributes
    ----------
    observatory_id : int
        The observatory ID to calculate visibility for.

    Methods
    -------
    get():
        Perform visibility calcualtion using get_ephem() or get_objvissap(),
        based on the visibility type. Return False if the visibility type is
        not 'tle' or 'objvissap'.
    post(api_token: str):
        Perform a POST request to create a new observatory visibility configuration record.
    put(api_token: str):
        Perform a PUT request to update an existing observatory visibility configuration record.
    delete(api_token: str):
        Perform a DELETE request to remove an existing observatory visibility configuration record.
    get_ephem():
        Calculate Ephemeris and SAA information for TLE based calculations.

    """

    observatory_id: int
    constraints: ObservatoryConstraints | None = Field(None, exclude=True)

    async def get(self):
        """
        Retrieves the visibility based on the specified visibility type.

        Returns
        -------
            If the visibility type is 'tle', returns the result of the 'get_ephem' method.
            If the visibility type is 'objvissap', returns the result of the 'get_objvissap' method.
            Otherwise, returns False.
        """
        # Fetch the Observatory Constraint configuration from the database
        self.constraints = ObservatoryConstraints(observatory_id=self.observatory_id)
        await self.constraints.get()

        # Return visibility for TLE based calculations
        if self.constraints.visibility_type == VisibilityType.tle:
            return await self.get_ephem()

        # Return visibility for VO ObjViSSAP based calculations
        elif self.constraints.visibility_type == VisibilityType.objvissap:
            return await self.get_objvissap()

        # Return visibilty for Custom calculation
        elif self.constraints.visibility_type == VisibilityType.custom:
            return await self.get_custom()

        # Raise an exception if the visibility type is invalid
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid visibility request",
            )

    async def get_ephem(self):
        # Calculate Ephemeris and SAA information. Note calculate this for
        # whole days to avoid having to calculate it multiple times (with
        # caching).
        if self.ephem is None:
            daybegin = floor_time(Time(self.begin), 1 * u.day).datetime
            dayend = ceil_time(Time(self.end), 1 * u.day).datetime
            self.ephem = ObservatoryEphem(
                observatory_id=self.observatory_id,
                begin=daybegin,
                end=dayend,
                step_size=self.step_size,
            )
            await self.ephem.get()

        # Calculate visibility
        return await super().get_ephem_vis()

    async def get_custom(self):
        observatory = await Observatory.record_from_id(self.observatory_id)
        if observatory is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Observatory not found",
            )
        if observatory.short_name in CUSTOM_OBSERVATORY_VISIBILITY_CLASSES:
            vis = CUSTOM_OBSERVATORY_VISIBILITY_CLASSES[observatory.short_name](
                begin=self.begin,
                end=self.end,
                ra=self.ra,
                dec=self.dec,
                hires=self.hires,
                min_vis=self.min_vis,
            )
            await vis.get()
            self.entries = vis.entries
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Custom visibility not found",
            )
        return True
