# Copyright © 2023 United States Government as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All Rights Reserved.


from datetime import datetime, timedelta

from httpx import HTTPStatusError
from spacetrack import AuthenticationError, SpaceTrackClient  # type: ignore[import-untyped]

from ..core.config import config
from ..core.schemas.tle import TLE
from .exceptions import SpaceTrackAuthenticationError


class TLEFetch:
    """
    Fetches Two-Line Element (TLE) data for a satellite at a specific epoch.
    Requires a Space-Track.org account to access the TLE data. If no spacetrack
    user or password is provided, the class will attempt to use the environment
    variables SPACETRACK_USER and SPACETRACK_PWD.

    Parameters
    ----------
    epoch
        Epoch of TLE to retrieve

    Attributes
    ----------
    satellite_name : str, optional
        Name of the satellite
    norad_id : int
        NORAD ID of the satellite
    epoch : datetime
        Epoch of the TLE
    spacetrack_user : str, optional
        Space-Track.org username
    spacetrack_pwd  : str, optional
        Space-Track.org password

    Methods
    -------
    get
        Get TLEs for given epoch
    """

    # Configuration parameters
    satellite_name: str | None
    norad_id: int
    epoch: datetime
    spacetrack_user: str | None
    spacetrack_pwd: str | None

    def __init__(
        self,
        norad_id: int,
        epoch: datetime,
        satellite_name: str | None = None,
        spacetrack_user: str | None = None,
        spacetrack_pwd: str | None = None,
    ):
        self.norad_id = norad_id
        self.epoch = epoch
        self.satellite_name = satellite_name
        self.spacetrack_user = spacetrack_user or config.SPACETRACK_USER
        self.spacetrack_pwd = spacetrack_pwd or config.SPACETRACK_PWD

    def get(self) -> TLE | None:
        """
        Return TLE epoch closest to the requested epoch, within +/- 7 days. If
        no TLE data is found, return None.

        Returns
        -------
        Optional[TLE]
            A TLE object containing the two-line element data for the specified
            satellite, or None if no data is found.

        Raises
        ------
        SpaceTrackAuthenticationError
            If space-track.org authentication fails.
        """
        # Build space-track.org query
        epoch_start = self.epoch - timedelta(days=7)
        epoch_stop = self.epoch + timedelta(days=7)

        # Log into space-track.org
        with SpaceTrackClient(
            identity=self.spacetrack_user, password=self.spacetrack_pwd
        ) as spacetrack_client:
            try:
                spacetrack_client.authenticate()
            except (AuthenticationError, HTTPStatusError) as e:
                raise SpaceTrackAuthenticationError("space-track.org authentication failed.") from e

            # Fetch the TLEs between the requested epochs
            tletext = spacetrack_client.gp_history(
                norad_cat_id=self.norad_id,
                orderby="epoch desc",
                limit=22,
                format="tle",
                epoch=f">{epoch_start},<{epoch_stop}",
            )

        # Check if we got a return
        if tletext == "":
            return None

        # Split the TLEs into individual lines
        tletext = tletext.splitlines()

        # Parse the results into a list of TLEEntry objects
        tles = [
            TLE(
                satellite_name=self.satellite_name,
                norad_id=self.norad_id,
                tle1=tletext[i],
                tle2=tletext[i + 1],
            )
            for i in range(0, len(tletext), 2)
        ]

        # Return the TLE that is closest to the requested epoch
        tles.sort(key=lambda x: abs(x.epoch - self.epoch))
        return tles[0] if tles else None


def get_tle(
    norad_id: int,
    epoch: datetime,
    spacetrack_user: str | None = None,
    spacetrack_pwd: str | None = None,
) -> TLE | None:
    """
    Gets the Two-Line Element (TLE) data for a satellite at a specific epoch.
    Credentials for space-track.org can be provided as arguments, or they can
    be set as environment variables SPACETRACK_USER and SPACETRACK_PWD.

    Parameters
    ----------
    norad_id : int
        The NORAD Catalog Number (NORAD ID) of the satellite.
    epoch : datetime
        The epoch timestamp for which to retrieve the TLE data.
    spacetrack_user : str, optional
        space-Track.org username.
    spacetrack_pwd : str, optional
        space-Track.org password.

    Returns
    -------
    TLE
        A TLE object containing the two-line element data for the specified
        satellite, or None if no data is found.

    Raises
    ------
    AuthenticationError
        If space-Track.org authentication fails.
    """

    tle = TLEFetch(
        norad_id=norad_id, epoch=epoch, spacetrack_user=spacetrack_user, spacetrack_pwd=spacetrack_pwd
    )
    return tle.get()
