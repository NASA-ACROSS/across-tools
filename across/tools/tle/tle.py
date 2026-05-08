# Copyright © 2023 United States Government as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All Rights Reserved.


import logging
import re
from datetime import datetime, timedelta
from typing import TypedDict

from httpx import HTTPStatusError
from spacetrack import AuthenticationError, SpaceTrackClient  # type: ignore[import-untyped]

from ..core.config import config
from ..core.schemas.tle import TLE
from .exceptions import SpaceTrackAuthenticationError


class NoradSatellite(TypedDict):
    """Satellite specification with name and NORAD ID."""

    name: str
    id: int


class TLEFetch:
    """
    Fetches Two-Line Element (TLE) data for one or more satellites at a specific epoch.
    Requires a Space-Track.org account to access the TLE data. If no spacetrack
    user or password is provided, the class will attempt to use the environment
    variables SPACETRACK_USER and SPACETRACK_PWD.

    Parameters
    ----------
    satellites : list[NoradSatellite]
        List of satellites to query, each with 'name' and 'id' keys.
        Only the id is used to query against spacetrack_client.gp norad_cat_id
    epoch : datetime
        Epoch of TLE to retrieve.
    spacetrack_user : str, optional
        Space-Track.org username. Falls back to SPACETRACK_USER config.
    spacetrack_pwd : str, optional
        Space-Track.org password. Falls back to SPACETRACK_PWD config.
    spacetrack_base_url : str, optional
        Optional Space-Track API base URL override. Default is None.

    Attributes
    ----------
    satellites : list[NoradSatellite]
        List of satellites with name and NORAD ID.
    epoch : datetime
        Epoch of the TLE
    spacetrack_user : str, optional
        Space-Track.org username
    spacetrack_pwd  : str, optional
        Space-Track.org password
    spacetrack_base_url : str, optional
        Space-Track API base URL override

    Methods
    -------
    get
        Get TLEs for given epoch

    Raises
    ------
    TypeError
        - satellites must be a list of dicts with 'name' and 'id' keys
        - each satellite must be a dict
        - each satellite dict must have 'name' and 'id' keys
        - satellite 'name' must be a string
        - satellite 'id' must be an integer
    ValueError
        - satellites list cannot be empty
    """

    # Configuration parameters
    satellites: list[NoradSatellite]
    epoch: datetime
    spacetrack_user: str | None
    spacetrack_pwd: str | None
    spacetrack_base_url: str | None

    def __init__(
        self,
        satellites: list[NoradSatellite],
        epoch: datetime,
        spacetrack_user: str | None = None,
        spacetrack_pwd: str | None = None,
        spacetrack_base_url: str | None = None,
    ):
        if not isinstance(satellites, list):
            raise TypeError("satellites must be a list of dicts with 'name' and 'id' keys")
        if not satellites:
            raise ValueError("satellites list cannot be empty")
        for sat in satellites:
            if not isinstance(sat, dict):
                raise TypeError("each satellite must be a dict")
            if "name" not in sat or "id" not in sat:
                raise TypeError("each satellite dict must have 'name' and 'id' keys")
            if not isinstance(sat["name"], str):
                raise TypeError("satellite 'name' must be a string")
            if not isinstance(sat["id"], int):
                raise TypeError("satellite 'id' must be an integer")

        self.satellites = satellites
        self.epoch = epoch
        self.spacetrack_user = spacetrack_user or config.SPACETRACK_USER
        self.spacetrack_pwd = spacetrack_pwd or config.SPACETRACK_PWD
        self.spacetrack_base_url = spacetrack_base_url

    @staticmethod
    def _extract_norad_id_from_tle1(tle1: str) -> int | None:
        """Extract NORAD ID from the first segment of TLE line 1."""
        match = re.match(r"^1\s+(\d+)", tle1)
        return int(match.group(1)) if match else None

    def get(self) -> list[TLE]:
        """
        Return TLE data for requested NORAD ID(s), within +/- 7 days.

        Aggregates response by NORAD ID
        by scanning the first section of TLE line 1, and only the
        newest TLE for each requested NORAD ID is retained.

        Returns
        -------
        list[TLE]
            TLE objects matching the requested NORAD ID(s). Empty when no data
            is found.

        Raises
        ------
        SpaceTrackAuthenticationError
            If space-track.org authentication fails.
        """
        if not self.satellites:
            return []

        # Build space-track.org query
        epoch_start = self.epoch - timedelta(days=7)
        epoch_stop = self.epoch + timedelta(days=7)

        spacetrack_client_args = {"identity": self.spacetrack_user, "password": self.spacetrack_pwd}

        # set base_url when provided upstream to override space-track server to access
        if self.spacetrack_base_url is not None:
            spacetrack_client_args["base_url"] = self.spacetrack_base_url

        # Log into space-track.org
        with SpaceTrackClient(**spacetrack_client_args) as spacetrack_client:
            try:
                spacetrack_client.authenticate()
            except (AuthenticationError, HTTPStatusError) as e:
                raise SpaceTrackAuthenticationError("space-track.org authentication failed.") from e

            # Extract NORAD IDs for the query
            norad_ids = [sat["id"] for sat in self.satellites]

            # Fetch the TLEs between the requested epochs
            tletext = spacetrack_client.gp(
                norad_cat_id=norad_ids,
                orderby="epoch desc",
                format="tle",
                epoch=f">{epoch_start},<{epoch_stop}",
            )

        # Check if we got a return
        if tletext == "":
            return []

        # Split the TLEs into individual lines and pair line1/line2 entries.
        lines = tletext.splitlines()
        tle_pairs = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]

        # Build map from norad_id to satellite name
        sat_name_map = {sat["id"]: sat["name"] for sat in self.satellites}
        requested_ids = set(sat_name_map.keys())
        seen_ids: set[int] = set()
        tles: list[TLE] = []

        # Filter to the first occurrence of each requested NORAD ID (newest due to orderby parameter)
        for tle1, tle2 in tle_pairs:
            parsed_id = self._extract_norad_id_from_tle1(tle1)
            if parsed_id is None:
                logging.warning("Could not parse norad_id from space-track", {"tle1": tle1, "tle2": tle2})
                continue
            if parsed_id not in requested_ids:
                logging.warning(f"Received norad_id {parsed_id} TLE from space-track that was not requested")
                continue
            if parsed_id not in seen_ids:
                tles.append(
                    TLE(
                        satellite_name=sat_name_map.get(parsed_id),
                        norad_id=parsed_id,
                        tle1=tle1,
                        tle2=tle2,
                    )
                )

                seen_ids.add(parsed_id)

                # all requested satellite ids have been parsed into TLEs from the response
                if seen_ids == requested_ids:
                    break

        return tles


def get_tle(
    satellites: list[NoradSatellite],
    epoch: datetime,
    spacetrack_user: str | None = None,
    spacetrack_pwd: str | None = None,
    spacetrack_base_url: str | None = None,
) -> list[TLE]:
    """
    Gets the Two-Line Element (TLE) data for one or more satellites at a specific epoch.
    Credentials for space-track.org can be provided as arguments, or they can
    be set as environment variables SPACETRACK_USER and SPACETRACK_PWD.

    Parameters
    ----------
    satellites : list[NoradSatellite]
        List of satellites to query, each with 'name' and 'id' keys.
    epoch : datetime
        The epoch timestamp for which to retrieve the TLE data.
    spacetrack_user : str, optional
        space-Track.org username.
    spacetrack_pwd : str, optional
        space-Track.org password.
    spacetrack_base_url : str, optional
        Optional Space-Track API base URL override.

    Returns
    -------
    list[TLE]
        TLE objects matching the requested satellites. Empty if no data is
        found.

    Raises
    ------
    TypeError
        - satellites must be a list of dicts with 'name' and 'id' keys
        - each satellite must be a dict
        - each satellite dict must have 'name' and 'id' keys
        - satellite 'name' must be a string
        - satellite 'id' must be an integer
    ValueError
        - satellites list cannot be empty
    """

    tle = TLEFetch(
        satellites=satellites,
        epoch=epoch,
        spacetrack_user=spacetrack_user,
        spacetrack_pwd=spacetrack_pwd,
        spacetrack_base_url=spacetrack_base_url,
    )
    return tle.get()
