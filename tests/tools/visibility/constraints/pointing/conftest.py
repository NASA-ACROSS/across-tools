from datetime import datetime, timedelta

import astropy.units as u  # type: ignore[import-untyped]
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]

from across.tools.core.schemas import Coordinate, Polygon
from across.tools.footprint import Footprint, Pointing
from across.tools.visibility.constraints.pointing import PointingConstraint


@pytest.fixture
def origin_sky_coord() -> SkyCoord:
    """Create a SkyCoord located at the center of the test pointing footprint."""
    return SkyCoord(ra=10 * u.deg, dec=10 * u.deg)


@pytest.fixture
def pointing_constraint(ephemeris_begin: datetime, ephemeris_end: datetime) -> PointingConstraint:
    """Create a PointingConstraint with a simple square footprint around origin_sky_coord."""
    footprint = Footprint(
        detectors=[
            Polygon(
                coordinates=[
                    Coordinate(ra=9.0, dec=9.0),
                    Coordinate(ra=11.0, dec=9.0),
                    Coordinate(ra=11.0, dec=11.0),
                    Coordinate(ra=9.0, dec=11.0),
                ]
            )
        ]
    )

    pointing = Pointing(
        footprint=footprint,
        start_time=ephemeris_begin - timedelta(minutes=1),
        end_time=ephemeris_end + timedelta(minutes=1),
    )

    return PointingConstraint(pointings=[pointing])
