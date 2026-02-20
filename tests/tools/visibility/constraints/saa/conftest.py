from datetime import datetime

import pytest
from shapely import Polygon

from across.tools.core.schemas.tle import TLE
from across.tools.ephemeris import Ephemeris
from across.tools.ephemeris.tle_ephem import TLEEphemeris
from across.tools.visibility.constraints.saa import SAAPolygonConstraint


@pytest.fixture
def saa_poly() -> Polygon:
    """Fixture for a basic SAA polygon."""
    return Polygon(
        [
            (39.0, -30.0),
            (36.0, -26.0),
            (28.0, -21.0),
            (6.0, -12.0),
            (-5.0, -6.0),
            (-21.0, 2.0),
            (-30.0, 3.0),
            (-45.0, 2.0),
            (-60.0, -2.0),
            (-75.0, -7.0),
            (-83.0, -10.0),
            (-87.0, -16.0),
            (-86.0, -23.0),
            (-83.0, -30.0),
        ]
    )


@pytest.fixture
def saa_polygon_constraint(saa_poly: Polygon) -> SAAPolygonConstraint:
    """Fixture for a basic SAAPolygonConstraint instance."""
    return SAAPolygonConstraint(polygon=saa_poly)


@pytest.fixture
def test_tle_ephemeris_no_compute(
    ephemeris_begin: datetime, ephemeris_end: datetime, ephemeris_step_size: int, test_tle: TLE
) -> Ephemeris:
    """Fixture for a non-computed TLE ephemeris instance."""
    return TLEEphemeris(begin=ephemeris_begin, end=ephemeris_end, step_size=ephemeris_step_size, tle=test_tle)
