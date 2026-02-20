from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.sun_angle import SunAngleConstraint


@pytest.fixture
def sun_angle_constraint() -> SunAngleConstraint:
    """Fixture to provide an instance of SunAngleConstraint for testing."""
    return SunAngleConstraint(min_angle=45.0, max_angle=170.0)


@pytest.fixture
def sun_constraint_result(
    sun_angle_constraint: SunAngleConstraint,
    sky_coord: SkyCoord,
    test_tle_ephemeris: Ephemeris,
    body_constraint_result_factory: Callable[
        [SunAngleConstraint, SkyCoord, Ephemeris], npt.NDArray[np.bool_]
    ],
) -> npt.NDArray[np.bool_]:
    """Constraint result over the full ephemeris timestamp."""
    return body_constraint_result_factory(sun_angle_constraint, sky_coord, test_tle_ephemeris)


@pytest.fixture
def sun_outside_constraint_coord(
    test_tle_ephemeris: Ephemeris,
    body_outside_constraint_coord_factory: Callable[[Ephemeris, Literal["sun", "moon"]], SkyCoord],
) -> SkyCoord:
    """Coordinate far from sun, expected outside sun-angle constraint."""
    return body_outside_constraint_coord_factory(test_tle_ephemeris, "sun")


@pytest.fixture
def sun_inside_constraint_coord(
    test_tle_ephemeris: Ephemeris,
    body_inside_constraint_coord_factory: Callable[[Ephemeris, Literal["sun", "moon"]], SkyCoord],
) -> SkyCoord:
    """Coordinate colocated with sun, expected inside sun-angle constraint."""
    return body_inside_constraint_coord_factory(test_tle_ephemeris, "sun")


@pytest.fixture
def sun_edge_constraint_coord(
    sun_angle_constraint: SunAngleConstraint,
    test_tle_ephemeris: Ephemeris,
    body_edge_constraint_coord_factory: Callable[
        [SunAngleConstraint, Ephemeris, Literal["sun", "moon"], int],
        SkyCoord,
    ],
) -> SkyCoord:
    """Coordinate near edge of sun-angle constraint."""
    return body_edge_constraint_coord_factory(
        sun_angle_constraint,
        test_tle_ephemeris,
        "sun",
        2,
    )
