from typing import Callable, Literal

import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.moon_angle import MoonAngleConstraint


@pytest.fixture
def moon_angle_constraint() -> MoonAngleConstraint:
    """Fixture to provide an instance of MoonAngleConstraint for testing."""
    return MoonAngleConstraint(min_angle=21.0, max_angle=170.0)


@pytest.fixture
def moon_constraint_result(
    moon_angle_constraint: MoonAngleConstraint,
    sky_coord: SkyCoord,
    test_tle_ephemeris: Ephemeris,
    body_constraint_result_factory: Callable[[MoonAngleConstraint, SkyCoord, Ephemeris], np.ndarray],
) -> np.ndarray:
    """Constraint result over the full ephemeris timestamp."""
    return body_constraint_result_factory(moon_angle_constraint, sky_coord, test_tle_ephemeris)


@pytest.fixture
def moon_outside_constraint_coord(
    test_tle_ephemeris: Ephemeris,
    body_outside_constraint_coord_factory: Callable[[Ephemeris, Literal["sun", "moon"]], SkyCoord],
) -> SkyCoord:
    """Coordinate far from moon, expected outside moon-angle constraint."""
    return body_outside_constraint_coord_factory(test_tle_ephemeris, "moon")


@pytest.fixture
def moon_inside_constraint_coord(
    test_tle_ephemeris: Ephemeris,
    body_inside_constraint_coord_factory: Callable[[Ephemeris, Literal["sun", "moon"]], SkyCoord],
) -> SkyCoord:
    """Coordinate colocated with moon, expected inside moon-angle constraint."""
    return body_inside_constraint_coord_factory(test_tle_ephemeris, "moon")


@pytest.fixture
def moon_edge_constraint_coord(
    moon_angle_constraint: MoonAngleConstraint,
    test_tle_ephemeris: Ephemeris,
    body_edge_constraint_coord_factory: Callable[
        [MoonAngleConstraint, Ephemeris, Literal["sun", "moon"], int],
        SkyCoord,
    ],
) -> SkyCoord:
    """Coordinate near edge of moon-angle constraint."""
    return body_edge_constraint_coord_factory(
        moon_angle_constraint,
        test_tle_ephemeris,
        "moon",
        3,
    )
