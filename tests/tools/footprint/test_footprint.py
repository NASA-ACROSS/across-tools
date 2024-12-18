from typing import Any

import pytest

from across.tools import Coordinate, Footprint, Polygon


def test_footprint_instantiation(simple_footprint: Footprint, simple_polygon: Polygon) -> None:
    """
    Tests the instantiation of a Footprint object, and its equivalence
    """
    footprint = Footprint(detectors=[simple_polygon])
    assert isinstance(footprint, Footprint), "Footprint instantiation type error"
    assert footprint == simple_footprint, "Footprint equivalence error"

    # test the detector types
    with pytest.raises(ValueError):
        Footprint(detectors=42)  # type: ignore
    # with pytest.raises(ValueError):
    #     Footprint(detectors=[])
    with pytest.raises(ValueError):
        Footprint(detectors=[42, 42])  # type: ignore


def test_footprint_projection(simple_coordinate: Coordinate, simple_footprint: Footprint) -> None:
    """
    Tests the projection of a footprint
    """
    zero_projected_footprint = simple_footprint.project(simple_coordinate, 0)
    assert isinstance(zero_projected_footprint, Footprint), "Footprint projection return type error"
    assert zero_projected_footprint == simple_footprint, "Footprint 0,0,0 projection equivalence error"

    fully_rotated_footprint = simple_footprint.project(simple_coordinate, 360)
    assert fully_rotated_footprint == simple_footprint, "Footprint 0,0,360 projection equivalence error"

    with pytest.raises(ValueError):
        simple_footprint.project(simple_coordinate, -361)
    with pytest.raises(ValueError):
        simple_footprint.project(simple_coordinate, 361)


def test_footprint_projection_expectation(
    precalculated_projections: Any, simple_footprint: Footprint
) -> None:
    """
    Tests the projection of a footprint with precalculated projections
        projection parameters:
            Coordinate(45, 0), roll_angle 0
            Coordinate(0, 45), roll_angle 0
            Coordinate(0, 0), roll_angle 45
        The `precalculated_projections` object is a pytest fixture with index fields:
            [0] -> Coordinate
            [1] -> RollAngle
            [2] -> precalculated projected Footprint
    """
    projected_footprint = simple_footprint.project(precalculated_projections[0], precalculated_projections[1])
    assert projected_footprint == precalculated_projections[2]


def test_footprint_query_pixels(
    simple_footprint: Footprint, ra45_dec45_coordinate: Coordinate, precalculated_hp_query_polygon: list[int]
) -> None:
    """
    Tests the footprint.query_pixels function
    """
    projected_footprint = simple_footprint.project(ra45_dec45_coordinate, 0)
    footprint_pixels = projected_footprint.query_pixels(order=9)

    assert isinstance(footprint_pixels, list), "Did not return a list"
    assert isinstance(footprint_pixels[0], int), "Did not return a list of integers"
    assert len(footprint_pixels) == len(set(footprint_pixels)), "Not a unique list of pixels was returned"
    assert precalculated_hp_query_polygon == footprint_pixels, "Did not return a correct result"

    with pytest.raises(ValueError):
        simple_footprint.query_pixels(order=15)
    with pytest.raises(ValueError):
        simple_footprint.query_pixels(order=-1)
