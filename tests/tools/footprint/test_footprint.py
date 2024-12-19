from typing import Any

import pytest

from across.tools import Coordinate, Footprint, Polygon


def test_should_instantiate_footprint(simple_polygon: Polygon) -> None:
    """
    Should return the instance of a `Footprint` when instantiating
    """
    footprint = Footprint(detectors=[simple_polygon])
    assert isinstance(footprint, Footprint)


def test_should_raise_value_error_with_invalid_detectors(invalid_detector: Any) -> None:
    """
    Should raise `ValueError` when instantiating with invalid detectors
    """
    with pytest.raises(ValueError):
        Footprint(detectors=invalid_detector)


def test_should_return_true_on_footprint_equality(
    simple_polygon: Polygon, simple_footprint: Footprint
) -> None:
    """
    Should return `true` when checking the equality on the same footprint
    """
    footprint = Footprint(detectors=[simple_polygon])
    assert footprint == simple_footprint


def test_project_should_return_footprint(origin_coordinate: Coordinate, simple_footprint: Footprint) -> None:
    """
    Project method should return a `Footprint` object
    """
    zero_projected_footprint = simple_footprint.project(origin_coordinate, 0)
    assert isinstance(zero_projected_footprint, Footprint)


def test_zero_projection_calculation_should_be_equal_to_original(
    origin_coordinate: Coordinate, simple_footprint: Footprint
) -> None:
    """
    Projection method with zero coordinate and zero roll angle should equal the original footprint
    """
    zero_projected_footprint = simple_footprint.project(origin_coordinate, 0)
    assert zero_projected_footprint == simple_footprint


def test_fully_rotated_projection_should_be_equal_original(
    origin_coordinate: Coordinate, simple_footprint: Footprint
) -> None:
    """
    Projection method with zero coordinate and 360 degree roll angle should equal original footprint
    """
    fully_rotated_footprint = simple_footprint.project(origin_coordinate, 360)
    assert fully_rotated_footprint == simple_footprint


def test_invalid_roll_angle_should_raise_value_error(
    origin_coordinate: Coordinate, simple_footprint: Footprint, invalid_roll_angle: Any
) -> None:
    """
    Projection method with invalid roll angles should raise value error
    """

    with pytest.raises(ValueError):
        simple_footprint.project(origin_coordinate, invalid_roll_angle)


def test_footprint_projection_expectation_should_equal(
    precalculated_projections: Any, simple_footprint: Footprint
) -> None:
    """
    Projection method with pre-calculated projections for simple footprint should equal precalculated
        results
    """
    projected_footprint = simple_footprint.project(
        precalculated_projections.coordinate, precalculated_projections.roll_angle
    )
    assert projected_footprint == precalculated_projections.projection


def test_query_pixels_should_return_list(
    simple_footprint: Footprint, ra45_dec45_coordinate: Coordinate
) -> None:
    """
    Footprint.query_pixels should return a list
    """
    projected_footprint = simple_footprint.project(ra45_dec45_coordinate, 0)
    footprint_pixels = projected_footprint.query_pixels(order=9)

    assert isinstance(footprint_pixels, list)


def test_query_pixels_should_return_list_of_ints(
    simple_footprint: Footprint, ra45_dec45_coordinate: Coordinate
) -> None:
    """
    Footprint.query_pixels should return a list of integers
    """
    projected_footprint = simple_footprint.project(ra45_dec45_coordinate, 0)
    footprint_pixels = projected_footprint.query_pixels(order=9)

    assert all([isinstance(pixel, int) for pixel in footprint_pixels])


def test_query_pixels_should_return_unique_set(
    simple_footprint: Footprint, ra45_dec45_coordinate: Coordinate
) -> None:
    """
    Footprint.query_pixels should return a unique list of integers
    """
    projected_footprint = simple_footprint.project(ra45_dec45_coordinate, 0)
    footprint_pixels = projected_footprint.query_pixels(order=9)

    assert len(footprint_pixels) == len(set(footprint_pixels))


def test_query_pixels_against_precalculated_result(
    simple_footprint: Footprint, ra45_dec45_coordinate: Coordinate, precalculated_hp_query_polygon: list[int]
) -> None:
    """
    Footprint.query_pixels should equal precalculated result with same parameters
    """
    projected_footprint = simple_footprint.project(ra45_dec45_coordinate, 0)
    footprint_pixels = projected_footprint.query_pixels(order=9)

    assert precalculated_hp_query_polygon == footprint_pixels


def test_query_pixels_should_raise_value_error_with_invalid_order(
    simple_footprint: Footprint, invalid_healpix_order: Any
) -> None:
    """
    Footprint.query_pixels should raise `ValueError` with invalid healpix order values
    """
    with pytest.raises(ValueError):
        simple_footprint.query_pixels(order=invalid_healpix_order)
