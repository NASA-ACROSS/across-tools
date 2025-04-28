from typing import Any

import pytest

from across.tools import Coordinate, Polygon
from across.tools.footprint import Footprint


class TestFootprintInstantiation:
    """
    Class to run set of Footprint Tests
    """

    @pytest.fixture(autouse=True)
    def setup(
        self, simple_polygon: Polygon, simple_footprint: Footprint, origin_coordinate: Coordinate
    ) -> None:
        """
        Init with fixtures
        """
        self.simple_polygon = simple_polygon
        self.simple_footprint = simple_footprint
        self.origin_coordinate = origin_coordinate

    def test_should_instantiate_footprint(self) -> None:
        """
        Should return the instance of a `Footprint` when instantiating
        """
        footprint = Footprint(detectors=[self.simple_polygon])
        assert isinstance(footprint, Footprint)

    def test_should_raise_value_error_with_invalid_detectors(self, invalid_detector: Any) -> None:
        """
        Should raise `ValueError` when instantiating with invalid detectors
        """
        with pytest.raises(ValueError):
            Footprint(detectors=invalid_detector)

    def test_should_return_true_on_footprint_equality(self) -> None:
        """
        Should return `true` when checking the equality on the same footprint
        """
        footprint = Footprint(detectors=[self.simple_polygon])
        assert footprint == self.simple_footprint

    def test_repr(self) -> None:
        """
        Should return a string representation of the Footprint object
        """
        assert (
            repr(self.simple_footprint)
            == "Footprint(\n\tPolygon(\n\t\tCoordinate(359.5, 0.5),\n\t\tCoordinate(0.5, 0.5),"
            + "\n\t\tCoordinate(0.5, -0.5),\n\t\tCoordinate(359.5, -0.5),"
            + "\n\t\tCoordinate(359.5, 0.5),\n\t),\n)"
        )

    def test_should_return_not_implemented_when_comparing_with_other_objects(self) -> None:
        """
        Should return `NotImplemented` when comparing with other objects
        """
        assert self.simple_footprint.__eq__("NotImplemented") == NotImplemented


class TestFootprintProjection:
    """
    Class to run set of `Footprint.project` tests
    """

    @pytest.fixture(autouse=True)
    def setup(
        self, simple_polygon: Polygon, simple_footprint: Footprint, origin_coordinate: Coordinate
    ) -> None:
        """
        Init with fixtures
        """

        self.simple_polygon = simple_polygon
        self.simple_footprint = simple_footprint
        self.origin_coordinate = origin_coordinate

    def test_should_return_footprint(self) -> None:
        """
        Project method should return a `Footprint` object
        """

        zero_projected_footprint = self.simple_footprint.project(self.origin_coordinate, 0)
        assert isinstance(zero_projected_footprint, Footprint)

    def test_zero_projection_should_be_equal_to_original(self) -> None:
        """
        Projection method with zero coordinate and zero roll angle should equal the original footprint
        """

        zero_projected_footprint = self.simple_footprint.project(self.origin_coordinate, 0)
        assert zero_projected_footprint == self.simple_footprint

    def test_fully_rotated_projection_should_be_equal_original(self) -> None:
        """
        Projection method with zero coordinate and 360 degree roll angle should equal original footprint
        """

        fully_rotated_footprint = self.simple_footprint.project(self.origin_coordinate, 360)
        assert fully_rotated_footprint == self.simple_footprint

    def test_invalid_roll_angle_should_raise_value_error(self, invalid_roll_angle: Any) -> None:
        """
        Projection method with invalid roll angles should raise value error
        """

        with pytest.raises(ValueError):
            self.simple_footprint.project(self.origin_coordinate, invalid_roll_angle)

    def test_should_equal_against_precalculated_result(self, precalculated_projections: Any) -> None:
        """
        Projection method with pre-calculated projections for simple footprint should equal precalculated
            results
        """

        projected_footprint = self.simple_footprint.project(
            precalculated_projections.coordinate, precalculated_projections.roll_angle
        )
        assert projected_footprint == precalculated_projections.projection


class TestFootprintQueryPixels:
    """
    Class to run set of `Footprint.query_pixels` tests
    """

    @pytest.fixture(autouse=True)
    def setup(self, simple_footprint: Footprint, ra45_dec45_coordinate: Coordinate) -> None:
        """
        Init with fixtures
        """

        self.simple_footprint = simple_footprint
        self.ra45_dec45_coordinate = ra45_dec45_coordinate

    def test_should_return_list(self) -> None:
        """
        Footprint.query_pixels should return a list
        """
        projected_footprint = self.simple_footprint.project(self.ra45_dec45_coordinate, 0)
        footprint_pixels = projected_footprint.query_pixels(order=9)

        assert isinstance(footprint_pixels, list)

    def test_should_return_list_of_ints(self) -> None:
        """
        Footprint.query_pixels should return a list of integers
        """
        projected_footprint = self.simple_footprint.project(self.ra45_dec45_coordinate, 0)
        footprint_pixels = projected_footprint.query_pixels(order=9)

        assert all([isinstance(pixel, int) for pixel in footprint_pixels])

    def test_should_return_unique_set(self) -> None:
        """
        Footprint.query_pixels should return a unique list of integers
        """
        projected_footprint = self.simple_footprint.project(self.ra45_dec45_coordinate, 0)
        footprint_pixels = projected_footprint.query_pixels(order=9)

        assert len(footprint_pixels) == len(set(footprint_pixels))

    def test_should_equal_against_precalculated_result(self, precalculated_hp_query_polygon: Any) -> None:
        """
        Footprint.query_pixels should equal precalculated result with same parameters
        """
        projected_footprint = self.simple_footprint.project(self.ra45_dec45_coordinate, 0)
        footprint_pixels = projected_footprint.query_pixels(order=9)

        assert precalculated_hp_query_polygon == footprint_pixels

    def test_should_raise_value_error_with_invalid_order(self, invalid_healpix_order: Any) -> None:
        """
        Footprint.query_pixels should raise `ValueError` with invalid healpix order values
        """
        with pytest.raises(ValueError):
            self.simple_footprint.query_pixels(order=invalid_healpix_order)

    def test_should_return_false_on_footprint_with_different_length(self, simple_polygon: Polygon) -> None:
        """
        Should return False when comparing footprints with different number of detectors
        """
        footprint1 = Footprint(detectors=[simple_polygon])
        footprint2 = Footprint(detectors=[simple_polygon, simple_polygon])
        assert footprint1 != footprint2
