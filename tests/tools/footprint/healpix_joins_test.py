from typing import Any

import pytest

from across.tools import Coordinate
from across.tools.footprint import Footprint, inner, outer, union


class TestInnerJoin:
    """
    Class to run set of healpix_joins.inner tests
    """

    @pytest.fixture(autouse=True)
    def setup(self, simple_footprint: Footprint) -> None:
        """
        Init with fixtures
        """
        self.simple_footprint = simple_footprint

    def test_should_return_list(self) -> None:
        """
        Should return a list when it is called with any overlap.
        """
        overlapping_footprints = [self.simple_footprint, self.simple_footprint]
        overlapping_inner_join_pixels = inner(overlapping_footprints, order=9)
        assert isinstance(overlapping_inner_join_pixels, list)

    def test_should_return_list_of_ints(self) -> None:
        """
        Should return a list of ints when it is called with any overlap.
        """
        overlapping_footprints = [self.simple_footprint, self.simple_footprint]
        overlapping_inner_join_pixels = inner(overlapping_footprints, order=9)
        assert all([isinstance(pixel, int) for pixel in overlapping_inner_join_pixels])

    def test_should_return_empty_with_non_overlapping_footprints(self) -> None:
        """
        Should return an empty list when the list of footprints do not overlap.
        """
        non_overlapping_footprints = [
            self.simple_footprint,
            self.simple_footprint.project(coordinate=Coordinate(ra=10, dec=10), roll_angle=0.0),
        ]
        non_overlapping_inner_join_pixels = inner(non_overlapping_footprints, order=9)
        assert len(non_overlapping_inner_join_pixels) == 0

    def test_should_return_empty_with_empty_footprint(self) -> None:
        """
        Should return an empty list when the list of footprint list are empty.
        """
        empty_footprint_list_pixels = inner([], order=9)
        assert len(empty_footprint_list_pixels) == 0

    def test_should_raise_value_error_with_invalid_healpix_order(self, invalid_healpix_order: Any) -> None:
        """
        Should raise ValueException when order is out of bounds of 0 <= order < 13.
        """
        with pytest.raises(ValueError):
            inner(footprints=[self.simple_footprint], order=invalid_healpix_order)


class TestOuterJoin:
    """
    Class to run set of healpix_joins.outer tests
    """

    @pytest.fixture(autouse=True)
    def setup(self, simple_footprint: Footprint) -> None:
        """
        Init with fixtures
        """
        self.simple_footprint = simple_footprint

    def test_should_return_list(self) -> None:
        """
        Should return a list when it is called with any non overlap.
        """
        non_overlapping_footprints = [
            self.simple_footprint,
            self.simple_footprint.project(coordinate=Coordinate(ra=10, dec=10), roll_angle=0.0),
        ]
        non_overlapping_outer_join_pixels = outer(non_overlapping_footprints, order=9)
        assert isinstance(non_overlapping_outer_join_pixels, list)

    def test_should_return_list_of_ints(self) -> None:
        """
        Should return a list of ints when it is called with any overlap.
        """
        non_overlapping_footprints = [
            self.simple_footprint,
            self.simple_footprint.project(coordinate=Coordinate(ra=10, dec=10), roll_angle=0.0),
        ]
        non_overlapping_outer_join_pixels = outer(non_overlapping_footprints, order=9)
        assert all([isinstance(pixel, int) for pixel in non_overlapping_outer_join_pixels])

    def test_should_return_empty_with_completely_overlapping_footprints(self) -> None:
        """
        Should return an empty list when the list of footprints do completely overlap.
        """
        overlapping_footprints = [self.simple_footprint, self.simple_footprint]
        overlapping_outer_join_pixels = outer(overlapping_footprints, order=9)
        assert len(overlapping_outer_join_pixels) == 0

    def test_should_return_empty_with_empty_footprint(self) -> None:
        """
        Should return an empty list when the list of footprint list are empty.
        """
        empty_footprint_list_pixels = outer([], order=9)
        assert len(empty_footprint_list_pixels) == 0

    def test_should_raise_value_error_with_invalid_healpix_order(self, invalid_healpix_order: Any) -> None:
        """
        Should raise ValueException when order is out of bounds of 0 <= order < 13.
        """
        with pytest.raises(ValueError):
            outer(footprints=[self.simple_footprint], order=invalid_healpix_order)


class TestUnionJoin:
    """
    Class to run set of healpix_joins.union tests
    """

    @pytest.fixture(autouse=True)
    def setup(self, simple_footprint: Footprint) -> None:
        """
        Init with fixtures
        """
        self.simple_footprint = simple_footprint

    def test_should_return_list(self) -> None:
        """
        Should return a list when it is called with any footprint.
        """
        non_overlapping_footprints = [
            self.simple_footprint,
            self.simple_footprint.project(coordinate=Coordinate(ra=10, dec=10), roll_angle=0.0),
        ]
        non_overlapping_outer_join_pixels = union(non_overlapping_footprints, order=9)
        assert isinstance(non_overlapping_outer_join_pixels, list)

    def test_should_return_list_of_ints(self) -> None:
        """
        Should return a list of ints when it is called with any overlap.
        """
        non_overlapping_footprints = [
            self.simple_footprint,
            self.simple_footprint.project(coordinate=Coordinate(ra=10, dec=10), roll_angle=0.0),
        ]
        non_overlapping_outer_join_pixels = union(non_overlapping_footprints, order=9)
        assert all([isinstance(pixel, int) for pixel in non_overlapping_outer_join_pixels])

    def test_should_return_empty_with_empty_footprint(self) -> None:
        """
        Should return an empty list when the list of footprint list are empty.
        """
        empty_footprint_list_pixels = union([], order=9)
        assert len(empty_footprint_list_pixels) == 0

    def test_should_raise_value_error_with_invalid_healpix_order(self, invalid_healpix_order: Any) -> None:
        """
        Should raise ValueException when order is out of bounds of 0 <= order < 13.
        """
        with pytest.raises(ValueError):
            union(footprints=[self.simple_footprint], order=invalid_healpix_order)
