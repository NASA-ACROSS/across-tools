import pytest

from across.tools import Coordinate, Footprint, inner, outer, union


def test_healpix_inner_join(simple_footprint: Footprint) -> None:
    """
    Tests the healpix_joins.inner()
    """
    # Should return a list of ints as healpix pixels when it is called with any overlap.
    overlapping_footprints = [simple_footprint, simple_footprint]
    overlapping_inner_join_pixels = inner(overlapping_footprints, order=9)
    assert isinstance(overlapping_inner_join_pixels, list), "healpix_joins.inner return type error"
    assert isinstance(overlapping_inner_join_pixels[0], int), "healpix_joins.inner return type error"

    # Should return an empty list when the list of footprints do not overlap.
    non_overlapping_footprints = [
        simple_footprint,
        simple_footprint.project(coordinate=Coordinate(10, 10), roll_angle=0.0),
    ]
    non_overlapping_inner_join_pixels = inner(non_overlapping_footprints, order=9)
    assert len(non_overlapping_inner_join_pixels) == 0, "healpix_joins.inner non overlap should return empty"

    empty_footprint_list_pixels = inner([], order=9)
    # Should return an empty list when the list of footprint list are empty.
    assert len(empty_footprint_list_pixels) == 0, "healpix_joins.inner empty footprints should return empty"

    # Should raise ValueException when order is out of bounds of 0 <= order < 13.
    with pytest.raises(ValueError):
        inner(footprints=[simple_footprint], order=15)
    with pytest.raises(ValueError):
        inner(footprints=[simple_footprint], order=-1)


def test_healpix_outer_join(simple_footprint: Footprint) -> None:
    """
    Tests the healpix_joins.outer()
    """

    # Should return a list of ints as healpix pixels when it is called with any non overlap.
    non_overlapping_footprints = [
        simple_footprint,
        simple_footprint.project(coordinate=Coordinate(10, 10), roll_angle=0.0),
    ]
    non_overlapping_outer_join_pixels = outer(non_overlapping_footprints, order=9)
    assert isinstance(non_overlapping_outer_join_pixels, list)
    assert isinstance(non_overlapping_outer_join_pixels[0], int)

    # Should return an empty list when the list of footprints do completely overlap.
    overlapping_footprints = [simple_footprint, simple_footprint]
    overlapping_outer_join_pixels = outer(overlapping_footprints, order=9)
    assert len(overlapping_outer_join_pixels) == 0

    # Should return an empty list when the list of footprint list are empty.
    empty_footprint_list_pixels = outer([], order=9)
    assert len(empty_footprint_list_pixels) == 0

    # Should raise ValueException when order is out of bounds of 0 <= order < 13.
    with pytest.raises(ValueError):
        outer(footprints=[simple_footprint], order=15)
    with pytest.raises(ValueError):
        outer(footprints=[simple_footprint], order=-1)


def test_healpix_union_join(simple_footprint: Footprint) -> None:
    """
    Tests the healpix_joins.union()
    """

    # Should return a list of ints as healpix pixels when it is called.
    non_overlapping_footprints = [
        simple_footprint,
        simple_footprint.project(coordinate=Coordinate(10, 10), roll_angle=0.0),
    ]
    non_overlapping_outer_union_pixels = union(non_overlapping_footprints, order=9)
    assert isinstance(non_overlapping_outer_union_pixels, list)
    assert isinstance(non_overlapping_outer_union_pixels[0], int)

    # Should return an empty list when the list of footprint list are empty.
    empty_footprint_list_pixels = union([], order=9)
    assert len(empty_footprint_list_pixels) == 0

    # Should raise ValueException when order is out of bounds of 0 <= order < 13.
    with pytest.raises(ValueError):
        union(footprints=[simple_footprint], order=15)
    with pytest.raises(ValueError):
        union(footprints=[simple_footprint], order=-1)
