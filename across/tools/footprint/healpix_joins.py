from ..core.math import find_duplicates
from .footprint import Footprint


def inner(footprints: list[Footprint], order: int = 10) -> list[int]:
    """
    Computes the inner join from a list of footprints.
    Returns the list of overlapping healpix-pixels (i.e. the duplicates)
    """
    total_pixels: list[int] = []

    for footprint in footprints:
        total_pixels.extend(footprint.query_pixels(order=order))

    return find_duplicates(total_pixels)


def outer(footprints: list[Footprint], order: int = 10) -> list[int]:
    """
    Computes the outer join from a list of footprints
    Returns the list of non-over lapping healpix pixels (i.e. the non-duplicates)
    """
    total_pixels: list[int] = []

    for footprint in footprints:
        total_pixels.extend(footprint.query_pixels(order=order))

    duplicates = find_duplicates(total_pixels)
    outer_pixels = list({x for x in total_pixels if x not in duplicates})

    return outer_pixels


def union(footprints: list[Footprint], order: int = 10) -> list[int]:
    """
    Computes the union of a list of footprints
    Returns the list of overlapping and non overlapping healpix pixels (i.e. all unique pixels)
    """
    total_pixels: list[int] = []

    for footprint in footprints:
        total_pixels.extend(footprint.query_pixels(order=order))

    return list(set(total_pixels))
