from typing import Any

import pytest

from across.tools import Coordinate, Footprint, Polygon


@pytest.fixture
def simple_polygon() -> Polygon:
    """
    Returns a very simple polygon center at 0, 0 with side = 1.0
    """
    coordinates = [
        Coordinate(-0.5, 0.5),
        Coordinate(0.5, 0.5),
        Coordinate(0.5, -0.5),
        Coordinate(-0.5, -0.5),
        Coordinate(-0.5, 0.5),
    ]
    return Polygon(coordinates)


@pytest.fixture
def simple_footprint() -> Footprint:
    """
    Instantiates  a simple footprint from a simple polygon
    """
    coordinates = [
        Coordinate(-0.5, 0.5),
        Coordinate(0.5, 0.5),
        Coordinate(0.5, -0.5),
        Coordinate(-0.5, -0.5),
        Coordinate(-0.5, 0.5),
    ]
    polygon = Polygon(coordinates)
    return Footprint(detectors=[polygon])


@pytest.fixture
def simple_coordinate() -> Coordinate:
    """
    Instantiates a coordinate at ra=0, dec=0
    """
    return Coordinate(ra=0, dec=0)


@pytest.fixture
def ra45_dec45_coordinate() -> Coordinate:
    """
    Instantiates a coordinate at ra=45, dec=45
    """
    return Coordinate(ra=45, dec=45)


def simple_footprint_projection_ra45_dec0_pos0() -> Footprint:
    """
    Instantiates a precalculated projected simple footprint at ra=45, dec=0, roll=0
    """
    return Footprint(
        [
            Polygon(
                [
                    Coordinate(44.5, 0.5),
                    Coordinate(45.5, 0.5),
                    Coordinate(45.5, -0.5),
                    Coordinate(44.5, -0.5),
                    Coordinate(44.5, 0.5),
                ]
            ),
        ]
    )


def simple_footprint_projection_ra0_dec45_pos0() -> Footprint:
    """
    Instantiates a precalculated projected simple footprint at ra=0, dec=45, roll=0
    """
    return Footprint(
        [
            Polygon(
                [
                    Coordinate(359.287, 45.498),
                    Coordinate(0.713, 45.498),
                    Coordinate(0.701, 44.498),
                    Coordinate(359.299, 44.498),
                    Coordinate(359.287, 45.498),
                ]
            ),
        ]
    )


def simple_footprint_projection_ra0_dec0_pos45() -> Footprint:
    """
    Instantiates a precalculated projected simple footprint at ra=0, dec=0, roll=45
    """
    return Footprint(
        [
            Polygon(
                [
                    Coordinate(359.293, 0.0),
                    Coordinate(360.0, 0.707),
                    Coordinate(0.707, -0.0),
                    Coordinate(0.0, -0.707),
                    Coordinate(359.293, 0.0),
                ]
            ),
        ]
    )


@pytest.fixture(
    params=[
        [Coordinate(ra=45, dec=0), 0, simple_footprint_projection_ra45_dec0_pos0()],
        [Coordinate(ra=0, dec=45), 0, simple_footprint_projection_ra0_dec45_pos0()],
        [Coordinate(ra=0, dec=0), 45, simple_footprint_projection_ra0_dec0_pos45()],
    ]
)
def precalculated_projections(request: pytest.FixtureRequest) -> Any:
    """
    Parameters to be passed into the projection tests
    """
    return request.param


@pytest.fixture()
def precalculated_hp_query_polygon() -> list[int]:
    """
    Precalculated hp.query_polygon for a simple footprint projected to Coordinate(45, 45)
    """
    return [
        463921,
        463922,
        463923,
        463924,
        463925,
        463926,
        456247,
        456248,
        456249,
        456250,
        456251,
        456252,
        456253,
        456254,
        463927,
        463928,
        471659,
        471660,
        471661,
        471662,
        471663,
        471664,
        471665,
        471666,
        448637,
        448638,
        448639,
        448640,
        448641,
        448642,
        448643,
        448644,
        461996,
        461997,
        461998,
        461999,
        462000,
        462001,
        462002,
        462003,
        462004,
        454338,
        454339,
        454340,
        454341,
        454342,
        454343,
        454344,
        454345,
        454346,
        469718,
        469719,
        469720,
        469721,
        469722,
        469723,
        469724,
        469725,
        469726,
        460076,
        460077,
        460078,
        460079,
        460080,
        460081,
        460082,
        460083,
        467782,
        467783,
        467784,
        467785,
        467786,
        467787,
        467788,
        467789,
        452434,
        452435,
        452436,
        452437,
        452438,
        452439,
        452440,
        452441,
        458159,
        458160,
        458161,
        458162,
        458163,
        458164,
        458165,
        458166,
        458167,
        465849,
        465850,
        465851,
        465852,
        465853,
        465854,
        465855,
        465856,
        465857,
        450533,
        450534,
        450535,
        450536,
        450537,
        450538,
        450539,
        450540,
        450541,
    ]
