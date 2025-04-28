from typing import Any

import pytest

from across.tools import Coordinate, Polygon
from across.tools.core.schemas import BaseSchema
from across.tools.footprint import Footprint


@pytest.fixture
def simple_polygon() -> Polygon:
    """
    Returns a very simple polygon center at 0, 0 with side = 1.0
    """
    coordinates = [
        Coordinate(ra=-0.5, dec=0.5),
        Coordinate(ra=0.5, dec=0.5),
        Coordinate(ra=0.5, dec=-0.5),
        Coordinate(ra=-0.5, dec=-0.5),
        Coordinate(ra=-0.5, dec=0.5),
    ]
    return Polygon(coordinates=coordinates)


@pytest.fixture
def simple_footprint(simple_polygon: Polygon) -> Footprint:
    """
    Instantiates  a simple footprint from a simple polygon
    """
    return Footprint(detectors=[simple_polygon])


@pytest.fixture(params=["bad detector", 42, [42, 42]])
def invalid_detector(request: pytest.FixtureRequest) -> Any:
    """
    Parameters to be passed into the projection tests
    """
    return request.param


@pytest.fixture
def origin_coordinate() -> Coordinate:
    """
    Instantiates a coordinate at ra=0, dec=0
    """
    return Coordinate(ra=0, dec=0)


@pytest.fixture(params=["bad roll_angle", -361, 361])
def invalid_roll_angle(request: pytest.FixtureRequest) -> Any:
    """
    Parameters to be passed into the projection tests
    """
    return request.param


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
        detectors=[
            Polygon(
                coordinates=[
                    Coordinate(ra=44.5, dec=0.5),
                    Coordinate(ra=45.5, dec=0.5),
                    Coordinate(ra=45.5, dec=-0.5),
                    Coordinate(ra=44.5, dec=-0.5),
                    Coordinate(ra=44.5, dec=0.5),
                ]
            ),
        ]
    )


def simple_footprint_projection_ra0_dec45_pos0() -> Footprint:
    """
    Instantiates a precalculated projected simple footprint at ra=0, dec=45, roll=0
    """
    return Footprint(
        detectors=[
            Polygon(
                coordinates=[
                    Coordinate(ra=359.28669, dec=45.4978),
                    Coordinate(ra=0.71331, dec=45.4978),
                    Coordinate(ra=0.70097, dec=44.49784),
                    Coordinate(ra=359.29903, dec=44.49784),
                    Coordinate(ra=359.28669, dec=45.4978),
                ]
            )
        ]
    )


def simple_footprint_projection_ra0_dec0_pos45() -> Footprint:
    """
    Instantiates a precalculated projected simple footprint at ra=0, dec=0, roll=45
    """
    return Footprint(
        detectors=[
            Polygon(
                coordinates=[
                    Coordinate(ra=359.2929, dec=1e-05),
                    Coordinate(ra=359.99999, dec=0.7071),
                    Coordinate(ra=0.7071, dec=0.0),
                    Coordinate(ra=0.0, dec=-0.7071),
                    Coordinate(ra=359.2929, dec=0.0),
                ]
            ),
        ]
    )


class PrecalculatedProjections(BaseSchema):
    """
    Class to represent a pre-calculated projection
    """

    coordinate: Coordinate
    roll_angle: float
    projection: Footprint


@pytest.fixture(
    params=[
        PrecalculatedProjections(
            coordinate=Coordinate(ra=45, dec=0),
            roll_angle=0,
            projection=simple_footprint_projection_ra45_dec0_pos0(),
        ),
        PrecalculatedProjections(
            coordinate=Coordinate(ra=0, dec=45),
            roll_angle=0,
            projection=simple_footprint_projection_ra0_dec45_pos0(),
        ),
        PrecalculatedProjections(
            coordinate=Coordinate(ra=0, dec=0),
            roll_angle=45,
            projection=simple_footprint_projection_ra0_dec0_pos45(),
        ),
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


@pytest.fixture(params=["bad healpix_order", -5, 15])
def invalid_healpix_order(request: pytest.FixtureRequest) -> Any:
    """
    Parameters to be passed into the projection tests
    """
    return request.param
