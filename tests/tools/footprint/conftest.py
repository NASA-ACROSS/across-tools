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
        197185,
        197187,
        197188,
        197189,
        197190,
        197191,
        197196,
        197197,
        197199,
        197200,
        197201,
        197202,
        197203,
        197204,
        197205,
        197206,
        197207,
        197208,
        197209,
        197210,
        197211,
        197212,
        197213,
        197214,
        197215,
        197232,
        197233,
        197235,
        197236,
        197237,
        197238,
        197239,
        197244,
        197245,
        196831,
        196847,
        196851,
        196852,
        196853,
        196854,
        196855,
        196856,
        196857,
        196858,
        196859,
        196860,
        196861,
        196862,
        196863,
        197376,
        197377,
        197378,
        197379,
        197380,
        197381,
        197382,
        197383,
        197384,
        197385,
        197386,
        197387,
        197388,
        197389,
        197390,
        197392,
        197393,
        197394,
        197408,
        197409,
        197410,
        196994,
        196995,
        197000,
        197001,
        197002,
        197003,
        197004,
        197006,
        197007,
        197024,
        197025,
        197026,
        197027,
        197028,
        197029,
        197030,
        197031,
        197032,
        197033,
        197034,
        197035,
        197036,
        197037,
        197038,
        197039,
        197040,
        197042,
        197043,
        197048,
        197049,
        197050,
        197051,
        197052,
        197054,
    ]


@pytest.fixture(params=["bad healpix_order", -5, 15])
def invalid_healpix_order(request: pytest.FixtureRequest) -> Any:
    """
    Parameters to be passed into the projection tests
    """
    return request.param
