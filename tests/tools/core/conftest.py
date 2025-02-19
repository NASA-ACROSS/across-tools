from typing import Any

import pytest

from across.tools.core.schemas.base import BaseSchema
from across.tools.core.schemas.coordinate import Coordinate


class DummyModel(BaseSchema):
    """Test model for BaseSchema."""

    pass


class DummyModelTwo(BaseSchema):
    """Test model for BaseSchema."""

    pass


@pytest.fixture
def test_model_class() -> type[DummyModel]:
    """Return a DummyModel class."""
    return DummyModel


@pytest.fixture
def test_model() -> BaseSchema:
    """Return a DummyModel instance."""
    return DummyModel()


@pytest.fixture
def test_model_two() -> BaseSchema:
    """Return a DummyModel instance."""
    return DummyModelTwo()


@pytest.fixture
def valid_coordinates() -> list[Coordinate]:
    """Return a list of valid coordinates."""
    return [
        Coordinate(ra=0, dec=0),
        Coordinate(ra=1, dec=1),
        Coordinate(ra=1, dec=0),
        Coordinate(ra=0, dec=0),
    ]


@pytest.fixture
def valid_polygon_data(valid_coordinates: list[Coordinate]) -> dict[str, Any]:
    """Return a dictionary containing valid polygon data."""
    return {"coordinates": valid_coordinates}


@pytest.fixture
def coord_standard() -> Coordinate:
    """Fixture for a standard coordinate with RA=10.0, Dec=20.0."""
    return Coordinate(ra=10.0, dec=20.0)


@pytest.fixture
def coord_small_values() -> Coordinate:
    """Fixture for a coordinate with very small values."""
    return Coordinate(ra=0.0000123, dec=0.0000456)


@pytest.fixture
def coord_large_values() -> Coordinate:
    """Fixture for a coordinate with extreme values."""
    return Coordinate(ra=123.456789, dec=45.678901)


@pytest.fixture
def coord_negative_ra() -> Coordinate:
    """Fixture for a coordinate with negative RA."""
    return Coordinate(ra=-45.678901, dec=0.0)


@pytest.fixture
def coord_extreme() -> Coordinate:
    """Fixture for a coordinate with extreme values."""
    return Coordinate(ra=360, dec=90)


@pytest.fixture
def coord_extreme_negative() -> Coordinate:
    """Fixture for a coordinate with extreme negative values."""
    return Coordinate(ra=-360, dec=-90)
