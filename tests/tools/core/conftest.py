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
