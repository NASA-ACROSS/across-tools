from typing import Any

import pytest

from across.tools.core.schemas.coordinate import Coordinate
from across.tools.core.schemas.polygon import Polygon


class TestPolygonSchema:
    """Test suite for the Polygon schema."""

    def test_valid_polygon(self, valid_polygon_data: dict[str, Any]) -> None:
        """Test Polygon with valid data."""
        polygon = Polygon(**valid_polygon_data)
        assert len(polygon.coordinates) == 4

    def test_empty_coordinates(self) -> None:
        """Test Polygon with empty coordinates list."""
        with pytest.raises(ValueError, match="Invalid polygon, coordinates cannot be empty."):
            Polygon(coordinates=[])

    def test_auto_close_polygon(self) -> None:
        """Test that polygon auto-closes when first and last points don't match."""
        coords = [Coordinate(ra=0, dec=0), Coordinate(ra=1, dec=1), Coordinate(ra=1, dec=0)]
        polygon = Polygon(coordinates=coords)
        assert polygon.coordinates[0] == polygon.coordinates[-1]
        assert len(polygon.coordinates) == 4

    def test_too_few_coordinates(self) -> None:
        """Test Polygon with less than 3 unique coordinates."""
        coords = [Coordinate(ra=0, dec=0), Coordinate(ra=1, dec=1), Coordinate(ra=0, dec=0)]
        with pytest.raises(ValueError, match="Invalid polygon, must contain more than 3 unique coordinates"):
            Polygon(coordinates=coords)

    def test_polygon_equality(self, valid_polygon_data: dict[str, Any]) -> None:
        """Test polygon equality comparison."""
        polygon1 = Polygon(**valid_polygon_data)
        polygon2 = Polygon(**valid_polygon_data)
        assert polygon1 == polygon2

    def test_polygon_inequality(self, valid_polygon_data: dict[str, Any]) -> None:
        """Test polygon inequality comparison."""
        polygon1 = Polygon(**valid_polygon_data)
        different_coords = [
            Coordinate(ra=0, dec=0),
            Coordinate(ra=2, dec=2),
            Coordinate(ra=2, dec=0),
            Coordinate(ra=0, dec=0),
        ]
        polygon2 = Polygon(coordinates=different_coords)
        assert polygon1 != polygon2

    def test_polygon_repr(self, valid_polygon_data: dict[str, Any]) -> None:
        """Test polygon string representation."""
        polygon = Polygon(**valid_polygon_data)
        repr_str = repr(polygon)
        assert repr_str.startswith("Polygon(")
        assert repr_str.endswith(")")
        assert "Coordinate" in repr_str

    def test_polygon_eq_non_polygon(self, valid_polygon_data: dict[str, Any]) -> None:
        """Test polygon equality with non-polygon object."""
        polygon = Polygon(**valid_polygon_data)
        assert (polygon == "not a polygon") is False

    def test_polygon_eq_different_length(self, valid_polygon_data: dict[str, Any]) -> None:
        """Test polygon equality with different length coordinates."""
        polygon1 = Polygon(**valid_polygon_data)
        coords = [
            Coordinate(ra=0, dec=0),
            Coordinate(ra=1, dec=1),
            Coordinate(ra=1, dec=0),
            Coordinate(ra=0.5, dec=0.5),
            Coordinate(ra=0, dec=0),
        ]
        polygon2 = Polygon(coordinates=coords)
        assert polygon1 != polygon2
