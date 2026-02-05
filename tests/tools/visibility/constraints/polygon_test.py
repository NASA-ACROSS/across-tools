from typing import Literal

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from shapely.geometry import Polygon as ShapelyPolygon

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.ephemeris.base import Ephemeris
from across.tools.visibility.constraints.polygon import PolygonConstraint


class MockConstraint(PolygonConstraint):
    """Dummy constraint class for testing"""

    short_name: str = "Test"
    name: Literal[ConstraintType.TEST] = ConstraintType.TEST
    polygon: ShapelyPolygon | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """Dummy call method for testing purposes."""
        return np.zeros(len(time), dtype=bool)


class TestPolygonConstraint:
    """Test suite for the PolygonConstraint class."""

    def test_polygon_constraint_init_with_polygon(self) -> None:
        """Test MockConstraint initialization with a Polygon."""

        shapely_polygon = ShapelyPolygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        constraint = MockConstraint(polygon=shapely_polygon)
        assert constraint.polygon == shapely_polygon

    def test_polygon_constraint_init_with_none(self) -> None:
        """Test MockConstraint initialization with None."""

        constraint = MockConstraint(polygon=None)
        assert constraint.polygon is None

    def test_polygon_constraint_init_default(self) -> None:
        """Test MockConstraint initialization with default value."""

        constraint = MockConstraint()
        assert constraint.polygon is None

    def test_serialize_polygon(self) -> None:
        """Test polygon serialization to list of tuples."""

        shapely_polygon = ShapelyPolygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        constraint = MockConstraint(polygon=shapely_polygon)
        serialized = constraint.serialize_polygon(shapely_polygon)

        expected = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        assert serialized == expected

    def test_serialize_polygon_with_holes(self) -> None:
        """Test polygon serialization for polygon with holes (exterior only)."""

        exterior = [(0, 0), (4, 0), (4, 4), (0, 4)]
        hole = [(1, 1), (3, 1), (3, 3), (1, 3)]
        shapely_polygon = ShapelyPolygon(exterior, [hole])
        constraint = MockConstraint(polygon=shapely_polygon)
        serialized = constraint.serialize_polygon(shapely_polygon)

        # Should only serialize exterior coordinates
        expected = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)]
        assert serialized == expected

    def test_serialize_triangle_polygon(self) -> None:
        """Test polygon serialization for a triangle."""

        shapely_polygon = ShapelyPolygon([(0, 0), (2, 0), (1, 2)])
        constraint = MockConstraint(polygon=shapely_polygon)
        serialized = constraint.serialize_polygon(shapely_polygon)

        expected = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0), (0.0, 0.0)]
        assert serialized == expected
