from pydantic import field_serializer
from shapely import Polygon

from .base import ConstraintABC


class PolygonConstraint(ConstraintABC):
    """
    Mixin class for constraints that are defined by a polygon. Mostly provides
    serialization and validation for the polygon.
    """

    polygon: Polygon | None = None

    @field_serializer("polygon")
    def serialize_polygon(self, polygon: Polygon) -> list[tuple[float, ...]]:
        """Serialize the polygon to a list of tuples"""
        return [co for co in polygon.exterior.coords]
