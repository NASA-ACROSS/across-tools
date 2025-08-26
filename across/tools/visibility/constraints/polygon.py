from pydantic import field_serializer, field_validator
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

    @field_validator("polygon", mode="before")
    @classmethod
    def validate_polygon(
        cls, v: Polygon | list[tuple[float, ...]] | tuple[tuple[float, ...], ...] | None
    ) -> Polygon | None:
        """Validate and convert polygon input"""
        if v is None:
            return v
        if isinstance(v, Polygon):
            return v
        if isinstance(v, (list, tuple)):
            return Polygon(v)
        raise ValueError(f"Cannot convert {type(v)} to Polygon")
