from __future__ import annotations

from typing import Any

import numpy as np
import pydantic as pyd


class BaseSchema(pyd.BaseModel):
    """
    Base class for schemas.

    This class provides a base implementation for schemas and defines the `from_attributes` method.
    Subclasses can inherit from this class and override the `from_attributes` method to define their
        own schema logic.
    """

    model_config = pyd.ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    def __hash__(self) -> int:
        return hash((type(self),) + tuple(self.__dict__.values()))


class Coordinate(BaseSchema):
    """
    Class that represents a point in spherical space
    """

    ra: float = pyd.Field(ge=-360, le=360)
    dec: float = pyd.Field(ge=-90, le=90)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-Init validations.
            Ensure the RA is positive, and rounded to an appropriate precision
        """
        self.ra = round(360 + self.ra, 5) if self.ra < 0 else round(self.ra, 5)
        self.dec = round(self.dec, 5)

    def __repr__(self) -> str:
        """
        Overrides the print statement
        """
        return f"{self.__class__.__name__}(ra={self.ra}, dec={self.dec})"

    def __eq__(self, other: object) -> bool:
        """
        Overrides the coordinate equals
        """
        if not isinstance(other, Coordinate):
            return NotImplemented

        ra_eq = np.isclose(self.ra, other.ra, atol=1e-5)
        dec_eq = np.isclose(self.dec, other.dec, atol=1e-5)
        return bool(ra_eq and dec_eq)


class Polygon(BaseSchema):
    """
    Class to represent a spherical polygon
    """

    coordinates: list[Coordinate]

    def __repr__(self) -> str:
        """
        Overrides the print statement
        """
        statement = f"{self.__class__.__name__}(\n"

        for coordinate in self.coordinates:
            statement += f"\t{coordinate.__class__.__name__}({coordinate.ra}, {coordinate.dec}),\n"

        statement += ")"

        return statement

    def __eq__(self, other: object) -> bool:
        """
        Overrides the coordinate equals
        """
        if not isinstance(other, Polygon):
            return NotImplemented

        if len(self.coordinates) != len(other.coordinates):
            return False

        else:
            equivalence: list[bool] = []
            for coordinate_iterable in range(len(self.coordinates)):
                equivalence.append(
                    self.coordinates[coordinate_iterable] == other.coordinates[coordinate_iterable]
                )
            return all(equivalence)


class RollAngle(BaseSchema):
    """
    Class to represent and validate a roll angle
        constraint: Must be (-360.0 >= a >= 360.0)
    """

    value: float = pyd.Field(ge=-360, le=360)


class HealpixOrder(BaseSchema):
    """
    Class to represent and validate a Healpix Order
        constraint: Must be (0 >= a >= 13)
    """

    value: int = pyd.Field(gt=0, lt=13, default=10)
