from typing import Any

from .base import BaseSchema
from .coordinate import Coordinate


class Polygon(BaseSchema):
    """
    Class to represent a spherical polygon
    """

    coordinates: list[Coordinate]

    def model_post_init(self, __context: Any) -> None:
        """
        Pydantic post-init hook

        Post-Init validations.
            1.) a polygon must contain a list of coordinates
            1.) a polygon's first and final coordinates must be the same (wrapping)
            2.) a polygon has more than 3 unique coordinates
        """

        if not len(self.coordinates):
            raise ValueError("Invalid polygon, coordinates cannot be empty.")

        first_coordinate = self.coordinates[0]
        last_coordinate = self.coordinates[len(self.coordinates) - 1]

        if first_coordinate != last_coordinate:
            self.coordinates.append(first_coordinate)

        if len(self.coordinates) < 4:
            raise ValueError("Invalid polygon, must contain more than 3 unique coordinates to be a polygon")

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
