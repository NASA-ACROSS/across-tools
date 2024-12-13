from __future__ import annotations


class Coordinate:
    """
    Class that represents a point in spherical space
    """

    ra: float
    dec: float

    def __init__(self, ra: float, dec: float) -> None:
        ra = round(360 + ra, 3) if ra < 0 else round(ra, 3)
        dec = round(dec, 3)

        invalid_ra = ra < -360 or ra > 360
        invalid_dec = dec < -90 or dec > 90

        if invalid_ra or invalid_dec:
            raise ValueError(
                "Invalid Coordinate object. \
                Coordinate.ra must be float (-360.0 >= a >= 360.0). \
                Coordinate.dec must be float (-90 >= a >= 90)"
            )

        self.ra = ra
        self.dec = dec

    def __repr__(self) -> str:
        """
        Overrides the print statement
        """
        return f"{self.__class__.__name__}({self.ra}, {self.dec})"

    def __eq__(self, other: object) -> bool:
        """
        Overrides the coordinate equals
        """
        if not isinstance(other, Coordinate):
            return NotImplemented

        ra_eq = self.ra == other.ra
        dec_eq = self.dec == other.dec
        return ra_eq and dec_eq

    def __hash__(self) -> int:
        """
        Overrides the hash function to enable the unique set function
        """
        return hash((self.ra, self.dec))


class Polygon:
    """
    Class to represent a spherical polygon
    """

    coordinates: list[Coordinate]

    def __init__(self, coordinates: list[Coordinate]) -> None:
        self.coordinates = coordinates

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


class RollAngle:
    """
    Class to represent and validate a roll angle
        constraint: Must be (-360.0 >= a >= 360.0)
    """

    value: float

    def __init__(self, value: float) -> None:
        invalid_value = value < -360 or value > 360
        if invalid_value:
            raise ValueError("Invalid value for RollAngle. Must be float (-360.0 >= a >= 360.0)")
        self.value = value


class HealpixOrder:
    """
    Class to represent and validate a Healpix Order
        constraint: Must be (0 >= a >= 13)
    """

    value: int

    def __init__(self, value: int) -> None:
        invalid_value = value < 0 or value > 13
        if invalid_value:
            raise ValueError("Invalid value for HealpixOrder. Must be int (0 >= a >= 13)")
        self.value = value
