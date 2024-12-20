from typing import Any

import numpy as np
from pydantic import Field

from .base import BaseSchema


class Coordinate(BaseSchema):
    """
    Class that represents a point in spherical space
    """

    ra: float = Field(ge=-360, le=360)
    dec: float = Field(ge=-90, le=90)

    def model_post_init(self, __context: Any) -> None:
        """
        Pydantic post-init hook

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
