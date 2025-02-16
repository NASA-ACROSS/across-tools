from datetime import datetime
from typing import Any

import numpy as np
from pydantic import Field, model_validator

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


class DateRangeSchema(BaseSchema):
    """Schema that defines date range, which is optional

    Parameters
    ----------
    begin
        The beginning date of the range, by default None
    end
        The end date of the range, by default None

    Methods
    -------
    check_dates(data: Any) -> Any
        Validates the date range and ensures that the begin and end dates are set correctly.

    """

    begin: datetime
    end: datetime

    @model_validator(mode="after")
    @classmethod
    def check_dates(cls, data: Any) -> Any:
        """Validates the date range and ensures that the begin and end dates are set correctly.

        Parameters
        ----------
        data
            The data to be validated.

        Returns
        -------
        Any
            The validated data.
        """
        if data.begin != data.end:
            assert data.begin <= data.end, "End date should not be before begin"

        return data
