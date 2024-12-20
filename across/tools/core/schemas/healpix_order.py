from pydantic import Field

from .base import BaseSchema


class HealpixOrder(BaseSchema):
    """
    Class to represent and validate a Healpix Order
        constraint: Must be (0 >= a >= 13)
    """

    value: int = Field(gt=0, lt=13, default=10)
