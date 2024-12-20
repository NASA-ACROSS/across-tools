from pydantic import Field

from .base import BaseSchema


class RollAngle(BaseSchema):
    """
    Class to represent and validate a roll angle
        constraint: Must be (-360.0 >= a >= 360.0)
    """

    value: float = Field(ge=-360, le=360)
