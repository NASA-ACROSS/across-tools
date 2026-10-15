from pydantic import Field

from .base import BaseSchema


class RollAngleConstrainingObject(BaseSchema):
    """
    Represents an object that constrains the roll angle.

    Parameters
    ----------
    name : str
        Name of the constraining object.
    min_angle : float
        Minimum permitted angle (degrees) from the object to the spacecraft vector.
    max_angle : float
        Maximum permitted angle (degrees) from the object to the spacecraft vector.
    """

    name: str = Field(description="Name of the constraining object", default="")
    min_angle: float = Field(description="Minimum permitted angle (degrees)", default=0.0, ge=0.0, le=180.0)
    max_angle: float = Field(description="Maximum permitted angle (degrees)", default=180.0, ge=0.0, le=180.0)
