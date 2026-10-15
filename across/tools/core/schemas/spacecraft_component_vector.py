from pydantic import Field

from .base import BaseSchema
from .roll_angle_constraining_object import RollAngleConstrainingObject


class SpacecraftComponentVector(BaseSchema):
    """
    Represents a spacecraft component unit vector.

    Parameters
    ----------
    name: str
        Name of the spacecraft component vector.
    x: float
        Scalar representing the x-direction of the spacecraft component in the spacecraft frame.
    y: float
        Scalar representing the y-direction of the spacecraft component in the spacecraft frame.
    z: float
        Scalar representing the z-direction of the spacecraft component in the spacecraft frame.
    constraining_bodies: list[RollAngleConstrainingObject]
        List of RollAngleConstrainingObjects that constrain the roll angle of the spacecraft component vector.
    """

    name: str = Field(
        description="Name of the spacecraft component vector",
        default="",
    )
    x: float = Field(
        description="Scalar representing the x-direction of the spacecraft component in the spacecraft frame",
        ge=-1.0,
        le=1.0,
    )
    y: float = Field(
        description="Scalar representing the y-direction of the spacecraft component in the spacecraft frame",
        ge=-1.0,
        le=1.0,
    )
    z: float = Field(
        description="Scalar representing the z-direction of the spacecraft component in the spacecraft frame",
        ge=-1.0,
        le=1.0,
    )
    constraining_bodies: list[RollAngleConstrainingObject] = Field(
        description="List of objects that constrain the roll angle of the spacecraft component vector",
    )
