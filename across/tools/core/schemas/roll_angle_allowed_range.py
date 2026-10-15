from pydantic import Field, model_validator

from .base import BaseSchema


class RollAngleAllowedRange(BaseSchema):
    """
    Represents the allowed roll angle range (degrees).

    Parameters
    ----------
    min_roll : float
        Minimum roll angle (degrees).
    max_roll : float
        Maximum roll angle (degrees).
    step_size : float
        Step size to iterate between min and max roll angles (degrees).
    """

    min_roll: float = Field(description="Minimum roll angle (degrees)", default=0.0, ge=0.0, le=360.0)
    max_roll: float = Field(description="Maximum roll angle (degrees)", default=360.0, ge=0.0, le=360.0)
    step_size: float = Field(
        description="Step size to iterate between min and max roll angles (degrees)",
        default=2.0,
        ge=0.1,
        le=360.0,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_roll_range(cls, values: dict[str, float]) -> dict[str, float]:
        """
        Validate that the min_roll is less than or equal to max_roll.

        Parameters
        ----------
        values : dict
            Dictionary containing the input values for the model.

        Returns
        -------
        dict
            The validated input values.
        """
        min_roll = values.get("min_roll", 0.0)
        max_roll = values.get("max_roll", 360.0)
        if min_roll > max_roll:
            raise ValueError(f"min_roll ({min_roll}) must be less than or equal to max_roll ({max_roll}).")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_step_size(cls, values: dict[str, float]) -> dict[str, float]:
        """
        Validate that the step_size is less than or equal to the range (max_roll - min_roll).

        Parameters
        ----------
        values : dict
            Dictionary containing the input values for the model.

        Returns
        -------
        dict
            The validated input values.
        """
        min_roll = values.get("min_roll", 0.0)
        max_roll = values.get("max_roll", 360.0)
        step_size = values.get("step_size", 2.0)
        if step_size > (max_roll - min_roll):
            raise ValueError(
                f"step_size ({step_size}) must be less than or equal to the range ({max_roll - min_roll})."
            )
        return values
