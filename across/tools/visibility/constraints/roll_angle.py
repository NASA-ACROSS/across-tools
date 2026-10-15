from typing import Literal, Self

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from ...core.enums import ComponentConstraintMode, ConstraintType
from ...core.schemas import RollAngleAllowedRange, SpacecraftComponentVector
from ...ephemeris import Ephemeris
from .base import ConstraintABC


def get_discontinuous_slice(time: Time, ephemeris: Ephemeris) -> npt.NDArray[np.int_]:
    """
    Return a slice for the part of the ephemeris that we're using.
    This handles discontinuous time ranges by returning a numpy array of indices,
    rather than a slice object. This is useful for evaluating roll angle constraints
    over time ranges that may not be contiguous in the ephemeris.

    Arguments
    ---------
    time : Time
        The time to calculate the slice for
    ephemeris : Ephemeris
        The spacecraft ephemeris

    Returns
    -------
        The slice for the ephemeris
    """
    # Ensure that time is an array-like object
    if time.isscalar:
        raise NotImplementedError("Scalar time not supported")

    # Find the indices for the ephemeris over the time range and return a slice for that range.
    # If the time range is discontinuous, return a numpy array of indices.
    return np.array([ephemeris.index(t) for t in time])


class RollAngleConstraint(ConstraintABC):
    """
    Constraint that avoids observing when no roll angles are permissible.

    Checks the separation between a given spacecraft component vector and
    a list of astronomical bodies to ensure the separation is within the allowable
    range for any roll angle at a given time slice.

    Parameters
    ----------
    constrained_components : list[SpacecraftComponentVector]
        List of spacecraft component unit vectors with bodies that constrain them.
    allowed_roll : RollAngleAllowedRange | None
        Allowed roll angle range and step size (degrees). If None, the optimal_roll_angle must be provided.
    optimal_roll_angle : float | None
        Optimal roll angle (degrees) that maximizes the visibility window.
        If None, the allowed_roll must be provided.
    constrain_all_components : bool
        Indicates whether all or any of the spacecraft components need to be constrained.

    Methods
    -------
    __call__(time, ephemeris, coordinate)
        Checks if the coordinate violates roll angle constraints over the given optimal roll angle
        or for the given roll angle range.
    """

    name: Literal[ConstraintType.ROLL] = ConstraintType.ROLL
    short_name: Literal["Roll Angle"] = "Roll Angle"
    constrained_components: list[SpacecraftComponentVector] = Field(
        description="List of spacecraft component unit vectors with bodies that constrain them"
    )
    allowed_roll: RollAngleAllowedRange | None = Field(
        description="Allowed roll angle range and step size (degrees)", default=RollAngleAllowedRange()
    )
    optimal_roll_angle: list[float] | None = Field(
        description="Optimal roll angle(s) (in degrees) that maximizes the visibility window",
        default=None,
    )
    component_constraint_mode: ComponentConstraintMode = Field(
        description="""
        Whether all or any of the spacecraft components need to be constrained
        for a target to be constrained
        """,
        default=ComponentConstraintMode.ALL,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_optimal_roll_angle(cls, values: dict) -> dict:  # type: ignore[type-arg]
        """
        Validate that the optimal_roll_angle is a list of valid floats, if provided.

        Parameters
        ----------
        values : dict
            Dictionary containing the input values for the model.

        Returns
        -------
        dict
            The validated input values.
        """
        optimal_roll_angle = values.get("optimal_roll_angle")
        if optimal_roll_angle is not None:
            try:
                for roll_angle in optimal_roll_angle:
                    if roll_angle < 0.0 or roll_angle > 360.0:
                        raise ValueError("Optimal roll angles must be between 0 and 360 degrees")
            except TypeError as e:
                raise ValueError("Optimal roll angle must be a list of floats") from e
        return values

    @model_validator(mode="after")
    def validate_provided_roll_info(self) -> Self:
        """
        Validate that either allowed_roll or optimal_roll_angle is provided, but not both.
        """
        if self.allowed_roll is None and self.optimal_roll_angle is None:
            raise ValueError("Either allowed_roll or optimal_roll_angle must be provided.")
        if self.allowed_roll is not None and self.optimal_roll_angle is not None:
            raise ValueError("Only one of allowed_roll or optimal_roll_angle can be provided.")
        return self

    def _compute_payload_axes(
        self,
        z_ref: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Computes the spacecraft payload x- and y-axes based on the boresight z-axis (z_ref).

        Parameters
        ----------
        z_ref : np.ndarray
            Vector representing the boresight direction in spacecraft coordinates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The computed x- and y-axes of the spacecraft payload in spacecraft coordinates.
        """
        north = np.zeros(z_ref.T.shape) + np.asarray([0.0, 0.0, 1.0])
        north_proj = north - (np.dot(north, z_ref) @ z_ref.T)
        north_norm = np.linalg.norm(north_proj)
        # Reference x axis
        x_ref = north_proj / north_norm
        # Reference y axis
        y_ref = np.cross(z_ref.T, x_ref)
        y_ref /= np.linalg.norm(y_ref)

        return x_ref, y_ref

    def _compute_constraint_per_roll(
        self,
        ephemeris: Ephemeris,
        component_vector_separations: dict[str, dict[str, SkyCoord]],
        i: npt.NDArray[np.int_],
        roll_ind: int,
    ) -> npt.NDArray[np.bool_]:
        """
        Compute the constraint per timestep for a givne roll index.
        For each constraining body, compare the component separation vector
        with the min and max angles for that body. If any body is outside of the
        allowed range, the constraint is violated.

        Parameters
        ----------
        ephemeris : Ephemeris
            The spacecraft ephemeris.
        component_vector_separations : dict[str, dict[str, SkyCoord]]
            Dictionary containing the angular separations between spacecraft component vectors
            and constraining bodies for each roll angle.
        i : slice
            Slice of ephemeris data to consider.
        roll_ind : int
            Index of the roll angle to consider.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates the roll angle violates one or more constraints.
        """
        # Master in_constraint to track whether all or any of the components are in constraint
        if self.component_constraint_mode == "all":
            in_constraint = np.ones(len(ephemeris.gcrs[i]), dtype=bool)
        else:
            in_constraint = np.zeros(len(ephemeris.gcrs[i]), dtype=bool)
        for component_name, body_separation_angles in component_vector_separations.items():
            component = next(comp for comp in self.constrained_components if comp.name == component_name)
            in_constraint_per_component = np.zeros(len(ephemeris.gcrs[i]), dtype=bool)
            for body in component.constraining_bodies:
                # Check if body separation violates the constraint constraint
                if body.name == "earth":
                    in_constraint_per_component |= np.logical_or(
                        body_separation_angles[body.name][roll_ind, :]
                        > ephemeris.earth_radius_angle[i] + body.max_angle * u.deg,
                        body_separation_angles[body.name][roll_ind, :]
                        < ephemeris.earth_radius_angle[i] + body.min_angle * u.deg,
                    )
                else:
                    in_constraint_per_component |= np.logical_or(
                        body_separation_angles[body.name][roll_ind, :] > body.max_angle * u.deg,
                        body_separation_angles[body.name][roll_ind, :] < body.min_angle * u.deg,
                    )
            if self.component_constraint_mode == "all":
                in_constraint &= in_constraint_per_component
            else:
                in_constraint |= in_constraint_per_component

        return in_constraint

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> npt.NDArray[np.bool_]:
        """
        Check if the coordinate is observable at any roll angle.

        Parameters
        ----------
        time : Time
            The time of observation.
        ephemeris : Ephemeris
            The ephemeris (used to get Earth location for body positions).
        coordinate : SkyCoord
            The coordinate to check.

        Returns
        -------
        npt.NDArray[np.bool_]
            Boolean array where True indicates the coordinate violates the constraint
            (is too close to a Solar System object).
        """
        if ephemeris.earth_location is None:
            raise ValueError("Earth location required for object positions")

        # Find the slice of ephemeris data we need
        i = get_discontinuous_slice(time, ephemeris)

        # Put target coordinate in satellite GCRS frame
        target_gcrs = coordinate.transform_to(ephemeris.gcrs)[i]
        # Assume the target coordinate is the boresight pointing direction,
        # and calculate its vector in cartesian coordinates
        z_ref = target_gcrs.cartesian.xyz.value
        z_ref /= np.linalg.norm(z_ref)
        # Compute payload x and y axes from boresight z
        x_ref, y_ref = self._compute_payload_axes(z_ref)

        # Build array of allowed rolls
        if self.allowed_roll is not None:
            roll = (
                (
                    np.arange(
                        self.allowed_roll.min_roll, self.allowed_roll.max_roll, self.allowed_roll.step_size
                    )
                    * u.deg
                )
                .to(u.rad)
                .value
            )
        elif self.optimal_roll_angle is not None:
            roll = np.asarray(np.deg2rad(self.optimal_roll_angle))

        # Compute rotation of x- and y-axes about z-axis by roll angle to get payload vectors
        x_payload = (np.cos(roll).reshape(-1, 1, 1) * x_ref.T.reshape(1, 3, -1)) + (
            np.sin(roll).reshape(-1, 1, 1) * y_ref.T.reshape(1, 3, -1)
        )
        y_payload = (-np.sin(roll).reshape(-1, 1, 1) * x_ref.T.reshape(1, 3, -1)) + (
            np.cos(roll).reshape(-1, 1, 1) * y_ref.T.reshape(1, 3, -1)
        )
        z_payload = np.zeros(x_payload.shape) + z_ref

        # Compute separation between the spacecraft component vectors and the constraining bodies
        # for each roll angle
        component_vector_separations: dict[str, dict[str, SkyCoord]] = {}
        for component in self.constrained_components:
            component_vector_position = (
                x_payload * component.x + y_payload * component.y + z_payload * component.z
            )
            component_vector_position /= np.linalg.norm(component_vector_position, axis=1, keepdims=True)

            # Transform from cartesian to SkyCoord component representation
            component_coords = SkyCoord(
                x=component_vector_position[:, 0, :],
                y=component_vector_position[:, 1, :],
                z=component_vector_position[:, 2, :],
                representation_type="cartesian",
            )
            # Get angular separations between the spacecraft component vector and each body
            body_separation_angles: dict[str, SkyCoord] = {}
            for body in component.constraining_bodies:
                body_coords = getattr(ephemeris, body.name)[i]
                body_separation_angles[body.name] = SkyCoord(body_coords.ra, body_coords.dec).separation(
                    component_coords
                )

            component_vector_separations[component.name] = body_separation_angles

        # Find optimal roll angle that maximizes the visibility window
        optimal_roll_angle_inds = []
        optimal_window = np.asarray([0])
        # Iterate over range of allowed roll angles and compute in_constraint for each one
        for roll_ind in range(len(roll)):
            in_constraint = self._compute_constraint_per_roll(
                ephemeris, component_vector_separations, i, roll_ind
            )

            # Compare length with stored optimal window, replace it and optimal roll angle index
            # if the new window is longer
            if np.sum(~in_constraint) > np.sum(optimal_window):
                optimal_window = ~in_constraint
                optimal_roll_angle_inds = [roll_ind]

            # Save all roll angles that yield the same optimal window length
            elif np.sum(~in_constraint) == np.sum(optimal_window):
                optimal_window = ~in_constraint
                optimal_roll_angle_inds.append(roll_ind)

        self.optimal_roll_angle = np.rad2deg(roll[optimal_roll_angle_inds])

        return ~optimal_window
