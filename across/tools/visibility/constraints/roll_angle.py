from typing import Literal, Self

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from ...core.enums import ComponentConstraintMode, ConstraintType
from ...core.schemas import RollAngleAllowedRange, RollAngleConstrainingObject, SpacecraftComponentVector
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
        z_axis = z_ref.T
        north = np.asarray([0.0, 0.0, 1.0])
        # Project north onto the plane perpendicular to the boresight, per timestep
        north_proj = north - (z_axis @ north)[:, np.newaxis] * z_axis
        # Reference x axis
        x_ref = north_proj / np.linalg.norm(north_proj, axis=1, keepdims=True)
        # Reference y axis
        y_ref = np.cross(z_axis, x_ref)
        y_ref /= np.linalg.norm(y_ref, axis=1, keepdims=True)

        return x_ref, y_ref

    def _compute_constraint(
        self,
        ephemeris: Ephemeris,
        roll_matrix: npt.NDArray[np.float64],
        payload_axes: npt.NDArray[np.float64],
        i: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.bool_]:
        """
        Compute the constraint for every roll angle and timestep at once.
        A component is constrained if any of its bodies is outside the allowed separation range.
        Components are then combined according to the component constraint mode.

        Parameters
        ----------
        ephemeris : Ephemeris
            The spacecraft ephemeris.
        roll_matrix : np.ndarray
            Roll design matrix with shape (n_roll, 3) with columns [cos(roll), sin(roll), 1].
        payload_axes : np.ndarray
            Payload x, y and z axes stacked with shape (3, 3, n_time), indexed [axis, xyz, time].
        i : np.ndarray
            Indices of ephemeris data to consider.

        Returns
        -------
        np.ndarray
            Boolean array with shape (n_roll, n_time), where True indicates the roll angle
            violates one or more constraints.
        """
        # Flatten the (component, body) pairs; component_index maps each pair to its component
        component_body_pairs = [
            (index, component, body)
            for index, component in enumerate(self.constrained_components)
            for body in component.constraining_bodies
        ]
        component_index = np.array([index for index, _, _ in component_body_pairs], dtype=int)
        violation = self._compute_component_body_violations(
            [component for _, component, _ in component_body_pairs],
            [body for _, _, body in component_body_pairs],
            roll_matrix,
            payload_axes,
            ephemeris,
            i,
        )

        # A component is constrained if any of its bodies is violated.
        component_ids = np.arange(len(self.constrained_components))
        # Membership is a (n_component, n_pair) matrix,
        # True for entries that correspond to the same component.
        membership = (component_ids[:, None] == component_index).astype(np.float32)
        # Use membership to evaluate which components are in constraint per roll per timestep
        component_in_constraint = (membership @ violation.astype(np.float32)) > 0

        # Evaluate if all or any of the components are in constraint per roll per timestep
        if self.component_constraint_mode == "all":
            return np.asarray(component_in_constraint.all(axis=1))
        return np.asarray(component_in_constraint.any(axis=1))

    def _compute_component_body_violations(
        self,
        components: list[SpacecraftComponentVector],
        bodies: list[RollAngleConstrainingObject],
        roll_matrix: npt.NDArray[np.float64],
        payload_axes: npt.NDArray[np.float64],
        ephemeris: Ephemeris,
        i: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.bool_]:
        """
        Compute whether each (component, body) pair is outside the allowed separation range,
        for every roll angle and timestep. `components[p]` and `bodies[p]` describe pair `p`.

        With the payload axes (x, y, z) at a given timestep, a component vector (cx, cy, cz)
        rotated by roll angle phi is
        v(phi) = cx*(cos(phi)*x + sin(phi)*y) + cy*(-sin(phi)*x + cos(phi)*y) + cz*z.
        Its dot product with the unit vector b towards a body is therefore linear in
        [cos(phi), sin(phi), 1]:

            v . b = [cos(phi), sin(phi), 1] @ M @ (bx, by, bz)

        where (bx, by, bz) are b's components along the payload axes and
        M = [[cx, cy, 0], [-cy, cx, 0], [0, 0, cz]] / |c|. Every pair is stacked along one axis,
        so batched products give the separation for all pairs, rolls and timesteps at once.

        Parameters
        ----------
        components : list[SpacecraftComponentVector]
            The component vector of each pair.
        bodies : list[RollAngleConstrainingObject]
            The constraining body of each pair.
        roll_matrix : np.ndarray
            Roll design matrix with shape (n_roll, 3) with columns [cos(roll), sin(roll), 1].
        payload_axes : np.ndarray
            Payload axes with shape (3, 3, n_time), indexed [axis, xyz, time].
        ephemeris : Ephemeris
            The spacecraft ephemeris.
        i : np.ndarray
            Indices of ephemeris data to consider.

        Returns
        -------
        np.ndarray
            Boolean array with shape (n_roll, n_pair, n_time), True where the separation is
            outside the allowed range.
        """
        n_pair, n_time = len(bodies), len(i)
        if n_pair == 0:
            return np.zeros((len(roll_matrix), 0, n_time), dtype=bool)

        # Unit vectors towards each body, shape (n_pair, 3, n_time)
        body_vector_by_name = {}
        for name in {body.name for body in bodies}:
            coords = getattr(ephemeris, name)[i]
            ra, dec = coords.ra.rad, coords.dec.rad
            body_vector_by_name[name] = np.stack(
                [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
            )
        body_unit_vectors = np.stack([body_vector_by_name[body.name] for body in bodies])

        # Body components along the payload axes, shape (n_pair, 3, n_time)
        # This is effectively a change of basis from the celestial frame to the payload frame.
        body_vectors_payload_frame = np.einsum("axn,pxn->pan", payload_axes, body_unit_vectors)

        # Normalized component vectors as a map onto the roll basis, shape (n_pair, 3, 3)
        cx = np.array([component.x for component in components])
        cy = np.array([component.y for component in components])
        cz = np.array([component.z for component in components])
        zero = np.zeros(n_pair)
        roll_basis_mapping = np.stack(
            [
                np.stack([cx, cy, zero], axis=-1),
                np.stack([-cy, cx, zero], axis=-1),
                np.stack([zero, zero, cz], axis=-1),
            ],
            axis=1,
        ) / np.sqrt(cx**2 + cy**2 + cz**2).reshape(-1, 1, 1)
        # Apply mapping transformation to body vectors
        # This is the M @ (bx, by, bz) term in the docstring above
        coefficients = np.einsum("pab,pbn->pan", roll_basis_mapping, body_vectors_payload_frame)
        # Calculate angular separation from dot product of the spacecraft design matrix and the coefficients
        # shape (n_roll, n_pair, n_time)
        separation = np.arccos(np.clip(np.einsum("ra,pan->rpn", roll_matrix, coefficients), -1.0, 1.0))

        # Allowed range per pair in radians, shape (n_pair, n_time)
        min_angle = np.deg2rad([body.min_angle for body in bodies])[:, np.newaxis] + np.zeros(n_time)
        max_angle = np.deg2rad([body.max_angle for body in bodies])[:, np.newaxis] + np.zeros(n_time)
        is_earth = np.array([body.name == "earth" for body in bodies], dtype=bool)
        if is_earth.any():
            # Earth's limits include its limb, so need to add it to the min and max angles
            earth_angular_radius = ephemeris.earth_radius_angle[i].to_value(u.rad)
            min_angle[is_earth] += earth_angular_radius
            max_angle[is_earth] += earth_angular_radius

        return np.asarray((separation > max_angle) | (separation < min_angle))

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
        z_ref /= np.linalg.norm(z_ref, axis=0)
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

        # Rolling the payload about its z-axis makes every separation a linear function of
        # [cos(roll), sin(roll), 1], so all rolls and timesteps can be evaluated in one shot.
        roll_matrix = np.stack([np.cos(roll), np.sin(roll), np.ones_like(roll)], axis=1)
        in_constraint = self._compute_constraint(
            ephemeris, roll_matrix, np.stack([x_ref.T, y_ref.T, z_ref]), i
        )

        # Find the roll angle(s) that maximize the visibility window. The returned window is
        # that of the last roll angle yielding the maximum number of visible timesteps.
        visible = ~in_constraint
        visible_counts = visible.sum(axis=1)
        optimal_roll_angle_inds = np.flatnonzero(visible_counts == visible_counts.max())
        optimal_window: npt.NDArray[np.bool_] = visible[optimal_roll_angle_inds[-1]]

        self.optimal_roll_angle = np.rad2deg(roll[optimal_roll_angle_inds])

        return ~optimal_window
