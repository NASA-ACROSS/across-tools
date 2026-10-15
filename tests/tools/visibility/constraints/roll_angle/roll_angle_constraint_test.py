import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.roll_angle import RollAngleConstraint, get_discontinuous_slice


class TestRollAngleConstraintAttributes:
    """Test suite for RollAngleConstraint attributes."""

    def test_constraint_short_name(self, roll_angle_constraint: RollAngleConstraint) -> None:
        """Test constraint short_name attribute."""
        assert roll_angle_constraint.short_name == "Roll Angle"

    def test_constraint_name_value(self, roll_angle_constraint: RollAngleConstraint) -> None:
        """Test constraint name.value attribute."""
        assert roll_angle_constraint.name.value == "Roll Angle"


class TestRollAngleConstraintInitialization:
    """Test construction of the constraint."""

    def test_constraint_initialization_default_allowed_roll(
        self, roll_angle_constraint: RollAngleConstraint
    ) -> None:
        """Test constraint initialization should handle default allowed roll."""
        assert roll_angle_constraint.allowed_roll is not None
        assert roll_angle_constraint.allowed_roll.min_roll == 0.0
        assert roll_angle_constraint.allowed_roll.max_roll == 360.0
        assert roll_angle_constraint.allowed_roll.step_size == 2.0

    def test_constraint_initialization_custom_optimal_roll(
        self, roll_angle_constraint_optimal_roll: RollAngleConstraint
    ) -> None:
        """Test constraint initialization should handle custom optimal roll."""
        assert roll_angle_constraint_optimal_roll.optimal_roll_angle == [45.0]
        assert roll_angle_constraint_optimal_roll.allowed_roll is None

    def test_constraint_initialization_bad_optimal_roll_type(
        self,
    ) -> None:
        """Test constraint initialization with bad optimal roll type should raise ValueError."""
        with pytest.raises(ValueError, match="Optimal roll angle must be a list of floats"):
            RollAngleConstraint(constrained_components=[], optimal_roll_angle=["not_a_string"])  # type: ignore

    def test_constraint_initialization_bad_optimal_roll_value(
        self,
    ) -> None:
        """Test constraint initialization with bad optimal roll value should raise ValueError."""
        with pytest.raises(ValueError, match="Optimal roll angles must be between 0 and 360 degrees"):
            RollAngleConstraint(constrained_components=[], optimal_roll_angle=[500.0])

    def test_constraint_initialization_without_provided_roll(
        self,
    ) -> None:
        """Test constraint initialization without provided roll should raise ValueError."""
        with pytest.raises(ValueError, match="Either allowed_roll or optimal_roll_angle must be provided"):
            RollAngleConstraint(constrained_components=[], optimal_roll_angle=None, allowed_roll=None)

    def test_constraint_initialization_with_too_many_provided_rolls(
        self,
    ) -> None:
        """Test constraint initialization with too many provided roll should raise ValueError."""
        with pytest.raises(
            ValueError, match="Only one of allowed_roll or optimal_roll_angle can be provided"
        ):
            RollAngleConstraint(constrained_components=[], optimal_roll_angle=[45.0])


class TestRollAngleConstraintCall:
    """Test evaluating the constraint."""

    def test_constraint_returns_array_like(
        self,
        begin_time_array: Time,
        test_tle_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        roll_angle_constraint: RollAngleConstraint,
    ) -> None:
        """Test that constraint returns array-like result."""
        result = roll_angle_constraint(begin_time_array, test_tle_ephemeris, sky_coord)
        assert hasattr(result, "dtype")

    def test_constraint_returns_boolean_array(
        self,
        begin_time_array: Time,
        test_tle_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        roll_angle_constraint: RollAngleConstraint,
    ) -> None:
        """Test that constraint returns boolean array-like result."""
        result = roll_angle_constraint(begin_time_array, test_tle_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_saves_optimal_roll_angle(
        self,
        begin_time_array: Time,
        test_tle_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        roll_angle_constraint: RollAngleConstraint,
    ) -> None:
        """Test that constraint call saves the optimal roll angle(s)."""
        roll_angle_constraint(begin_time_array, test_tle_ephemeris, sky_coord)
        assert roll_angle_constraint.optimal_roll_angle is not None
        assert len(roll_angle_constraint.optimal_roll_angle) > 0

    def test_constraint_optimized_roll_returns_array_like(
        self,
        begin_time_array: Time,
        test_tle_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        roll_angle_constraint_optimal_roll: RollAngleConstraint,
    ) -> None:
        """Test that constrain with optimized roll returns array-like result."""
        result = roll_angle_constraint_optimal_roll(begin_time_array, test_tle_ephemeris, sky_coord)
        assert hasattr(result, "dtype")

    def test_constraint_optimized_roll_returns_boolean_array(
        self,
        begin_time_array: Time,
        test_tle_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        roll_angle_constraint_optimal_roll: RollAngleConstraint,
    ) -> None:
        """Test that constraint with optimized roll returns boolean array-like result."""
        result = roll_angle_constraint_optimal_roll(begin_time_array, test_tle_ephemeris, sky_coord)
        assert result.dtype == bool

    def test_constraint_constrain_any_components_returns_array_like(
        self,
        begin_time_array: Time,
        test_tle_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        roll_angle_constraint_constrain_any_components: RollAngleConstraint,
    ) -> None:
        """Test that constraint requiring any components to be constrained returns array-like result."""
        result = roll_angle_constraint_constrain_any_components(
            begin_time_array, test_tle_ephemeris, sky_coord
        )
        assert hasattr(result, "dtype")

    def test_constraint_constrain_any_components_returns_boolean_array(
        self,
        begin_time_array: Time,
        test_tle_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        roll_angle_constraint_constrain_any_components: RollAngleConstraint,
    ) -> None:
        """
        Test that constraint requiring any components to be constrained returns boolean array-like result.
        """
        result = roll_angle_constraint_constrain_any_components(
            begin_time_array, test_tle_ephemeris, sky_coord
        )
        assert result.dtype == bool

    def test_constraint_calculation_using_ephemeris_without_earth_position_raises_error(
        self,
        begin_time_array: Time,
        test_tle_ephemeris: Ephemeris,
        sky_coord: SkyCoord,
        roll_angle_constraint_constrain_any_components: RollAngleConstraint,
    ) -> None:
        """
        Test that calling the constraint with an ephemeris missing an earth location raises a ValueError .
        """
        test_tle_ephemeris.earth_location = None
        with pytest.raises(ValueError, match="Earth location required for object positions"):
            roll_angle_constraint_constrain_any_components(begin_time_array, test_tle_ephemeris, sky_coord)


class TestGetDiscontinuousSlice:
    """Test the get_discontinuous_slice method."""

    def test_should_return_array(
        self,
        begin_time_array: Time,
        test_tle_ephemeris: Ephemeris,
    ) -> None:
        """Test that get_discontinuous_slice returns an array."""
        result = get_discontinuous_slice(begin_time_array, test_tle_ephemeris)
        assert hasattr(result, "dtype")

    def test_should_return_array_of_indices(
        self,
        begin_time_array: Time,
        test_tle_ephemeris: Ephemeris,
    ) -> None:
        """Test that get_discontinuous_slice returns an array of indices."""
        result = get_discontinuous_slice(begin_time_array, test_tle_ephemeris)
        assert result.dtype == int

    def test_scalar_time_should_raise_value_error(
        self,
        begin_time_array: Time,
        test_tle_ephemeris: Ephemeris,
    ) -> None:
        """Test that get_discontinuous_slice raises NotImplementedError for scalar time input."""
        with pytest.raises(NotImplementedError):
            get_discontinuous_slice(begin_time_array[0], test_tle_ephemeris)
