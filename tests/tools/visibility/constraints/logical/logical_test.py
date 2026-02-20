"""
Tests for logical constraint operators.

Tests for the AndConstraint, OrConstraint, and NotConstraint classes,
as well as the __and__, __or__, and __invert__ operators on ConstraintABC.
Tests cover both operator creation and constraint evaluation logic.
"""

from typing import cast

import numpy as np
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.core.enums import ConstraintType
from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints import (
    AndConstraint,
    NotConstraint,
    OrConstraint,
    XorConstraint,
)
from across.tools.visibility.constraints.base import ConstraintABC


class TestAndOperator:
    """Test suite for the & operator creating AndConstraint."""

    def test_and_operator_creates_and_constraint(self, logical_and_constraint: AndConstraint) -> None:
        """Test that & operator creates an AndConstraint."""
        assert isinstance(logical_and_constraint, AndConstraint)

    def test_and_operator_with_two_constraints_count(self, logical_and_constraint: AndConstraint) -> None:
        """Test & operator creates AndConstraint with two constraints."""
        assert len(logical_and_constraint.constraints) == 2

    def test_and_operator_with_two_constraints_first(
        self,
        logical_and_constraint: AndConstraint,
        logical_sun_constraint: ConstraintABC,
    ) -> None:
        """Test & operator keeps sun as first constraint."""
        assert logical_and_constraint.constraints[0] is logical_sun_constraint

    def test_and_operator_with_two_constraints_second(
        self,
        logical_and_constraint: AndConstraint,
        logical_moon_constraint: ConstraintABC,
    ) -> None:
        """Test & operator keeps moon as second constraint."""
        assert logical_and_constraint.constraints[1] is logical_moon_constraint


class TestOrOperator:
    """Test suite for the | operator creating OrConstraint."""

    def test_or_operator_creates_or_constraint(self, logical_or_constraint: OrConstraint) -> None:
        """Test that | operator creates an OrConstraint."""
        assert isinstance(logical_or_constraint, OrConstraint)

    def test_or_operator_with_two_constraints_count(self, logical_or_constraint: OrConstraint) -> None:
        """Test | operator creates OrConstraint with two constraints."""
        assert len(logical_or_constraint.constraints) == 2

    def test_or_operator_with_two_constraints_first(
        self,
        logical_or_constraint: OrConstraint,
        logical_sun_constraint: ConstraintABC,
    ) -> None:
        """Test | operator keeps sun as first constraint."""
        assert logical_or_constraint.constraints[0] is logical_sun_constraint

    def test_or_operator_with_two_constraints_second(
        self,
        logical_or_constraint: OrConstraint,
        logical_moon_constraint: ConstraintABC,
    ) -> None:
        """Test | operator keeps moon as second constraint."""
        assert logical_or_constraint.constraints[1] is logical_moon_constraint


class TestNotOperator:
    """Test suite for the ~ operator creating NotConstraint."""

    def test_not_operator_creates_not_constraint(self, logical_not_constraint: NotConstraint) -> None:
        """Test that ~ operator creates a NotConstraint."""
        assert isinstance(logical_not_constraint, NotConstraint)

    def test_not_operator_with_constraint(
        self,
        logical_not_constraint: NotConstraint,
        logical_sun_constraint: ConstraintABC,
    ) -> None:
        """Test ~ operator creates NotConstraint with correct constraint."""
        assert logical_not_constraint.constraint is logical_sun_constraint


class TestXorOperator:
    """Test suite for the ^ operator creating XorConstraint."""

    def test_xor_operator_creates_xor_constraint(self, logical_xor_constraint: XorConstraint) -> None:
        """Test that ^ operator creates an XorConstraint."""
        assert isinstance(logical_xor_constraint, XorConstraint)

    def test_xor_operator_with_two_constraints_count(self, logical_xor_constraint: XorConstraint) -> None:
        """Test ^ operator creates XorConstraint with two constraints."""
        assert len(logical_xor_constraint.constraints) == 2

    def test_xor_operator_with_two_constraints_first(
        self,
        logical_xor_constraint: XorConstraint,
        logical_sun_constraint: ConstraintABC,
    ) -> None:
        """Test ^ operator keeps sun as first constraint."""
        assert logical_xor_constraint.constraints[0] is logical_sun_constraint

    def test_xor_operator_with_two_constraints_second(
        self,
        logical_xor_constraint: XorConstraint,
        logical_moon_constraint: ConstraintABC,
    ) -> None:
        """Test ^ operator keeps moon as second constraint."""
        assert logical_xor_constraint.constraints[1] is logical_moon_constraint


class TestConstraintTypeValues:
    """Test suite for constraint type enum values."""

    def test_constraint_type_for_and(self, logical_and_constraint: AndConstraint) -> None:
        """Test that AND constraint has correct ConstraintType."""
        assert logical_and_constraint.name == ConstraintType.AND

    def test_constraint_type_for_or(self, logical_or_constraint: OrConstraint) -> None:
        """Test that OR constraint has correct ConstraintType."""
        assert logical_or_constraint.name == ConstraintType.OR

    def test_constraint_type_for_not(self, logical_not_constraint: NotConstraint) -> None:
        """Test that NOT constraint has correct ConstraintType."""
        assert logical_not_constraint.name == ConstraintType.NOT

    def test_constraint_type_for_xor(self, logical_xor_constraint: XorConstraint) -> None:
        """Test that XOR constraint has correct ConstraintType."""
        assert logical_xor_constraint.name == ConstraintType.XOR


class TestAndConstraint:
    """Test suite for AndConstraint functionality."""

    def test_and_constraint_empty_constraints(
        self, mock_ephemeris: Ephemeris, time_array: Time, sky_coord: SkyCoord
    ) -> None:
        """Test AndConstraint with empty constraints list."""
        and_constraint = AndConstraint(constraints=[])
        result = and_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)  # All False since empty

    def test_and_constraint_single_constraint(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
    ) -> None:
        """Test AndConstraint with a single constraint returns same result."""
        and_constraint = AndConstraint(constraints=[true_constraint])
        result = and_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        expected = true_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.array_equal(result, expected)

    def test_and_constraint_logic_true_and_true(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
    ) -> None:
        """Test AndConstraint logic for True AND True."""
        and_constraint = AndConstraint(constraints=[true_constraint, true_constraint])
        result = and_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

    def test_and_constraint_logic_true_and_false(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test AndConstraint logic for True AND False."""
        and_constraint = AndConstraint(constraints=[true_constraint, false_constraint])
        result = and_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)

    def test_and_constraint_logic_false_and_false(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test AndConstraint logic for False AND False."""
        and_constraint = AndConstraint(constraints=[false_constraint, false_constraint])
        result = and_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)


class TestOrConstraint:
    """Test suite for OrConstraint functionality."""

    def test_or_constraint_empty_constraints(
        self, mock_ephemeris: Ephemeris, time_array: Time, sky_coord: SkyCoord
    ) -> None:
        """Test OrConstraint with empty constraints list."""
        or_constraint = OrConstraint(constraints=[])
        result = or_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)  # All False since empty

    def test_or_constraint_single_constraint(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
    ) -> None:
        """Test OrConstraint with a single constraint returns same result."""
        or_constraint = OrConstraint(constraints=[true_constraint])
        result = or_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        expected = true_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.array_equal(result, expected)

    def test_or_constraint_logic_true_or_true(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
    ) -> None:
        """Test OrConstraint logic for True OR True."""
        or_constraint = OrConstraint(constraints=[true_constraint, true_constraint])
        result = or_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

    def test_or_constraint_logic_true_or_false(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test OrConstraint logic for True OR False."""
        or_constraint = OrConstraint(constraints=[true_constraint, false_constraint])
        result = or_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

    def test_or_constraint_logic_false_or_false(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test OrConstraint logic for False OR False."""
        or_constraint = OrConstraint(constraints=[false_constraint, false_constraint])
        result = or_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)


class TestNotConstraint:
    """Test suite for NotConstraint functionality."""

    def test_not_constraint_logic_not_true(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
    ) -> None:
        """Test NotConstraint logic for NOT True."""
        not_constraint = NotConstraint(constraint=true_constraint)
        result = not_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)

    def test_not_constraint_logic_not_false(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test NotConstraint logic for NOT False."""
        not_constraint = NotConstraint(constraint=false_constraint)
        result = not_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)


class TestXorConstraint:
    """Test suite for XorConstraint functionality."""

    def test_xor_constraint_empty_constraints(
        self, mock_ephemeris: Ephemeris, time_array: Time, sky_coord: SkyCoord
    ) -> None:
        """Test XorConstraint with empty constraints list."""
        xor_constraint = XorConstraint(constraints=[])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)  # All False since empty

    def test_xor_constraint_single_constraint(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
    ) -> None:
        """Test XorConstraint with a single constraint returns same result."""
        xor_constraint = XorConstraint(constraints=[true_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        expected = true_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.array_equal(result, expected)

    def test_xor_constraint_logic_true_xor_true(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
    ) -> None:
        """Test XorConstraint logic for True XOR True."""
        xor_constraint = XorConstraint(constraints=[true_constraint, true_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)

    def test_xor_constraint_logic_true_xor_false(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test XorConstraint logic for True XOR False."""
        xor_constraint = XorConstraint(constraints=[true_constraint, false_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

    def test_xor_constraint_logic_false_xor_true(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test XorConstraint logic for False XOR True."""
        xor_constraint = XorConstraint(constraints=[false_constraint, true_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

    def test_xor_constraint_logic_false_xor_false(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test XorConstraint logic for False XOR False."""
        xor_constraint = XorConstraint(constraints=[false_constraint, false_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)

    def test_xor_constraint_three_constraints_all_true(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
    ) -> None:
        """Test XorConstraint with three True constraints."""
        xor_constraint = XorConstraint(constraints=[true_constraint, true_constraint, true_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

    def test_xor_constraint_three_constraints_two_true_one_false(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test XorConstraint with two True and one False constraint."""
        xor_constraint = XorConstraint(constraints=[true_constraint, true_constraint, false_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)

    def test_xor_constraint_three_constraints_one_true_two_false(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test XorConstraint with one True and two False constraints."""
        xor_constraint = XorConstraint(constraints=[true_constraint, false_constraint, false_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

    def test_xor_constraint_three_constraints_all_false(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test XorConstraint with three False constraints."""
        xor_constraint = XorConstraint(constraints=[false_constraint, false_constraint, false_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)


class TestComplexConstraintCombinations:
    """Test suite for complex constraint combinations."""

    def test_nested_and_or_is_or_constraint(
        self,
        logical_sun_constraint: ConstraintABC,
        logical_moon_constraint: ConstraintABC,
    ) -> None:
        """Test nested (A & B) | C is an OrConstraint."""
        combined = (logical_sun_constraint & logical_moon_constraint) | logical_sun_constraint
        assert isinstance(combined, OrConstraint)

    def test_nested_and_or_first_child_is_and_constraint(
        self,
        logical_sun_constraint: ConstraintABC,
        logical_moon_constraint: ConstraintABC,
    ) -> None:
        """Test nested (A & B) | C keeps AndConstraint as first child."""
        combined = cast(
            OrConstraint,
            (logical_sun_constraint & logical_moon_constraint) | logical_sun_constraint,
        )
        assert isinstance(combined.constraints[0], AndConstraint)

    def test_nested_or_and_is_and_constraint(
        self,
        logical_sun_constraint: ConstraintABC,
        logical_moon_constraint: ConstraintABC,
    ) -> None:
        """Test nested (A | B) & C is an AndConstraint."""
        combined = (logical_sun_constraint | logical_moon_constraint) & logical_sun_constraint
        assert isinstance(combined, AndConstraint)

    def test_nested_or_and_first_child_is_or_constraint(
        self,
        logical_sun_constraint: ConstraintABC,
        logical_moon_constraint: ConstraintABC,
    ) -> None:
        """Test nested (A | B) & C keeps OrConstraint as first child."""
        combined = cast(
            AndConstraint,
            (logical_sun_constraint | logical_moon_constraint) & logical_sun_constraint,
        )
        assert isinstance(combined.constraints[0], OrConstraint)

    def test_double_negation_is_not_constraint(self, logical_sun_constraint: ConstraintABC) -> None:
        """Test double negation remains a NotConstraint wrapper."""
        double_negated = ~(~logical_sun_constraint)
        assert isinstance(double_negated, NotConstraint)

    def test_double_negation_inner_constraint_is_not_constraint(
        self, logical_sun_constraint: ConstraintABC
    ) -> None:
        """Test double negation inner constraint is also NotConstraint."""
        double_negated = cast(NotConstraint, ~(~logical_sun_constraint))
        assert isinstance(double_negated.constraint, NotConstraint)

    def test_complex_and_or_combination(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test complex combination: (A & B) | C."""
        # (True & True) | False = True | False = True
        combined = (true_constraint & true_constraint) | false_constraint
        result = combined(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

    def test_complex_not_combination(
        self, mock_ephemeris: Ephemeris, time_array: Time, sky_coord: SkyCoord, true_constraint: ConstraintABC
    ) -> None:
        """Test complex combination: ~(A & B)."""
        # ~(True & True) = ~True = False
        combined = ~(true_constraint & true_constraint)
        result = combined(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)

    def test_demorgan_law(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test De Morgan's law: ~(A & B) == ~A | ~B."""
        constraint_a = true_constraint
        constraint_b = false_constraint

        # ~(A & B)
        left = ~(constraint_a & constraint_b)
        result_left = left(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)

        # ~A | ~B
        right = ~constraint_a | ~constraint_b
        result_right = right(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)

        assert np.array_equal(result_left, result_right)

    def test_nested_xor_combination_is_or_constraint(
        self,
        logical_sun_constraint: ConstraintABC,
        logical_moon_constraint: ConstraintABC,
    ) -> None:
        """Test nested (A ^ B) | C is an OrConstraint."""
        combined = (logical_sun_constraint ^ logical_moon_constraint) | logical_sun_constraint
        assert isinstance(combined, OrConstraint)

    def test_nested_xor_combination_first_child_is_xor_constraint(
        self,
        logical_sun_constraint: ConstraintABC,
        logical_moon_constraint: ConstraintABC,
    ) -> None:
        """Test nested (A ^ B) | C keeps XorConstraint as first child."""
        combined = cast(
            OrConstraint,
            (logical_sun_constraint ^ logical_moon_constraint) | logical_sun_constraint,
        )
        assert isinstance(combined.constraints[0], XorConstraint)

    def test_complex_xor_combination_true_xor_false_and_true(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test (True ^ False) & True evaluates to True."""
        combined = (true_constraint ^ false_constraint) & true_constraint
        result = combined(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

    def test_complex_xor_combination_true_xor_true_and_true(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
    ) -> None:
        """Test (True ^ True) & True evaluates to False."""
        combined = (true_constraint ^ true_constraint) & true_constraint
        result = combined(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)
