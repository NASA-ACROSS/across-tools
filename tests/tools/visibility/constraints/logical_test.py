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
    MoonAngleConstraint,
    NotConstraint,
    OrConstraint,
    SunAngleConstraint,
    XorConstraint,
)
from across.tools.visibility.constraints.base import ConstraintABC

from .conftest import DummyConstraint


class TestAndOperator:
    """Test suite for the & operator creating AndConstraint."""

    def test_and_operator_creates_and_constraint(self) -> None:
        """Test that & operator creates an AndConstraint."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)
        combined = sun & moon
        assert isinstance(combined, AndConstraint)

    def test_and_operator_with_two_constraints(self) -> None:
        """Test & operator creates AndConstraint with correct constraints list."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)
        combined = cast(AndConstraint, sun & moon)
        assert len(combined.constraints) == 2
        assert combined.constraints[0] is sun
        assert combined.constraints[1] is moon


class TestOrOperator:
    """Test suite for the | operator creating OrConstraint."""

    def test_or_operator_creates_or_constraint(self) -> None:
        """Test that | operator creates an OrConstraint."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)
        combined = sun | moon
        assert isinstance(combined, OrConstraint)

    def test_or_operator_with_two_constraints(self) -> None:
        """Test | operator creates OrConstraint with correct constraints list."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)
        combined = cast(OrConstraint, sun | moon)
        assert len(combined.constraints) == 2
        assert combined.constraints[0] is sun
        assert combined.constraints[1] is moon


class TestNotOperator:
    """Test suite for the ~ operator creating NotConstraint."""

    def test_not_operator_creates_not_constraint(self) -> None:
        """Test that ~ operator creates a NotConstraint."""
        sun = SunAngleConstraint(min_angle=45)
        negated = ~sun
        assert isinstance(negated, NotConstraint)

    def test_not_operator_with_constraint(self) -> None:
        """Test ~ operator creates NotConstraint with correct constraint."""
        sun = SunAngleConstraint(min_angle=45)
        negated = cast(NotConstraint, ~sun)
        assert negated.constraint is sun


class TestXorOperator:
    """Test suite for the ^ operator creating XorConstraint."""

    def test_xor_operator_creates_xor_constraint(self) -> None:
        """Test that ^ operator creates an XorConstraint."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)
        combined = sun ^ moon
        assert isinstance(combined, XorConstraint)

    def test_xor_operator_with_two_constraints(self) -> None:
        """Test ^ operator creates XorConstraint with correct constraints list."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)
        combined = cast(XorConstraint, sun ^ moon)
        assert len(combined.constraints) == 2
        assert combined.constraints[0] is sun
        assert combined.constraints[1] is moon


class TestConstraintTypeValues:
    """Test suite for constraint type enum values."""

    def test_constraint_type_for_and(self) -> None:
        """Test that AND constraint has correct ConstraintType."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)
        and_constraint = sun & moon
        assert and_constraint.name == ConstraintType.AND

    def test_constraint_type_for_or(self) -> None:
        """Test that OR constraint has correct ConstraintType."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)
        or_constraint = sun | moon
        assert or_constraint.name == ConstraintType.OR

    def test_constraint_type_for_not(self) -> None:
        """Test that NOT constraint has correct ConstraintType."""
        sun = SunAngleConstraint(min_angle=45)
        not_constraint = ~sun
        assert not_constraint.name == ConstraintType.NOT

    def test_constraint_type_for_xor(self) -> None:
        """Test that XOR constraint has correct ConstraintType."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)
        xor_constraint = sun ^ moon
        assert xor_constraint.name == ConstraintType.XOR


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
        self, mock_ephemeris: Ephemeris, time_array: Time, sky_coord: SkyCoord
    ) -> None:
        """Test AndConstraint with a single constraint returns same result."""
        dummy = DummyConstraint()
        and_constraint = AndConstraint(constraints=[dummy])
        result = and_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        expected = dummy(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.array_equal(result, expected)

    def test_and_constraint_logic(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test that AndConstraint correctly applies AND logic."""
        # True AND True = True
        and_constraint = AndConstraint(constraints=[true_constraint, true_constraint])
        result = and_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

        # True AND False = False
        and_constraint = AndConstraint(constraints=[true_constraint, false_constraint])
        result = and_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)

        # False AND False = False
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
        self, mock_ephemeris: Ephemeris, time_array: Time, sky_coord: SkyCoord
    ) -> None:
        """Test OrConstraint with a single constraint returns same result."""
        dummy = DummyConstraint()
        or_constraint = OrConstraint(constraints=[dummy])
        result = or_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        expected = dummy(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.array_equal(result, expected)

    def test_or_constraint_logic(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test that OrConstraint correctly applies OR logic."""
        # True OR True = True
        or_constraint = OrConstraint(constraints=[true_constraint, true_constraint])
        result = or_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

        # True OR False = True
        or_constraint = OrConstraint(constraints=[true_constraint, false_constraint])
        result = or_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

        # False OR False = False
        or_constraint = OrConstraint(constraints=[false_constraint, false_constraint])
        result = or_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)


class TestNotConstraint:
    """Test suite for NotConstraint functionality."""

    def test_not_constraint_logic(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test that NotConstraint correctly applies NOT logic."""
        # NOT True = False
        not_constraint = NotConstraint(constraint=true_constraint)
        result = not_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)

        # NOT False = True
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
        self, mock_ephemeris: Ephemeris, time_array: Time, sky_coord: SkyCoord
    ) -> None:
        """Test XorConstraint with a single constraint returns same result."""
        dummy = DummyConstraint()
        xor_constraint = XorConstraint(constraints=[dummy])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        expected = dummy(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.array_equal(result, expected)

    def test_xor_constraint_logic(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test that XorConstraint correctly applies XOR logic."""
        # True XOR True = False
        xor_constraint = XorConstraint(constraints=[true_constraint, true_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)

        # True XOR False = True
        xor_constraint = XorConstraint(constraints=[true_constraint, false_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

        # False XOR True = True
        xor_constraint = XorConstraint(constraints=[false_constraint, true_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

        # False XOR False = False
        xor_constraint = XorConstraint(constraints=[false_constraint, false_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)

    def test_xor_constraint_three_constraints(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test that XorConstraint with three constraints returns True for odd violations."""
        # True XOR True XOR True = True (odd number: 3)
        xor_constraint = XorConstraint(constraints=[true_constraint, true_constraint, true_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

        # True XOR True XOR False = False (even number: 2)
        xor_constraint = XorConstraint(constraints=[true_constraint, true_constraint, false_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)

        # True XOR False XOR False = True (odd number: 1)
        xor_constraint = XorConstraint(constraints=[true_constraint, false_constraint, false_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

        # False XOR False XOR False = False (even number: 0)
        xor_constraint = XorConstraint(constraints=[false_constraint, false_constraint, false_constraint])
        result = xor_constraint(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)


class TestComplexConstraintCombinations:
    """Test suite for complex constraint combinations."""

    def test_nested_and_or(self) -> None:
        """Test nested (A & B) | C combination creates correct structure."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)

        # (A & B) | C
        combined = (sun & moon) | sun
        assert isinstance(combined, OrConstraint)
        assert isinstance(combined.constraints[0], AndConstraint)

    def test_nested_or_and(self) -> None:
        """Test nested (A | B) & C combination creates correct structure."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)

        # (A | B) & C
        combined = (sun | moon) & sun
        assert isinstance(combined, AndConstraint)
        assert isinstance(combined.constraints[0], OrConstraint)

    def test_double_negation(self) -> None:
        """Test double negation: ~~A = A (in structure)."""
        sun = SunAngleConstraint(min_angle=45)

        # ~~A
        double_negated = ~(~sun)
        assert isinstance(double_negated, NotConstraint)
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

    def test_nested_xor_combination(self) -> None:
        """Test nested (A ^ B) | C combination creates correct structure."""
        sun = SunAngleConstraint(min_angle=45)
        moon = MoonAngleConstraint(min_angle=10)

        # (A ^ B) | C
        combined = (sun ^ moon) | sun
        assert isinstance(combined, OrConstraint)
        assert isinstance(combined.constraints[0], XorConstraint)

    def test_complex_xor_combination(
        self,
        mock_ephemeris: Ephemeris,
        time_array: Time,
        sky_coord: SkyCoord,
        true_constraint: ConstraintABC,
        false_constraint: ConstraintABC,
    ) -> None:
        """Test complex combination: (A ^ B) & C."""
        # (True ^ False) & True = True & True = True
        combined = (true_constraint ^ false_constraint) & true_constraint
        result = combined(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(result)

        # (True ^ True) & True = False & True = False
        combined = (true_constraint ^ true_constraint) & true_constraint
        result = combined(time=time_array, ephemeris=mock_ephemeris, coordinate=sky_coord)
        assert np.all(~result)
