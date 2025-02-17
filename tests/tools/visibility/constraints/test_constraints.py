from datetime import datetime
from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import ValidationError

from across.tools.ephemeris import Ephemeris
from across.tools.visibility.constraints.base import Constraint, get_slice


class DummyConstraint(Constraint):
    """Dummy concrete implementation for testing"""

    short_name: Literal["dummy"] = "dummy"
    name: Literal["Dummy Constraint"] = "Dummy Constraint"

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """Dummy implementation of the constraint"""
        return np.array([True])


class TestConstraint:
    """Test the Constraint base class"""

    def test_constraint_is_instance_of_constraint(self) -> None:
        """
        Test that a DummyConstraint instance is properly recognized as a subclass of Constraint.

        Verifies the inheritance relationship between DummyConstraint and the base Constraint class.
        """
        constraint: DummyConstraint = DummyConstraint()
        assert isinstance(constraint, Constraint)

    def test_constraint_min_angle(self) -> None:
        """
        Test that the min_angle parameter is correctly set in the constraint.

        Creates a constraint with min_angle=10 and verifies it's properly stored.
        """
        constraint: DummyConstraint = DummyConstraint(min_angle=10, max_angle=20)
        assert constraint.min_angle == 10

    def test_constraint_max_angle(self) -> None:
        """
        Test that the max_angle parameter is correctly set in the constraint.

        Creates a constraint with max_angle=20 and verifies it's properly stored.
        """
        constraint: DummyConstraint = DummyConstraint(min_angle=10, max_angle=20)
        assert constraint.max_angle == 20

    def test_constraint_short_name(self) -> None:
        """
        Test that the short_name class attribute is correctly accessible.

        Verifies that the short_name property returns the expected value 'dummy'.
        """
        constraint: DummyConstraint = DummyConstraint(min_angle=10, max_angle=20)
        assert constraint.short_name == "dummy"

    def test_constraint_name(self) -> None:
        """
        Test that the name class attribute is correctly accessible.

        Verifies that the name property returns the expected value 'Dummy Constraint'.
        """
        constraint: DummyConstraint = DummyConstraint(min_angle=10, max_angle=20)
        assert constraint.name == "Dummy Constraint"

    def test_constraint_optional_min_angle(self) -> None:
        """
        Test that min_angle is optional and defaults to None.

        Creates a constraint without specifying min_angle and verifies it's None.
        """
        constraint: DummyConstraint = DummyConstraint()
        assert constraint.min_angle is None

    def test_constraint_optional_max_angle(self) -> None:
        """
        Test that max_angle is optional and defaults to None.

        Creates a constraint without specifying max_angle and verifies it's None.
        """
        constraint: DummyConstraint = DummyConstraint()
        assert constraint.max_angle is None

    def test_get_slice_scalar_time(self) -> None:
        """
        Test that get_slice raises NotImplementedError for scalar time input.

        Attempts to call get_slice with a scalar time value and verifies it raises
        the expected exception.
        """
        scalar_time: Time = Time(datetime(2023, 1, 1))
        mock_ephemeris: Ephemeris = MagicMock()

        with pytest.raises(NotImplementedError):
            get_slice(scalar_time, mock_ephemeris)

    def test_abstract_constraint(self) -> None:
        """
        Test that the abstract Constraint class cannot be instantiated directly.

        Attempts to create an instance of the abstract base class and verifies
        it raises TypeError.
        """
        with pytest.raises(TypeError):
            Constraint()  # type: ignore

    def test_constraint_validation_min_angle(self) -> None:
        """
        Test that constraint validation fails with invalid min_angle value.

        Attempts to create a constraint with an invalid min_angle and verifies
        it raises ValidationError.
        """
        with pytest.raises(ValidationError):
            DummyConstraint(min_angle="invalid")  # type: ignore

    def test_constraint_validation_max_angle(self) -> None:
        """
        Test that constraint validation fails with invalid max_angle value.

        Attempts to create a constraint with an invalid max_angle and verifies
        it raises ValidationError.
        """
        with pytest.raises(ValidationError):
            DummyConstraint(max_angle="invalid")  # type: ignore
