from typing import Literal, cast

import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.core.enums.constraint_type import ConstraintType
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


class DummyConstraint(ConstraintABC):
    """Dummy constraint for testing purposes."""

    short_name: str = "Dummy"
    name: Literal[ConstraintType.UNKNOWN] = ConstraintType.UNKNOWN
    min_angle: float | None = None
    max_angle: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """Dummy implementation of the constraint."""
        return np.zeros(len(time), dtype=bool)


class TrueConstraint(DummyConstraint):
    """Constraint that always returns True (always violated)."""

    short_name: Literal["True"] = "True"

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """Always return True."""
        return np.ones(len(time), dtype=bool)


class FalseConstraint(DummyConstraint):
    """Constraint that always returns False (never violated)."""

    short_name: Literal["False"] = "False"

    def __call__(self, time: Time, ephemeris: Ephemeris, coordinate: SkyCoord) -> np.typing.NDArray[np.bool_]:
        """Always return False."""
        return np.zeros(len(time), dtype=bool)


@pytest.fixture
def true_constraint() -> TrueConstraint:
    """Fixture for a TrueConstraint instance that always returns True."""
    return TrueConstraint()


@pytest.fixture
def false_constraint() -> FalseConstraint:
    """Fixture for a FalseConstraint instance that always returns False."""
    return FalseConstraint()


@pytest.fixture
def logical_sun_constraint() -> SunAngleConstraint:
    """Sun constraint used by logical-operator structure tests."""
    return SunAngleConstraint(min_angle=45)


@pytest.fixture
def logical_moon_constraint() -> MoonAngleConstraint:
    """Moon constraint used by logical-operator structure tests."""
    return MoonAngleConstraint(min_angle=10)


@pytest.fixture
def logical_and_constraint(
    logical_sun_constraint: SunAngleConstraint,
    logical_moon_constraint: MoonAngleConstraint,
) -> AndConstraint:
    """(Sun & Moon) combined constraint."""
    return cast(AndConstraint, logical_sun_constraint & logical_moon_constraint)


@pytest.fixture
def logical_or_constraint(
    logical_sun_constraint: SunAngleConstraint,
    logical_moon_constraint: MoonAngleConstraint,
) -> OrConstraint:
    """(Sun | Moon) combined constraint."""
    return cast(OrConstraint, logical_sun_constraint | logical_moon_constraint)


@pytest.fixture
def logical_xor_constraint(
    logical_sun_constraint: SunAngleConstraint,
    logical_moon_constraint: MoonAngleConstraint,
) -> XorConstraint:
    """(Sun ^ Moon) combined constraint."""
    return cast(XorConstraint, logical_sun_constraint ^ logical_moon_constraint)


@pytest.fixture
def logical_not_constraint(logical_sun_constraint: SunAngleConstraint) -> NotConstraint:
    """(~Sun) negated constraint."""
    return cast(NotConstraint, ~logical_sun_constraint)
