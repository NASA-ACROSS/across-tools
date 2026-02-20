import pytest

from across.tools.visibility.constraints.airmass import AirmassConstraint


@pytest.fixture
def airmass_constraint() -> AirmassConstraint:
    """Default airmass constraint with default max_air_mass (2.0)."""
    return AirmassConstraint()


@pytest.fixture
def airmass_constraint_custom() -> AirmassConstraint:
    """Custom airmass constraint with max_air_mass=1.5."""
    return AirmassConstraint(max_air_mass=1.5)


@pytest.fixture
def airmass_constraint_high_threshold() -> AirmassConstraint:
    """Airmass constraint with high threshold (50.0) for no-violation tests."""
    return AirmassConstraint(max_air_mass=50.0)


@pytest.fixture
def airmass_constraint_low_threshold() -> AirmassConstraint:
    """Airmass constraint with low threshold (1.2) for violation tests."""
    return AirmassConstraint(max_air_mass=1.2)


@pytest.fixture
def airmass_constraint_high_max() -> AirmassConstraint:
    """Airmass constraint with high max_air_mass (10.0) for value range tests."""
    return AirmassConstraint(max_air_mass=10.0)
