import pytest

from across.tools.visibility.constraints.earth_limb import EarthLimbConstraint


@pytest.fixture
def earth_limb_constraint() -> EarthLimbConstraint:
    """Fixture to provide an instance of EarthLimbConstraint for testing."""
    return EarthLimbConstraint(min_angle=33.0, max_angle=170.0)
