import pytest

from across.tools.core.enums import ComponentConstraintMode
from across.tools.core.schemas import RollAngleConstrainingObject, SpacecraftComponentVector
from across.tools.visibility.constraints.roll_angle import RollAngleConstraint


@pytest.fixture
def roll_angle_constraining_object_earth() -> RollAngleConstrainingObject:
    """Fixture for a basic RollAngleConstrainingObject instance."""
    return RollAngleConstrainingObject(name="earth", min_angle=20.0)


@pytest.fixture
def roll_angle_constraining_object_sun() -> RollAngleConstrainingObject:
    """Fixture for a basic RollAngleConstrainingObject instance."""
    return RollAngleConstrainingObject(name="sun", min_angle=90.0)


@pytest.fixture
def spacecraft_component_vector(
    roll_angle_constraining_object_earth: RollAngleConstrainingObject,
    roll_angle_constraining_object_sun: RollAngleConstrainingObject,
) -> SpacecraftComponentVector:
    """Fixture for a basic SpacecraftComponentVector instance."""
    return SpacecraftComponentVector(
        name="Test Vector",
        x=1.0,
        y=0.0,
        z=0.0,
        constraining_bodies=[roll_angle_constraining_object_sun, roll_angle_constraining_object_earth],
    )


@pytest.fixture
def roll_angle_constraint(spacecraft_component_vector: SpacecraftComponentVector) -> RollAngleConstraint:
    """Fixture for a basic RollAngleConstraint instance."""
    return RollAngleConstraint(constrained_components=[spacecraft_component_vector])


@pytest.fixture
def roll_angle_constraint_optimal_roll(
    spacecraft_component_vector: SpacecraftComponentVector,
) -> RollAngleConstraint:
    """Fixture for a RollAngleConstraint instance with a user-provided optimal roll."""
    return RollAngleConstraint(
        constrained_components=[spacecraft_component_vector], optimal_roll_angle=[45.0], allowed_roll=None
    )


@pytest.fixture
def roll_angle_constraint_constrain_any_components(
    spacecraft_component_vector: SpacecraftComponentVector,
) -> RollAngleConstraint:
    """Fixture for a RollAngleConstraint instance requiring only one component to be constrained."""
    return RollAngleConstraint(
        constrained_components=[spacecraft_component_vector],
        component_constraint_mode=ComponentConstraintMode.ANY,
    )
