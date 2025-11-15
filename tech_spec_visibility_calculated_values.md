# ACROSS Visibility Calculated Values

## Introduction

The visibility calculation system in ACROSS tools currently computes constraint
checks (e.g., Sun angle, Moon angle, Earth limb) but only returns boolean
arrays indicating whether constraints are violated. This feature proposes
to extend the system to also record and output the calculated values (e.g.,
actual angles) used in these computations, allowing users to access to more detailed
visibility data, rather than just windows and boolean yes/no. This will allow
in future people to plot things like airmass over time for a target, and angles
from sun / earth moon that might be important in higher level calcuations.

### Terminology

- **Constraint**: A rule defining when an observation is not visible (e.g., too close to the Sun).
- **Calculated Values**: The numerical results of constraint computations (e.g., angular separations) before applying thresholds.
- **Visibility Window**: A time period when an observation is visible based on all constraints.
- **Ephemeris**: Orbital data for spacecraft or celestial bodies used in calculations.

### Goal of the Solution

Expose calculated values (e.g., Sun angles, Moon angles) alongside boolean constraint results.

### Acceptance Criteria / User Stories

1. **Record Calculated Values**: All constraint classes (Sun, Moon, Earth, Alt-Az) must populate `computed_values` with arrays of calculated metrics during evaluation.
2. **Merge Values in Visibility**: The `Visibility` class must collect and merge calculated values from all active constraints into a single output structure.
3. **JSON Output**: Calculated values must be serializable to JSON, with arrays of numerical values (e.g., angles in degrees) for each constraint type.
4. **Backward Compatibility**: Existing boolean constraint logic and API must remain unchanged; calculated values are additive.
5. **Support Multiple Constraints**: The output must dynamically include values only for active constraints (e.g., if no Moon constraint, no `moon_angle` array).
6. **Unit Handling**: Values must be in appropriate units (e.g., degrees for angles) and easily convertible for plotting/serialization.

### Assumptions

1. Users need access to raw calculated values for analysis (e.g., plotting angle trends over time).
2. Astropy units are acceptable for internal storage but must be converted to
   plain numbers for JSON output, on the user end the client will add units
   back in as appropriate.
3. The feature builds on existing constraint infrastructure without major refactoring.

## Solution

### Interfaces

This adds a new Schema the Visibility model with the name `computed_values`,
the Schema is `VisibilityComputedValues`:

```python
class VisibilityComputedValues(BaseSchema):
    """
    A class to hold computed values for the SunAngleConstraint.
    """

    sun_angle: u.Quantity | None = Field(
        default=None, description="Angular distance between the Sun and the coordinate"
    )
    moon_angle: u.Quantity | None = Field(
        default=None, description="Angular distance between the Moon and the coordinate"
    )
    earth_angle: u.Quantity | None = Field(
        default=None, description="Angular distance between the Earth and the coordinate"
    )
    alt_az: SkyCoord | None = Field(default=None, description="AltAz coordinates of the coordinate")
```

### Code examples

```python
from across.tools.visibility import EphemerisVisibility, compute_ephemeris_visibility

# Compute visibility with constraints
visibility = compute_ephemeris_visibility(
    begin=start_time,
    end=end_time,
    ephemeris=ephemeris,
    constraints=[sun_constraint, moon_constraint],
    coordinate=target_coord
)

# Access calculated values
sun_angles = visibility.computed_values.sun_angle  # Array of angles in degrees
moon_angles = visibility.computed_values.moon_angle  # Array of angles in degrees

# JSON output includes calculated values
json_output = visibility.model_dump_json()
# Contains: {"computed_values": {"sun_angle": [45.2, 46.1, ...], "moon_angle": [30.5, 31.2, ...]}, ...}
```

### JSON Serialization Example

When `visibility.model_dump_json()` is called, the output includes the calculated values in a nested structure:

```json
{
  "ra": 100.0,
  "dec": 45.0,
  "begin": "2023-01-01T00:00:00.000",
  "end": "2023-01-02T00:00:00.000",
  "visibility_windows": [...],
  "computed_values": {
    "sun_angle": [45.2, 46.1, 47.0, ...],
    "moon_angle": [30.5, 31.2, 32.1, ...],
    "earth_angle": [15.3, 16.0, 16.8, ...]
  },
  ...
}
```

The `computed_values` dict merges values from all constraints being calculated.

```python
from across.tools.visibility.constraints import SunAngleConstraint

# Constraints automatically record values during evaluation
constraint = SunAngleConstraint(min_angle=45.0)
result = constraint(time_array, ephemeris, coordinate)
# constraint.computed_values.sun_angle now holds the angle array
```

### Gotchas for Discussion

Adding new constraints means updating `VisibilityComputedValues` schema. This
technically means an API change, perhaps `VisibilityComputedValues` should be a
freeform `dict` rather than a fixed format?

### Technical Requirements

1. **Update Constraint Base Class**: Ensure `ConstraintABC` has a `computed_values` attribute (already implemented) and that all subclasses populate it during `__call__`.
2. **Implement Merging Logic**: Complete `_merge_computed_values` in `EphemerisVisibility` to collect arrays from constraints.
3. **Add Serialization Support**: Ensure `VisibilityComputedValues` serializes arrays correctly, excluding `None` values.
4. **Unit Tests**: Add tests verifying calculated values are recorded, merged, and output in JSON for various constraint combinations.
