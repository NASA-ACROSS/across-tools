from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from astropy import units as u  # type: ignore[import-untyped]
from astropy.coordinates import AltAz, EarthLocation, SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from across.tools.core.schemas.base import BaseSchema
from across.tools.core.schemas.coordinate import Coordinate
from across.tools.core.schemas.custom_types import (
    AstropyAltAz,
    AstropyAngles,
    AstropyDateTime,
    AstropySkyCoords,
    AstropyTimeDelta,
    NumpyArray,
)
from across.tools.core.schemas.visibility import VisibilityComputedValues


class DummyModel(BaseSchema):
    """Test model for BaseSchema."""

    pass


class DummyModelTwo(BaseSchema):
    """Test model for BaseSchema."""

    pass


# Concrete model classes for custom type tests
class ModelWithTimestamp(BaseSchema):
    """Model with AstropyDateTime field."""

    timestamp: AstropyDateTime


class ModelWithTimestamps(BaseSchema):
    """Model with multiple timestamps."""

    timestamps: AstropyDateTime


class ModelWithDuration(BaseSchema):
    """Model with AstropyTimeDelta field."""

    duration: AstropyTimeDelta


class ModelWithAltitude(BaseSchema):
    """Model with single AstropyAngles field."""

    altitude: AstropyAngles


class ModelWithAltitudes(BaseSchema):
    """Model with multiple AstropyAngles fields."""

    altitudes: AstropyAngles


class ModelWithCoordinate(BaseSchema):
    """Model with single AstropySkyCoords field."""

    coordinate: AstropySkyCoords


class ModelWithCoordinates(BaseSchema):
    """Model with multiple AstropySkyCoords fields."""

    coordinates: AstropySkyCoords


class ModelWithAltAz(BaseSchema):
    """Model with single AstropyAltAz field."""

    coordinate: AstropyAltAz


class ModelWithAltAzs(BaseSchema):
    """Model with multiple AstropyAltAz fields."""

    coordinates: AstropyAltAz


class ModelWithValues(BaseSchema):
    """Model with NumpyArray field."""

    values: NumpyArray


class ModelWithValue(BaseSchema):
    """Model with single NumpyArray value field."""

    value: NumpyArray


class ModelWithMatrix(BaseSchema):
    """Model with 2D NumpyArray matrix field."""

    matrix: NumpyArray


class ModelWithData(BaseSchema):
    """Model with dict of NumpyArray data field."""

    data: dict[str, NumpyArray]


class ModelWithOptionalValues(BaseSchema):
    """Model with optional NumpyArray values field."""

    values: NumpyArray | None = None


class ComprehensiveModel(BaseSchema):
    """Model with multiple custom types for integration testing."""

    timestamp: AstropyDateTime
    angles: AstropyAngles
    coordinates: AstropySkyCoords
    values: NumpyArray


class OptionalModel(BaseSchema):
    """Model with optional custom type fields."""

    angle: AstropyAngles | None = None
    coordinate: AstropySkyCoords | None = None
    values: NumpyArray | None = None


class AngleModel(BaseSchema):
    """Model with AstropyAngles field for error testing."""

    angle: AstropyAngles


class TimeModel(BaseSchema):
    """Model with AstropyDateTime field for error testing."""

    time: AstropyDateTime


class CoordModel(BaseSchema):
    """Model with AstropySkyCoords field for error testing."""

    coord: AstropySkyCoords


class AltAzModel(BaseSchema):
    """Model with AstropyAltAz field for error testing."""

    coord: AstropyAltAz


class MultiModel(BaseSchema):
    """Model with multiple custom type validators for independent error testing."""

    angle: AstropyAngles
    time: AstropyDateTime
    coord: AstropySkyCoords
    altaz: AstropyAltAz | None = None


@pytest.fixture
def test_model_class() -> type[DummyModel]:
    """Return a DummyModel class."""
    return DummyModel


@pytest.fixture
def test_model() -> BaseSchema:
    """Return a DummyModel instance."""
    return DummyModel()


@pytest.fixture
def test_model_two() -> BaseSchema:
    """Return a DummyModel instance."""
    return DummyModelTwo()


@pytest.fixture
def model_with_timestamp() -> type[ModelWithTimestamp]:
    """Return ModelWithTimestamp class."""
    return ModelWithTimestamp


@pytest.fixture
def model_with_timestamps() -> type[ModelWithTimestamps]:
    """Return ModelWithTimestamps class."""
    return ModelWithTimestamps


@pytest.fixture
def model_with_duration() -> type[ModelWithDuration]:
    """Return ModelWithDuration class."""
    return ModelWithDuration


@pytest.fixture
def model_with_altitude() -> type[ModelWithAltitude]:
    """Return ModelWithAltitude class."""
    return ModelWithAltitude


@pytest.fixture
def model_with_altitudes() -> type[ModelWithAltitudes]:
    """Return ModelWithAltitudes class."""
    return ModelWithAltitudes


@pytest.fixture
def model_with_coordinate() -> type[ModelWithCoordinate]:
    """Return ModelWithCoordinate class."""
    return ModelWithCoordinate


@pytest.fixture
def model_with_coordinates() -> type[ModelWithCoordinates]:
    """Return ModelWithCoordinates class."""
    return ModelWithCoordinates


@pytest.fixture
def model_with_altaz() -> type[ModelWithAltAz]:
    """Return ModelWithAltAz class."""
    return ModelWithAltAz


@pytest.fixture
def model_with_altazs() -> type[ModelWithAltAzs]:
    """Return ModelWithAltAzs class."""
    return ModelWithAltAzs


@pytest.fixture
def model_with_values() -> type[ModelWithValues]:
    """Return ModelWithValues class."""
    return ModelWithValues


@pytest.fixture
def model_with_value() -> type[ModelWithValue]:
    """Return ModelWithValue class."""
    return ModelWithValue


@pytest.fixture
def model_with_matrix() -> type[ModelWithMatrix]:
    """Return ModelWithMatrix class."""
    return ModelWithMatrix


@pytest.fixture
def model_with_data() -> type[ModelWithData]:
    """Return ModelWithData class."""
    return ModelWithData


@pytest.fixture
def model_with_optional_values() -> type[ModelWithOptionalValues]:
    """Return ModelWithOptionalValues class."""
    return ModelWithOptionalValues


@pytest.fixture
def comprehensive_model() -> type[ComprehensiveModel]:
    """Return ComprehensiveModel class."""
    return ComprehensiveModel


@pytest.fixture
def optional_model() -> type[OptionalModel]:
    """Return OptionalModel class."""
    return OptionalModel


@pytest.fixture
def angle_model() -> type[AngleModel]:
    """Return AngleModel class."""
    return AngleModel


@pytest.fixture
def time_model() -> type[TimeModel]:
    """Return TimeModel class."""
    return TimeModel


@pytest.fixture
def coord_model() -> type[CoordModel]:
    """Return CoordModel class."""
    return CoordModel


@pytest.fixture
def altaz_model() -> type[AltAzModel]:
    """Return AltAzModel class."""
    return AltAzModel


@pytest.fixture
def multi_model() -> type[MultiModel]:
    """Return MultiModel class."""
    return MultiModel


@pytest.fixture
def valid_coordinates() -> list[Coordinate]:
    """Return a list of valid coordinates."""
    return [
        Coordinate(ra=0, dec=0),
        Coordinate(ra=1, dec=1),
        Coordinate(ra=1, dec=0),
        Coordinate(ra=0, dec=0),
    ]


@pytest.fixture
def valid_polygon_data(valid_coordinates: list[Coordinate]) -> dict[str, Any]:
    """Return a dictionary containing valid polygon data."""
    return {"coordinates": valid_coordinates}


@pytest.fixture
def coord_standard() -> Coordinate:
    """Fixture for a standard coordinate with RA=10.0, Dec=20.0."""
    return Coordinate(ra=10.0, dec=20.0)


@pytest.fixture
def coord_small_values() -> Coordinate:
    """Fixture for a coordinate with very small values."""
    return Coordinate(ra=0.0000123, dec=0.0000456)


@pytest.fixture
def coord_large_values() -> Coordinate:
    """Fixture for a coordinate with extreme values."""
    return Coordinate(ra=123.456789, dec=45.678901)


@pytest.fixture
def coord_negative_ra() -> Coordinate:
    """Fixture for a coordinate with negative RA."""
    return Coordinate(ra=-45.678901, dec=0.0)


@pytest.fixture
def coord_extreme() -> Coordinate:
    """Fixture for a coordinate with extreme values."""
    return Coordinate(ra=360, dec=90)


@pytest.fixture
def coord_extreme_negative() -> Coordinate:
    """Fixture for a coordinate with extreme negative values."""
    return Coordinate(ra=-360, dec=-90)


@pytest.fixture
def altaz_frame() -> AltAz:
    """Create a standard AltAz frame for testing."""
    location = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.m)
    time = Time("2020-01-01 12:00:00")
    return AltAz(location=location, obstime=time)


@pytest.fixture
def standard_time() -> Time:
    """Create a standard Time object for testing."""
    return Time("2020-01-01 12:00:00")


@pytest.fixture
def reference_datetime_string() -> str:
    """Reference datetime string for testing."""
    return "2020-01-01T12:00:00"


@pytest.fixture
def single_angle() -> Any:
    """Single angle quantity for testing."""
    return 45.0 * u.deg


@pytest.fixture
def angle_array() -> Any:
    """Array of angles for testing."""
    return [30.0, 45.0, 60.0] * u.deg


@pytest.fixture
def numpy_array_1d() -> npt.NDArray[np.float64]:
    """1D numpy array for testing."""

    return np.array([1.0, 2.0, 3.0], dtype=np.float64)


@pytest.fixture
def numpy_array_2d() -> npt.NDArray[np.float64]:
    """2D numpy array for testing."""

    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)


@pytest.fixture
def numpy_dict_arrays() -> dict[str, npt.NDArray[np.float64]]:
    """Dictionary of numpy arrays for testing."""

    return {
        "mars": np.array([0.5, 0.6, 0.7], dtype=np.float64),
        "venus": np.array([1.0, 1.1], dtype=np.float64),
    }


@pytest.fixture
def numpy_scalar() -> np.float64:
    """Numpy scalar for testing."""

    return np.float64(42.5)


@pytest.fixture
def python_list() -> list[float]:
    """Python list for testing."""
    return [1.0, 2.0, 3.0]


@pytest.fixture
def skycoord_single() -> SkyCoord:
    """Single SkyCoord for testing."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    return SkyCoord(ra=10 * u.deg, dec=20 * u.deg)


@pytest.fixture
def skycoord_multiple() -> SkyCoord:
    """Multiple SkyCoords for testing."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    return SkyCoord(ra=[10, 20, 30] * u.deg, dec=[20, 30, 40] * u.deg)


@pytest.fixture
def skycoord_comprehensive() -> SkyCoord:
    """SkyCoord for comprehensive model testing."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    return SkyCoord(ra=[10, 20] * u.deg, dec=[20, 30] * u.deg)


@pytest.fixture
def visibility_computed_values(altaz_frame: AltAz, angle_array: Any) -> VisibilityComputedValues:
    """VisibilityComputedValues fixture for integration testing."""
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from across.tools.core.schemas.visibility import VisibilityComputedValues

    airmass_values = np.array([1.0, 1.2, 1.5])
    return VisibilityComputedValues(
        sun_angle=angle_array,
        moon_angle=[60.0] * u.deg,  # Use array for consistency
        alt_az=SkyCoord(alt=[30, 45] * u.deg, az=[45, 90] * u.deg, frame=altaz_frame),
        air_mass=airmass_values,
        body_magnitude={"mars": np.array([0.5, 0.6])},
    )


@pytest.fixture
def visibility_computed_values_json(visibility_computed_values: VisibilityComputedValues) -> dict[str, Any]:
    """Serialized VisibilityComputedValues as dict."""
    import json

    json_str = visibility_computed_values.model_dump_json()
    return cast(dict[str, Any], json.loads(json_str))
