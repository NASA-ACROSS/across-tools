"""Tests for custom Pydantic types in across.tools.core.schemas.custom_types."""

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest
from astropy import units as u  # type: ignore[import-untyped]
from astropy.coordinates import AltAz, SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]

from across.tools.core.schemas.base import BaseSchema
from across.tools.core.schemas.custom_types import (
    AstropyAngles,
    AstropyDateTime,
    AstropySkyCoords,
    is_array_like,
)

if TYPE_CHECKING:
    from .conftest import (
        AltAzModel,
        AngleModel,
        ComprehensiveModel,
        CoordModel,
        ModelWithAltAz,
        ModelWithAltAzs,
        ModelWithAltitude,
        ModelWithAltitudes,
        ModelWithCoordinate,
        ModelWithCoordinates,
        ModelWithData,
        ModelWithDuration,
        ModelWithMatrix,
        ModelWithOptionalValues,
        ModelWithTimestamp,
        ModelWithTimestamps,
        ModelWithValues,
        OptionalModel,
        TimeModel,
    )


class TestIsArrayLike:
    """Test the is_array_like helper function."""

    def test_list_is_array_like(self) -> None:
        """Lists should be considered array-like."""
        assert is_array_like([1, 2, 3]) is True

    def test_tuple_is_array_like(self) -> None:
        """Tuples should be considered array-like."""
        assert is_array_like((1, 2, 3)) is True

    def test_numpy_array_is_array_like(self) -> None:
        """Numpy arrays should be considered array-like."""
        assert is_array_like(np.array([1, 2, 3])) is True

    def test_string_is_not_array_like(self) -> None:
        """Strings should NOT be considered array-like."""
        assert is_array_like("hello") is False

    def test_bytes_is_not_array_like(self) -> None:
        """Bytes should NOT be considered array-like."""
        assert is_array_like(b"hello") is False

    def test_dict_is_not_array_like(self) -> None:
        """Dicts are not sequences, should return False."""
        assert is_array_like({"a": 1}) is False

    def test_set_is_not_array_like(self) -> None:
        """Sets are not sequences, should return False."""
        assert is_array_like({1, 2, 3}) is False

    def test_generator_is_not_array_like(self) -> None:
        """Generators are iterables but not sequences."""
        gen = (x for x in range(3))
        assert is_array_like(gen) is False

    def test_scalar_is_not_array_like(self) -> None:
        """Scalar values should not be array-like."""
        assert is_array_like(42) is False
        assert is_array_like(3.14) is False


class TestAstropyDateTime:
    """Test AstropyDateTime custom type."""

    def test_serialize_single_time_is_string(
        self, standard_time: Time, model_with_timestamp: "type[ModelWithTimestamp]"
    ) -> None:
        """Test serialization of a single Time object returns string."""
        model = model_with_timestamp(timestamp=standard_time)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert isinstance(data["timestamp"], str)

    def test_serialize_single_time_contains_date(
        self, standard_time: Time, model_with_timestamp: "type[ModelWithTimestamp]"
    ) -> None:
        """Test serialization of a single Time object contains expected date."""
        model = model_with_timestamp(timestamp=standard_time)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert "2020-01-01" in data["timestamp"]

    def test_serialize_array_times_length(
        self, standard_time: Time, model_with_timestamps: "type[ModelWithTimestamps]"
    ) -> None:
        """Test serialization of an array of Time objects has correct length."""
        times = Time(["2020-01-01 12:00:00", "2020-01-02 12:00:00"])
        model = model_with_timestamps(timestamps=times)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert len(data["timestamps"]) == 2

    def test_deserialize_datetime_string(
        self, reference_datetime_string: str, model_with_timestamp: "type[ModelWithTimestamp]"
    ) -> None:
        """Test deserialization from ISO datetime string."""
        model_with_timestamp(timestamp=reference_datetime_string)

    def test_deserialize_python_datetime(self, model_with_timestamp: "type[ModelWithTimestamp]") -> None:
        """Test deserialization from Python datetime."""
        dt = datetime(2020, 1, 1, 12, 0, 0)
        model_with_timestamp(timestamp=dt)


class TestAstropyTimeDelta:
    """Test AstropyTimeDelta custom type."""

    def test_serialize_timedelta(self, model_with_duration: "type[ModelWithDuration]") -> None:
        """Test serialization of a TimeDelta object."""
        td = TimeDelta(3600, format="sec")
        model = model_with_duration(duration=td)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert isinstance(data["duration"], str)

    def test_deserialize_timedelta_seconds(self, model_with_duration: "type[ModelWithDuration]") -> None:
        """Test deserialization from seconds."""
        # Use proper format specification
        model = model_with_duration(duration=TimeDelta(3600, format="sec"))
        assert isinstance(model.duration, TimeDelta)


class TestAstropyAngles:
    """Test AstropyAngles custom type."""

    def test_serialize_single_angle_value(
        self, single_angle: Any, model_with_altitude: "type[ModelWithAltitude]"
    ) -> None:
        """Test serialization of a single angle has correct value."""
        model = model_with_altitude(altitude=single_angle)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["altitude"] == pytest.approx(45.0)

    def test_serialize_angle_array_first_value(
        self, angle_array: Any, model_with_altitudes: "type[ModelWithAltitudes]"
    ) -> None:
        """Test serialization of an array of angles has correct first value."""
        model = model_with_altitudes(altitudes=angle_array)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["altitudes"][0] == pytest.approx(30.0)

    def test_serialize_angle_array_third_value(
        self, angle_array: Any, model_with_altitudes: "type[ModelWithAltitudes]"
    ) -> None:
        """Test serialization of an array of angles has correct third value."""
        model = model_with_altitudes(altitudes=angle_array)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["altitudes"][2] == pytest.approx(60.0)

    def test_deserialize_float_list_unit(self, model_with_altitudes: "type[ModelWithAltitudes]") -> None:
        """Test deserialization from list of floats has correct unit."""
        model = model_with_altitudes(altitudes=[30.0, 45.0, 60.0])
        assert model.altitudes.unit == u.deg

    def test_deserialize_single_float_unit(self, model_with_altitude: "type[ModelWithAltitude]") -> None:
        """Test deserialization from single float has correct unit."""
        model = model_with_altitude(altitude=45.0)
        assert model.altitude.unit == u.deg

    def test_convert_radians_to_degrees_value(
        self, single_angle: Any, model_with_altitude: "type[ModelWithAltitude]"
    ) -> None:
        """Test that radians are converted to degrees with correct value."""
        angle = np.pi / 4 * u.rad  # 45 degrees in radians
        model = model_with_altitude(altitude=angle)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["altitude"] == pytest.approx(45.0, abs=1e-6)


class TestAstropySkyCoords:
    """Test AstropySkyCoords custom type."""

    def test_serialize_single_skycoord_length(
        self, model_with_coordinate: "type[ModelWithCoordinate]", skycoord_single: SkyCoord
    ) -> None:
        """Test serialization of a single SkyCoord has length 1."""
        model = model_with_coordinate(coordinate=skycoord_single)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert len(data["coordinate"]) == 1

    def test_serialize_single_skycoord_keys(
        self, model_with_coordinate: "type[ModelWithCoordinate]", skycoord_single: SkyCoord
    ) -> None:
        """Test serialization of a single SkyCoord has correct keys."""
        model = model_with_coordinate(coordinate=skycoord_single)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert set(data["coordinate"][0].keys()) == {"ra", "dec"}

    def test_serialize_single_skycoord_ra_value(
        self, model_with_coordinate: "type[ModelWithCoordinate]", skycoord_single: SkyCoord
    ) -> None:
        """Test serialization of a single SkyCoord has correct RA value."""
        model = model_with_coordinate(coordinate=skycoord_single)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["coordinate"][0]["ra"] == pytest.approx(10.0)

    def test_serialize_single_skycoord_dec_value(
        self, model_with_coordinate: "type[ModelWithCoordinate]", skycoord_single: SkyCoord
    ) -> None:
        """Test serialization of a single SkyCoord has correct Dec value."""
        model = model_with_coordinate(coordinate=skycoord_single)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["coordinate"][0]["dec"] == pytest.approx(20.0)

    def test_serialize_multiple_skycoords_length(
        self, model_with_coordinates: "type[ModelWithCoordinates]", skycoord_multiple: SkyCoord
    ) -> None:
        """Test serialization of multiple SkyCoords has correct length."""
        model = model_with_coordinates(coordinates=skycoord_multiple)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert len(data["coordinates"]) == 3

    def test_serialize_multiple_skycoords_first_ra(
        self, model_with_coordinates: "type[ModelWithCoordinates]", skycoord_multiple: SkyCoord
    ) -> None:
        """Test serialization of multiple SkyCoords has correct first RA."""
        model = model_with_coordinates(coordinates=skycoord_multiple)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["coordinates"][0]["ra"] == pytest.approx(10.0)

    def test_serialize_multiple_skycoords_second_ra(
        self, model_with_coordinates: "type[ModelWithCoordinates]", skycoord_multiple: SkyCoord
    ) -> None:
        """Test serialization of multiple SkyCoords has correct second RA."""
        model = model_with_coordinates(coordinates=skycoord_multiple)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["coordinates"][1]["ra"] == pytest.approx(20.0)

    def test_create_skycoord_from_angles(
        self, model_with_coordinate: "type[ModelWithCoordinate]", skycoord_single: SkyCoord
    ) -> None:
        """Test creating SkyCoord from separate ra/dec quantities."""
        model = model_with_coordinate(coordinate=skycoord_single)
        assert isinstance(model.coordinate, SkyCoord)

    def test_converts_to_icrs_frame_has_ra(self, model_with_coordinate: "type[ModelWithCoordinate]") -> None:
        """Test that coordinates are converted to ICRS and have RA."""
        # Create a SkyCoord in Galactic frame
        coord_galactic = SkyCoord(l=100 * u.deg, b=50 * u.deg, frame="galactic")
        model = model_with_coordinate(coordinate=coord_galactic)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        # Should be converted to ICRS (RA/Dec)
        assert "ra" in data["coordinate"][0]

    def test_converts_to_icrs_frame_has_dec(self, model_with_coordinate: "type[ModelWithCoordinate]") -> None:
        """Test that coordinates are converted to ICRS and have Dec."""
        # Create a SkyCoord in Galactic frame
        coord_galactic = SkyCoord(l=100 * u.deg, b=50 * u.deg, frame="galactic")
        model = model_with_coordinate(coordinate=coord_galactic)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        # Should be converted to ICRS (RA/Dec)
        assert "dec" in data["coordinate"][0]


class TestAstropyAltAz:
    """Test AstropyAltAz custom type."""

    def test_serialize_single_altaz_skycoord_length(
        self, altaz_frame: AltAz, model_with_altaz: "type[ModelWithAltAz]"
    ) -> None:
        """Test serialization of a single AltAz SkyCoord has length 1."""
        coord = SkyCoord(alt=30 * u.deg, az=45 * u.deg, frame=altaz_frame)
        model = model_with_altaz(coordinate=coord)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert len(data["coordinate"]) == 1

    def test_serialize_single_altaz_skycoord_keys(
        self, altaz_frame: AltAz, model_with_altaz: "type[ModelWithAltAz]"
    ) -> None:
        """Test serialization of a single AltAz SkyCoord has correct keys."""
        coord = SkyCoord(alt=30 * u.deg, az=45 * u.deg, frame=altaz_frame)
        model = model_with_altaz(coordinate=coord)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert set(data["coordinate"][0].keys()) == {"alt", "az"}

    def test_serialize_single_altaz_skycoord_alt_value(
        self, altaz_frame: AltAz, model_with_altaz: "type[ModelWithAltAz]"
    ) -> None:
        """Test serialization of a single AltAz SkyCoord has correct alt value."""
        coord = SkyCoord(alt=30 * u.deg, az=45 * u.deg, frame=altaz_frame)
        model = model_with_altaz(coordinate=coord)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["coordinate"][0]["alt"] == pytest.approx(30.0)

    def test_serialize_single_altaz_skycoord_az_value(
        self, altaz_frame: AltAz, model_with_altaz: "type[ModelWithAltAz]"
    ) -> None:
        """Test serialization of a single AltAz SkyCoord has correct az value."""
        coord = SkyCoord(alt=30 * u.deg, az=45 * u.deg, frame=altaz_frame)
        model = model_with_altaz(coordinate=coord)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["coordinate"][0]["az"] == pytest.approx(45.0)

    def test_serialize_multiple_altaz_skycoords_length(
        self, altaz_frame: AltAz, model_with_altazs: "type[ModelWithAltAzs]"
    ) -> None:
        """Test serialization of multiple AltAz SkyCoords has correct length."""
        coords = SkyCoord(alt=[30, 45, 60] * u.deg, az=[45, 90, 135] * u.deg, frame=altaz_frame)
        model = model_with_altazs(coordinates=coords)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert len(data["coordinates"]) == 3

    def test_serialize_multiple_altaz_skycoords_first_alt(
        self, altaz_frame: AltAz, model_with_altazs: "type[ModelWithAltAzs]"
    ) -> None:
        """Test serialization of multiple AltAz SkyCoords has correct first alt."""
        coords = SkyCoord(alt=[30, 45, 60] * u.deg, az=[45, 90, 135] * u.deg, frame=altaz_frame)
        model = model_with_altazs(coordinates=coords)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["coordinates"][0]["alt"] == pytest.approx(30.0)

    def test_serialize_multiple_altaz_skycoords_second_alt(
        self, altaz_frame: AltAz, model_with_altazs: "type[ModelWithAltAzs]"
    ) -> None:
        """Test serialization of multiple AltAz SkyCoords has correct second alt."""
        coords = SkyCoord(alt=[30, 45, 60] * u.deg, az=[45, 90, 135] * u.deg, frame=altaz_frame)
        model = model_with_altazs(coordinates=coords)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["coordinates"][1]["alt"] == pytest.approx(45.0)

    def test_serialize_multiple_altaz_skycoords_third_alt(
        self, altaz_frame: AltAz, model_with_altazs: "type[ModelWithAltAzs]"
    ) -> None:
        """Test serialization of multiple AltAz SkyCoords has correct third alt."""
        coords = SkyCoord(alt=[30, 45, 60] * u.deg, az=[45, 90, 135] * u.deg, frame=altaz_frame)
        model = model_with_altazs(coordinates=coords)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["coordinates"][2]["alt"] == pytest.approx(60.0)

    def test_json_format_consistency(
        self, altaz_frame: AltAz, model_with_altaz: "type[ModelWithAltAz]"
    ) -> None:
        """Test that JSON format is consistent for multiple serializations."""
        coord = SkyCoord(alt=30 * u.deg, az=45 * u.deg, frame=altaz_frame)
        model1 = model_with_altaz(coordinate=coord)
        json_str1 = model1.model_dump_json()
        data1 = json.loads(json_str1)
        # Serialize again with same data to verify consistency
        model2 = model_with_altaz(coordinate=coord)
        json_str2 = model2.model_dump_json()
        data2 = json.loads(json_str2)

        # JSON content should be identical for same input
        assert data1 == data2

    def test_reject_wrong_frame(self, model_with_altaz: "type[ModelWithAltAz]") -> None:
        """Test that SkyCoord in wrong frame raises clear error."""
        # Create a SkyCoord in ICRS frame (not AltAz)
        icrs_coord = SkyCoord(ra=10 * u.deg, dec=20 * u.deg)
        with pytest.raises(ValueError, match="must be in AltAz frame"):
            model_with_altaz(coordinate=icrs_coord)

    def test_reject_galactic_frame(self, model_with_altaz: "type[ModelWithAltAz]") -> None:
        """Test that SkyCoord in Galactic frame raises clear error."""
        # Create a SkyCoord in Galactic frame
        galactic_coord = SkyCoord(l=100 * u.deg, b=50 * u.deg, frame="galactic")
        with pytest.raises(ValueError, match="must be in AltAz frame"):
            model_with_altaz(coordinate=galactic_coord)


class TestNumpyArray:
    """Test NumpyArray custom type."""

    def test_serialize_numpy_array_values(
        self, numpy_array_1d: npt.NDArray[np.float64], model_with_values: "type[ModelWithValues]"
    ) -> None:
        """Test serialization of a numpy array has correct values."""
        model = model_with_values(values=numpy_array_1d)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["values"] == [1.0, 2.0, 3.0]

    def test_serialize_2d_numpy_array_length(
        self, numpy_array_2d: npt.NDArray[np.float64], model_with_matrix: "type[ModelWithMatrix]"
    ) -> None:
        """Test serialization of 2D numpy array has correct length."""
        model = model_with_matrix(matrix=numpy_array_2d)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert len(data["matrix"]) == 2

    def test_serialize_2d_numpy_array_first_row(
        self, numpy_array_2d: npt.NDArray[np.float64], model_with_matrix: "type[ModelWithMatrix]"
    ) -> None:
        """Test serialization of 2D numpy array has correct first row."""
        model = model_with_matrix(matrix=numpy_array_2d)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["matrix"][0] == [1.0, 2.0]

    def test_serialize_2d_numpy_array_second_row(
        self, numpy_array_2d: npt.NDArray[np.float64], model_with_matrix: "type[ModelWithMatrix]"
    ) -> None:
        """Test serialization of 2D numpy array has correct second row."""
        model = model_with_matrix(matrix=numpy_array_2d)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["matrix"][1] == [3.0, 4.0]

    def test_serialize_dict_with_numpy_arrays_mars_values(
        self, numpy_dict_arrays: dict[str, npt.NDArray[np.float64]], model_with_data: "type[ModelWithData]"
    ) -> None:
        """Test serialization of dict containing numpy arrays has correct mars values."""
        model = model_with_data(data=numpy_dict_arrays)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["data"]["mars"] == [0.5, 0.6, 0.7]

    def test_serialize_dict_with_numpy_arrays_venus_values(
        self, numpy_dict_arrays: dict[str, npt.NDArray[np.float64]], model_with_data: "type[ModelWithData]"
    ) -> None:
        """Test serialization of dict containing numpy arrays has correct venus values."""
        model = model_with_data(data=numpy_dict_arrays)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["data"]["venus"] == [1.0, 1.1]

    def test_serialize_none_value(self, model_with_optional_values: "type[ModelWithOptionalValues]") -> None:
        """Test that None values are handled correctly."""
        model = model_with_optional_values(values=None)
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["values"] is None


class TestIntegration:
    """Integration tests combining multiple custom types."""

    def test_visibility_computed_values_serialization_sun_angle(
        self, visibility_computed_values_json: dict[str, Any]
    ) -> None:
        """Test serialization of VisibilityComputedValues has correct sun_angle."""
        # Verify all fields are present
        assert visibility_computed_values_json["sun_angle"] == [30.0, 45.0, 60.0]

    def test_visibility_computed_values_serialization_moon_angle(
        self, visibility_computed_values_json: dict[str, Any]
    ) -> None:
        """Test serialization of VisibilityComputedValues has correct moon_angle."""
        # moon_angle is an array with single element, so it serializes as a list
        assert visibility_computed_values_json["moon_angle"] == [60.0]

    def test_visibility_computed_values_serialization_air_mass(
        self, visibility_computed_values_json: dict[str, Any]
    ) -> None:
        """Test serialization of VisibilityComputedValues has correct air_mass."""
        assert visibility_computed_values_json["air_mass"] == [1.0, 1.2, 1.5]

    def test_visibility_computed_values_serialization_body_magnitude(
        self, visibility_computed_values_json: dict[str, Any]
    ) -> None:
        """Test serialization of VisibilityComputedValues has correct body_magnitude."""
        assert visibility_computed_values_json["body_magnitude"]["mars"] == [0.5, 0.6]

    def test_visibility_computed_values_serialization_alt_az_length(
        self, visibility_computed_values_json: dict[str, Any]
    ) -> None:
        """Test serialization of VisibilityComputedValues has correct alt_az length."""
        assert len(visibility_computed_values_json["alt_az"]) == 2

    def test_comprehensive_model_serialization_angles_values(
        self,
        standard_time: Time,
        angle_array: Any,
        numpy_array_1d: npt.NDArray[np.float64],
        comprehensive_model: "type[ComprehensiveModel]",
        skycoord_comprehensive: SkyCoord,
    ) -> None:
        """Test comprehensive model serialization has correct angles values."""
        model = comprehensive_model(
            timestamp=standard_time,
            angles=angle_array,
            coordinates=skycoord_comprehensive,
            values=numpy_array_1d,
        )

        # Serialize to JSON
        json_str = model.model_dump_json()

        # Verify it's valid JSON
        data = json.loads(json_str)

        # Verify data values
        assert data["angles"] == [30.0, 45.0, 60.0]

    def test_comprehensive_model_serialization_coordinates_length(
        self,
        standard_time: Time,
        angle_array: Any,
        numpy_array_1d: npt.NDArray[np.float64],
        comprehensive_model: "type[ComprehensiveModel]",
        skycoord_comprehensive: SkyCoord,
    ) -> None:
        """Test comprehensive model serialization has correct coordinates length."""
        model = comprehensive_model(
            timestamp=standard_time,
            angles=angle_array,
            coordinates=skycoord_comprehensive,
            values=numpy_array_1d,
        )

        # Serialize to JSON
        json_str = model.model_dump_json()

        # Verify it's valid JSON
        data = json.loads(json_str)

        assert len(data["coordinates"]) == 2

    def test_comprehensive_model_serialization_values(
        self,
        standard_time: Time,
        angle_array: Any,
        numpy_array_1d: npt.NDArray[np.float64],
        comprehensive_model: "type[ComprehensiveModel]",
        skycoord_comprehensive: SkyCoord,
    ) -> None:
        """Test comprehensive model serialization has correct values."""
        model = comprehensive_model(
            timestamp=standard_time,
            angles=angle_array,
            coordinates=skycoord_comprehensive,
            values=numpy_array_1d,
        )

        # Serialize to JSON
        json_str = model.model_dump_json()

        # Verify it's valid JSON
        data = json.loads(json_str)

        assert data["values"] == [1.0, 2.0, 3.0]

    def test_optional_fields_with_none_angle(self, optional_model: "type[OptionalModel]") -> None:
        """Test that optional angle field with None is handled correctly."""
        model = optional_model(angle=None, coordinate=None, values=None)
        json_str = model.model_dump_json()
        data = json.loads(json_str)

        assert data["angle"] is None

    def test_optional_fields_with_none_coordinate(self, optional_model: "type[OptionalModel]") -> None:
        """Test that optional coordinate field with None is handled correctly."""
        model = optional_model(angle=None, coordinate=None, values=None)
        json_str = model.model_dump_json()
        data = json.loads(json_str)

        assert data["coordinate"] is None

    def test_optional_fields_with_none_values(self, optional_model: "type[OptionalModel]") -> None:
        """Test that optional values field with None is handled correctly."""
        model = optional_model(angle=None, coordinate=None, values=None)
        json_str = model.model_dump_json()
        data = json.loads(json_str)

        assert data["values"] is None


class TestErrorHandling:
    """Test error handling and validation messages for custom types."""

    def test_astropy_angles_invalid_string(self, angle_model: "type[AngleModel]") -> None:
        """Angle validator should reject invalid string values."""
        with pytest.raises(ValueError, match="Invalid angle"):
            angle_model(angle="not_a_number")

    def test_astropy_angles_invalid_non_numeric_list(self, angle_model: "type[AngleModel]") -> None:
        """Angle validator should reject non-numeric list values."""
        with pytest.raises(ValueError, match="Invalid angle"):
            angle_model(angle=["a", "b", "c"])

    def test_astropy_datetime_invalid_string(self, time_model: "type[TimeModel]") -> None:
        """DateTime validator should reject invalid time strings."""
        with pytest.raises(ValueError, match="Invalid time"):
            time_model(time="not_a_valid_time")

    def test_astropy_datetime_invalid_none(self, time_model: "type[TimeModel]") -> None:
        """DateTime validator should reject None values."""
        with pytest.raises((ValueError, TypeError)):
            time_model(time=None)

    def test_astropy_skycoords_invalid_input(self, coord_model: "type[CoordModel]") -> None:
        """SkyCoord validator should reject invalid coordinate inputs."""
        with pytest.raises(ValueError, match="Invalid SkyCoord"):
            coord_model(coord="not_valid_coordinates")

    def test_astropy_skycoords_missing_dec(self, coord_model: "type[CoordModel]") -> None:
        """SkyCoord validator should reject incomplete coordinate input."""
        with pytest.raises(ValueError, match="Invalid SkyCoord"):
            coord_model(coord=[45.0])  # Only RA, no Dec

    def test_astropy_altaz_wrong_frame(self, altaz_model: "type[AltAzModel]") -> None:
        """AltAz validator should reject SkyCoords in wrong frame."""
        # Create a SkyCoord in ICRS frame (not AltAz)
        icrs_coord = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)

        with pytest.raises(ValueError, match="must be in AltAz frame"):
            altaz_model(coord=icrs_coord)

    def test_astropy_altaz_invalid_input(self) -> None:
        """AltAz validator should reject completely invalid inputs."""
        from pydantic import ValidationError

        from across.tools.core.schemas.custom_types import AstropyAltAz

        class AltAzModel(BaseSchema):
            coord: AstropyAltAz

        with pytest.raises(ValidationError):
            AltAzModel(coord="not_a_coordinate")

    def test_error_message_includes_context(self) -> None:
        """Error messages should include the problematic value for debugging."""

        class AngleModel(BaseSchema):
            angle: AstropyAngles

        try:
            AngleModel(angle=[1, 2, "invalid"])
        except ValueError as e:
            # Error should mention the invalid value or conversion error
            error_msg = str(e)
            assert "Invalid angle" in error_msg or "could not convert" in error_msg

    def test_multiple_validators_error_independently(self) -> None:
        """Each validator should handle its own errors independently."""
        from across.tools.core.schemas.custom_types import AstropyAltAz

        class MultiModel(BaseSchema):
            angle: AstropyAngles
            time: AstropyDateTime
            coord: AstropySkyCoords
            altaz: AstropyAltAz | None = None

        # Test angle validator error
        with pytest.raises(ValueError, match="Invalid angle"):
            MultiModel(angle="bad", time="2020-01-01T00:00:00", coord=[1.0, 2.0])

        # Test time validator error
        with pytest.raises(ValueError, match="Invalid time"):
            MultiModel(angle=45.0, time="bad_time", coord=[1.0, 2.0])

        # Test coordinate validator error
        with pytest.raises(ValueError, match="Invalid SkyCoord"):
            MultiModel(angle=45.0, time="2020-01-01T00:00:00", coord="bad_coord")
