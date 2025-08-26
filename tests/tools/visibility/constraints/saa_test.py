import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from shapely import Polygon

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.ephemeris.base import Ephemeris
from across.tools.visibility.constraints.saa import SAAPolygonConstraint


class TestSAAPolygonConstraintInstantiation:
    """Test suite for instantiating the SAAPolygonConstraint class."""

    def test_saa_polygon_constraint_short_name(self, saa_polygon_constraint: SAAPolygonConstraint) -> None:
        """Test that SAAPolygonConstraint has correct short_name."""
        assert saa_polygon_constraint.short_name == "SAA"

    def test_saa_polygon_constraint_name_value(self, saa_polygon_constraint: SAAPolygonConstraint) -> None:
        """Test that SAAPolygonConstraint has correct name value."""
        assert saa_polygon_constraint.name.value == ConstraintType.SAA.value

    def test_saa_polygon_constraint_polygon_not_none(
        self, saa_polygon_constraint: SAAPolygonConstraint
    ) -> None:
        """Test that SAAPolygonConstraint polygon is not None."""
        assert saa_polygon_constraint.polygon is not None

    def test_saa_polygon_constraint_polygon_coordinates(
        self, saa_polygon_constraint: SAAPolygonConstraint
    ) -> None:
        """Test that SAAPolygonConstraint polygon has correct coordinates."""
        assert isinstance(saa_polygon_constraint.polygon, Polygon)
        assert saa_polygon_constraint.polygon.exterior.coords[0] == (39.0, -30.0)

    def test_saa_polygon_constraint_instantiation_from_json(
        self, saa_polygon_constraint: SAAPolygonConstraint
    ) -> None:
        """Test that SAAPolygonConstraint can be instantiated from JSON."""
        json_data = saa_polygon_constraint.model_dump_json()
        saa_constraint = SAAPolygonConstraint.model_validate_json(json_data)
        assert saa_constraint is not None
        assert isinstance(saa_constraint, SAAPolygonConstraint)

    def test_saa_polygon_constraint_instantiation_from_dict_bad_polygon_type(
        self, saa_polygon_constraint: SAAPolygonConstraint
    ) -> None:
        """Test that SAAPolygonConstraint raises ValidationError with invalid polygon data."""
        model_dict = saa_polygon_constraint.model_dump()
        model_dict["polygon"] = 1

        with pytest.raises(ValueError):
            SAAPolygonConstraint.model_validate(model_dict)


class TestSAAPolygonConstraintCompute:
    """Test suite for the computing constraints with the SAAPolygonConstraint class."""

    def test_saa_polygon_constraint_call_returns_ndarray(
        self, saa_polygon_constraint: SAAPolygonConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns numpy ndarray."""
        result = saa_polygon_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert isinstance(result, np.ndarray)

    def test_saa_polygon_constraint_call_raises_value_error_if_ephemeris_not_computed(
        self,
        saa_polygon_constraint: SAAPolygonConstraint,
        sky_coord: SkyCoord,
        test_tle_ephemeris_no_compute: Ephemeris,
    ) -> None:
        """Test that __call__ method raises ValueError for invalid coordinates."""
        with pytest.raises(ValueError):
            saa_polygon_constraint(
                time=test_tle_ephemeris_no_compute.timestamp,
                ephemeris=test_tle_ephemeris_no_compute,
                coordinate=sky_coord,
            )

    def test_saa_polygon_constraint_returns_boolean_dtype(
        self, saa_polygon_constraint: SAAPolygonConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns boolean array."""
        result = saa_polygon_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert result.dtype == np.bool_

    def test_saa_polygon_constraint_result_length_matches_timestamp(
        self, saa_polygon_constraint: SAAPolygonConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that result length matches timestamp length."""
        result = saa_polygon_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert len(result) == len(test_tle_ephemeris.timestamp)

    def test_saa_polygon_constraint_contains_false_values(
        self, saa_polygon_constraint: SAAPolygonConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that result contains False values indicating no SAA presence."""
        result = saa_polygon_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert np.False_ in result

    def test_saa_polygon_constraint_contains_true_values(
        self, saa_polygon_constraint: SAAPolygonConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that result contains True values indicating SAA presence."""
        result = saa_polygon_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert np.True_ in result

    def test_saa_polygon_constraint_outside_saa_returns_false(
        self, saa_polygon_constraint: SAAPolygonConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that result is False for times outside the SAA."""
        outside_time = test_tle_ephemeris.timestamp[0]
        result = saa_polygon_constraint(
            time=Time([outside_time]), ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert result[0] is np.False_

    def test_saa_polygon_constraint_inside_saa_returns_true(
        self, saa_polygon_constraint: SAAPolygonConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that result is True for a time inside the SAA."""
        inside_time = test_tle_ephemeris.timestamp[-1]
        result = saa_polygon_constraint(
            time=Time([inside_time]), ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert result[0] is np.True_
