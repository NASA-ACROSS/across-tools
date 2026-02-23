from datetime import datetime, timedelta

import numpy as np
import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]

from across.tools.core.enums.constraint_type import ConstraintType
from across.tools.ephemeris.base import Ephemeris
from across.tools.footprint import Footprint
from across.tools.footprint.schemas import Pointing
from across.tools.visibility.constraints.pointing import PointingConstraint


class TestPointingConstraintInstantiation:
    """Test suite for instantiating the PointingConstraint class."""

    def test_pointing_constraint_short_name(self, pointing_constraint: PointingConstraint) -> None:
        """Test that PointingConstraint has correct short_name."""
        assert pointing_constraint.short_name == "Pointing"

    def test_pointing_constraint_name_value(self, pointing_constraint: PointingConstraint) -> None:
        """Test that PointingConstraint has correct name value."""
        assert pointing_constraint.name.value == ConstraintType.POINTING.value

    def test_pointing_constraint_pointings_not_none(self, pointing_constraint: PointingConstraint) -> None:
        """Test that PointingConstraint pointings is not None."""
        assert pointing_constraint.pointings is not None

    def test_pointing_constraint_pointing_type(self, pointing_constraint: PointingConstraint) -> None:
        """Test that PointingConstraint pointing has correct type."""
        assert isinstance(pointing_constraint.pointings[0], Pointing)

    def test_pointing_constraint_pointing_footprint_type(
        self, pointing_constraint: PointingConstraint
    ) -> None:
        """Test that PointingConstraint pointing footprint has correct type."""
        assert isinstance(pointing_constraint.pointings[0].footprint, Footprint)

    def test_pointing_constraint_instantiation_from_json(
        self, pointing_constraint: PointingConstraint
    ) -> None:
        """Test that PointingConstraint can be instantiated from JSON."""
        json_data = pointing_constraint.model_dump_json()
        new_pointing_constraint = PointingConstraint.model_validate_json(json_data)
        assert isinstance(new_pointing_constraint, PointingConstraint)

    def test_pointing_constraint_instantiation_from_dict_bad_polygon_type(
        self, pointing_constraint: PointingConstraint
    ) -> None:
        """Test that PointingConstraint raises ValidationError with invalid pointing data."""
        model_dict = pointing_constraint.model_dump()
        model_dict["pointings"] = 1

        with pytest.raises(ValueError):
            PointingConstraint.model_validate(model_dict)


class TestPointingConstraintCompute:
    """Test suite for the computing constraints with the PointingConstraint class."""

    def test_pointing_constraint_call_returns_ndarray(
        self, pointing_constraint: PointingConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns numpy ndarray."""
        result = pointing_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert isinstance(result, np.ndarray)

    def test_pointing_constraint_returns_boolean_dtype(
        self, pointing_constraint: PointingConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that __call__ method returns boolean array."""
        result = pointing_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert result.dtype == np.bool_

    def test_pointing_constraint_result_length_matches_timestamp(
        self, pointing_constraint: PointingConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that result length matches timestamp length."""
        result = pointing_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert len(result) == len(test_tle_ephemeris.timestamp)

    def test_pointing_constraint_inside_pointing_footprint_returns_false(
        self,
        pointing_constraint: PointingConstraint,
        origin_sky_coord: SkyCoord,
        test_tle_ephemeris: Ephemeris,
    ) -> None:
        """Test that result is False for a coordinate inside the pointing footprint."""
        result = pointing_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=origin_sky_coord
        )
        assert result[0] is np.False_

    def test_pointing_constraint_outside_pointing_footprint_returns_true(
        self, pointing_constraint: PointingConstraint, sky_coord: SkyCoord, test_tle_ephemeris: Ephemeris
    ) -> None:
        """Test that result is True for a coordinate outside the pointing footprint."""
        result = pointing_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=sky_coord
        )
        assert np.True_ in result

    def test_pointing_constraint_outside_pointing_time_returns_true(
        self,
        pointing_constraint: PointingConstraint,
        origin_sky_coord: SkyCoord,
        test_tle_ephemeris: Ephemeris,
        ephemeris_begin: datetime,
    ) -> None:
        """
        Test that result is False for a coordinate inside the pointing footprint but
        pointing time outside ephemeris timestamp.
        """
        pointing_constraint.pointings[0].start_time = ephemeris_begin + timedelta(days=1)
        pointing_constraint.pointings[0].end_time = ephemeris_begin + timedelta(days=2)
        result = pointing_constraint(
            time=test_tle_ephemeris.timestamp, ephemeris=test_tle_ephemeris, coordinate=origin_sky_coord
        )
        assert result[0] is np.True_
