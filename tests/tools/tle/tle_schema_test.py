from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from across.tools.core.schemas.tle import TLE, TLEBase


class TestTLESchema:
    """Test suite for the TLE schema."""

    def test_tle_base_norad_id(self, valid_tle_data: dict[str, Any]) -> None:
        """Test TLEBase norad_id with valid data."""
        tle_base = TLEBase(**valid_tle_data)
        assert tle_base.norad_id == 25544

    def test_tle_base_satellite_name(self, valid_tle_data: dict[str, Any]) -> None:
        """Test TLEBase satellite_name with valid data."""
        tle_base = TLEBase(**valid_tle_data)
        assert tle_base.satellite_name == "ISS (ZARYA)"

    def test_tle_base_tle1(self, valid_tle_data: dict[str, Any]) -> None:
        """Test TLEBase tle1 with valid data."""
        tle_base = TLEBase(**valid_tle_data)
        assert tle_base.tle1 == "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"

    def test_tle_base_tle2(self, valid_tle_data: dict[str, Any]) -> None:
        """Test TLEBase tle2 with valid data."""
        tle_base = TLEBase(**valid_tle_data)
        assert tle_base.tle2 == "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    def test_tle_base_norad_id_optional(self, valid_tle_data: dict[str, Any]) -> None:
        """Test TLEBase with optional fields set to None."""
        tle_base = TLEBase(tle1=valid_tle_data["tle1"], tle2=valid_tle_data["tle2"])
        assert tle_base.norad_id is None

    def test_tle_base_satellite_name_optional(self, valid_tle_data: dict[str, Any]) -> None:
        """Test TLEBase with optional fields set to None."""
        tle_base = TLEBase(tle1=valid_tle_data["tle1"], tle2=valid_tle_data["tle2"])
        assert tle_base.satellite_name is None

    def test_tle_base_invalid_tle1_length(self) -> None:
        """Test TLEBase with invalid tle1 length."""
        with pytest.raises(ValueError):
            TLEBase(
                tle1="12345",
                tle2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
            )

    def test_tle_base_invalid_tle2_length(self) -> None:
        """Test TLEBase with invalid tle2 length."""
        with pytest.raises(ValueError):
            TLEBase(
                tle1="1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
                tle2="12345",
            )

    def test_tle_epoch_year_calculation(self, valid_tle_data: dict[str, Any]) -> None:
        """Test the epoch calculation of the TLE class."""
        tle = TLE(**valid_tle_data)
        assert tle.epoch.year == 2008

    def test_tle_epoch_month_calculation(self, valid_tle_data: dict[str, Any]) -> None:
        """Test the epoch calculation of the TLE class."""
        tle = TLE(**valid_tle_data)
        assert tle.epoch.month == 9

    def test_tle_epoch_day_calculation(self, valid_tle_data: dict[str, Any]) -> None:
        """Test the epoch calculation of the TLE class."""
        tle = TLE(**valid_tle_data)
        assert tle.epoch.day == 20

    def test_tle_epoch_day_matches_calculated_value(self, valid_tle_data: dict[str, Any]) -> None:
        """Test the epoch calculation of the TLE class."""
        with pytest.raises(ValidationError):
            TLE(**valid_tle_data, epoch=datetime(2008, 9, 21))

    def test_tle_line_one_doesnt_start_with_one(self, valid_tle_data: dict[str, Any]) -> None:
        """Test that TLE line 1 must start with '1'."""
        with pytest.raises(ValidationError):
            TLE(
                tle2=valid_tle_data["tle2"],
                tle1="2 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            )

    def test_tle_line_two_doesnt_start_with_two(self, valid_tle_data: dict[str, Any]) -> None:
        """Test that TLE line 2 must start with '2'."""
        with pytest.raises(ValidationError):
            TLE(
                tle1=valid_tle_data["tle1"],
                tle2="1 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
            )
