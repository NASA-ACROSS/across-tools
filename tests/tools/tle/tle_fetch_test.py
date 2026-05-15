from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from httpx import HTTPStatusError
from pydantic import ValidationError
from spacetrack import AuthenticationError  # type: ignore[import-untyped]

from across.tools.core.schemas.tle import TLE
from across.tools.tle.exceptions import SpaceTrackAuthenticationError
from across.tools.tle.tle import TLEFetch, get_tle


class TestTLEFetch:
    """Test suite for the TLEFetch class."""

    def test_init_satellites(self, tle_fetch_object: TLEFetch) -> None:
        """Test TLEFetch initialization satellites."""
        assert tle_fetch_object.satellites == [{"name": "ISS", "id": 25544}]

    def test_init_epoch(self, tle_fetch_object: TLEFetch) -> None:
        """Test TLEFetch initialization epoch."""
        assert tle_fetch_object.epoch == datetime(2008, 9, 20)

    def test_get_returns_correct_satellite_name_from_input(self, tle_fetch_object: TLEFetch) -> None:
        """Test TLEFetch returns satellite name from input satellites list."""
        # This is now tested in the actual get() calls, since satellite_name
        # is no longer an instance attribute but derived from satellites
        assert tle_fetch_object.satellites[0]["name"] == "ISS"

    def test_init_spacetrack_user(self, tle_fetch_object: TLEFetch) -> None:
        """Test TLEFetch initialization spacetrack_user."""
        assert tle_fetch_object.spacetrack_user == "user"

    def test_init_spacetrack_pwd(self, tle_fetch_object: TLEFetch) -> None:
        """Test TLEFetch initialization spacetrack_pwd."""
        assert tle_fetch_object.spacetrack_pwd == "pass"

    @patch("across.tools.core.config.config.SPACETRACK_PWD", "env_pass")
    @patch("across.tools.core.config.config.SPACETRACK_USER", "env_user")
    @patch.dict("os.environ", {"SPACETRACK_USER": "env_user", "SPACETRACK_PWD": "env_pass"}, clear=True)
    def test_init_with_env_vars_user(self) -> None:
        """Test TLEFetch initialization with environment variable user."""
        tle_fetch = TLEFetch(satellites=[{"name": "ISS", "id": 25544}], epoch=datetime(2008, 9, 20))
        assert tle_fetch.spacetrack_user == "env_user"

    @patch("across.tools.core.config.config.SPACETRACK_PWD", "env_pass")
    @patch("across.tools.core.config.config.SPACETRACK_USER", "env_user")
    @patch.dict("os.environ", {"SPACETRACK_USER": "env_user", "SPACETRACK_PWD": "env_pass"})
    def test_init_with_env_vars_pwd(self) -> None:
        """Test TLEFetch initialization with environment variable password."""
        tle_fetch = TLEFetch(satellites=[{"name": "ISS", "id": 25544}], epoch=datetime(2008, 9, 20))
        assert tle_fetch.spacetrack_pwd == "env_pass"

    def test_init_raises_validation_error_if_satellites_not_list(self) -> None:
        """Test TLEFetch initialization raises ValidationErr if satellites is not a list."""
        with pytest.raises(ValidationError):
            TLEFetch(satellites={"name": "ISS", "id": 25544}, epoch=datetime(2008, 9, 20))  # type: ignore

    def test_init_raises_validation_error_if_satellite_missing_keys(self) -> None:
        """Test TLEFetch initialization raises ValidationErr if satellite dict lacks name or id."""
        with pytest.raises(ValidationError):
            TLEFetch(satellites=[{"name": "ISS"}], epoch=datetime(2008, 9, 20))  # type: ignore

    def test_init_raises_validation_error_if_satellite_name_not_string(self) -> None:
        """Test TLEFetch initialization raises ValidationErr if satellite name is not a string."""
        with pytest.raises(ValidationError):
            TLEFetch(satellites=[{"name": 123, "id": 25544}], epoch=datetime(2008, 9, 20))  # type: ignore

    def test_get_returns_tle_list_type(
        self,
        valid_spacetrack_tle_response: str,
        tle_fetch_object: TLEFetch,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """Test TLEFetch get method returns a list."""
        mock_spacetrack_instance.gp.return_value = valid_spacetrack_tle_response

        result = tle_fetch_object.get()
        assert isinstance(result, list)

    def test_get_returns_tle_list_length(
        self,
        valid_spacetrack_tle_response: str,
        tle_fetch_object: TLEFetch,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """Test TLEFetch get method returns one TLE for one response pair."""
        mock_spacetrack_instance.gp.return_value = valid_spacetrack_tle_response

        result = tle_fetch_object.get()
        assert len(result) == 1

    def test_get_returns_tle_list_item_type(
        self,
        valid_spacetrack_tle_response: str,
        tle_fetch_object: TLEFetch,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """Test TLEFetch get method returns TLE entries."""
        mock_spacetrack_instance.gp.return_value = valid_spacetrack_tle_response

        result = tle_fetch_object.get()
        assert isinstance(result[0], TLE)

    def test_get_returns_correct_norad_id(
        self,
        valid_spacetrack_tle_response: str,
        tle_fetch_object: TLEFetch,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """Test TLEFetch get method returns correct norad_id."""
        mock_spacetrack_instance.gp.return_value = valid_spacetrack_tle_response

        result = tle_fetch_object.get()
        assert result
        assert result[0].norad_id == 25544

    def test_get_returns_correct_satellite_name(
        self,
        valid_spacetrack_tle_response: str,
        tle_fetch_object: TLEFetch,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """Test TLEFetch get method returns correct satellite name."""
        mock_spacetrack_instance.gp.return_value = valid_spacetrack_tle_response

        result = tle_fetch_object.get()
        assert result
        assert result[0].satellite_name == "ISS"

    def test_get_empty_response(
        self,
        empty_spacetrack_tle_response: str,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """Test TLEFetch get method with an empty response."""
        mock_spacetrack_instance.gp.return_value = empty_spacetrack_tle_response

        tle_fetch = TLEFetch(
            satellites=[{"name": "ISS", "id": 25544}],
            epoch=datetime(2008, 9, 20),
            spacetrack_user="user",
            spacetrack_pwd="pass",
        )

        result = tle_fetch.get()
        assert result == []

    def test_get_list_satellites_returns_two_entries(
        self,
        multi_norad_spacetrack_tle_response: str,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """List satellites query returns one entry per requested satellite."""
        mock_spacetrack_instance.gp.return_value = multi_norad_spacetrack_tle_response

        tle_fetch = TLEFetch(
            satellites=[{"name": "ISS", "id": 25544}, {"name": "SWIFT", "id": 28485}],
            epoch=datetime(2008, 9, 20),
            spacetrack_user="user",
            spacetrack_pwd="pass",
        )

        result = tle_fetch.get()
        assert len(result) == 2

    def test_get_list_satellites_returns_matching_ids(
        self,
        multi_norad_spacetrack_tle_response: str,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """List satellites query returns only requested satellites."""
        mock_spacetrack_instance.gp.return_value = multi_norad_spacetrack_tle_response

        tle_fetch = TLEFetch(
            satellites=[{"name": "ISS", "id": 25544}, {"name": "SWIFT", "id": 28485}],
            epoch=datetime(2008, 9, 20),
            spacetrack_user="user",
            spacetrack_pwd="pass",
        )

        result = tle_fetch.get()
        assert {tle.norad_id for tle in result} == {25544, 28485}

    def test_get_list_satellites_keeps_newest_first_match(
        self,
        multi_norad_spacetrack_tle_response: str,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """List satellites query keeps newest entry per satellite based on ordering."""
        mock_spacetrack_instance.gp.return_value = multi_norad_spacetrack_tle_response

        tle_fetch = TLEFetch(
            satellites=[{"name": "ISS", "id": 25544}, {"name": "SWIFT", "id": 28485}],
            epoch=datetime(2008, 9, 20),
            spacetrack_user="user",
            spacetrack_pwd="pass",
        )

        result = tle_fetch.get()
        assert result[0].tle1.split()[3] == "08264.51782528"

    def test_get_filters_to_newest_epoch_for_same_norad_id(
        self,
        duplicate_norad_spacetrack_tle_response: str,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """When Space-Track returns multiple TLEs for same NORAD ID, keep only the newest."""
        mock_spacetrack_instance.gp.return_value = duplicate_norad_spacetrack_tle_response

        tle_fetch = TLEFetch(
            satellites=[{"name": "ISS", "id": 25544}],
            epoch=datetime(2008, 9, 20),
            spacetrack_user="user",
            spacetrack_pwd="pass",
        )

        result = tle_fetch.get()
        assert len(result) == 1
        assert result[0].tle1.split()[3] == "08264.51782528"

    def test_get_authentication_error(self, mock_spacetrack: MagicMock) -> None:
        """Test TLEFetch get method with an authentication error."""
        mock_client = MagicMock()
        mock_client.authenticate.side_effect = AuthenticationError()
        mock_spacetrack.return_value.__enter__.return_value = mock_client

        tle_fetch = TLEFetch(
            satellites=[{"name": "ISS", "id": 25544}],
            epoch=datetime(2008, 9, 20),
            spacetrack_user="user",
            spacetrack_pwd="pass",
        )

        with pytest.raises(SpaceTrackAuthenticationError):
            tle_fetch.get()

    def test_get_http_error(self, mock_spacetrack: MagicMock) -> None:
        """Test TLEFetch get method with an HTTP error."""
        mock_client = MagicMock()
        mock_client.authenticate.side_effect = HTTPStatusError(
            "Error", request=MagicMock(), response=MagicMock()
        )
        mock_spacetrack.return_value.__enter__.return_value = mock_client

        tle_fetch = TLEFetch(
            satellites=[{"name": "ISS", "id": 25544}],
            epoch=datetime(2008, 9, 20),
            spacetrack_user="user",
            spacetrack_pwd="pass",
        )

        with pytest.raises(SpaceTrackAuthenticationError):
            tle_fetch.get()


class TestGetTLE:
    """Test suite for the get_tle function."""

    def test_get_tle_returns_tle_list_type(
        self,
        valid_spacetrack_tle_response: str,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """Test get_tle returns a list."""
        mock_spacetrack_instance.gp.return_value = valid_spacetrack_tle_response

        result = get_tle(
            satellites=[{"name": "ISS", "id": 25544}],
            epoch=datetime(2008, 9, 20),
            spacetrack_user="test_user",
            spacetrack_pwd="test_pass",
        )

        assert isinstance(result, list)

    def test_get_tle_returns_tle_list_length(
        self,
        valid_spacetrack_tle_response: str,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """Test get_tle returns one element for one TLE pair."""
        mock_spacetrack_instance.gp.return_value = valid_spacetrack_tle_response

        result = get_tle(
            satellites=[{"name": "ISS", "id": 25544}],
            epoch=datetime(2008, 9, 20),
            spacetrack_user="test_user",
            spacetrack_pwd="test_pass",
        )

        assert len(result) == 1

    def test_get_tle_returns_tle_item_type(
        self,
        valid_spacetrack_tle_response: str,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """Test get_tle returns TLE entries."""
        mock_spacetrack_instance.gp.return_value = valid_spacetrack_tle_response

        result = get_tle(
            satellites=[{"name": "ISS", "id": 25544}],
            epoch=datetime(2008, 9, 20),
            spacetrack_user="test_user",
            spacetrack_pwd="test_pass",
        )

        assert isinstance(result[0], TLE)

    def test_get_tle_no_results(
        self,
        empty_spacetrack_tle_response: str,
        mock_spacetrack_instance: MagicMock,
    ) -> None:
        """Test when no TLEs are found"""
        mock_spacetrack_instance.gp.return_value = empty_spacetrack_tle_response

        result = get_tle(
            satellites=[{"name": "UNKNOWN", "id": 99999}],
            epoch=datetime(2008, 9, 20),
            spacetrack_user="test_user",
            spacetrack_pwd="test_pass",
        )

        assert result == []
