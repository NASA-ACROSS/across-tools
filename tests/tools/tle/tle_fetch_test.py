from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from httpx import HTTPStatusError
from spacetrack import AuthenticationError  # type: ignore[import-untyped]

from across.tools.core.schemas.tle import TLE
from across.tools.tle.exceptions import SpaceTrackAuthenticationError
from across.tools.tle.tle import TLEFetch, get_tle


class TestTLEFetch:
    """Test suite for the TLEFetch class."""

    def test_init_norad_id(self, tle_fetch_object: TLEFetch) -> None:
        """Test TLEFetch initialization norad_id."""
        assert tle_fetch_object.norad_id == 25544

    def test_init_epoch(self, tle_fetch_object: TLEFetch) -> None:
        """Test TLEFetch initialization epoch."""
        assert tle_fetch_object.epoch == datetime(2008, 9, 20)

    def test_init_satellite_name(self, tle_fetch_object: TLEFetch) -> None:
        """Test TLEFetch initialization satellite_name."""
        assert tle_fetch_object.satellite_name == "ISS"

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
        tle_fetch = TLEFetch(norad_id=25544, epoch=datetime(2008, 9, 20))
        assert tle_fetch.spacetrack_user == "env_user"

    @patch("across.tools.core.config.config.SPACETRACK_PWD", "env_pass")
    @patch("across.tools.core.config.config.SPACETRACK_USER", "env_user")
    @patch.dict("os.environ", {"SPACETRACK_USER": "env_user", "SPACETRACK_PWD": "env_pass"})
    def test_init_with_env_vars_pwd(self) -> None:
        """Test TLEFetch initialization with environment variable password."""
        tle_fetch = TLEFetch(norad_id=25544, epoch=datetime(2008, 9, 20))
        assert tle_fetch.spacetrack_pwd == "env_pass"

    def test_get_returns_tle_instance(
        self, mock_spacetrack: MagicMock, valid_spacetrack_tle_response: str, tle_fetch_object: TLEFetch
    ) -> None:
        """Test TLEFetch get method returns TLE instance."""
        mock_client = MagicMock()
        mock_client.gp_history.return_value = valid_spacetrack_tle_response
        mock_spacetrack.return_value.__enter__.return_value = mock_client

        result = tle_fetch_object.get()
        assert isinstance(result, TLE)

    def test_get_returns_correct_norad_id(
        self, mock_spacetrack: MagicMock, valid_spacetrack_tle_response: str, tle_fetch_object: TLEFetch
    ) -> None:
        """Test TLEFetch get method returns correct norad_id."""
        mock_client = MagicMock()
        mock_client.gp_history.return_value = valid_spacetrack_tle_response
        mock_spacetrack.return_value.__enter__.return_value = mock_client

        result = tle_fetch_object.get()
        assert result is not None
        assert result.norad_id == 25544

    def test_get_returns_correct_satellite_name(
        self, mock_spacetrack: MagicMock, valid_spacetrack_tle_response: str, tle_fetch_object: TLEFetch
    ) -> None:
        """Test TLEFetch get method returns correct satellite name."""
        mock_client = MagicMock()
        mock_client.gp_history.return_value = valid_spacetrack_tle_response
        mock_spacetrack.return_value.__enter__.return_value = mock_client

        result = tle_fetch_object.get()
        assert result is not None
        assert result.satellite_name == "ISS"

    def test_get_empty_response(self, mock_spacetrack: MagicMock) -> None:
        """Test TLEFetch get method with an empty response."""
        mock_client = MagicMock()
        mock_client.gp_history.return_value = ""
        mock_spacetrack.return_value.__enter__.return_value = mock_client

        tle_fetch = TLEFetch(
            norad_id=25544, epoch=datetime(2008, 9, 20), spacetrack_user="user", spacetrack_pwd="pass"
        )

        result = tle_fetch.get()
        assert result is None

    def test_get_authentication_error(self, mock_spacetrack: MagicMock) -> None:
        """Test TLEFetch get method with an authentication error."""
        mock_client = MagicMock()
        mock_client.authenticate.side_effect = AuthenticationError()
        mock_spacetrack.return_value.__enter__.return_value = mock_client

        tle_fetch = TLEFetch(
            norad_id=25544, epoch=datetime(2008, 9, 20), spacetrack_user="user", spacetrack_pwd="pass"
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
            norad_id=25544, epoch=datetime(2008, 9, 20), spacetrack_user="user", spacetrack_pwd="pass"
        )

        with pytest.raises(SpaceTrackAuthenticationError):
            tle_fetch.get()


class TestGetTLE:
    """Test suite for the get_tle function."""

    def test_get_tle_returns_tle_instance(
        self, mock_spacetrack: MagicMock, valid_spacetrack_tle_response: str
    ) -> None:
        """Test get_tle returns TLE instance"""
        mock_client = MagicMock()
        mock_client.gp_history.return_value = valid_spacetrack_tle_response
        mock_spacetrack.return_value.__enter__.return_value = mock_client

        result = get_tle(
            norad_id=25544,
            epoch=datetime(2008, 9, 20),
            spacetrack_user="test_user",
            spacetrack_pwd="test_pass",
        )

        assert isinstance(result, TLE)

    def test_get_tle_no_results(self, mock_spacetrack: MagicMock) -> None:
        """Test when no TLEs are found"""
        mock_spacetrack.authenticate.return_value = True
        mock_spacetrack.gp_history.return_value = ""

        result = get_tle(
            norad_id=99999,
            epoch=datetime(2008, 9, 20),
            spacetrack_user="test_user",
            spacetrack_pwd="test_pass",
        )

        assert result is None
