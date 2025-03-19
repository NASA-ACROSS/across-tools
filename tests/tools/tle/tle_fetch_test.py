from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from httpx import HTTPStatusError
from spacetrack import AuthenticationError  # type: ignore[import-untyped]

from across.tools.core.schemas.tle import TLE
from across.tools.tle.exceptions import SpaceTrackAuthenticationError
from across.tools.tle.tle import TLEFetch


class TestTLEFetch:
    """Test suite for the TLEFetch class."""

    def test_init_with_credentials(self) -> None:
        """Test TLEFetch initialization with credentials."""
        tle_fetch = TLEFetch(
            norad_id=25544,
            epoch=datetime(2008, 9, 20),
            satellite_name="ISS",
            spacetrack_user="user",
            spacetrack_pwd="pass",
        )

        assert tle_fetch.norad_id == 25544
        assert tle_fetch.epoch == datetime(2008, 9, 20)
        assert tle_fetch.satellite_name == "ISS"
        assert tle_fetch.spacetrack_user == "user"
        assert tle_fetch.spacetrack_pwd == "pass"

    @patch.dict("os.environ", {"SPACETRACK_USER": "env_user", "SPACETRACK_PWD": "env_pass"})
    def test_init_with_env_vars(self) -> None:
        """Test TLEFetch initialization with environment variables."""
        tle_fetch = TLEFetch(norad_id=25544, epoch=datetime(2008, 9, 20))

        assert tle_fetch.spacetrack_user == "env_user"
        assert tle_fetch.spacetrack_pwd == "env_pass"

    def test_get_success(self, mock_spacetrack: MagicMock, valid_spacetrack_tle_response: str) -> None:
        """Test TLEFetch get method with a successful response."""
        mock_client = MagicMock()
        mock_client.tle.return_value = valid_spacetrack_tle_response
        mock_spacetrack.return_value.__enter__.return_value = mock_client

        tle_fetch = TLEFetch(
            norad_id=25544,
            epoch=datetime(2008, 9, 20),
            satellite_name="ISS",
            spacetrack_user="user",
            spacetrack_pwd="pass",
        )

        result = tle_fetch.get()

        assert isinstance(result, TLE)
        assert result.norad_id == 25544
        assert result.satellite_name == "ISS"

    def test_get_empty_response(self, mock_spacetrack: MagicMock) -> None:
        """Test TLEFetch get method with an empty response."""
        mock_client = MagicMock()
        mock_client.tle.return_value = ""
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
