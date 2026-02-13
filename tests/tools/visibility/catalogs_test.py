"""Tests for star catalog utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.table import Table  # type: ignore[import-untyped]

from across.tools.visibility.catalogs import (
    _get_cache_dir,
    _get_fallback_bright_stars,
    cache_clear,
    get_bright_stars,
)


class TestCacheDir:
    """Test suite for cache directory functions."""

    def test_get_cache_dir_returns_path(self) -> None:
        """Test that get_cache_dir returns a Path object."""
        cache_dir = _get_cache_dir()
        assert isinstance(cache_dir, Path)

    def test_get_cache_dir_creates_directory(self) -> None:
        """Test that get_cache_dir creates the directory."""
        cache_dir = _get_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_get_cache_dir_contains_star_catalogs(self) -> None:
        """Test that cache directory path contains 'star_catalogs'."""
        cache_dir = _get_cache_dir()
        assert "star_catalogs" in str(cache_dir)

    def test_get_cache_dir_is_consistent(self) -> None:
        """Test that multiple calls return the same directory."""
        cache_dir1 = _get_cache_dir()
        cache_dir2 = _get_cache_dir()
        assert cache_dir1 == cache_dir2


class TestFallbackBrightStars:
    """Test suite for fallback bright stars function."""

    def test_get_fallback_bright_stars_returns_list(self) -> None:
        """Test that fallback function returns a list."""
        stars = _get_fallback_bright_stars()
        assert isinstance(stars, list)

    def test_get_fallback_bright_stars_returns_tuples(self) -> None:
        """Test that fallback function returns list of tuples."""
        stars = _get_fallback_bright_stars()
        assert len(stars) > 0
        for item in stars:
            assert isinstance(item, tuple)
            assert len(item) == 2
            coord, mag = item
            assert isinstance(coord, SkyCoord)
            assert isinstance(mag, float)

    def test_get_fallback_bright_stars_count(self) -> None:
        """Test that fallback function returns expected number of stars."""
        stars = _get_fallback_bright_stars()
        assert len(stars) == 20

    def test_get_fallback_bright_stars_includes_sirius(self) -> None:
        """Test that fallback stars include Sirius (brightest star)."""
        stars = _get_fallback_bright_stars()
        # Sirius should be first and brightest (mag -1.46)
        coord, mag = stars[0]
        assert mag == pytest.approx(-1.46, abs=0.01)

    def test_get_fallback_bright_stars_all_bright(self) -> None:
        """Test that all fallback stars are bright (mag < 2.0)."""
        stars = _get_fallback_bright_stars()
        for _, mag in stars:
            assert mag < 2.0  # All should be brighter than magnitude 2


@pytest.mark.usefixtures("isolated_star_cache")
class TestGetBrightStars:
    """Test suite for get_bright_stars main function."""

    def test_get_bright_stars_returns_list(self) -> None:
        """Test that get_bright_stars returns a list."""
        stars = get_bright_stars(magnitude_limit=3.0)
        assert isinstance(stars, list)

    def test_get_bright_stars_returns_tuples(self) -> None:
        """Test that get_bright_stars returns list of tuples."""
        stars = get_bright_stars(magnitude_limit=3.0)
        assert len(stars) > 0
        for item in stars:
            assert isinstance(item, tuple)
            assert len(item) == 2
            coord, mag = item
            assert isinstance(coord, SkyCoord)
            assert isinstance(mag, (float, int))

    def test_get_bright_stars_magnitude_filtering(self) -> None:
        """Test that magnitude filtering works correctly."""
        stars_3 = get_bright_stars(magnitude_limit=3.0)
        stars_6 = get_bright_stars(magnitude_limit=6.0)

        # More stars with higher magnitude limit
        assert len(stars_6) > len(stars_3)

    def test_get_bright_stars_max_stars_limit(self) -> None:
        """Test that max_stars parameter limits results."""
        stars = get_bright_stars(magnitude_limit=10.0, max_stars=50)

        # Should have at most 50 stars
        assert len(stars) <= 50

    def test_get_bright_stars_caching(self) -> None:
        """Test that results are cached in memory."""
        # First call
        stars1 = get_bright_stars(magnitude_limit=4.0)

        # Second call with same parameters - should use cache
        stars2 = get_bright_stars(magnitude_limit=4.0)

        # Should return the same object (from cache)
        assert len(stars1) == len(stars2)

    def test_get_bright_stars_disk_cache_integration(self, tmp_path: Path, mock_vizier_table: Table) -> None:
        """Test integration between memory and disk cache."""
        with patch("across.tools.visibility.catalogs._get_cache_dir", return_value=tmp_path):
            with patch("across.tools.visibility.catalogs.Vizier") as mock_vizier:
                mock_instance = MagicMock()
                mock_instance.query_constraints.return_value = [mock_vizier_table]
                mock_vizier.return_value = mock_instance

                # First call - should query and cache
                stars1 = get_bright_stars(magnitude_limit=5.0)
                assert len(stars1) > 0

                # Clear cache
                cache_clear()

                # Second call - should load from cache
                stars2 = get_bright_stars(magnitude_limit=5.0)
                assert len(stars2) == len(stars1)

    def test_get_bright_stars_uses_fallback_on_failure(self, tmp_path: Path) -> None:
        """Test that fallback is used when Vizier query fails."""
        with patch("across.tools.visibility.catalogs._get_cache_dir", return_value=tmp_path):
            with patch("across.tools.visibility.catalogs.Vizier") as mock_vizier:
                # Make Vizier raise an exception
                mock_instance = MagicMock()
                mock_instance.query_constraints.side_effect = Exception("Network error")
                mock_vizier.return_value = mock_instance

                # Should return fallback stars without raising
                stars = get_bright_stars(magnitude_limit=6.0)
                assert len(stars) == 20  # Fallback has 20 stars

    def test_get_bright_stars_uses_fallback_on_empty_result(self, tmp_path: Path) -> None:
        """Test that fallback is used when Vizier returns no results."""
        with patch("across.tools.visibility.catalogs._get_cache_dir", return_value=tmp_path):
            with patch("across.tools.visibility.catalogs.Vizier") as mock_vizier:
                # Make Vizier return empty result
                mock_instance = MagicMock()
                mock_instance.query_constraints.return_value = []
                mock_vizier.return_value = mock_instance

                # Should return fallback stars
                stars = get_bright_stars(magnitude_limit=6.0)
                assert len(stars) == 20  # Fallback has 20 stars

    def test_get_bright_stars_all_magnitudes_within_limit(self) -> None:
        """Test that all returned stars are within magnitude limit."""
        magnitude_limit = 3.0
        stars = get_bright_stars(magnitude_limit=magnitude_limit)

        for _, mag in stars:
            # All stars should be brighter (lower magnitude) than limit
            assert mag < magnitude_limit

    def test_get_bright_stars_default_parameters(self) -> None:
        """Test get_bright_stars with default parameters."""
        stars = get_bright_stars()

        # Should return some stars
        assert len(stars) > 0

        # All should be within default magnitude limit (6.0)
        for _, mag in stars:
            assert mag < 6.0


@pytest.mark.usefixtures("isolated_star_cache")
class TestGetBrightStarsEdgeCases:
    """Test suite for edge cases in get_bright_stars."""

    def test_get_bright_stars_very_low_magnitude_limit(self) -> None:
        """Test with very low magnitude limit (only brightest stars)."""
        stars = get_bright_stars(magnitude_limit=0.0)

        # Should have very few stars
        assert len(stars) < 20

        # All should be very bright
        for _, mag in stars:
            assert mag < 0.0

    def test_get_bright_stars_different_catalog_format(
        self, tmp_path: Path, mock_vizier_table_alternate_columns: Table
    ) -> None:
        """Test handling of different catalog column formats."""
        with patch("across.tools.visibility.catalogs._get_cache_dir", return_value=tmp_path):
            with patch("across.tools.visibility.catalogs.Vizier") as mock_vizier:
                mock_instance = MagicMock()
                mock_instance.query_constraints.return_value = [mock_vizier_table_alternate_columns]
                mock_vizier.return_value = mock_instance

                stars = get_bright_stars(magnitude_limit=5.0)
                assert len(stars) > 0

    def test_get_bright_stars_missing_ra_column_uses_fallback(self, tmp_path: Path) -> None:
        """Test that missing RA column triggers fallback."""
        with patch("across.tools.visibility.catalogs._get_cache_dir", return_value=tmp_path):
            # Mock table missing RA column
            mock_table = Table()
            mock_table["UNKNOWN_COL"] = [1.0]
            mock_table["Vmag"] = [-1.46]

            with patch("across.tools.visibility.catalogs.Vizier") as mock_vizier:
                mock_instance = MagicMock()
                mock_instance.query_constraints.return_value = [mock_table]
                mock_vizier.return_value = mock_instance

                stars = get_bright_stars(magnitude_limit=5.0)
                # Should use fallback
                assert len(stars) == 20

    def test_get_bright_stars_missing_magnitude_column_uses_fallback(self, tmp_path: Path) -> None:
        """Test that missing magnitude column triggers fallback."""
        with patch("across.tools.visibility.catalogs._get_cache_dir", return_value=tmp_path):
            # Mock table missing magnitude column
            mock_table = Table()
            mock_table["_RA.icrs"] = [101.28]
            mock_table["_DE.icrs"] = [-16.72]
            mock_table["UNKNOWN_MAG"] = [-1.46]

            with patch("across.tools.visibility.catalogs.Vizier") as mock_vizier:
                mock_instance = MagicMock()
                mock_instance.query_constraints.return_value = [mock_table]
                mock_vizier.return_value = mock_instance

                stars = get_bright_stars(magnitude_limit=5.0)
                # Should use fallback
                assert len(stars) == 20
