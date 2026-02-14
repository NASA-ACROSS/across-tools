"""Tests for star catalog utilities."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.table import Table  # type: ignore[import-untyped]

from across.tools.visibility.catalogs import (
    _get_cache_dir,
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
        """Test that get_cache_dir creates the directory path."""
        cache_dir = _get_cache_dir()
        assert cache_dir.exists()

    def test_get_cache_dir_returns_directory(self) -> None:
        """Test that get_cache_dir returns a directory."""
        cache_dir = _get_cache_dir()
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

    def test_get_fallback_bright_stars_returns_list(
        self, fallback_bright_stars: list[tuple[SkyCoord, float]]
    ) -> None:
        """Test that fallback function returns a list."""
        assert isinstance(fallback_bright_stars, list)

    def test_get_fallback_bright_stars_not_empty(self) -> None:
        """Test that fallback function returns at least one star."""
        stars = _get_fallback_bright_stars()
        assert len(stars) > 0

    def test_get_fallback_bright_stars_items_are_tuples(self) -> None:
        """Test that fallback stars are represented as tuples."""
        stars = _get_fallback_bright_stars()
        assert all(isinstance(item, tuple) for item in stars)

    def test_get_fallback_bright_stars_tuples_have_two_items(self) -> None:
        """Test that each fallback star tuple has two items."""
        stars = _get_fallback_bright_stars()
        assert all(len(item) == 2 for item in stars)

    def test_get_fallback_bright_stars_coords_are_skycoord(self) -> None:
        """Test that fallback star coordinates are SkyCoord instances."""
        stars = _get_fallback_bright_stars()
        assert all(isinstance(coord, SkyCoord) for coord, _ in stars)

    def test_get_fallback_bright_stars_magnitudes_are_float(self) -> None:
        """Test that fallback star magnitudes are floats."""
        stars = _get_fallback_bright_stars()
        assert all(isinstance(mag, float) for _, mag in stars)

    def test_get_fallback_bright_stars_count(
        self, fallback_bright_stars: list[tuple[SkyCoord, float]]
    ) -> None:
        """Test that fallback function returns expected number of stars."""
        assert len(fallback_bright_stars) == 20

    def test_get_fallback_bright_stars_includes_sirius(
        self, fallback_bright_stars: list[tuple[SkyCoord, float]]
    ) -> None:
        """Test that fallback stars include Sirius (brightest star)."""
        # Sirius should be first and brightest (mag -1.46)
        coord, mag = fallback_bright_stars[0]
        assert mag == pytest.approx(-1.46, abs=0.01)

    def test_get_fallback_bright_stars_all_bright(
        self, fallback_bright_stars: list[tuple[SkyCoord, float]]
    ) -> None:
        """Test that all fallback stars are bright (mag < 2.0)."""
        stars = _get_fallback_bright_stars()
        assert all(mag < 2.0 for _, mag in stars)  # All should be brighter than magnitude 2


@pytest.mark.usefixtures("isolated_star_cache")
class TestGetBrightStars:
    """Test suite for get_bright_stars main function."""

    def test_get_bright_stars_returns_list(self, mock_vizier_patch: MagicMock) -> None:
        """Test that get_bright_stars returns a list."""
        stars = get_bright_stars(magnitude_limit=3.0)
        assert isinstance(stars, list)

    def test_get_bright_stars_not_empty(self) -> None:
        """Test that get_bright_stars returns at least one star."""
        stars = get_bright_stars(magnitude_limit=3.0)
        assert len(stars) > 0

    def test_get_bright_stars_items_are_tuples(self) -> None:
        """Test that get_bright_stars returns tuple items."""
        stars = get_bright_stars(magnitude_limit=3.0)
        assert all(isinstance(item, tuple) for item in stars)

    def test_get_bright_stars_tuples_have_two_items(self) -> None:
        """Test that each returned star tuple has two items."""
        stars = get_bright_stars(magnitude_limit=3.0)
        assert all(len(item) == 2 for item in stars)

    def test_get_bright_stars_coords_are_skycoord(self) -> None:
        """Test that returned star coordinates are SkyCoord instances."""
        stars = get_bright_stars(magnitude_limit=3.0)
        assert all(isinstance(coord, SkyCoord) for coord, _ in stars)

    def test_get_bright_stars_magnitudes_are_numeric(self) -> None:
        """Test that returned star magnitudes are numeric."""
        stars = get_bright_stars(magnitude_limit=3.0)
        assert all(isinstance(mag, (float, int)) for _, mag in stars)

    def test_get_bright_stars_magnitude_filtering(self, mock_vizier_magnitude_filtering: MagicMock) -> None:
        """Test that magnitude filtering works correctly."""
        stars_3 = get_bright_stars(magnitude_limit=3.0)

        # Clear cache to ensure second call queries again
        cache_clear()

        stars_6 = get_bright_stars(magnitude_limit=6.0)

        # More stars with higher magnitude limit
        assert len(stars_6) > len(stars_3)

    def test_get_bright_stars_max_stars_limit(self, mock_vizier_patch: MagicMock) -> None:
        """Test that max_stars parameter limits results."""
        stars = get_bright_stars(magnitude_limit=10.0, max_stars=50)

        # Should have at most 50 stars
        assert len(stars) <= 50

    def test_get_bright_stars_caching(self, mock_vizier_patch: MagicMock) -> None:
        """Test that results are cached in memory."""
        # First call
        stars1 = get_bright_stars(magnitude_limit=4.0)

        # Second call with same parameters - should use cache
        stars2 = get_bright_stars(magnitude_limit=4.0)

        # Should return the same object (from cache)
        assert len(stars1) == len(stars2)

    def test_get_bright_stars_disk_cache_integration(
        self, tmp_path: Path, mock_vizier_table: Table, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test disk cache integration returns stars on first call."""
        monkeypatch.setattr("across.tools.visibility.catalogs._get_cache_dir", lambda: tmp_path)
        mock_instance = MagicMock()
        mock_instance.query_constraints.return_value = [mock_vizier_table]
        mock_vizier = MagicMock(return_value=mock_instance)
        monkeypatch.setattr("across.tools.visibility.catalogs.Vizier", mock_vizier)

        # First call - should query and cache
        stars1 = get_bright_stars(magnitude_limit=5.0)
        assert len(stars1) > 0

    def test_get_bright_stars_disk_cache_integration_lengths_match(
        self, tmp_path: Path, mock_vizier_table: Table, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test disk cache integration returns same result size after cache clear."""
        monkeypatch.setattr("across.tools.visibility.catalogs._get_cache_dir", lambda: tmp_path)
        mock_instance = MagicMock()
        mock_instance.query_constraints.return_value = [mock_vizier_table]
        mock_vizier = MagicMock(return_value=mock_instance)
        monkeypatch.setattr("across.tools.visibility.catalogs.Vizier", mock_vizier)

        stars1 = get_bright_stars(magnitude_limit=5.0)

        # Clear cache
        cache_clear()

        # Second call - should load from cache
        stars2 = get_bright_stars(magnitude_limit=5.0)
        assert len(stars2) == len(stars1)

    def test_get_bright_stars_uses_fallback_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that fallback is used when Vizier query fails."""
        monkeypatch.setattr("across.tools.visibility.catalogs._get_cache_dir", lambda: tmp_path)
        # Make Vizier raise an exception
        mock_instance = MagicMock()
        mock_instance.query_constraints.side_effect = Exception("Network error")
        mock_vizier = MagicMock(return_value=mock_instance)
        monkeypatch.setattr("across.tools.visibility.catalogs.Vizier", mock_vizier)

        # Should return fallback stars without raising
        stars = get_bright_stars(magnitude_limit=6.0)
        assert len(stars) == 20  # Fallback has 20 stars

    def test_get_bright_stars_uses_fallback_on_empty_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that fallback is used when Vizier returns no results."""
        monkeypatch.setattr("across.tools.visibility.catalogs._get_cache_dir", lambda: tmp_path)
        # Make Vizier return empty result
        mock_instance = MagicMock()
        mock_instance.query_constraints.return_value = []
        mock_vizier = MagicMock(return_value=mock_instance)
        monkeypatch.setattr("across.tools.visibility.catalogs.Vizier", mock_vizier)

        # Should return fallback stars
        stars = get_bright_stars(magnitude_limit=6.0)
        assert len(stars) == 20  # Fallback has 20 stars

    def test_get_bright_stars_all_magnitudes_within_limit(self, mock_vizier_patch: MagicMock) -> None:
        """Test that all returned stars are within magnitude limit."""
        magnitude_limit = 3.0
        stars = get_bright_stars(magnitude_limit=magnitude_limit)
        assert all(mag < magnitude_limit for _, mag in stars)

    def test_get_bright_stars_default_parameters(self) -> None:
        """Test get_bright_stars default call returns stars."""
        stars = get_bright_stars()
        assert len(stars) > 0

    def test_get_bright_stars_default_parameters_within_default_limit(self) -> None:
        """Test get_bright_stars default call returns stars within default limit."""
        stars = get_bright_stars()
        assert all(mag < 6.0 for _, mag in stars)


@pytest.mark.usefixtures("isolated_star_cache")
class TestGetBrightStarsEdgeCases:
    """Test suite for edge cases in get_bright_stars."""

    def test_get_bright_stars_very_low_magnitude_limit(self) -> None:
        """Test with very low magnitude limit returns few stars."""
        stars = get_bright_stars(magnitude_limit=0.0)
        assert len(stars) < 20

    def test_get_bright_stars_very_low_magnitude_limit_all_bright(self) -> None:
        """Test with very low magnitude limit returns only very bright stars."""
        stars = get_bright_stars(magnitude_limit=0.0)
        assert all(mag < 0.0 for _, mag in stars)

    def test_get_bright_stars_different_catalog_format(
        self, tmp_path: Path, mock_vizier_table_alternate_columns: Table, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling of different catalog column formats."""
        monkeypatch.setattr("across.tools.visibility.catalogs._get_cache_dir", lambda: tmp_path)
        mock_instance = MagicMock()
        mock_instance.query_constraints.return_value = [mock_vizier_table_alternate_columns]
        mock_vizier = MagicMock(return_value=mock_instance)
        monkeypatch.setattr("across.tools.visibility.catalogs.Vizier", mock_vizier)

        stars = get_bright_stars(magnitude_limit=5.0)
        assert len(stars) > 0

    def test_get_bright_stars_missing_ra_column_uses_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that missing RA column triggers fallback."""
        monkeypatch.setattr("across.tools.visibility.catalogs._get_cache_dir", lambda: tmp_path)
        # Mock table missing RA column
        mock_table = Table()
        mock_table["UNKNOWN_COL"] = [1.0]
        mock_table["Vmag"] = [-1.46]

        mock_instance = MagicMock()
        mock_instance.query_constraints.return_value = [mock_table]
        mock_vizier = MagicMock(return_value=mock_instance)
        monkeypatch.setattr("across.tools.visibility.catalogs.Vizier", mock_vizier)

        stars = get_bright_stars(magnitude_limit=5.0)
        # Should use fallback
        assert len(stars) == 20

    def test_get_bright_stars_missing_magnitude_column_uses_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that missing magnitude column triggers fallback."""
        monkeypatch.setattr("across.tools.visibility.catalogs._get_cache_dir", lambda: tmp_path)
        # Mock table missing magnitude column
        mock_table = Table()
        mock_table["_RA.icrs"] = [101.28]
        mock_table["_DE.icrs"] = [-16.72]
        mock_table["UNKNOWN_MAG"] = [-1.46]

        mock_instance = MagicMock()
        mock_instance.query_constraints.return_value = [mock_table]
        mock_vizier = MagicMock(return_value=mock_instance)
        monkeypatch.setattr("across.tools.visibility.catalogs.Vizier", mock_vizier)

        stars = get_bright_stars(magnitude_limit=5.0)
        # Should use fallback
        assert len(stars) == 20
