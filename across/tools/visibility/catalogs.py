"""Star catalog utilities for visibility constraints."""

import hashlib
import pickle
from functools import lru_cache
from pathlib import Path
from typing import cast

import astropy.units as u  # type: ignore[import-untyped]
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astroquery.vizier import Vizier  # type: ignore[import-untyped]


def _get_cache_dir() -> Path:
    """
    Get the cache directory for star catalog data.

    Creates the directory if it doesn't exist. Uses platform-appropriate
    cache location.

    Returns
    -------
    Path
        Path to the cache directory.
    """
    # Use XDG cache directory on Linux/macOS, or user's home on Windows
    cache_home = Path.home() / ".cache"
    if not cache_home.exists():
        cache_home = Path.home()

    cache_dir = cache_home / "across-tools" / "star_catalogs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_key(magnitude_limit: float, catalog: str, max_stars: int | None) -> str:
    """
    Generate a unique cache key based on function arguments.

    Parameters
    ----------
    magnitude_limit : float
        Magnitude limit for stars.
    catalog : str
        Catalog identifier.
    max_stars : int or None
        Maximum number of stars.

    Returns
    -------
    str
        Cache key (hash of arguments).
    """
    # Create a string representation of the arguments
    key_str = f"{magnitude_limit}_{catalog}_{max_stars}"
    # Hash it to create a valid filename
    hash_obj = hashlib.sha256(key_str.encode())
    return hash_obj.hexdigest()[:16]  # Use first 16 characters


def _load_from_disk_cache(cache_key: str) -> list[tuple[SkyCoord, float]] | None:
    """
    Load star data from disk cache.

    Parameters
    ----------
    cache_key : str
        Cache key identifying the cached data.

    Returns
    -------
    list[tuple[SkyCoord, float]] or None
        Cached star data if available, None otherwise.
    """
    cache_file = _get_cache_dir() / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return cast(list[tuple[SkyCoord, float]], pickle.load(f))
        except Exception:
            # If cache is corrupted, remove it
            cache_file.unlink(missing_ok=True)
    return None


def _save_to_disk_cache(cache_key: str, data: list[tuple[SkyCoord, float]]) -> None:
    """
    Save star data to disk cache.

    Parameters
    ----------
    cache_key : str
        Cache key identifying the data.
    data : list[tuple[SkyCoord, float]]
        Star data to cache.
    """
    cache_file = _get_cache_dir() / f"{cache_key}.pkl"
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        # If we can't save to cache, continue without caching
        pass


@lru_cache(maxsize=10)
def get_bright_stars(
    magnitude_limit: float = 6.0,
    catalog: str = "I/239/hip_main",
    max_stars: int | None = None,
) -> list[tuple[SkyCoord, float]]:
    """
    Retrieve bright stars from a catalog filtered by magnitude.

    This function queries the Hipparcos catalog via Vizier and returns
    stars brighter than the specified magnitude limit along with their
    magnitudes. Results are cached both in memory (LRU cache) and on disk
    to minimize network calls to astroquery.

    Parameters
    ----------
    magnitude_limit : float, optional
        Magnitude limit for stars to retrieve. Stars brighter (lower magnitude)
        than this value will be returned. Default is 6.0 (naked eye visible).
    catalog : str, optional
        Vizier catalog identifier. Default is "I/239/hip_main" (Hipparcos Main Catalog).
        Other options include:
        - "I/259/tyc2" (Tycho-2 Catalogue)
        - "V/50/catalog" (Yale Bright Star Catalog)
    max_stars : int, optional
        Maximum number of stars to retrieve. If None, retrieves all stars
        brighter than magnitude_limit. Default is None.

    Returns
    -------
    list[tuple[SkyCoord, float]]
        List of tuples containing (sky coordinate, magnitude) for bright stars.

    Examples
    --------
    >>> # Get all stars brighter than magnitude 6
    >>> stars = get_bright_stars(magnitude_limit=6.0)
    >>> for coord, mag in stars[:5]:
    ...     print(f"Star at {coord.ra.deg:.2f}, {coord.dec.deg:.2f} with mag {mag:.2f}")
    >>> # Get brightest 100 stars
    >>> bright_100 = get_bright_stars(magnitude_limit=10.0, max_stars=100)

    Notes
    -----
    Results are cached using a two-level cache:
    1. In-memory LRU cache (up to 10 different parameter combinations)
    2. On-disk pickle cache (persistent across sessions)

    Disk cache location: ~/.cache/across-tools/star_catalogs/
    (or ~/across-tools/star_catalogs/ on systems without ~/.cache)

    The Hipparcos catalog contains ~118,000 stars. Filtering by magnitude
    significantly reduces the number of stars to consider:
    - mag < 3: ~200 stars
    - mag < 6: ~5,000 stars
    - mag < 8: ~25,000 stars
    """
    # Generate cache key for disk caching
    cache_key = _get_cache_key(magnitude_limit, catalog, max_stars)

    # Try to load from disk cache first
    cached_data = _load_from_disk_cache(cache_key)
    if cached_data is not None:
        return cached_data

    # Not in disk cache, query from Vizier
    # Configure Vizier query
    v = Vizier(
        row_limit=max_stars if max_stars is not None else -1,
    )

    # Query the catalog
    # For Hipparcos, filter by Vmag (V magnitude in Johnson system)
    try:
        result = v.query_constraints(
            catalog=catalog,
            Vmag=f"<{magnitude_limit}",
        )
    except Exception:
        # If catalog query fails, use fallback
        result = None

    if not result or len(result) == 0:
        # Fallback to hardcoded bright stars if catalog query fails
        fallback_data = _get_fallback_bright_stars()
        # Don't cache fallback data to disk
        return fallback_data

    # Extract coordinates from the catalog
    star_table = result[0]

    # Try to find RA/Dec columns (different catalogs use different names)
    ra_col = None
    dec_col = None

    # Common RA column names
    for col_name in ["_RA.icrs", "RA_ICRS", "RAJ2000", "RA(ICRS)", "_RAJ2000", "ra", "RAICRS"]:
        if col_name in star_table.colnames:
            ra_col = col_name
            break

    # Common Dec column names
    for col_name in ["_DE.icrs", "DE_ICRS", "DEJ2000", "DE(ICRS)", "_DEJ2000", "dec", "DEICRS"]:
        if col_name in star_table.colnames:
            dec_col = col_name
            break

    if ra_col is None or dec_col is None:
        # If we can't find RA/Dec columns, use fallback
        return _get_fallback_bright_stars()

    # Create SkyCoord objects from RA and Dec
    # Check if columns already have units
    ra_data = star_table[ra_col]
    dec_data = star_table[dec_col]

    # If data has units, use directly; otherwise assume degrees
    if hasattr(ra_data, "unit") and ra_data.unit is not None:
        ra_values = ra_data
        dec_values = dec_data
    else:
        ra_values = ra_data * u.deg
        dec_values = dec_data * u.deg

    stars = SkyCoord(
        ra=ra_values,
        dec=dec_values,
        frame="icrs",
    )

    # Find magnitude column
    mag_col = None
    for col_name in ["Vmag", "Bmag", "mag", "V"]:
        if col_name in star_table.colnames:
            mag_col = col_name
            break

    if mag_col is None:
        # If we can't find magnitude column, use fallback
        return _get_fallback_bright_stars()

    magnitudes = star_table[mag_col]

    # Convert to list of tuples (SkyCoord, magnitude)
    star_data = [
        (SkyCoord(ra=star.ra, dec=star.dec, frame="icrs"), float(mag)) for star, mag in zip(stars, magnitudes)
    ]

    # Save to disk cache for future use
    _save_to_disk_cache(cache_key, star_data)

    return star_data


def _get_fallback_bright_stars() -> list[tuple[SkyCoord, float]]:
    """
    Provide a fallback list of very bright stars if catalog query fails.

    This includes the 20 brightest stars in the sky, suitable for
    basic bright star avoidance when network access is unavailable.

    Returns
    -------
    list[tuple[SkyCoord, float]]
        List of tuples containing (sky coordinate, magnitude) for the brightest stars (magnitude < ~1.5).
    """
    return [
        (SkyCoord(ra="06h45m08.9s", dec="-16d42m58.0s", frame="icrs"), -1.46),  # Sirius
        (SkyCoord(ra="06h23m57.1s", dec="-52d41m44.4s", frame="icrs"), -0.74),  # Canopus
        (SkyCoord(ra="14h39m36.5s", dec="-60d50m02.3s", frame="icrs"), -0.27),  # Alpha Centauri
        (SkyCoord(ra="14h15m39.7s", dec="19d10m56.7s", frame="icrs"), -0.05),  # Arcturus
        (SkyCoord(ra="18h36m56.3s", dec="38d47m01.3s", frame="icrs"), 0.03),  # Vega
        (SkyCoord(ra="05h16m41.4s", dec="45d59m52.8s", frame="icrs"), 0.08),  # Capella
        (SkyCoord(ra="05h14m32.3s", dec="-08d12m05.9s", frame="icrs"), 0.13),  # Rigel
        (SkyCoord(ra="07h39m18.1s", dec="05d13m30.0s", frame="icrs"), 0.34),  # Procyon
        (SkyCoord(ra="05h55m10.3s", dec="07d24m25.4s", frame="icrs"), 0.50),  # Betelgeuse
        (SkyCoord(ra="01h37m42.8s", dec="-57d14m12.3s", frame="icrs"), 0.46),  # Achernar
        (SkyCoord(ra="14h03m49.4s", dec="-60d22m22.3s", frame="icrs"), 0.61),  # Hadar
        (SkyCoord(ra="12h26m35.9s", dec="-63d05m56.7s", frame="icrs"), 0.77),  # Acrux
        (SkyCoord(ra="20h41m25.9s", dec="45d16m49.3s", frame="icrs"), 0.77),  # Altair
        (SkyCoord(ra="16h29m24.5s", dec="-26d25m55.2s", frame="icrs"), 0.85),  # Aldebaran
        (SkyCoord(ra="13h25m11.6s", dec="-11d09m40.8s", frame="icrs"), 0.98),  # Spica
        (SkyCoord(ra="08h09m32.0s", dec="-47d20m11.7s", frame="icrs"), 1.06),  # Antares
        (SkyCoord(ra="22h57m39.0s", dec="-29d37m20.0s", frame="icrs"), 1.16),  # Fomalhaut
        (SkyCoord(ra="07h24m05.7s", dec="-28d58m19.5s", frame="icrs"), 1.14),  # Pollux
        (SkyCoord(ra="08h44m42.2s", dec="-59d30m34.1s", frame="icrs"), 1.25),  # Deneb
        (SkyCoord(ra="12h15m08.7s", dec="-17d32m30.9s", frame="icrs"), 1.25),  # Mimosa
    ]
