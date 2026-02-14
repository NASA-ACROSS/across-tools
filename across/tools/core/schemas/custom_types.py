from collections.abc import Sequence
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
from astropy import units as u  # type: ignore[import-untyped]
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from pydantic import BeforeValidator, PlainSerializer


def is_array_like(obj: Any) -> bool:
    """Check if the object is sequence-like (has __len__ and __getitem__).

    Returns True for sequences like lists, tuples, and numpy arrays,
    but False for strings, bytes, and non-sequence iterables.
    """
    # Numpy arrays are array-like but don't inherit from Sequence
    if isinstance(obj, np.ndarray):
        return True
    # Check for standard sequences, excluding strings and bytes
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


# Custom pydantic type to handle serialization of Astropy Time
def validate_astropy_datetime(v: Any) -> Time:
    """Convert input to an Astropy Time object."""
    try:
        return Time(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid time: {v!r}") from e


def serialize_astropy_datetime(v: Time) -> str | list[str]:
    """Serialize Time object to ISO format string(s)."""
    try:
        return v.utc.datetime.isoformat() if v.isscalar else [t.isoformat() for t in v.utc.to_datetime()]
    except Exception as e:
        raise ValueError(f"Failed to serialize Time: {e}") from e


AstropyDateTime = Annotated[
    Time,
    BeforeValidator(validate_astropy_datetime),
    PlainSerializer(serialize_astropy_datetime, return_type=str | list[str], when_used="json-unless-none"),
]


def validate_astropy_timedelta(v: Any) -> TimeDelta:
    """Convert input to an Astropy TimeDelta object."""
    try:
        return TimeDelta(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid timedelta: {v!r}") from e


def serialize_astropy_timedelta(v: TimeDelta) -> str:
    """Serialize TimeDelta to string."""
    try:
        return v.to_value(u.s).__str__() if hasattr(v, "to_value") else str(v)
    except Exception as e:
        raise ValueError(f"Failed to serialize TimeDelta: {e}") from e


AstropyTimeDelta = Annotated[
    TimeDelta,
    BeforeValidator(validate_astropy_timedelta),
    PlainSerializer(serialize_astropy_timedelta, return_type=str, when_used="json-unless-none"),
]


def validate_astropy_angles(v: Any) -> u.Quantity:
    """Convert input to Astropy Angle Quantity in degrees."""
    try:
        if isinstance(v, u.Quantity):
            return v.to(u.deg)
        return u.Quantity(v, unit=u.deg)
    except (TypeError, ValueError, u.UnitConversionError) as e:
        raise ValueError(f"Invalid angle: {v!r}") from e


def serialize_astropy_angles_to_list(v: u.Quantity) -> list[float] | float:
    """Serialize Angle Quantity to float or list in degrees."""
    return v.to(u.deg).value.tolist()  # type: ignore[no-any-return]


AstropyAngles = Annotated[
    # Ensures the runtime type is an Astropy Quantity
    u.Quantity,
    BeforeValidator(validate_astropy_angles),
    PlainSerializer(
        serialize_astropy_angles_to_list,
        return_type=list[float] | float,
        when_used="json-unless-none",
    ),
]


# Custom pydantic type to handle serialization of numpy arrays to lists
def serialize_numpy_array(v: Any) -> list[Any] | float | int | Any:
    """Serialize a numpy array or scalar to a Python list or scalar value.

    Args:
        v: A value that may be a numpy array, numpy scalar, or Python value.

    Returns:
        A Python list (if input was ndarray), Python scalar (if numpy scalar),
        or the input value unchanged (if already a Python type).
    """
    if isinstance(v, np.ndarray):
        # Convert numpy array to list
        return v.tolist()
    elif isinstance(v, (np.integer, np.floating)):
        # Convert numpy scalar to Python scalar
        return v.item()
    else:
        # Already a Python type, return as-is
        return v


NumpyArray = Annotated[
    npt.NDArray[Any],
    PlainSerializer(serialize_numpy_array, return_type=list, when_used="json-unless-none"),
]


# Custom pydantic type to handle serialization of Astropy SkyCoord objects.
# JSON serialization will convert these to lists of dicts with ICRS RA/Dec in degrees.
def validate_astropy_skycoords(v: Any) -> SkyCoord:
    """Convert input to Astropy SkyCoord object."""
    try:
        return v if isinstance(v, SkyCoord) else SkyCoord(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid SkyCoord: {v!r}") from e


def serialize_astropy_skycoords_to_list(v: SkyCoord) -> list[dict[str, float]]:
    """Serialize SkyCoord to list of dicts with ICRS RA/Dec in degrees."""
    icrs = v.icrs
    if v.isscalar:
        return [{"ra": float(icrs.ra.deg), "dec": float(icrs.dec.deg)}]
    return [{"ra": float(ra), "dec": float(dec)} for ra, dec in zip(icrs.ra.deg, icrs.dec.deg)]


AstropySkyCoords = Annotated[
    SkyCoord,
    BeforeValidator(validate_astropy_skycoords),
    PlainSerializer(
        serialize_astropy_skycoords_to_list, return_type=list[dict[str, float]], when_used="json-unless-none"
    ),
]


# Custom pydantic type to handle serialization of Astropy SkyCoord objects
# with AltAz coordinates. JSON serialization will convert these to lists of
# dicts with altitude/azimuth in degrees.
def validate_astropy_altaz(v: Any) -> SkyCoord:
    """Convert input to AltAz SkyCoord and validate frame."""
    try:
        if isinstance(v, SkyCoord):
            _ = v.alt  # Verify AltAz frame by accessing alt/az
            _ = v.az
            return v
        return SkyCoord(v)
    except AttributeError as e:
        frame_name = v.frame.name if isinstance(v, SkyCoord) and hasattr(v, "frame") else "unknown"
        raise ValueError(f"SkyCoord must be in AltAz frame, got: {frame_name}") from e
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to validate AltAz SkyCoord: {v!r}") from e


def serialize_astropy_altaz_to_list(v: SkyCoord) -> list[dict[str, float]]:
    """Serialize SkyCoord to list of dicts with AltAz in degrees."""
    try:
        if v.isscalar:
            return [{"alt": float(v.alt.deg), "az": float(v.az.deg)}]
        return [{"alt": float(alt), "az": float(az)} for alt, az in zip(v.alt.deg, v.az.deg)]
    except AttributeError as e:
        frame_name = v.frame.name if hasattr(v, "frame") else "unknown"
        raise AttributeError(f"Cannot serialize {frame_name} frame as AltAz") from e


AstropyAltAz = Annotated[
    SkyCoord,
    BeforeValidator(validate_astropy_altaz),
    PlainSerializer(
        serialize_astropy_altaz_to_list,
        return_type=list[dict[str, float]],
        when_used="json-unless-none",
    ),
]
