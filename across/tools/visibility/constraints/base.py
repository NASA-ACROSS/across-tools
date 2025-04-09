from astropy.time import Time  # type: ignore[import-untyped]

from ...ephemeris import Ephemeris


def get_slice(time: Time, ephemeris: Ephemeris) -> slice:
    """
    Return a slice for what the part of the ephemeris that we're using.

    Arguments
    ---------
    time : Time
        The time to calculate the slice for
    ephemeris : Ephemeris
        The spacecraft ephemeris

    Returns
    -------
        The slice for the ephemeris
    """
    # Ensure that time is an array-like object
    if time.isscalar:
        raise NotImplementedError("Scalar time not supported")

    # Find the indices for the start and end of the time range and return a
    # slice for that range
    return slice(ephemeris.index(time[0]), ephemeris.index(time[-1]) + 1)
