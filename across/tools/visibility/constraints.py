from typing import Literal

import astropy.units as u  # type: ignore[import]
import numpy as np
from astropy.coordinates import AltAz, SkyCoord  # type: ignore[import]
from astropy.time import Time  # type: ignore[import]
from pydantic import field_serializer
from shapely import Polygon, points

from across.tools.core.enums import TwilightType

from ..core.schemas.base import BaseSchema
from ..ephemeris import Ephemeris


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
    # If we're just passing a single time, we can just find the index for that
    if time.isscalar:
        # Find the index for the time and return a slice for that index
        index = ephemeris.ephindex(time)
        return slice(index, index + 1)
    else:
        # Find the indices for the start and end of the time range and return a
        # slice for that range
        return slice(ephemeris.ephindex(time[0]), ephemeris.ephindex(time[-1]) + 1)


class PolygonConstraint(BaseSchema):
    """
    Mixin class for constraints that are defined by a polygon. Mostly provides
    serialization and validation for the polygon.
    """

    polygon: Polygon | None = None

    @field_serializer("polygon")
    def serialize_dt(self, polygon) -> list[tuple[float, float]]:
        """Serialize the polygon to a list of tuples"""
        return [co for co in polygon.exterior.coords]


class SAAPolygonConstraint(PolygonConstraint):
    """
    Polygon based SAA constraint. The SAA is defined by a Shapely Polygon, and
    this constraint will calculate for a given set of times and a given
    ephemeris whether the spacecraft is in that SAA polygon.

    Attributes
    ----------
    polygon
        Shapely Polygon object defining the SAA polygon.
    """

    polygon: Polygon | None = None
    name: Literal["South Atlantic Anomaly"] = "South Atlantic Anomaly"
    short_name: Literal["SAA"] = "SAA"

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.ndarray:
        """
        Evaluate the constraint at the given time(s) and ephemeris position(s).

        Parameters
        ----------
        time : Time
            The time(s) at which to evaluate the constraint.
        ephemeris : Ephemeris
            The ephemeris position(s) at which to evaluate the constraint.
        skycoord : SkyCoord
            The sky coordinate(s) at which to evaluate the constraint.

        Returns
        -------
        ndarray
            A boolean array indicating whether the constraint is satisfied at
            the given time(s) and position(s). If time is scalar, returns a
            single boolean value.
        """
        # Find a slice what the part of the ephemeris that we're using
        i = get_slice(time, ephemeris)
        if ephemeris.longitude is None or ephemeris.latitude is None:
            raise ValueError("Ephemeris must contain longitude and latitude")
        in_constraint = np.zeros(len(ephemeris.longitude[i]), dtype=bool)
        if self.polygon is not None:
            in_constraint |= self.polygon.contains(points(ephemeris.longitude[i], ephemeris.latitude[i]))

        if self.polygon is not None:
            in_constraint |= self.polygon.contains(points(ephemeris.longitude[i], ephemeris.latitude[i]))
        # Return the result as True or False, or an array of True/False
        return in_constraint[0] if time.isscalar else in_constraint


class EarthLimbConstraint(BaseSchema):
    """
    For a given Earth limb avoidance angle, is a given coordinate inside this
    constraint?

    Parameters
    ----------
    name
        The name of the constraint.
    short_name
        The short name of the constraint.
    min_angle
        The minimum angle from the Earth limb that the spacecraft can point.

    Methods
    -------
    __call__(coord, ephemeris, earth_radius_angle=None)
        Checks if a given coordinate is inside the constraint.
    """

    short_name: Literal["Earth"] = "Earth"
    name: Literal["Earth Limb Constraint"] = "Earth Limb Constraint"
    min_angle: float | None = None
    max_angle: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.ndarray:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the Earth limb constraint. This is done by checking if the
        separation between the Earth and the spacecraft is less than the
        Earth's angular radius plus the minimum angle.

        NOTE: Assumes a circular approximation for Earth.

        Parameters
        ----------
        skycoord : SkyCoord
            The coordinate to check.
        time : Time
            The time to check.
        ephemeris : Ephemeris
            The ephemeris object.

        Returns
        -------
        bool
            `True` if the coordinate is inside the constraint, `False`
            otherwise.

        """
        # Find a slice what the part of the ephemeris that we're using
        i = get_slice(time, ephemeris)

        # Calculate the angular distance between the center of the Earth and
        # the object. Note that creating the SkyCoord here from ra/dec stored
        # in the ephemeris `earth` is 3x faster than just doing the separation
        # directly with `earth`.
        assert ephemeris.earth is not None and ephemeris.earth_radius_angle is not None

        in_constraint = np.zeros(len(ephemeris.earth[i]), dtype=bool)

        if self.min_angle is not None:
            in_constraint |= (
                SkyCoord(ephemeris.earth[i].ra, ephemeris.earth[i].dec).separation(skycoord)
                < ephemeris.earth_radius_angle[i] + self.min_angle * u.deg
            )
        if self.max_angle is not None:
            in_constraint |= (
                SkyCoord(ephemeris.earth[i].ra, ephemeris.earth[i].dec).separation(skycoord)
                > ephemeris.earth_radius_angle[i] + self.max_angle * u.deg
            )

        # Return the result as True or False, or an array of True/False
        return in_constraint[0] if time.isscalar and skycoord.isscalar else in_constraint


class SunConstraint(BaseSchema):
    """
    For a given Sun avoidance angle, is a given coordinate inside this
    constraint?

    Parameters
    ----------
    min_angle
        The minimum angle from the Sun that the spacecraft can point.

    Methods
    -------
    __call__(coord, ephemeris, sun_radius_angle=None)
        Checks if a given coordinate is inside the constraint.

    """

    short_name: Literal["Sun"] = "Sun"
    name: Literal["Sun Angle"] = "Sun Angle"
    min_angle: float | None = None
    max_angle: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.ndarray:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the Sun constraint. This is done by checking if the
        separation between the Sun and the spacecraft is less than the
        a given minimum angle or greater than a maximum angle (essentially an
        anti-Sun constraint).

        Parameters
        ----------
        skycoord : SkyCoord
            The coordinate to check.
        time : Time
            The time to check.
        ephemeris : Ephemeris
            The ephemeris object.

        Returns
        -------
        bool
            `True` if the coordinate is inside the constraint, `False`
            otherwise.

        """
        # Find a slice what the part of the ephemeris that we're using
        i = get_slice(time, ephemeris)

        # Calculate the angular distance between the center of the Sun and
        # the object. Note that creating the SkyCoord here from ra/dec stored
        # in the ephemeris `sun` is 3x faster than just doing the separation
        # directly with `sun`.
        assert ephemeris.sun is not None

        # Construct the constraint based on the minimum and maximum angles
        in_constraint = np.zeros(len(ephemeris.sun[i]), dtype=bool)

        if self.min_angle is not None:
            in_constraint = (
                SkyCoord(ephemeris.sun[i].ra, ephemeris.sun[i].dec).separation(skycoord)
                < self.min_angle * u.deg
            )
        if self.max_angle is not None:
            in_constraint |= (
                SkyCoord(ephemeris.sun[i].ra, ephemeris.sun[i].dec).separation(skycoord)
                > self.max_angle * u.deg
            )

        # Return the result as True or False, or an array of True/False
        return in_constraint[0] if time.isscalar and skycoord.isscalar else in_constraint


class MoonConstraint(BaseSchema):
    """
    For a given Moon avoidance angle, is a given coordinate inside this
    constraint?

    Parameters
    ----------
    min_angle
        The minimum angle from the Moon that the spacecraft can point.

    Methods
    -------
    __call__(coord, ephemeris, moon_radius_angle=None)
        Checks if a given coordinate is inside the constraint.

    """

    short_name: Literal["Moon"] = "Moon"
    name: Literal["Moon Angle"] = "Moon Angle"
    min_angle: float | None = None
    max_angle: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.ndarray:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the Moon constraint. This is done by checking if the
        separation between the Moon and the spacecraft is less than the
        Moon's angular radius plus the minimum angle.

        Parameters
        ----------
        skycoord : SkyCoord:
            The coordinate to check.
        time : Time
            The time to check.
        ephemeris : Ephemeris
            The ephemeris object.

        Returns
        -------
        bool
            `True` if the coordinate is inside the constraint, `False`
            otherwise.

        """
        # Find a slice what the part of the ephemeris that we're using
        i = get_slice(time, ephemeris)

        # Calculate the angular distance between the center of the Moon and
        # the object. Note that creating the SkyCoord here from ra/dec stored
        # in the ephemeris `moon` is 3x faster than just doing the separation
        # directly with `moon`.
        assert ephemeris.moon is not None

        in_constraint = np.zeros(len(ephemeris.moon[i]), dtype=bool)
        if self.min_angle is None:
            in_constraint |= (
                SkyCoord(ephemeris.moon[i].ra, ephemeris.moon[i].dec).separation(skycoord)
                < self.min_angle * u.deg
            )
        if self.max_angle is not None:
            in_constraint |= (
                SkyCoord(ephemeris.moon[i].ra, ephemeris.moon[i].dec).separation(skycoord)
                > self.max_angle * u.deg
            )

        # Return the result as True or False, or an array of True/False
        return in_constraint[0] if time.isscalar and skycoord.isscalar else in_constraint


class PoleConstraint(BaseSchema):
    """
    For a given pole avoidance angle, is a given coordinate inside this
    constraint?

    Parameters
    ----------
    min_angle
        The minimum angle from the pole that the spacecraft can point.

    Methods
    -------
    __call__(coord, ephemeris, polesize=None)
        Checks if a given coordinate is inside the constraint.

    """

    short_name: Literal["Pole"] = "Pole"
    name: Literal["Orbit Pole Angle"] = "Orbit Pole Angle"
    min_angle: float | None = None
    max_angle: float | None = None
    earth_limb_pole: bool = True

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.ndarray:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the Pole constraint. The pole constraint is defined as the
        minimum angle away from the poles of the orbit that the spacecraft can
        be pointed.

        This is done by checking if the separation between the Pole and the
        spacecraft is less than the Pole's angular radius plus the minimum
        angle.

        Parameters
        ----------
        skycoord
            The coordinate to check.
        time
            The time to check.
        ephemeris
            The ephemeris object.

        Returns
        -------
        bool
            `True` if the coordinate is inside the constraint, `False`
            otherwise.

        """
        # Find a slice what the part of the ephemeris that we're using
        i = get_slice(time, ephemeris)

        # Calculate the angular distance between the center of the Pole and the
        # object. Note that creating the SkyCoord here from ra/dec stored in
        # the ephemeris `pole` is 3x faster than just doing the separation
        # directly with `pole`.

        # Find the vector of the orbit pole, which is cross product of the
        # orbit vector and the velocity vector
        assert (
            ephemeris.gcrs.cartesian.xyz.value is not None
            and ephemeris.gcrs.velocity.d_xyz.value is not None
            and ephemeris.earth_radius_angle is not None
        )

        # Calculate the coordinate of the orbit pole
        polevec = ephemeris.gcrs.cartesian.xyz.value[i].cross(ephemeris.gcrs.velocity.d_xyz.value[i])
        pole = SkyCoord(polevec / polevec.norm())

        # If this is an Earth Limb driven pole constraint, then we need to
        # calculate the pole from the Earth limb constraint, in this case
        # min_angle is the minimum angle from the Earth limb that the
        # spacecraft can point.
        if self.earth_limb_pole and self.min_angle is not None:
            min_angle = ephemeris.earth_radius_angle[i].to_value("deg") + (self.min_angle - 90)
        else:
            # Assume self.min_angle is minimum angle we can be from the pole
            min_angle = self.min_angle

        # Calculate constraint as being within the min_angle of the northern or
        # southern pole
        polesep = pole.separation(skycoord)
        # If the pole separation is greater than 90 degrees, then we're
        # actually closer to the other pole
        index = np.where(polesep.value > 90)
        polesep[index] = 180 * u.deg - polesep[index]

        in_constraint = np.zeros(len(polesep), dtype=bool)
        if self.min_angle is not None:
            in_constraint |= polesep.value < min_angle
        if self.max_angle is not None:
            in_constraint |= polesep.value > self.max_angle

        # Return the result as True or False, or an array of True/False
        return in_constraint[0] if time.isscalar and skycoord.isscalar else in_constraint


class RamConstraint(BaseSchema):
    """
    For a given Ram avoidance angle, is a given coordinate inside this
    constraint?

    Parameters
    ----------
    min_angle
        The minimum angle from the Ram that the spacecraft can point.

    Methods
    -------
    __call__(coord, ephemeris, ramsize=None)
        Checks if a given coordinate is inside the constraint.

    """

    name: Literal["Ram Direction Constraint"] = "Ram Direction Constraint"
    short_name: Literal["Ram"] = "Ram"
    min_angle: float | None = None
    max_angle: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.ndarray:
        """
        Check for a given time, ephemeris and coordinate if positions given are
        inside the Ram direction constraint. Ram direction is the the direction
        the spacecraft is travelling in, and is often avoided due to an
        increased chance of particles entering the telescope if pointed along
        the direction of motion. This is done by checking if the separation
        between the spacecraft velocity vector and the spacecraft pointing is
        less than the minimum angle allowed or the maximum allowed angle.

        Parameters
        ----------
        skycoord
            The coordinate to check.
        time
            The time to check.
        ephemeris
            The ephemeris object.

        Returns
        -------
        bool
            `True` if the coordinate is inside the constraint, `False`
            otherwise.

        """
        # Find a slice what the part of the ephemeris that we're using
        i = get_slice(time, ephemeris)

        # Calculate the angular distance between the center of the Ram
        # direction and the object. Note that creating the SkyCoord here from
        # ra/dec stored in the ephemeris `ram` is 3x faster than just doing the
        # separation directly with `ram`.
        assert ephemeris.gcrs.velocity.d_xyz.value is not None

        in_constraint = np.zeros(len(ephemeris.gcrs.velocity.d_xyz.value[i]), dtype=bool)

        if self.min_angle is None:
            in_constraint |= (
                SkyCoord(ephemeris.gcrs.velocity.d_xyz.value[i]).separation(skycoord) < self.min_angle * u.deg
            )
        if self.max_angle is not None:
            in_constraint |= (
                SkyCoord(ephemeris.gcrs.velocity.d_xyz.value[i]).separation(skycoord) > self.max_angle * u.deg
            )

        # Return the result as True or False, or an array of True/False
        return in_constraint[0] if time.isscalar and skycoord.isscalar else in_constraint


class SpaceCraftDayConstraint(BaseSchema):
    """
    A constraint that determines if a spacecraft is in daylight. This
    constraint checks whether the spacecraft (observatory) is in daytime,
    defined as when any part of the Sun is above the Earth's limb from the
    spacecraft's perspective.

    Parameters
    ----------
    whole_sun : bool, optional
        If True, considers the entire solar disk when calculating the
        constraint. If False, only considers the center of the Sun, so
        nighttime begins when the Sun is >50% below the horizon. Default is
        True.

    Attributes
    ----------
    name : Literal["Spacecraft Daytime"]
        Full name of the constraint
    short_name : Literal["Day"]
        Abbreviated name of the constraint

    Methods
    -------
    __call__(time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.ndarray
        Evaluates the daytime constraint at given time(s)

    Notes
    -----
    The constraint uses ephemeris data to calculate the angular separation
    between the Earth and Sun as seen from the spacecraft's location. It
    accounts for the apparent sizes of both the Earth and Sun when determining
    if the spacecraft is in daylight.
    """

    name: Literal["Spacecraft Daytime"] = "Spacecraft Daytime"
    short_name: Literal["Day"] = "Day"
    whole_sun: bool = True

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.ndarray:
        """
        For a given time, ephemeris, check if the observatory is in daytime.
        Daytime is defined as the time when any part of the Sun is above the
        Earth Limb.

        Parameters
        ----------
        skycoord
            The coordinate to check.
        time
            The time to check.
        ephemeris
            The ephemeris object.

        Returns
        -------
        bool
            `True` if the coordinate is inside the constraint, `False`
            otherwise.

        """
        # Find a slice what the part of the ephemeris that we're using
        i = get_slice(time, ephemeris)

        # Calculate the angular distance between the center of the Ram
        # direction and the object. Note that creating the SkyCoord here from
        # ra/dec stored in the ephemeris `ram` is 3x faster than just doing the
        # separation directly with `ram`.
        assert (
            ephemeris.sun is not None
            and ephemeris.earth is not None
            and ephemeris.earth_radius_angle is not None
            and ephemeris.sun_radius_angle is not None
            and ephemeris.earth_location is not None
        )
        # Calculate the seperation between the Earth and the Sun
        sun_earth_sep = ephemeris.sun[i].separation(ephemeris.earth[i])
        if self.whole_sun:
            in_constraint = sun_earth_sep > ephemeris.earth_radius_angle - ephemeris.sun_radius_angle
        else:
            in_constraint = sun_earth_sep > ephemeris.earth_radius_angle

        # Return the
        return in_constraint[0] if time.isscalar else in_constraint


class DayConstraint(BaseSchema):
    """
    For a given daytime constraint, is a given coordinate inside this
    constraint?

    Parameters
    ----------
    None
    """

    short_name: Literal["Day"] = "Day"
    name: Literal["Daytime"] = "Daytime"

    def __call__(
        self,
        time: Time,
        ephemeris: Ephemeris,
        twilight_type: TwilightType,
        horizon_dip: bool = False,
    ) -> np.ndarray:
        """
        For a given time, ephemeris, check if it is daytime at the observatory.
        Daytime is defined by the twilight type, and whether the Sun is above
        the horizon.

        Parameters
        ----------
        skycoord : SkyCoord
            The coordinate to check.
        time : Time
            The time to check.
        ephemeris : Ephemeris
            The ephemeris object.
        twilight_type : TwilightType
            The type of twilight to consider bounding "daytime". Options are
            "astronomical", "nautical", "civil", and "sunrise".
        horizon_dip : bool, optional
            If True, consider the dip of the horizon due to the elevation of
            the observatory. Default is False.

        Returns
        -------
        np.ndarray
            Boolean array of `True` if the coordinate is inside the constraint, `False`
            otherwise.
        """
        # Find a slice what the part of the ephemeris that we're using
        i = get_slice(time, ephemeris)

        # Calculate the angular distance between the center of the Ram
        # direction and the object. Note that creating the SkyCoord here from
        # ra/dec stored in the ephemeris `ram` is 3x faster than just doing the
        # separation directly with `ram`.
        assert (
            ephemeris.sun is not None
            and ephemeris.earth is not None
            and ephemeris.timestamp is not None
            and ephemeris.earth_location is not None
            and ephemeris.sun_radius_angle is not None
        )
        sun_alt = (
            ephemeris.sun[i]
            .transform_to(AltAz(obstime=ephemeris.timestamp[i], location=ephemeris.earth_location))
            .alt
        )

        # Calculate the horizon dip due to elevation of the observatory
        if horizon_dip:
            height = ephemeris.earth_location.height.to_value(u.m)
            dip_horizon_degrees = np.sqrt(2 * height) * 0.034 * u.deg
        else:
            dip_horizon_degrees = 0 * u.deg

        # Depending on the twilight type, calculate if it is considered day time
        if twilight_type == TwilightType.ASTRONOMICAL:
            in_constraint = sun_alt > (-18 * u.deg - dip_horizon_degrees)
        elif twilight_type == TwilightType.NAUTICAL:
            in_constraint = sun_alt > (-12 * u.deg - dip_horizon_degrees)
        elif twilight_type == TwilightType.CIVIL:
            in_constraint = sun_alt > (-6 * u.deg - dip_horizon_degrees)
        elif twilight_type == TwilightType.SUNRISE:
            in_constraint = sun_alt > 0 * u.deg

        # Return the
        return in_constraint[0] if time.isscalar else in_constraint


class AltAzConstraint(PolygonConstraint):
    """
    For a given Alt/Az constraint, is a given coordinate inside this
    constraint? Constraint is both defined by a polygon exclusion region and a
    minimum and maximum altitude. By default the minimum and maximum altitude
    values are 0 and 90 degrees respectively. Polygon restriction regions can
    be combined with minimum and maximum altitude restrictions.

    Parameters
    ----------
    polygon
        The polygon defining the exclusion region.
    alt_min
        The minimum altitude in degrees.
    alt_max
        The maximum altitude in degrees.
    """

    short_name: Literal["Alt/Az"] = "Alt/Az"
    name: Literal["Altitude/Azimuth Avoidance"] = "Altitude/Azimuth Avoidance"
    polygon: Polygon | None
    alt_min: float | None
    alt_max: float | None

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.ndarray:
        """
        Calculate the Alt/Az constraint for a given time, ephemeris, and sky coordinates.

        Parameters
        ----------
        time : Time
            The time for which to calculate the constraint.
        ephemeris : Ephemeris
            The ephemeris containing the Earth location.
        skycoord : SkyCoord
            The sky coordinates to calculate the constraint for.

        Returns
        -------
        np.ndarray
            The calculated constraint values as a NumPy array.
        """
        # Get the range of the ephemeris that we're using
        i = get_slice(time, ephemeris)

        # Convert the sky coordinates to Alt/Az coordinates
        assert ephemeris.earth_location is not None
        alt_az = skycoord.transform_to(AltAz(obstime=time[i], location=ephemeris.earth_location))

        # Initialize the constraint array as all False
        in_constraint = np.zeros(len(alt_az), dtype=bool)

        # Calculate the basic Alt/Az min/max constraints
        if self.alt_min is not None:
            in_constraint |= alt_az.alt < self.alt_min * u.deg
        if self.alt_max is not None:
            in_constraint |= alt_az.alt > self.alt_max * u.deg

        # If a polygon is defined, then check if the Alt/Az is inside the polygon
        if self.polygon is not None:
            in_constraint |= self.polygon.contains(points(alt_az.alt, alt_az.az))

        # Return the value as a scalar or array
        return in_constraint[0] if time.isscalar and skycoord.isscalar else in_constraint


class MoonPhaseConstraint(BaseSchema):
    """
    Constraint based on the current phase of the Moon, as observed from the
    given observatory ephemeris.

    Parameters
    ----------
    min_phase
        The minimum moon phase value.
    max_phase
        The maximum moon phase value.
    """

    name: Literal["Moon Phase"] = "Moon Phase"
    short_name: Literal["Moon Phase"] = "Moon Phase"
    min_phase_angle: float | None = None
    max_phase_angle: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris) -> np.ndarray:
        """
        Calculate the Moon phase constraint for a given time and ephemeris.

        Parameters
        ----------
        time : Time
            The time for which to calculate the constraint.
        ephemeris : Ephemeris
            The ephemeris containing the Moon phase.

        Returns
        -------
        np.ndarray
            The calculated constraint values as a NumPy array.
        """
        i = get_slice(time, ephemeris)
        assert (
            ephemeris.sun is not None
            and ephemeris.moon is not None
            and ephemeris.gcrs.cartesian.xyz.value is not None
        )
        # Calculate the moon phase angle
        sun_moon = ephemeris.sun.cartesian[i] - ephemeris.moon.cartesian[i]
        sun_moon /= sun_moon.norm()
        observatory_moon = ephemeris.gcrs.cartesian.xyz.value[i] - ephemeris.moon.cartesian[i]
        observatory_moon /= observatory_moon.norm()
        phase_angle = np.arccos(sun_moon.dot(observatory_moon))

        # Set up the constraint array
        in_constraint = np.zeros(len(phase_angle), dtype=bool)

        # Calculate the basic Moon phase constraint
        if self.min_phase_angle is not None:
            in_constraint |= phase_angle > self.min_phase_angle
        if self.max_phase_angle is not None:
            in_constraint |= phase_angle < self.max_phase_angle

        # Return the value as a scalar or array
        return in_constraint[0] if time.isscalar else in_constraint


class AirMassConstraint(BaseSchema):
    """
    For a given limits on airmass, is a given skycoord coordinate constrained?

    Parameters
    ----------
    airmass_min
        The minimum altitude in degrees.
    airmass_max
        The maximum altitude in degrees.
    """

    short_name: Literal["Airmass"] = "Airmass"
    name: Literal["Airmass"] = "Airmass"
    airmass_max: float | None = None
    airmass_min: float | None = None

    def __call__(self, time: Time, ephemeris: Ephemeris, skycoord: SkyCoord) -> np.ndarray:
        """
        Calculate the Alt/Az constraint for a given time, ephemeris, and sky coordinates.

        Parameters
        ----------
        time : Time
            The time for which to calculate the constraint.
        ephemeris : Ephemeris
            The ephemeris containing the Earth location.
        skycoord : SkyCoord
            The sky coordinates to calculate the constraint for.

        Returns
        -------
        np.ndarray
            The calculated constraint values as a NumPy array.
        """
        # Get the range of the ephemeris that we're using
        i = get_slice(time, ephemeris)

        # Convert the sky coordinates to Alt/Az coordinates
        assert ephemeris.earth_location is not None
        alt_az = skycoord.transform_to(AltAz(obstime=time[i], location=ephemeris.earth_location))

        # Initialize the constraint array as all False
        in_constraint = np.zeros(len(alt_az), dtype=bool)

        # Calculate the basic Alt/Az min/max constraints
        if self.airmass_max is not None:
            in_constraint |= alt_az.secz < self.airmass_max
        if self.airmass_min is not None:
            in_constraint |= alt_az.secz > self.airmass_min

        # Return the value as a scalar or array
        return in_constraint[0] if time.isscalar and skycoord.isscalar else in_constraint
