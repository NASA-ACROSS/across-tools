from __future__ import annotations

import numpy as np

from ..core.math import x_rot, y_rot, z_rot
from ..core.schemas import BaseSchema, Coordinate, Polygon


class CartesianVector(BaseSchema):
    """
    Class to represent a 3D cartesian vector
    """

    x: float
    y: float
    z: float

    def rotate(self, coordinate: Coordinate, roll_angle: float) -> CartesianVector:
        """
        Method that performs matrix rotations on a cartesian vector
        """
        vector_array = np.asarray([self.x, self.y, self.z])

        rotated_vector = (
            vector_array @ x_rot(-1.0 * roll_angle) @ y_rot(coordinate.dec) @ z_rot(-1.0 * coordinate.ra)
        )

        rotated_x_value, rotated_y_value, rotated_z_value = rotated_vector.flat

        return CartesianVector(x=float(rotated_x_value), y=float(rotated_y_value), z=float(rotated_z_value))

    def to_spherical_coordinate(self) -> Coordinate:
        """
        Method to transform a cartesian vector to a spherical coordinate
        """
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        normalized_x = np.asarray(self.x / r)
        normalized_y = np.asarray(self.y / r)
        normalized_z = np.asarray(self.z / r)
        theta = np.arctan2(normalized_y, normalized_x)
        phi = np.arccos(normalized_z)
        dec = round(90 - np.rad2deg(phi), 5)
        ra = round(360 + np.rad2deg(theta), 5) if theta < 0 else round(np.rad2deg(theta), 5)

        return Coordinate(ra=ra, dec=dec)


def detector_to_cartesian_vectors(detector: Polygon) -> list[CartesianVector]:
    """
    Method to convert spherical detector to a list of cartesian unit vectors
    """
    detector_ra_values = np.asarray([coord.ra for coord in detector.coordinates])
    detector_dec_values = np.asarray([coord.dec for coord in detector.coordinates])

    phi = np.deg2rad(90 - detector_dec_values)
    theta = np.deg2rad(detector_ra_values)

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    cartesian_vectors: list[CartesianVector] = []

    for idx in range(x.shape[0]):
        cartesian_vectors.append(CartesianVector(x=x[idx], y=y[idx], z=z[idx]))

    return cartesian_vectors


def project_detector(detector: Polygon, coordinate: Coordinate, roll_angle: float) -> Polygon:
    """
    Method to project a polygon detector onto a sphere
    """
    projected_coordinates: list[Coordinate] = []
    detector_cartesian_vectors = detector_to_cartesian_vectors(detector=detector)

    for vector in detector_cartesian_vectors:
        rotated_vector = vector.rotate(coordinate=coordinate, roll_angle=roll_angle)
        spherical_coordinate = rotated_vector.to_spherical_coordinate()
        projected_coordinates.append(spherical_coordinate)

    return Polygon(coordinates=projected_coordinates)
