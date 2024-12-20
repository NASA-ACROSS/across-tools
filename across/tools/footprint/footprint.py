from __future__ import annotations

import astropy.coordinates  # type: ignore[import-untyped]
import healpy as hp  # type: ignore[import-untyped]
import numpy as np

from ..core.schemas import BaseSchema, Coordinate, HealpixOrder, Polygon, RollAngle
from .projection import project_detector


class Footprint(BaseSchema):
    """
    Class to represent a astronomical instrument's imaging footprint
    """

    detectors: list[Polygon]

    def __repr__(self) -> str:
        """
        Overrides the print statement
        """
        statement = f"{self.__class__.__name__}(\n"

        for detector in self.detectors:
            statement += f"\t{detector.__class__.__name__}(\n"

            for coordinate in detector.coordinates:
                statement += f"\t\t{coordinate.__class__.__name__}({coordinate.ra}, {coordinate.dec}),\n"

            statement += "\t),\n"

        statement += ")"

        return statement

    def __eq__(self, other: object) -> bool:
        """
        Overrides the Footprint equals
        """
        if not isinstance(other, Footprint):
            return NotImplemented

        if len(self.detectors) != len(other.detectors):
            return False
        else:
            equivalence: list[bool] = []
            for detector_iterable in range(len(self.detectors)):
                equivalence.append(self.detectors[detector_iterable] == other.detectors[detector_iterable])
            return all(equivalence)

    def project(self, coordinate: Coordinate, roll_angle: float) -> Footprint:
        """
        Method to project an astronomical instrument footprint on a sphere.
        """
        angle = RollAngle(value=roll_angle)

        projected_detectors = []

        for detector in self.detectors:
            projected_detectors.append(
                project_detector(detector=detector, coordinate=coordinate, roll_angle=angle.value)
            )

        return Footprint(detectors=projected_detectors)

    def query_pixels(self, order: int = 10) -> list[int]:
        """
        Method to query the healpix pixels in a footprint at a given healpix order
        """
        hp_order = HealpixOrder(value=order)

        pixels_in_footprint: list[int] = []
        nside = hp.order2nside(order=hp_order.value)

        for detector in self.detectors:
            detector_ra_values = np.deg2rad([coordinate.ra for coordinate in detector.coordinates][:-1])
            detector_dec_values = np.deg2rad([coordinate.dec for coordinate in detector.coordinates][:-1])
            cartesian_polygon = astropy.coordinates.spherical_to_cartesian(
                1.0, detector_dec_values, detector_ra_values
            )
            healpy_polygon = np.array(cartesian_polygon).T
            pixels_queried = hp.query_polygon(nside, healpy_polygon, inclusive=True)
            pixels_in_footprint.extend(pixels_queried.tolist())

        unique_pixels = list(set(pixels_in_footprint))
        return unique_pixels
