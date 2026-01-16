from __future__ import annotations

from typing import Any

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
from astropy.coordinates import Angle  # type: ignore[import-untyped]
from mocpy import MOC  # type: ignore[import-untyped]

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
        Projects the footprint to a new coordinate with a given roll angle.

        Args:
            coordinate (Coordinate):
                The new center coordinate to project to.
            roll_angle (float):
                The roll angle in degrees to apply during projection.
        Returns:
            Footprint:
                The projected footprint.
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
        Convert a Footprint into HEALPix pixels using MOC.

        Args:
            footprint (Footprint):
                The footprint to convert.
            order (int):
                HEALPix order (10 = NSIDE 1024)

        Returns:
            list[int]:
                HEALPix NESTED pixel indices at requested order.
        """
        hp_order = HealpixOrder(value=order)
        all_pixels: list[int] = []
        for detector in self.detectors:
            lon = Angle(np.array([coord.ra for coord in detector.coordinates]) * u.deg, unit=u.deg)
            lat = Angle(np.array([coord.dec for coord in detector.coordinates]) * u.deg, unit=u.deg)

            # Build MOC from polygon
            moc = MOC.from_polygon(
                lon=lon,
                lat=lat,
                max_depth=hp_order.value,
            )
            # Get HEALPix pixels by flattening MOC
            moc_pixels: np.ndarray[Any, Any] = moc.flatten()
            all_pixels.extend(moc_pixels.tolist())

        return list(set(all_pixels))
