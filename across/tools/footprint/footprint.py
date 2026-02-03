from __future__ import annotations

from typing import Any

import astropy.units as u  # type: ignore[import-untyped]
import numpy as np
import plotly.graph_objects as go
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

    def contains(self, coordinate: Coordinate, order: int = 10) -> bool:
        """
        Tests if a point exists in a footprint.

        Args:
            footprint (Footprint):
                The footprint to convert.
            coordinate (Coordinate):
                The coordinate to check for containment.
            order (int):
                HEALPix order (10 = NSIDE 1024)

        Returns:
            bool:
                True if the coordinate is contained within the footprint, False otherwise.
        """
        hp_order = HealpixOrder(value=order)
        for detector in self.detectors:
            lon = Angle(np.array([coord.ra for coord in detector.coordinates]) * u.deg, unit=u.deg)
            lat = Angle(np.array([coord.dec for coord in detector.coordinates]) * u.deg, unit=u.deg)

            # Build MOC from polygon
            moc = MOC.from_polygon(
                lon=lon,
                lat=lat,
                max_depth=hp_order.value,
            )
            # Query polygon containment
            coord_lon = Angle([coordinate.ra * u.deg], unit=u.deg)
            coord_lat = Angle([coordinate.dec * u.deg], unit=u.deg)
            if any(moc.contains(coord_lon, coord_lat)):
                return True

        return False

    def plot(
        self,
        fig: go.Figure | None = None,
        name: str | None = None,
        color: str | None = None,
        lat_axis_tick: int = 30,
        lon_axis_tick: int = 60,
    ) -> go.Figure:
        """
        Method to plot the footprint using plotly

        Parameters
        ----------
        fig : go.Figure, optional
            An existing plotly figure to add the footprint to, by default None
        name : str | None, optional
            The name to assign to the detector traces, by default None
        color : str | None, optional
            The color to assign to the detector traces, by default None
        lat_axis_tick : int, optional
            The latitude axis tick interval, by default 30
        lon_axis_tick : int, optional
            The longitude axis tick interval, by default 60
        Returns
        -------
        go.Figure
            The plotly figure containing the footprint plot
        """

        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                title="Footprint Visualization",
                geo=dict(
                    projection_type="mollweide",
                    showland=False,
                    showcountries=False,
                    showcoastlines=False,
                    lataxis=dict(showgrid=True, dtick=lat_axis_tick),
                    lonaxis=dict(showgrid=True, dtick=lon_axis_tick),
                ),
            )

        for i, detector in enumerate(self.detectors):
            ra_values = [coord.ra for coord in detector.coordinates] + [detector.coordinates[0].ra]
            dec_values = [coord.dec for coord in detector.coordinates] + [detector.coordinates[0].dec]

            # Check if trace with same name already exists
            name_exists = False
            if name:
                for trace in fig.data:
                    name_exists = trace.name == name  # type: ignore[attr-defined]
                    break

            # Only show legend for first detector if name exists
            show_legend = i == 0 and not name_exists

            # Set legend group to name or unique id
            legend_group = name if name else f"footprint-{id(self)}"

            fig.add_trace(
                go.Scattergeo(
                    lon=ra_values,
                    lat=dec_values,
                    mode="lines",
                    fill="none",
                    name=name if name else legend_group,
                    legendgroup=legend_group,
                    line=dict(color=color) if color else None,
                    showlegend=show_legend,
                )
            )

        return fig
