from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional

import astropy.units as u  # type: ignore[import-untyped]
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from .schema import VisWindow


class Visibility(ABC):
    """
    Abstract base class for visibility calculations.
    """

    # Parameters
    ra: float = Field(ge=0, lt=360)
    dec: float = Field(ge=-90, le=90)
    begin: Time
    end: Time
    min_vis: Optional[int] = None
    hires: bool = True

    # Computed values
    timestamp: Time
    entries: list[VisWindow] = []

    def __init__(
        self, begin: str, end: str, ra: float, dec: float, hires: bool = False, min_vis: Optional[int] = None
    ):
        self.begin = Time(begin)
        self.end = Time(end)
        self.ra = ra
        self.dec = dec
        self.hires = hires
        self.min_vis = min_vis

    def __getitem__(self, i: int) -> VisWindow:
        return self.entries[i]

    def __len__(self) -> int:
        return len(self.timestamp)

    @cached_property
    def skycoord(self) -> SkyCoord:
        """
        Coordinates as an astropy SkyCoord object.

        Returns
        -------
        SkyCoord
            Array of RA/Dec coordinates.
        """
        return SkyCoord(self.ra * u.deg, self.dec * u.deg)

    def visible(self, t: Time) -> bool:
        """
        For a given time, is the target visible?

        Parameters
        ----------
        t
            Time to check

        Returns
        -------
            True if visible, False if not
        """
        return any(t >= win.begin and t <= win.end for win in self.entries)

    @abstractmethod
    async def get(self) -> None:
        """
        Abstract method to perform visibility calculation.
        """
        raise NotImplementedError
