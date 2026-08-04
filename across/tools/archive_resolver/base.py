from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

from ..core.schemas.base import BaseSchema

# TODO: Archive for Euclid
# TESS is on MAST but not the MAST UI with urlparams
# Not ingesting data for SPHEREx, SPARCS
# Fermi and Rubin and survey instruments - can ignore for now


class MASTMission(Enum):
    """
    Enum containing valid missions whose data are hosted on MAST.
    """

    HST = "HST"
    JWST = "JWST"


class HEASARCTable(Enum):
    """
    Enum containing valid tables for the HEASARC archive, corresponding to different observatories.
    """

    CHANMASTER = "chanmaster"
    IXMASTER = "ixmaster"
    NICERMASTR = "nicermastr"
    NUMASTER = "numaster"
    SWIFTMASTR = "swiftmastr"
    XMMMASTER = "xmmmaster"
    XRISMMASTR = "xrismmastr"


class ArchiveResolver(ABC, BaseSchema):
    """
    AbstractBaseClass for resolving observation IDs into links to
    archive data.

    Parameters
    ----------
    external_observation_id: str
        External ID of the observation.
    archive_url_template: str
        URL template for the archive data, must contain '{external_id}' placeholder.

    Attributes
    ----------
    archive_url: str
        The resolved archive URL after substituting the sanitized external_observation_id
        into the archive_url_template.

    Methods
    -------
    _sanitize_id()
        Logic to transform the external_observation_id into a format suitable for the archive URL.
    construct_archive_url
        Constructs the valid archive URL pointing to the data for the observation by observation ID.
    """

    external_observation_id: str
    archive_url_template: str = ""
    archive_url: str | None = None

    @abstractmethod
    def _sanitize_id(self) -> None:
        """
        Abstract method to sanitize the external_observation_id into a format suitable for the archive URL.
        """
        raise NotImplementedError("Subclasses must implement this method.")  # pragma: no cover

    @abstractmethod
    def construct_archive_url(self) -> None:
        """
        Abstract method to construct the valid archive URL pointing to the
        data for the observation by observation ID. This method should first call
        _sanitize_id() to ensure the ID is in the correct format before constructing the URL.
        """
        raise NotImplementedError("Subclasses must implement this method.")  # pragma: no cover


class MASTArchiveResolver(ArchiveResolver):
    """
    Archive resolver for the MAST archive.
    Handles HST and JWST external observation IDs.
    """

    name: Literal["MAST"] = "MAST"
    archive_url_template: str = (
        "https://mast.stsci.edu/search/ui/#/{mission}/results?{proposal_keyword}_id={external_observation_id}"
    )
    mission: MASTMission

    def _sanitize_id(self) -> None:
        """
        Sanitize the external_observation_id for MAST archive URLs.

        This method truncates the ID to the first 5 characters.
        """
        self.external_observation_id = self.external_observation_id[:5]

    def construct_archive_url(self) -> None:
        """
        Constructs the valid archive URL for MAST by substituting the sanitized
        external_observation_id into the archive_url_template.
        """
        self._sanitize_id()
        proposal_keyword = "proposal" if self.mission == MASTMission.HST else "program"
        self.archive_url = self.archive_url_template.format(
            mission=self.mission.value.lower(),
            proposal_keyword=proposal_keyword,
            external_observation_id=self.external_observation_id,
        )


class HEASARCArchiveResolver(ArchiveResolver):
    """
    Archive resolver for the HEASARC archive.
    Handles external observation IDs for HEASARC data.

    Observatories that use HEASARC include Chandra, Swift, NuSTAR, XMM,
    NICER, IXPE, and XRISM.
    """

    name: Literal["HEASARC"] = "HEASARC"
    heasarc_table: HEASARCTable
    archive_url_template: str = "https://heasarc.gsfc.nasa.gov/xamin/?table={heasarc_table}&constraint=obsid%3D%27{external_observation_id}%27"

    def _sanitize_id(self) -> None:
        """
        Sanitize the external_observation_id for HEASARC archive URLs.

        This method is not needed for the external observation IDs that ACROSS stores.
        """
        pass

    def construct_archive_url(self) -> None:
        """
        Constructs the valid archive URL for HEASARC by substituting the sanitized
        external_observation_id into the archive_url_template.
        """
        self.archive_url = self.archive_url_template.format(
            heasarc_table=self.heasarc_table.value.lower(),
            external_observation_id=self.external_observation_id,
        )
