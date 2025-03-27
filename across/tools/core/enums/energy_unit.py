from enum import Enum


class EnergyUnit(str, Enum):
    """
    Enum to represent the bandpass energy types
    """

    eV = "eV"  # noqa: N815
    keV = "keV"  # noqa: N815
    MeV = "MeV"
    GeV = "GeV"
    TeV = "TeV"
