from enum import Enum


class FrequencyUnit(str, Enum):
    """
    Enum to represent the bandpass frequency types
    """

    Hz = "Hz"
    kHz = "kHz"  # noqa: N815
    MHz = "MHz"
    GHz = "GHz"
    THz = "THz"
