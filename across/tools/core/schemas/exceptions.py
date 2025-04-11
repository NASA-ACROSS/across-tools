class MinMaxValueError(ValueError):
    """
    Exception raise for invalid bandpass min/max values
    """

    def __init__(self, message: str):
        super().__init__(message)


class BandwidthValueError(ValueError):
    """
    Exception raise for invalid bandpass central_wavelength/bandwidth values
    """

    def __init__(self, message: str):
        super().__init__(message)
