from typing import Any, Literal

from astropy import units as u  # type: ignore[import-untyped]

from ..enums import EnergyUnit, FrequencyUnit, WavelengthUnit
from .base import BaseSchema
from .exceptions import BandwidthValueError, MinMaxValueError


class BaseBandpass(BaseSchema):
    """
    A base class for defining bandpass filters with a specified range.

    Attributes:
        filter_name (str): The name of the filter, if provided.
        min (float | None): The minimum value of the bandpass range.
        max (float | None): The maximum value of the bandpass range.
    """

    filter_name: str | None = None
    min: float | None = None
    max: float | None = None


class WavelengthBandpass(BaseBandpass):
    """
    A class representing a bandpass filter defined in terms of wavelength.

    Inherits from `BaseBandpass`, this class specializes the filter to operate in the
    wavelength domain and provides additional functionality for wavelength-based filters.

    Attributes:
        type (Literal['WAVELENGTH']): A constant string indicating the type of the bandpass filter.
        central_wavelength (float | None): The central wavelength of the filter.
        bandwidth (float | None): The bandwidth of the filter.
        unit (WavelengthUnit): The unit of measurement for the wavelength.

    Methods:
        model_post_init(__context: Any) -> None:
            Performs validation and calculation of central wavelength and bandwidth based on the min/max range
    """

    type: Literal["WAVELENGTH"] = "WAVELENGTH"
    central_wavelength: float | None = None
    peak_wavelength: float | None = None
    bandwidth: float | None = None
    unit: WavelengthUnit

    def model_post_init(self, __context: Any) -> None:
        """
        Validates the min and max values of the wavelength bandpass and calculates the central wavelength
        and bandwidth if they are not provided. Also ensures the values are positive and that the max
        wavelength is greater than the min wavelength. Lastly, it converts the units to angstroms.

        Raises:
            ValueError: If the min/max values are invalid or if the calculated values for central wavelength
                        or bandwidth are non-positive.
        """
        if (self.min and not self.max) or (self.max and not self.min):
            raise MinMaxValueError("Both min and max must be defined.")

        if self.min and self.max:
            if self.max < self.min:
                raise MinMaxValueError("Max wavelength cannot be less than min wavelength.")

            if not all([self.min > 0, self.max > 0]):
                raise MinMaxValueError("Wavelength values must be positive.")

            self.bandwidth = 0.5 * (self.max - self.min)
            self.central_wavelength = self.min + self.bandwidth

        if not (self.central_wavelength and self.bandwidth):
            raise BandwidthValueError("Both central wavelength and bandwidth must be defined.")

        if not all([self.central_wavelength > 0, self.bandwidth > 0]):
            raise BandwidthValueError("Central wavelength and bandwidth must be positive.")

        self.bandwidth = float(
            (self.bandwidth * u.Unit(self.unit.value)).to(u.Unit(WavelengthUnit.ANGSTROM.value)).value
        )

        self.central_wavelength = float(
            (self.central_wavelength * u.Unit(self.unit.value))
            .to(u.Unit(WavelengthUnit.ANGSTROM.value))
            .value
        )

        self.min = self.central_wavelength - self.bandwidth

        self.max = self.central_wavelength + self.bandwidth

        self.unit = WavelengthUnit.ANGSTROM


class EnergyBandpass(BaseBandpass):
    """
    A class representing a bandpass filter defined in terms of energy.

    Inherits from `BaseBandpass`, this class specializes the filter to operate in the energy domain.

    Attributes:
        type (Literal['ENERGY']): A constant string indicating the type of the bandpass filter.
        unit (EnergyUnit): The unit of measurement for the energy.

    Methods:
        model_post_init(__context: Any) -> None:
            Ensures the min and max energy values are positive and valid.
    """

    type: Literal["ENERGY"] = "ENERGY"
    unit: EnergyUnit

    def model_post_init(self, __context: Any) -> None:
        """
        Validates that the min and max energy values are positive.

        Raises:
            ValueError: If the min or max energy values are not defined, are non-positive, and max is greater
            than min.
        """
        if not (self.min and self.max):
            raise MinMaxValueError("Both min and max energy values must be defined.")

        if not all([self.min > 0, self.max > 0]):
            raise MinMaxValueError("Energy values must be positive.")

        if self.max < self.min:
            raise MinMaxValueError("Max wavelength cannot be less than min wavelength.")


class FrequencyBandpass(BaseBandpass):
    """
    A class representing a bandpass filter defined in terms of frequency.

    Inherits from `BaseBandpass`, this class specializes the filter to operate in the frequency domain.

    Attributes:
        type (Literal['FREQUENCY']): A constant string indicating the type of the bandpass filter.
        unit (FrequencyUnit): The unit of measurement for the frequency.

    Methods:
        model_post_init(__context: Any) -> None:
            Ensures the min and max frequency values are positive and valid.
    """

    type: Literal["FREQUENCY"] = "FREQUENCY"
    unit: FrequencyUnit

    def model_post_init(self, __context: Any) -> None:
        """
        Validates that the min and max frequency values are positive.

        Raises:
            ValueError: If the min or max energy values are not defined, are non-positive, and max is greater
            than min.
        """
        if not (self.min and self.max):
            raise MinMaxValueError("Both min and max frequency values must be defined.")

        if not all([self.min > 0, self.max > 0]):
            raise MinMaxValueError("Frequency values must be positive.")

        if self.max < self.min:
            raise MinMaxValueError("Max wavelength cannot be less than min wavelength.")


def convert_to_wave(bandpass: EnergyBandpass | FrequencyBandpass) -> WavelengthBandpass:
    """
    Converts a given EnergyBandpass or FrequencyBandpass to a WavelengthBandPass.

    Args:
        bandpass (EnergyBandpass | FrequencyBandpass): The bandpass filter in energy or frequency domain.

    Returns:
        WavelengthBandPass: The corresponding bandpass filter in the wavelength domain.

    Important Note:
        When converting from Energy/Frequency to wavelength, the min/max values are inverted
        in relation to their corresponding wavelengths. Thus, the min/max values are switched during
        conversion.
    """
    bandpass_min_angstrom = (
        (bandpass.max * u.Unit(bandpass.unit.value))
        .to(u.Unit(WavelengthUnit.ANGSTROM.value), equivalencies=u.spectral())
        .value
    )

    bandpass_max_angstrom = (
        (bandpass.min * u.Unit(bandpass.unit.value))
        .to(u.Unit(WavelengthUnit.ANGSTROM.value), equivalencies=u.spectral())
        .value
    )

    return WavelengthBandpass(
        min=bandpass_min_angstrom,
        max=bandpass_max_angstrom,
        unit=WavelengthUnit.ANGSTROM,
        filter_name=bandpass.filter_name,
    )
