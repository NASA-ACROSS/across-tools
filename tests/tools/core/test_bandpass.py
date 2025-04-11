import numpy as np
import pytest

from across.tools import EnergyBandpass, FrequencyBandpass, WavelengthBandpass, convert_to_wave, enums


class TestBandpassSchema:
    """
    Class that tests the functionality of the bandpass schemas
    """

    class TestWavelengthBandpass:
        """
        Class that tests the instantiation of the WavelengthBandpass schema
        """

        @pytest.mark.parametrize(
            "min, max",
            [
                (None, 1),
                (1, None),
                (-1, 1),
                (1, -1),
                (2, 1),
            ],
        )
        def test_should_throw_value_error_min_max(self, min: float | None, max: float | None) -> None:
            """
            Parameterized tests that validate bad instantiations for WavelengthBandpass.
                Should raise ValueError when min=None, max=None
                Should raise ValueError when min < 0
                Should raise ValueError when max < 0
                Should raise ValueError when min > max
            """
            with pytest.raises(ValueError):
                WavelengthBandpass(min=min, max=max, unit=enums.WavelengthUnit.ANGSTROM)

        @pytest.mark.parametrize(
            "central_wavelength, bandwidth",
            [(None, None), (None, 1), (1, None), (-1, 1), (1, -1)],
        )
        def test_should_throw_value_error_central_wavelength_bandwidth(
            self, central_wavelength: float | None, bandwidth: float | None
        ) -> None:
            """
            Parameterized tests that validate bad instantiations for WavelengthBandpass.
                Should raise ValueError when central_wavelength=None, bandwidth=None
                Should raise ValueError when central_wavelength < 0
                Should raise ValueError when bandwidth < 0
            """
            with pytest.raises(ValueError):
                WavelengthBandpass(
                    central_wavelength=central_wavelength,
                    bandwidth=bandwidth,
                    unit=enums.WavelengthUnit.ANGSTROM,
                )

        @pytest.mark.parametrize(
            "min, max, unit, central_wavelength, bandwidth",
            [
                (5000.0, 6000.0, enums.WavelengthUnit.ANGSTROM, 5500.0, 500.0),
                (500, 600, enums.WavelengthUnit.NANOMETER, 5500, 500),
                (0.5, 0.6, enums.WavelengthUnit.MICRON, 5500, 500),
                (0.0005, 0.0006, enums.WavelengthUnit.MILLIMETER, 5500, 500),
            ],
        )
        def test_should_convert_min_max_to_correct_central_wavelength_bandwidth(
            self,
            min: float,
            max: float,
            unit: enums.WavelengthUnit,
            central_wavelength: float,
            bandwidth: float,
        ) -> None:
            """
            Parameterized test that confirms whether or not WavelengthBandpass converts
            the appropriate min max to central wavelength and bandpass.
            """
            wavelength_bandpass = WavelengthBandpass(min=min, max=max, unit=unit)
            assert all(
                [
                    np.isclose(wavelength_bandpass.central_wavelength, central_wavelength, 0.001),  # type:ignore
                    np.isclose(wavelength_bandpass.bandwidth, bandwidth, 0.001),  # type:ignore
                ]
            )

    class TestEnergyBandpass:
        """
        Class that tests the instantiation of the EnergyBandpass schema
        """

        @pytest.mark.parametrize(
            "min, max",
            [
                (None, None),
                (None, 1),
                (1, None),
                (-1, 1),
                (1, -1),
            ],
        )
        def test_should_throw_value_error_min_max(self, min: float | None, max: float | None) -> None:
            """
            Parameterized tests that validate bad instantiations for EnergyBandpass.
                Should raise ValueError when min=None, max=None
                Should raise ValueError when min < 0
                Should raise ValueError when max < 0
            """
            with pytest.raises(ValueError):
                EnergyBandpass(min=min, max=max, unit=enums.EnergyUnit.eV)

    class TestFrequencyBandpass:
        """
        Class that tests the instantiation of the FrequencyBandpass schema
        """

        @pytest.mark.parametrize(
            "min, max",
            [
                (None, None),
                (None, 1),
                (1, None),
                (-1, 1),
                (1, -1),
            ],
        )
        def test_should_throw_value_error_min_max(self, min: float | None, max: float | None) -> None:
            """
            Parameterized tests that validate bad instantiations for FrequencyBandpass.
                Should raise ValueError when min=None, max=None
                Should raise ValueError when min < 0
                Should raise ValueError when max < 0
            """
            with pytest.raises(ValueError):
                FrequencyBandpass(min=min, max=max, unit=enums.FrequencyUnit.Hz)

    class TestConvertToWave:
        """
        Class that tests the bandpass.convert_to_wave method
        """

        @pytest.mark.parametrize(
            "min, max, unit, min_wave, max_wave",
            [
                (1, 10, enums.EnergyUnit.eV, 1239.842, 12398.420),
                (1, 10, enums.EnergyUnit.keV, 1.23984198, 12.3984198),
                (1, 10, enums.EnergyUnit.MeV, 1.23984198e-3, 1.23984198e-2),
                (1, 10, enums.EnergyUnit.GeV, 1.23984198e-6, 1.23984198e-5),
                (1, 10, enums.EnergyUnit.TeV, 1.23984198e-9, 1.23984198e-8),
            ],
        )
        def test_convert_energy_to_wavelength(
            self, min: float, max: float, unit: enums.EnergyUnit, min_wave: float, max_wave: float
        ) -> None:
            """
            Parameterized test that confirms that the convert_to_wave method
            converts Energy min/max to appropriate Wavelength min/max
            """
            energy_bandpass = EnergyBandpass(min=min, max=max, unit=unit)
            wavelength_bandpass = convert_to_wave(energy_bandpass)
            assert all(
                [
                    np.isclose(wavelength_bandpass.min, min_wave),  # type:ignore
                    np.isclose(wavelength_bandpass.max, max_wave),  # type:ignore
                ]
            )

        @pytest.mark.parametrize(
            "min, max, unit, min_wave, max_wave",
            [
                (1, 10, enums.FrequencyUnit.Hz, 2.997924e17, 2.997924e18),
                (1, 10, enums.FrequencyUnit.kHz, 299792457999999.9, 2997924579999999.5),
                (1, 10, enums.FrequencyUnit.MHz, 299792457999.9, 2997924579999.9),
                (1, 10, enums.FrequencyUnit.GHz, 299792457.9, 2997924580),
                (1, 10, enums.FrequencyUnit.THz, 299792.4579, 2997924.579),
            ],
        )
        def test_convert_frequency_to_wavelength(
            self, min: float, max: float, unit: enums.FrequencyUnit, min_wave: float, max_wave: float
        ) -> None:
            """
            Parameterized test that confirms that the convert_to_wave method
            converts Frequency min/max to appropriate Wavelength min/max
            """
            energy_bandpass = FrequencyBandpass(min=min, max=max, unit=unit)
            wavelength_bandpass = convert_to_wave(energy_bandpass)
            assert all(
                [
                    np.isclose(wavelength_bandpass.min, min_wave),  # type:ignore
                    np.isclose(wavelength_bandpass.max, max_wave),  # type:ignore
                ]
            )
