from datetime import datetime
from unittest.mock import patch

import astropy.units as u  # type: ignore[import-untyped]  # type: ignore[import-untyped]
import numpy as np
import pytest
from astropy.coordinates import (  # type: ignore[import-untyped]
    Latitude,
    Longitude,
)
from astropy.table import Table  # type: ignore[import-untyped]
from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from astropy.units import Quantity as uQuantity
from numpy.typing import NDArray

from across.tools.core.schemas.tle import TLE
from across.tools.ephemeris import (
    Ephemeris,
    compute_ground_ephemeris,
    compute_jpl_ephemeris,
    compute_spice_ephemeris,
    compute_tle_ephemeris,
)
from across.tools.ephemeris.ground_ephem import GroundEphemeris


class MockEphemeris(Ephemeris):
    """Mock class for testing the Ephemeris class."""

    def prepare_data(self) -> None:
        """Mock method to prepare data."""
        pass


class NotImplementedEphemeris(Ephemeris):
    """Mock class for testing the Ephemeris class."""

    def prepare_data(self) -> None:
        """Mock method to prepare data."""
        super().prepare_data()  # type: ignore[safe-super]


@pytest.fixture
def mock_ephemeris_class() -> type[MockEphemeris]:
    """Fixture for the MockEphemeris class."""
    return MockEphemeris


@pytest.fixture
def not_implemented_ephemeris_class() -> type[NotImplementedEphemeris]:
    """Fixture for the NotImplementedEphemeris class."""
    return NotImplementedEphemeris


@pytest.fixture
def ephemeris_begin() -> Time:
    """Fixture to provide a begin datetime for testing."""
    return Time("2025-02-12 00:00:00", scale="utc")


@pytest.fixture
def ephemeris_end() -> Time:
    """Fixture to provide an end datetime for testing."""
    return Time("2025-02-12 00:05:00", scale="utc")


@pytest.fixture
def mock_ephemeris(
    ephemeris_begin: Time, ephemeris_end: Time, ephemeris_step_size: TimeDelta
) -> MockEphemeris:
    """Fixture to provide a MockEphemeris instance."""
    return MockEphemeris(begin=ephemeris_begin, end=ephemeris_end, step_size=ephemeris_step_size)


@pytest.fixture
def ephemeris_step_size() -> TimeDelta:
    """Fixture to provide a step_size for testing."""
    return TimeDelta(60 * u.s)


@pytest.fixture
def keck_latitude() -> u.Quantity:
    """Fixture to provide a latitude for testing."""
    return 20.879 * u.deg


@pytest.fixture
def keck_longitude() -> u.Quantity:
    """Fixture to provide a longitude for testing."""
    return 155.6655 * u.deg


@pytest.fixture
def keck_height() -> u.Quantity:
    """Fixture to provide a height for testing."""
    return 4160 * u.m


@pytest.fixture
def hubble_tle() -> TLE:
    """Fixture to provide a TLE for the Hubble Space Telescope."""
    return TLE(
        satellite_name="HST",
        tle1="1 20580U 90037B   25043.65115181  .00008438  00000+0  34020-3 0  9990",
        tle2="2 20580  28.4676 167.7952 0002864 136.5792 223.5028 15.22991033713929",
    )


@pytest.fixture
def hubble_naif_id() -> int:
    """Fixture to provide the NAIF ID for the Hubble Space Telescope."""
    return -48


@pytest.fixture
def hubble_gcrs_value_km() -> NDArray[np.float64]:
    """Fixture to provide the GCRS position of the Hubble satellite."""
    return np.array(
        [
            [-6601.79957494, -6463.92010459, -6297.43794083, -6103.08764513, -5881.72745354, -5634.33546338],
            [-1415.26655239, -1803.68208289, -2184.11634725, -2554.88519298, -2914.34708129, -3260.9103933],
            [1281.48161724, 1479.25441952, 1670.46341691, 1854.25957144, 2029.826681, 2196.38502342],
        ]
    )


@pytest.fixture
def hubble_itrs_value_km() -> NDArray[np.float64]:
    """Fixture to provide the Earth location of the Hubble satellite."""
    return np.array(
        [
            [4331.13875516, 4007.24602751, 3667.39060024, 3312.88335429, 2945.09455093, 2565.44849455],
            [5183.55079613, 5387.41092415, 5570.43884626, 5731.96023048, 5871.38387356, 5988.20382911],
            [1265.29459209, 1463.38942926, 1654.99066752, 1839.24753875, 2015.3418068, 2182.49142062],
        ]
    )


@pytest.fixture
def hubble_horizons_vectors() -> Table:
    """Fixture to provide the Hubble Space Telescope vectors from
    astroquery.jplephemeris's Horizons.vectors method."""

    return Table(
        [
            np.array(
                [
                    "Hubble Space Telescope (spacecraft)",
                    "Hubble Space Telescope (spacecraft)",
                    "Hubble Space Telescope (spacecraft)",
                    "Hubble Space Telescope (spacecraft)",
                    "Hubble Space Telescope (spacecraft)",
                    "Hubble Space Telescope (spacecraft)",
                ]
            ),
            np.array(
                [
                    2460718.500800752,
                    2460718.501495197,
                    2460718.502189641,
                    2460718.502884086,
                    2460718.50357853,
                    2460718.504272975,
                ]
            )
            * u.d,
            np.array(
                [
                    "A.D. 2025-Feb-12 00:01:09.1850",
                    "A.D. 2025-Feb-12 00:02:09.1850",
                    "A.D. 2025-Feb-12 00:03:09.1850",
                    "A.D. 2025-Feb-12 00:04:09.1850",
                    "A.D. 2025-Feb-12 00:05:09.1850",
                    "A.D. 2025-Feb-12 00:06:09.1850",
                ]
            ),
            np.array(
                np.array(
                    [
                        -4.413109685504098e-05,
                        -4.320951686348332e-05,
                        -4.209674074411141e-05,
                        -4.079767809571204e-05,
                        -3.931806519583499e-05,
                        -3.76644395176696e-05,
                    ]
                )
            )
            * u.AU,
            np.array(
                [
                    -9.459679304384228e-06,
                    -1.20560763138425e-05,
                    -1.459912674999905e-05,
                    -1.707757297872861e-05,
                    -1.948044216827302e-05,
                    -2.179709512380254e-05,
                ]
            )
            * u.AU,
            np.array(
                [
                    8.565141089418774e-06,
                    9.887171518916822e-06,
                    1.116533134540432e-05,
                    1.239394518573508e-05,
                    1.356755710675922e-05,
                    1.468095498206593e-05,
                ]
            )
            * u.AU,
            np.array(
                [
                    0.001187350603237076,
                    0.001465801721275105,
                    0.001737789196464366,
                    0.002002106051621462,
                    0.002257579357790702,
                    0.002503075493910799,
                ]
            )
            * u.AU
            / u.d,
            np.array(
                [
                    -0.003771715115954226,
                    -0.003703120624754096,
                    -0.003618134112267888,
                    -0.003517129872519882,
                    -0.003400553506769829,
                    -0.003268919895516453,
                ]
            )
            * u.AU
            / u.d,
            np.array(
                [
                    0.001932503216081898,
                    0.001873512471782746,
                    0.001806203302310515,
                    0.001730873692075622,
                    0.001647857439270663,
                    0.001557522639078362,
                ]
            )
            * u.AU
            / u.d,
            np.array(
                [
                    2.653221126234431e-07,
                    2.653074405706743e-07,
                    2.652928068499324e-07,
                    2.652782970600778e-07,
                    2.652639978997604e-07,
                    2.652499963858138e-07,
                ]
            )
            * u.d,
            np.array(
                [
                    4.593909973053949e-05,
                    4.593655934335226e-05,
                    4.593402559314974e-05,
                    4.593151330091518e-05,
                    4.592903747805448e-05,
                    4.592661319106531e-05,
                ]
            )
            * u.AU,
            np.array(
                [
                    -3.650623268807726e-06,
                    -3.651561928277356e-06,
                    -3.631391291522598e-06,
                    -3.589746103254491e-06,
                    -3.526451748811547e-06,
                    -3.441531501265534e-06,
                ]
            )
            * u.AU
            / u.d,
        ],
        names=(
            "targetname",
            "datetime_jd",
            "datetime_str",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "lighttime",
            "range",
            "range_rate",
        ),
    )


@pytest.fixture
def hubble_mocked_spkezr() -> NDArray[np.float64]:
    """Fixture to provide the SPK ephemeris states for the Hubble Telescope."""

    return np.array(
        [
            [-6.60191801e03, -1.41514824e03, 1.28132705e03, 2.05584676e00, -6.53056183e00, 3.34604582e00],
            [-6.46405158e03, -1.80356370e03, 1.47909998e03, 2.53797285e00, -6.41179340e00, 3.24390588e00],
            [-6.29758261e03, -2.18399862e03, 1.67030997e03, 3.00890741e00, -6.26464290e00, 3.12736294e00],
            [-6.10324558e03, -2.55476889e03, 1.85410797e03, 3.46656062e00, -6.08975839e00, 2.99693296e00],
            [-5.88189862e03, -2.91423299e03, 2.02967781e03, 3.90890161e00, -5.88791142e00, 2.85319389e00],
            [-5.63451972e03, -3.26079933e03, 2.19623975e03, 4.33396756e00, -5.65999348e00, 2.69678307e00],
        ]
    )


@pytest.fixture
def hubble_spice_kernel_url() -> str:
    """Fixture to provide the spice kernel URL for the Hubble Space Telescope."""
    return "https://naif.jpl.nasa.gov/pub/naif/HST/kernels/spk/hst.bsp"


@pytest.fixture
def keck_ground_ephemeris(
    ephemeris_begin: datetime,
    ephemeris_end: datetime,
    ephemeris_step_size: int,
    keck_latitude: u.Quantity,
    keck_longitude: u.Quantity,
    keck_height: u.Quantity,
) -> Ephemeris:
    """Fixture to provide the ground ephemeris for the Keck Observatory."""
    return compute_ground_ephemeris(
        begin=ephemeris_begin,
        end=ephemeris_end,
        step_size=ephemeris_step_size,
        latitude=keck_latitude,
        longitude=keck_longitude,
        height=keck_height,
    )


@pytest.fixture
def hubble_tle_ephemeris(
    ephemeris_begin: datetime, ephemeris_end: datetime, ephemeris_step_size: int, hubble_tle: TLE
) -> Ephemeris:
    """Fixture to provide the TLE ephemeris for the Hubble satellite."""
    return compute_tle_ephemeris(
        begin=ephemeris_begin,
        end=ephemeris_end,
        step_size=ephemeris_step_size,
        tle=hubble_tle,
    )


@pytest.fixture
def hubble_spice_ephemeris(
    ephemeris_begin: datetime,
    ephemeris_end: datetime,
    ephemeris_step_size: int,
    hubble_naif_id: int,
    hubble_spice_kernel_url: str,
    hubble_mocked_spkezr: NDArray[np.float64],
) -> Ephemeris:
    """Fixture to provide the spice ephemeris for the Hubble Space Telescope."""
    mock_start_et = 7.92590469e08
    mock_end_et = 7.92590769e08

    with (
        patch("across.tools.ephemeris.spice_ephem.spice.furnsh") as mock_furnsh,
        patch("across.tools.ephemeris.spice_ephem.download_file") as mock_download_file,
        patch("across.tools.ephemeris.spice_ephem.spice.str2et") as mock_str2et,
        patch("across.tools.ephemeris.spice_ephem.spice.spkezr") as mock_spkezr,
        patch("across.tools.ephemeris.spice_ephem.spice.str2et") as mock_str2et,
    ):
        mock_furnsh.return_value = None
        mock_download_file.return_value = None
        mock_str2et.return_value = None
        mock_spkezr.side_effect = [(v,) for v in hubble_mocked_spkezr]
        mock_str2et.side_effect = [mock_start_et, mock_end_et]

        ephemeris = compute_spice_ephemeris(
            begin=ephemeris_begin,
            end=ephemeris_end,
            step_size=ephemeris_step_size,
            naif_id=hubble_naif_id,
            spice_kernel_url=hubble_spice_kernel_url,
        )
        assert mock_furnsh.called
        assert mock_download_file.called
        assert mock_str2et.called
        assert mock_spkezr.called
        return ephemeris


@pytest.fixture
def hubble_jpl_ephemeris(
    ephemeris_begin: datetime,
    ephemeris_end: datetime,
    ephemeris_step_size: int,
    hubble_naif_id: int,
    hubble_horizons_vectors: Table,
) -> Ephemeris:
    """Fixture to provide the JPL ephemeris for the Hubble satellite."""
    with patch("across.tools.ephemeris.jpl_ephem.jpl.Horizons") as mock_horizons_class:
        mock_instance = mock_horizons_class.return_value
        mock_instance.vectors.return_value = hubble_horizons_vectors

        return compute_jpl_ephemeris(
            begin=ephemeris_begin,
            end=ephemeris_end,
            step_size=ephemeris_step_size,
            naif_id=hubble_naif_id,
        )


@pytest.fixture
def ground_ephemeris_params() -> tuple[str, str, int, Latitude, Longitude, uQuantity]:
    """Fixture for ground ephemeris parameters."""
    begin = "2023-01-01T00:00:00"
    end = "2023-01-01T00:01:00"
    step_size = 60
    latitude = Latitude(34.2 * u.deg)
    longitude = Longitude(-118.2 * u.deg)
    height = 100 * u.m
    return begin, end, step_size, latitude, longitude, height


@pytest.fixture
def ground_ephemeris(
    ground_ephemeris_params: tuple[str, str, int, Latitude, Longitude, uQuantity],
) -> GroundEphemeris:
    """Fixture for a GroundEphemeris object with prepared data."""
    begin, end, step_size, latitude, longitude, height = ground_ephemeris_params
    ephemeris = GroundEphemeris(begin, end, step_size, latitude, longitude, height)
    ephemeris.prepare_data()
    return ephemeris
