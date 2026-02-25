from datetime import datetime

import pytest
from astropy.time import Time  # type: ignore[import-untyped]


@pytest.fixture
def scalar_time() -> Time:
    """Fixture for a scalar Time instance."""
    return Time(datetime(2023, 1, 1))
