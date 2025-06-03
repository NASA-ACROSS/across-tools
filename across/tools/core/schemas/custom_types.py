from typing import Annotated

from astropy.time import Time, TimeDelta  # type: ignore[import-untyped]
from pydantic import PlainSerializer, PlainValidator

# Custom pydantic type to handle serialization of Astropy Time
AstropyDateTime = Annotated[
    Time,
    PlainValidator(lambda x: Time(x)),
    PlainSerializer(lambda x: x.utc.datetime if x.isscalar else x.utc.to_datetime().tolist()),
]

AstropyTimeDelta = Annotated[
    TimeDelta,
    PlainValidator(lambda x: TimeDelta(x)),
    PlainSerializer(lambda x: x.to_datetime()),
]
