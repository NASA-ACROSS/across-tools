from ..core.schemas import AstropyDateTime, BaseSchema
from .footprint import Footprint


class Pointing(BaseSchema):
    """
    Class describing an astronomical pointing
    A pointing is described as a footprint valid
    for a given time range
    """

    footprint: Footprint
    start_time: AstropyDateTime
    end_time: AstropyDateTime
