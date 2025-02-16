import io
from typing import Any

import httpx
import numpy as np
from astropy.io.votable import parse_single_table  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from across.tools.visibility.schema import VisWindow

from .base import Visibility


class ObjVisSAPVisibilityBase(Visibility):
    """
    Calculate visibility of a given object using the VO ObjVisSAP protocol

    Parameters
    ----------
    ra
        Right Ascension in decimal degrees
    dec
        Declination in decimal degrees
    begin
        Start time of visibility search
    end
        End time of visibility search
    objvissap_url
        URL of the VO ObjVisSAP visibility tool
    """

    objvissap_url: str | None = Field(None, exclude=True)
    objvissap_default_params: dict[str, Any] = Field({}, exclude=True)
    inconstraint: np.typing.NDArray[np.bool_] = Field(np.array([]), exclude=True)
    timestamp: np.typing.NDArray[np.bool_] = Field(np.array([]), exclude=True)

    async def get_objvissap(self) -> None:
        """
        Asynchronously retrieves object visibility data from the ObjVisSAP service.

        This method constructs a query to the ObjVisSAP service using the object's RA, Dec,
        start time, and end time. It then parses the VOTABLE response, extracting visibility
        windows and storing them as VisWindow objects in the `entries` list.
        The times are converted to numpy datetime64 format.

        Raises
        ------
        ValueError
            If the ObjVisSAP service returns a non-200 status code,
            indicating that the service is offline or unavailable.
        """
        # Check that the ObjVisSAP URL is set
        if self.objvissap_url is None:
            raise ValueError("ObjVisSAP URL not set.")

        # Construct the ObjVisSAP query parameters
        params = {
            "s_ra": self.ra,
            "s_dec": self.dec,
            "t_min": Time(self.begin).mjd,
            "t_max": Time(self.end).mjd,
            "min_vis": 1,
        }
        # Perform the query
        async with httpx.AsyncClient() as client:
            r = await client.get(self.objvissap_url, params=dict(params, **self.objvissap_default_params))

        # If successful, parse the results
        if r.status_code == 200:
            # Parse VOTABLE into VisWindow objects
            xmlvo = r.text
            # Fix broke non-conformant VO from XMM
            xmlvo = xmlvo.replace('datatype="timestamp"', 'datatype="float"')
            xmlvo = xmlvo.replace('name="t_end"', 'name="t_stop"')

            votablefile = io.BytesIO(xmlvo.encode())
            votable = parse_single_table(votablefile)

            for i in range(len(votable.array.data)):
                vw = VisWindow(
                    begin=Time(votable.array.data["t_start"][i], format="mjd").datetime,
                    end=Time(votable.array.data["t_stop"][i], format="mjd").datetime,
                    visibility=votable.array.data["t_visibility"][i],
                    initial="Window",
                    final="Window",
                )
                self.entries.append(vw)
        else:
            raise ValueError("Visibility tool offline.")
