import io
from typing import Any

import httpx
import numpy as np
from astropy.io.votable import parse_single_table  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import Field

from ..core.schemas.visibility import ConstraintType
from .base import Visibility


class VOVisibility(Visibility):
    """
    A class that handles visibility calculations using ObjVisSAP service.
    This class extends the Visibility base class to calculate visibility windows using the
    ObjVisSAP service. It queries the service with position and time range parameters and
    parses the returned VOTable format data into visibility windows.
    Parameters
    ----------
    objvissap_url : str, optional
        The URL of the ObjVisSAP service endpoint. Must be set before calling prepare_data.
    objvissap_default_params : dict, optional
        Additional default parameters to include in ObjVisSAP queries, defaults to empty dict.
    Attributes
    ----------
    visibility_windows : list[VisibilityWindows]
        List of VisWindow objects representing visibility periods.
    Methods
    -------
    prepare_data()
        Queries the ObjVisSAP service and processes results into visibility windows.
    Raises
    ------
    ValueError
        If ObjVisSAP URL is not set or if the service is offline/unreachable.
    Notes
    -----
    The class expects the ObjVisSAP service to return data in VOTable format and handles
    some known format inconsistencies in the response.
    """

    objvissap_url: str = Field(..., exclude=True)
    objvissap_default_params: dict[str, Any] = Field({}, exclude=True)

    def _constraint(self, index: int) -> ConstraintType:
        """
        For a given index, return the constraint at that time.
        """
        return ConstraintType.VISIBILITY

    def prepare_data(self) -> None:
        """
        Prepares visibility data by querying the ObjVisSAP service and parsing the results.
        This method performs the following steps:
        1. Validates that the ObjVisSAP URL is configured
        2. Constructs query parameters including RA, Dec and time range
        3. Queries the ObjVisSAP service via HTTP GET
        4. Parses the returned VOTable XML response
        5. Creates VisWindow objects from the parsed data

        Raises
        ------
        ValueError
            If ObjVisSAP URL is not set or if visibility service is offline

        Returns
        -------
        None
            Results are stored in self.visibility_windows as VisWindow objects


        Notes
        -----
        - Converts input times to MJD format for the query
        - Applies fixes for non-conformant VO data from XMM
        - Each visibility window includes start/end times and visibility time
        """

        # Check that the ObjVisSAP URL is set
        if self.objvissap_url is None:
            raise ValueError("ObjVisSAP URL not set.")

        # Check that timestamp is set
        if self.timestamp is None:
            raise ValueError("Timestamp not set.")

        # Construct the ObjVisSAP query parameters
        params = {
            "s_ra": self.ra,
            "s_dec": self.dec,
            "t_min": Time(self.begin).mjd,
            "t_max": Time(self.end).mjd,
            "min_vis": self.min_vis,
        }
        # Perform the query
        with httpx.Client() as client:
            r = client.get(self.objvissap_url, params=dict(params, **self.objvissap_default_params))

        # If successful, parse the results
        if r.status_code == 200:
            # Parse VOTABLE into VisWindow objects
            xmlvo = r.text
            # Fix broke non-conformant VO from XMM
            xmlvo = xmlvo.replace('datatype="timestamp"', 'datatype="float"')
            xmlvo = xmlvo.replace('name="t_end"', 'name="t_stop"')

            votablefile = io.BytesIO(xmlvo.encode())
            votable = parse_single_table(votablefile)

            # Compute the inconstraint array from the returned VOTable data
            self.inconstraint = np.array(
                [
                    np.where(
                        np.bitwise_and(
                            votable.array.data["t_stop"] >= t.mjd, votable.array.data["t_start"] <= t.mjd
                        )
                    )[0].size
                    > 0
                    for t in self.timestamp
                ]
            )
        else:
            raise ValueError("Visibility tool offline.")

        # Calculate the constraint by constraint dictionary
        self.calculated_constraints = {"Visibility": self.inconstraint}
