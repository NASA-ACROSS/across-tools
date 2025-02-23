from across.tools.visibility.base import Visibility


class FootprintVisibility(Visibility):
    """
    A class to calculate the visibility of a target based on a observatory
    footprint and schedule.
    """

    def _constraint(self, i: int) -> str:
        """
        For a given index, return the constraint at that time. For a footprint
        visibility, this is always Field of View (FOV).
        """
        return "FOV"

    def prepare_data(self) -> None:
        """
        Prepare the data for the visibility calculation.
        """
        raise NotImplementedError
