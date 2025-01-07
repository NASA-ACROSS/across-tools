from collections import Counter

import numpy as np
import numpy.typing as npt


def find_duplicates(input_list: list[int]) -> list[int]:
    """
    Finds duplicates in a list.
    Taken from https://stackoverflow.com/questions/9835762/
    """
    return [item for item, count in Counter(input_list).items() if count > 1]


def x_rot(theta_deg: float) -> npt.NDArray[np.float64]:
    """
    Performs matrix rotation around the cartesian x direction
    """
    theta = np.deg2rad(theta_deg)
    return np.asarray([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])


def y_rot(theta_deg: float) -> npt.NDArray[np.float64]:
    """
    Performs matrix rotation around the cartesian y direction
    """
    theta = np.deg2rad(theta_deg)
    return np.asarray([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])


def z_rot(theta_deg: float) -> npt.NDArray[np.float64]:
    """
    Performs matrix rotation around the cartesian z direction
    """
    theta = np.deg2rad(theta_deg)
    return np.asarray([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
