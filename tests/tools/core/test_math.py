import numpy as np
import pytest

from across.tools.core.math import find_duplicates, x_rot, y_rot, z_rot


class TestXRotation:
    """Test suite for x_rot function"""

    @pytest.mark.parametrize(
        "angle,expected",
        [
            (0, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
            (90, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])),
            (180, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])),
            (360, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
            (-90, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])),
        ],
    )
    def test_x_rot_angles(self, angle: float, expected: np.typing.NDArray[np.float64]) -> None:
        """Test rotation matrix for various angles"""
        actual = x_rot(angle)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_x_rot_return_type(self) -> None:
        """Test return type is numpy array"""
        result = x_rot(45)
        assert isinstance(result, np.ndarray)

    def test_x_rot_dtype(self) -> None:
        """Test array dtype is float64"""
        result = x_rot(45)
        assert result.dtype == np.float64


class TestZRotation:
    """Test suite for z_rot function"""

    @pytest.mark.parametrize(
        "angle,expected",
        [
            (0, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
            (90, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
            (180, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])),
            (360, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
            (-90, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])),
        ],
    )
    def test_z_rot_angles(self, angle: float, expected: np.typing.NDArray[np.float64]) -> None:
        """Test rotation matrix for various angles"""
        actual = z_rot(angle)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_z_rot_return_type(self) -> None:
        """Test return type is numpy array"""
        result = z_rot(45)
        assert isinstance(result, np.ndarray)

    def test_z_rot_dtype(self) -> None:
        """Test array dtype is float64"""
        result = z_rot(45)
        assert result.dtype == np.float64


class TestYRotation:
    """Test suite for y_rot function"""

    @pytest.mark.parametrize(
        "angle,expected",
        [
            (0, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
            (90, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])),
            (180, np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])),
            (360, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
            (-90, np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])),
        ],
    )
    def test_y_rot_angles(self, angle: float, expected: np.typing.NDArray[np.float64]) -> None:
        """Test rotation matrix for various angles"""
        actual = y_rot(angle)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_y_rot_return_type(self) -> None:
        """Test return type is numpy array"""
        result = y_rot(45)
        assert isinstance(result, np.ndarray)

    def test_y_rot_dtype(self) -> None:
        """Test array dtype is float64"""
        result = y_rot(45)
        assert result.dtype == np.float64


class TestFindDuplicates:
    """Test suite for find_duplicates function"""

    def test_find_duplicates_empty_list(self) -> None:
        """Test with empty list"""
        assert find_duplicates([]) == []

    def test_find_duplicates_no_duplicates(self) -> None:
        """Test with list containing no duplicates"""
        assert find_duplicates([1, 2, 3, 4]) == []

    def test_find_duplicates_single_duplicate(self) -> None:
        """Test with list containing single duplicate"""
        assert find_duplicates([1, 2, 2, 3]) == [2]

    def test_find_duplicates_multiple_duplicates(self) -> None:
        """Test with list containing multiple duplicates"""
        assert find_duplicates([1, 1, 2, 2, 3]) == [1, 2]

    def test_find_duplicates_multiple_occurrences(self) -> None:
        """Test with list containing elements appearing more than twice"""
        assert find_duplicates([1, 1, 1, 2, 2, 3]) == [1, 2]

    def test_find_duplicates_return_type(self) -> None:
        """Test return type is list"""
        result = find_duplicates([1, 2, 2, 3])
        assert isinstance(result, list)
