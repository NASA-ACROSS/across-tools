import numpy as np

from across.tools.core.math import find_duplicates, x_rot, y_rot, z_rot


class TestXRotation:
    """Test suite for x_rot function"""

    def test_x_rot_zero_degrees(self) -> None:
        """Test rotation matrix for 0 degrees"""
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actual = x_rot(0)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_x_rot_90_degrees(self) -> None:
        """Test rotation matrix for 90 degrees"""
        expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        actual = x_rot(90)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_x_rot_180_degrees(self) -> None:
        """Test rotation matrix for 180 degrees"""
        expected = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        actual = x_rot(180)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_x_rot_360_degrees(self) -> None:
        """Test rotation matrix for 360 degrees"""
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actual = x_rot(360)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_x_rot_negative_angle(self) -> None:
        """Test rotation matrix for -90 degrees"""
        expected = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        actual = x_rot(-90)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_x_rot_return_type(self) -> None:
        """Test return type is numpy array"""
        result = x_rot(45)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64


class TestZRotation:
    """Test suite for z_rot function"""

    def test_z_rot_zero_degrees(self) -> None:
        """Test rotation matrix for 0 degrees"""
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actual = z_rot(0)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_z_rot_90_degrees(self) -> None:
        """Test rotation matrix for 90 degrees"""
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        actual = z_rot(90)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_z_rot_180_degrees(self) -> None:
        """Test rotation matrix for 180 degrees"""
        expected = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        actual = z_rot(180)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_z_rot_360_degrees(self) -> None:
        """Test rotation matrix for 360 degrees"""
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actual = z_rot(360)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_z_rot_negative_angle(self) -> None:
        """Test rotation matrix for -90 degrees"""
        expected = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        actual = z_rot(-90)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_z_rot_return_type(self) -> None:
        """Test return type is numpy array"""
        result = z_rot(45)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64


class TestYRotation:
    """Test suite for y_rot function"""

    def test_y_rot_zero_degrees(self) -> None:
        """Test rotation matrix for 0 degrees"""
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actual = y_rot(0)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_y_rot_90_degrees(self) -> None:
        """Test rotation matrix for 90 degrees"""
        expected = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        actual = y_rot(90)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_y_rot_180_degrees(self) -> None:
        """Test rotation matrix for 180 degrees"""
        expected = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        actual = y_rot(180)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_y_rot_360_degrees(self) -> None:
        """Test rotation matrix for 360 degrees"""
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actual = y_rot(360)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_y_rot_negative_angle(self) -> None:
        """Test rotation matrix for -90 degrees"""
        expected = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        actual = y_rot(-90)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_y_rot_return_type(self) -> None:
        """Test return type is numpy array"""
        result = y_rot(45)
        assert isinstance(result, np.ndarray)
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
