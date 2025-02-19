from across.tools.core.schemas.coordinate import Coordinate


class TestCoordinate:
    """Test suite for the Coordinate class testing various coordinate operations and validations."""

    def test_positive_ra_rounding(self, coord_large_values: Coordinate) -> None:
        """Test rounding behavior of positive right ascension values to 5 decimal places."""
        assert coord_large_values.ra == 123.45679

    def test_positive_dec_rounding(self, coord_large_values: Coordinate) -> None:
        """Test rounding behavior of positive declination values to 5 decimal places."""
        assert coord_large_values.dec == 45.67890

    def test_negative_ra_conversion(self, coord_negative_ra: Coordinate) -> None:
        """Test conversion of negative right ascension to its positive equivalent (360 - |RA|)."""
        assert coord_negative_ra.ra == 314.32110

    def test_negative_ra_dec_zero(self, coord_negative_ra: Coordinate) -> None:
        """Test that declination remains zero when provided with negative RA and zero dec."""
        assert coord_negative_ra.dec == 0.0

    def test_extreme_negative_ra(self, coord_extreme_negative: Coordinate) -> None:
        """Test handling of extreme negative right ascension value (-360)."""
        assert coord_extreme_negative.ra == 0.0

    def test_extreme_negative_dec(self, coord_extreme_negative: Coordinate) -> None:
        """Test handling of extreme negative declination value (-90)."""
        assert coord_extreme_negative.dec == -90.0

    def test_extreme_positive_ra(self, coord_extreme: Coordinate) -> None:
        """Test handling of extreme positive right ascension value (360)."""
        assert coord_extreme.ra == 360.0

    def test_extreme_positive_dec(self, coord_extreme: Coordinate) -> None:
        """Test handling of extreme positive declination value (90)."""
        assert coord_extreme.dec == 90.0

    def test_small_ra_rounding(self, coord_small_values: Coordinate) -> None:
        """Test rounding behavior of very small right ascension values."""
        assert coord_small_values.ra == 0.00001

    def test_small_dec_rounding(self, coord_small_values: Coordinate) -> None:
        """Test rounding behavior of very small declination values."""
        assert coord_small_values.dec == 0.00005

    def test_string_representation(self, coord_standard: Coordinate) -> None:
        """Test the string representation (repr) of the Coordinate class."""
        assert repr(coord_standard) == "Coordinate(ra=10.0, dec=20.0)"

    def test_exact_equality(self, coord_standard: Coordinate) -> None:
        """Test exact equality comparison between two identical coordinates."""
        coord2 = Coordinate(ra=10.0, dec=20.0)
        assert coord_standard == coord2

    def test_equality_within_tolerance(self, coord_standard: Coordinate) -> None:
        """Test equality comparison with coordinates that differ within acceptable tolerance."""
        coord3 = Coordinate(ra=10.00001, dec=20.00001)
        assert coord_standard == coord3

    def test_inequality_ra(self, coord_standard: Coordinate) -> None:
        """Test inequality comparison between coordinates with different RA values."""
        coord4 = Coordinate(ra=11.0, dec=20.0)
        assert coord_standard != coord4

    def test_inequality_dec(self, coord_standard: Coordinate) -> None:
        """Test inequality comparison between coordinates with different declination values."""
        coord5 = Coordinate(ra=10.0, dec=21.0)
        assert coord_standard != coord5

    def test_inequality_with_string(self, coord_standard: Coordinate) -> None:
        """Test inequality comparison between a coordinate and a string."""
        assert coord_standard != "not a coordinate"

    def test_inequality_with_number(self, coord_standard: Coordinate) -> None:
        """Test inequality comparison between a coordinate and a number."""
        assert coord_standard != 42
