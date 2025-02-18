from across.tools.core.schemas.coordinate import Coordinate


class TestCoordinate:
    """Test suite for the Coordinate class testing various coordinate operations and validations."""

    def test_positive_ra_rounding(self) -> None:
        """Test rounding behavior of positive right ascension values to 5 decimal places."""
        coord: Coordinate = Coordinate(ra=123.456789, dec=45.678901)
        assert coord.ra == 123.45679

    def test_positive_dec_rounding(self) -> None:
        """Test rounding behavior of positive declination values to 5 decimal places."""
        coord: Coordinate = Coordinate(ra=123.456789, dec=45.678901)
        assert coord.dec == 45.67890

    def test_negative_ra_conversion(self) -> None:
        """Test conversion of negative right ascension to its positive equivalent (360 - |RA|)."""
        coord: Coordinate = Coordinate(ra=-45.678901, dec=0.0)
        assert coord.ra == 314.32110

    def test_negative_ra_dec_zero(self) -> None:
        """Test that declination remains zero when provided with negative RA and zero dec."""
        coord: Coordinate = Coordinate(ra=-45.678901, dec=0.0)
        assert coord.dec == 0.0

    def test_extreme_negative_ra(self) -> None:
        """Test handling of extreme negative right ascension value (-360)."""
        coord: Coordinate = Coordinate(ra=-360, dec=-90)
        assert coord.ra == 0.0

    def test_extreme_negative_dec(self) -> None:
        """Test handling of extreme negative declination value (-90)."""
        coord: Coordinate = Coordinate(ra=-360, dec=-90)
        assert coord.dec == -90.0

    def test_extreme_positive_ra(self) -> None:
        """Test handling of extreme positive right ascension value (360)."""
        coord: Coordinate = Coordinate(ra=360, dec=90)
        assert coord.ra == 360.0

    def test_extreme_positive_dec(self) -> None:
        """Test handling of extreme positive declination value (90)."""
        coord: Coordinate = Coordinate(ra=360, dec=90)
        assert coord.dec == 90.0

    def test_small_ra_rounding(self) -> None:
        """Test rounding behavior of very small right ascension values."""
        coord: Coordinate = Coordinate(ra=0.0000123, dec=0.0000456)
        assert coord.ra == 0.00001

    def test_small_dec_rounding(self) -> None:
        """Test rounding behavior of very small declination values."""
        coord: Coordinate = Coordinate(ra=0.0000123, dec=0.0000456)
        assert coord.dec == 0.00005

    def test_string_representation(self) -> None:
        """Test the string representation (repr) of the Coordinate class."""
        coord: Coordinate = Coordinate(ra=10.0, dec=20.0)
        assert repr(coord) == "Coordinate(ra=10.0, dec=20.0)"

    def test_exact_equality(self) -> None:
        """Test exact equality comparison between two identical coordinates."""
        coord1: Coordinate = Coordinate(ra=10.0, dec=20.0)
        coord2: Coordinate = Coordinate(ra=10.0, dec=20.0)
        assert coord1 == coord2

    def test_equality_within_tolerance(self) -> None:
        """Test equality comparison with coordinates that differ within acceptable tolerance."""
        coord1: Coordinate = Coordinate(ra=10.0, dec=20.0)
        coord3: Coordinate = Coordinate(ra=10.00001, dec=20.00001)
        assert coord1 == coord3

    def test_inequality_ra(self) -> None:
        """Test inequality comparison between coordinates with different RA values."""
        coord1: Coordinate = Coordinate(ra=10.0, dec=20.0)
        coord4: Coordinate = Coordinate(ra=11.0, dec=20.0)
        assert coord1 != coord4

    def test_inequality_dec(self) -> None:
        """Test inequality comparison between coordinates with different declination values."""
        coord1: Coordinate = Coordinate(ra=10.0, dec=20.0)
        coord5: Coordinate = Coordinate(ra=10.0, dec=21.0)
        assert coord1 != coord5

    def test_inequality_with_string(self) -> None:
        """Test inequality comparison between a coordinate and a string."""
        coord1: Coordinate = Coordinate(ra=10.0, dec=20.0)
        assert coord1 != "not a coordinate"

    def test_inequality_with_number(self) -> None:
        """Test inequality comparison between a coordinate and a number."""
        coord1: Coordinate = Coordinate(ra=10.0, dec=20.0)
        assert coord1 != 42
