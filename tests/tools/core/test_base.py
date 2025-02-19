from across.tools.core.schemas.base import BaseSchema


class TestBaseSchema:
    """Test suite for the BaseSchema class."""

    def test_base_schema_hash(self, test_model: BaseSchema) -> None:
        """Test that BaseSchema objects can be hashed."""
        hash(test_model)  # Should not raise exception

    def test_base_schema_hash_equality(self, test_model: BaseSchema) -> None:
        """Test that identical BaseSchema objects have same hash."""
        assert hash(test_model) == hash(test_model)

    def test_base_schema_hash_inequality(self, test_model: BaseSchema, test_model_two: BaseSchema) -> None:
        """Test that different BaseSchema objects have different hashes."""
        assert hash(test_model) != hash(test_model_two)

    def test_base_schema_config_from_attributes(self, test_model_class: type[BaseSchema]) -> None:
        """Test that BaseSchema has correct model config."""
        assert test_model_class.model_config["from_attributes"] is True

    def test_base_schema_config_arbitrary_types_allowed(self, test_model_class: type[BaseSchema]) -> None:
        """Test that BaseSchema has correct model config."""
        assert test_model_class.model_config["arbitrary_types_allowed"] is True
