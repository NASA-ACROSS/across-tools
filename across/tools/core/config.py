from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """Base configuration for the application."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Config(BaseConfig):
    """Configuration for the application.
    The Config class extends BaseConfig and provides configuration settings for the application,
    including environment, hosting, logging, and request-related parameters.

    Attributes
    ----------
    SPACETRACK_USER : str
        Space-Track.org username
    SPACETRACK_PWD  : str
        Space-Track.org password
    """

    SPACETRACK_USER: str | None = None
    SPACETRACK_PWD: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


config = Config()
