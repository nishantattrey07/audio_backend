# src/config.py
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Configure logging for the settings module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Centralized application settings.
    Reads configuration from environment variables, which can be populated
    from a .env file for local development.
    """
    # This tells pydantic-settings to look for a .env file.
    # It will automatically override with system environment variables if they exist.
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # PostgreSQL Settings
    POSTGRES_HOST: str
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    # Redis Settings
    REDIS_HOST: str
    REDIS_PORT: int = 6379
    # Use Optional[str] = "" for passwords that might be empty
    REDIS_PASSWORD: str = ""

    # This is a helper property to construct the DSN string required by psycopg2
    @property
    def postgres_dsn(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

# Create a single, importable instance of the settings
try:
    settings = Settings()
    logger.info("✅ Configuration loaded successfully.")
except Exception as e:
    logger.error(f"🔴 FATAL: Could not load configuration. Missing environment variables? Error: {e}")
    # Exit if config fails to load, as the app cannot run.
    exit(1)