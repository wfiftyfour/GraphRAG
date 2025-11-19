"""Configuration management."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Manage application configuration."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._config = {}

    def load(self, config_file: str = "graphrag_config.yaml") -> Dict[str, Any]:
        """Load main configuration."""
        config_path = self.config_dir / config_file

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        # Load environment variables
        self._load_env()

        return self._config

    def _load_env(self):
        """Load environment variables."""
        from dotenv import load_dotenv
        load_dotenv()

    def get(self, *keys, default=None) -> Any:
        """Get nested config value."""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value

    def get_env(self, key: str, default: str = None) -> str:
        """Get environment variable."""
        return os.getenv(key, default)

    @property
    def llm(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self._config.get('llm', {})

    @property
    def embedding(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return self._config.get('embedding', {})

    @property
    def graph(self) -> Dict[str, Any]:
        """Get graph configuration."""
        return self._config.get('graph', {})

    @property
    def search(self) -> Dict[str, Any]:
        """Get search configuration."""
        return self._config.get('search', {})
