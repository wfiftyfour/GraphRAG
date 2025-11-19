"""Configuration management."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Manage application configuration."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._configs = {}

    def load(self, config_name: str) -> Dict[str, Any]:
        """Load a configuration file."""
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self._configs[config_name] = config
        return config

    def get(self, config_name: str, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if config_name not in self._configs:
            self.load(config_name)

        return self._configs[config_name].get(key, default)

    def load_env(self):
        """Load environment variables."""
        from dotenv import load_dotenv
        load_dotenv()

    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get an environment variable."""
        return os.getenv(key, default)
