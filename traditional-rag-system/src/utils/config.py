"""Configuration loader."""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Load and manage configuration."""

    def __init__(self, config_path: str = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to config file (default: configs/rag_config.yaml)
        """
        if config_path is None:
            # Default to configs/rag_config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "rag_config.yaml"

        self.config_path = Path(config_path)
        self.config = None

    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        return self.config

    def get(self, key: str, default=None):
        """Get configuration value by key (supports nested keys with dot notation)."""
        if self.config is None:
            self.load()

        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def __getitem__(self, key):
        """Allow dict-like access."""
        if self.config is None:
            self.load()
        return self.config[key]
