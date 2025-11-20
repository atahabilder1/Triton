#!/usr/bin/env python3
"""
Configuration Management for Triton
Centralized config loading and access
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Centralized configuration manager"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_path: Path to config file (defaults to root config.yaml)
        """
        if config_path is None:
            # Look for config.yaml in project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"

        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Examples:
            config.get('data.train_dir')
            config.get('training.batch_size')
            config.get('models.static_encoder')
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_data_path(self, key: str) -> Path:
        """Get data path as Path object"""
        path = self.get(f'data.{key}')
        if path is None:
            raise ValueError(f"Data path '{key}' not found in config")
        return Path(path)

    def get_model_path(self, key: str) -> Path:
        """Get model path as Path object"""
        path = self.get(f'models.{key}')
        if path is None:
            raise ValueError(f"Model path '{key}' not found in config")
        return Path(path)

    def get_training_config(self, training_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get training configuration

        Args:
            training_type: Type of training ('static', 'dynamic', 'semantic', 'full')
                         If None, returns common training config

        Returns:
            Training configuration dict
        """
        if training_type:
            # Get type-specific config and merge with common config
            common = self.get('training', {})
            specific = self.get(f'training.{training_type}', {})

            # Create merged config (specific overrides common)
            config = {k: v for k, v in common.items() if not isinstance(v, dict)}
            config.update(specific)
            return config
        return self.get('training', {})

    def get_architecture_config(self, model_name: str) -> Dict[str, Any]:
        """Get architecture config for specific model"""
        return self.get(f'architecture.{model_name}', {})

    def get_vulnerability_classes(self) -> list:
        """Get list of vulnerability classes"""
        return self.get('vulnerabilities.classes', [])

    def get_num_classes(self) -> int:
        """Get number of vulnerability classes"""
        return self.get('vulnerabilities.num_classes', 11)

    @property
    def train_dir(self) -> Path:
        """Training data directory"""
        return self.get_data_path('train_dir')

    @property
    def val_dir(self) -> Path:
        """Validation data directory"""
        return self.get_data_path('val_dir')

    @property
    def test_dir(self) -> Path:
        """Test data directory"""
        return self.get_data_path('test_dir')

    @property
    def cache_dir(self) -> Path:
        """Cache directory"""
        return self.get_data_path('cache_dir')

    @property
    def checkpoints_dir(self) -> Path:
        """Checkpoints directory"""
        return self.get_model_path('checkpoints_dir')

    @property
    def batch_size(self) -> int:
        """Training batch size"""
        return self.get('training.batch_size', 16)

    @property
    def learning_rate(self) -> float:
        """Learning rate"""
        return self.get('training.learning_rate', 0.001)

    @property
    def num_epochs(self) -> int:
        """Number of training epochs"""
        return self.get('training.num_epochs', 50)

    @property
    def device(self) -> str:
        """Training device"""
        return self.get('training.device', 'auto')

    @property
    def num_workers(self) -> int:
        """Number of data loader workers"""
        return self.get('training.num_workers', 4)

    def __repr__(self) -> str:
        return f"Config({self._config})"


# Global config instance
_global_config: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration file

    Args:
        config_path: Path to config file (optional)

    Returns:
        Config object
    """
    global _global_config
    _global_config = Config(config_path)
    return _global_config


def get_config() -> Config:
    """
    Get global config instance

    Returns:
        Config object

    Raises:
        RuntimeError: If config not loaded yet
    """
    global _global_config
    if _global_config is None:
        # Auto-load default config
        _global_config = Config()
    return _global_config


# Convenience functions
def get_data_path(key: str) -> Path:
    """Get data path from config"""
    return get_config().get_data_path(key)


def get_model_path(key: str) -> Path:
    """Get model path from config"""
    return get_config().get_model_path(key)


def get_training_config() -> Dict[str, Any]:
    """Get training configuration"""
    return get_config().get_training_config()
