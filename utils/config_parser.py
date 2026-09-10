"""
Configuration parser for loading, validating, and updating YAML settings.
"""

from pathlib import Path
from typing import Any, Dict, Union
import yaml
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """Loads and validates a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration successfully from {path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config file {path}: {e}")
            raise e
