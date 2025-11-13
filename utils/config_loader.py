from pathlib import Path
import os
import yaml
from typing import Optional

from logger.custom_logger import CustomLogger


logger = CustomLogger().get_logger(__file__)


def _project_root() -> Path:
    """
    Get project root path (parent of utils/).
    Example: .../FloatChat
    """
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load YAML config reliably irrespective of CWD.

    Priority:
      1. Explicit config_path argument
      2. CONFIG_PATH env variable
      3. <project_root>/config/config.yaml
    """
    env_path = os.getenv("CONFIG_PATH")
    if config_path is None:
        config_path = env_path or str(_project_root() / "config" / "config.yaml")

    path = Path(config_path)
    if not path.is_absolute():
        path = _project_root() / path

    if not path.exists():
        logger.error("❌ Config file not found", path=str(path))
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    logger.info("✅ Config loaded", path=str(path), keys=list(config.keys()))
    return config
