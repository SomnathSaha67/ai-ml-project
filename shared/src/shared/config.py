"""Configuration utilities."""

import os
from pathlib import Path
from typing import Optional


def get_env_var(key: str, default: str = None) -> Optional[str]:
    """Get environment variable with optional default."""
    return os.environ.get(key, default)


def get_weka_jar_path() -> Optional[str]:
    """Get Weka JAR path from environment or None."""
    return get_env_var("WEKA_JAR_PATH")


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent.parent


def get_submission_dir() -> Path:
    """Get submission directory."""
    return get_project_root() / "submission"


def get_reports_dir() -> Path:
    """Get shared reports directory."""
    return get_project_root() / "reports"
