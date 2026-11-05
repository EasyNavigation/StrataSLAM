"""Reading a robot profile for the CLI tools: the profile is the whole configuration (ADR-0053).

Validation itself is done by the C++ core (`strataslam_ros.validate_profile`); this module only needs
the topics, frames and ranges the doctor checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_profile(path: str | Path) -> dict[str, Any]:
    """Profile as a dict. A section whose keys are all commented out (`output:`, `lidar:`) is null in YAML; the core
    reads it as present and empty, so a sensor written that way is enabled with its defaults."""
    prof: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    sensors = prof.get("sensors") or {}
    prof["sensors"] = sensors
    for section in (prof, sensors):
        for key, value in list(section.items()):
            if value is None:
                section[key] = {}
    return prof


def enabled(prof: dict[str, Any], sensor: str) -> bool:
    """A present sensor section is enabled unless it says otherwise (ADR-0002)."""
    section = prof.get("sensors", {}).get(sensor)
    if section is None:
        return False
    return bool(section.get("enabled", True)) if isinstance(section, dict) else bool(section)
