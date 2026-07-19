"""Optional prior-information configuration loading for PKGPT."""

import json
import os
from typing import Any, Dict


def load_prior_info(path: str) -> Dict[str, Any]:
    """Load an optional JSON or YAML prior-information file."""
    if not os.path.isfile(path):
        raise ValueError(f"Prior-information file not found: {path}")

    extension = os.path.splitext(path)[1].lower()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            if extension == ".json":
                data = json.load(handle)
            elif extension in {".yaml", ".yml"}:
                try:
                    import yaml
                except ImportError as exc:
                    raise ValueError(
                        "YAML prior-information files require PyYAML. "
                        "Install it with: pip install pyyaml, or use JSON."
                    ) from exc
                data = yaml.safe_load(handle)
            else:
                raise ValueError("Prior-information file must be .json, .yaml, or .yml")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read prior-information file: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Prior-information file must contain one top-level object/mapping")
    return data
