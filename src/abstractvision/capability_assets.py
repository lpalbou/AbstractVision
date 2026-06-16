from __future__ import annotations

import json
import os
import pkgutil
from pathlib import Path
from typing import Any, Optional


def load_json_asset(
    *,
    package: str,
    default_asset_path: str,
    asset_path: Optional[str] = None,
    env_var: Optional[str] = None,
    label: str,
) -> Any:
    configured = str(
        asset_path
        or (os.getenv(env_var) if env_var else "")
        or default_asset_path
    ).strip()
    if not configured:
        raise RuntimeError(f"{label} path is empty.")

    fs_candidate = _filesystem_candidate(configured)
    if fs_candidate is not None:
        if not fs_candidate.is_file():
            raise RuntimeError(f"{label} not found: {fs_candidate}")
        return json.loads(fs_candidate.read_text(encoding="utf-8"))

    raw = pkgutil.get_data(package, configured)
    if raw is None:
        raise RuntimeError(f"{label} not found: {package}/{configured}")
    return json.loads(raw.decode("utf-8"))


def _filesystem_candidate(value: str) -> Optional[Path]:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if value.startswith(("./", "../", "~")):
        return path
    if path.exists():
        return path
    return None
