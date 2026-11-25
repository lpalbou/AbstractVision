#!/usr/bin/env python3
"""Rewrite `src/abstractvision/assets/vision_model_hub_facts.json` from the Hugging Face Hub.

For every repository a download preset names (and every stable-diffusion.cpp companion), this reads
the Hub's public metadata (`https://huggingface.co/api/models/<repo>?blobs=true`, no token, no file
download) and records whether the repository is gated and, for each non-Diffusers preset target,
how many bytes `abstractvision download` fetches with that preset's patterns. Diffusers presets get
no size: the files they fetch depend on the host (the checkpoint variant the backend loads).

Usage: python scripts/refresh_hub_facts.py [--check]
`--check` prints the differences and exits 1 when the asset is out of date, without writing it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "src" / "abstractvision" / "assets" / "vision_model_hub_facts.json"
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub.utils import filter_repo_objects  # noqa: E402

from abstractvision.model_downloads import _SDCPP_BUNDLES, _all_presets  # noqa: E402


def _fetch(repo_id: str) -> Dict[str, Any]:
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo_id}?blobs=true",
        headers={"User-Agent": "abstractvision-hub-facts"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        return {"error": int(e.code)}


def _selected_bytes(siblings: List[Dict[str, Any]], allow: Optional[List[str]], ignore: Optional[List[str]]) -> int:
    chosen = filter_repo_objects(
        items=siblings,
        allow_patterns=allow or None,
        ignore_patterns=ignore or None,
        key=lambda sibling: str(sibling["rfilename"]),
    )
    return sum(int(sibling.get("size") or 0) for sibling in chosen)


def build() -> Dict[str, Any]:
    presets = _all_presets()
    repos = sorted({p.repo_id for p in presets} | {c.repo_id for b in _SDCPP_BUNDLES.values() for c in b.components})
    with ThreadPoolExecutor(8) as pool:
        metadata = dict(zip(repos, pool.map(_fetch, repos)))
    out: Dict[str, Any] = {}
    for repo_id in repos:
        data = metadata[repo_id]
        if "error" in data:
            print(f"{repo_id}: HTTP {data['error']} (left out)", file=sys.stderr)
            continue
        siblings = list(data.get("siblings") or [])
        entry: Dict[str, Any] = {"gated": bool(data.get("gated"))}
        sizes: Dict[str, int] = {}
        for preset in presets:
            if preset.repo_id != repo_id or preset.target == "diffusers":
                continue
            allow = list(preset.allow_patterns) or None
            ignore = ["*/*"] if preset.target == "gguf" and not any("/" in p for p in (allow or ())) else None
            sizes[preset.target] = _selected_bytes(siblings, allow, ignore)
        if sizes:
            entry["download_bytes"] = dict(sorted(sizes.items()))
        out[repo_id] = entry
    return {
        "schema_version": 1,
        "source": "https://huggingface.co/api/models/<repo_id>?blobs=true",
        "checked": _dt.date.today().isoformat(),
        "repos": out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Report differences; do not write.")
    args = parser.parse_args()
    fresh = build()
    current = json.loads(ASSET.read_text(encoding="utf-8")) if ASSET.is_file() else {"repos": {}}
    changed = sorted(
        repo
        for repo in set(fresh["repos"]) | set(current.get("repos", {}))
        if fresh["repos"].get(repo) != current.get("repos", {}).get(repo)
    )
    for repo in changed:
        print(f"{repo}: {current.get('repos', {}).get(repo)} -> {fresh['repos'].get(repo)}")
    if args.check:
        return 1 if changed else 0
    ASSET.write_text(json.dumps(fresh, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {ASSET} ({len(fresh['repos'])} repositories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
