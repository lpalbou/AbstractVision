#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BundleItem:
    path: str
    fenced_lang: Optional[str] = None


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "llms-full.txt"

# Keep this list aligned with the "Documentation" section of llms.txt.
BUNDLE: list[BundleItem] = [
    BundleItem("llms.txt"),
    BundleItem("README.md"),
    BundleItem("docs/README.md"),
    BundleItem("docs/getting-started.md"),
    BundleItem("docs/api.md"),
    BundleItem("docs/architecture.md"),
    BundleItem("docs/faq.md"),
    BundleItem("docs/reference/backends.md"),
    BundleItem("docs/reference/configuration.md"),
    BundleItem("docs/reference/capabilities-registry.md"),
    BundleItem("docs/adr/README.md"),
    BundleItem("docs/adr/0005_curated_capability_registry_and_download_catalog.md"),
    BundleItem("docs/adr/0006_operator_control_configuration_precedence_and_explicit_network_use.md"),
    BundleItem("docs/reference/artifacts.md"),
    BundleItem("docs/reference/abstractcore-integration.md"),
    BundleItem("playground/README.md"),
    BundleItem("CONTRIBUTING.md"),
    BundleItem("SECURITY.md"),
    BundleItem("ACKNOWLEDGMENTS.md"),
    BundleItem("CHANGELOG.md"),
    BundleItem("pyproject.toml", fenced_lang="toml"),
]


def _read_text(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _strip_trailing_newlines(s: str) -> str:
    return s.rstrip("\n")


def main() -> None:
    header_lines: list[str] = [
        "# AbstractVision — llms-full",
        "",
        "> Single-file, agent-oriented context bundle generated from this repository’s docs and metadata.",
        "",
        "How to use:",
        "- If you only need a map of where to look, use `llms.txt`.",
        "- If you need an all-in-one context bundle (for offline models or constrained retrieval), use this file.",
        "",
        "Notes:",
        "- This is a **convention** (many projects publish an `llms-full.txt`). It is not part of the core `llms.txt` spec.",
        "- Relative links inside each file section are authored for that file’s original location. Use the `FILE: …` marker to interpret link paths.",
        "- If you change docs or packaging metadata, regenerate this file by running:",
        "  - `python scripts/generate_llms_full.py`",
        "",
        "Included files:",
    ]
    for item in BUNDLE:
        header_lines.append(f"- `{item.path}`")
    header_lines.append("")

    parts: list[str] = ["\n".join(header_lines)]
    for item in BUNDLE:
        content = _strip_trailing_newlines(_read_text(item.path))
        parts.append(f"--- 8< --- FILE: {item.path} --- 8< ---")
        if item.fenced_lang:
            parts.append(f"```{item.fenced_lang}")
            parts.append(content)
            parts.append("```")
        else:
            parts.append(content)
        parts.append(f"--- >8 --- END FILE: {item.path} --- >8 ---")
        parts.append("")

    OUT.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
