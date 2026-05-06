from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class HfFileSpec:
    repo_id: str
    filename: str
    repo_type: str = "model"
    revision: Optional[str] = None
    rename_to: Optional[str] = None


@dataclass(frozen=True)
class HfSnapshotSpec:
    repo_id: str
    repo_type: str = "model"
    revision: Optional[str] = None
    allow_patterns: Optional[Sequence[str]] = None
    ignore_patterns: Optional[Sequence[str]] = None
    local_dir_name: Optional[str] = None


@dataclass(frozen=True)
class ModelSetSpec:
    name: str
    description: str
    files: Sequence[HfFileSpec] = ()
    snapshots: Sequence[HfSnapshotSpec] = ()


def _default_out_dir() -> Path:
    # Prefer the parent AbstractFramework repo root's `untracked/` folder when running from a mono-checkout.
    # This script lives at: abstractvision/scripts/download_model_sets.py
    abstractvision_repo = Path(__file__).resolve().parents[1]
    parent = abstractvision_repo.parent
    untracked = parent / "untracked"
    if untracked.is_dir():
        return untracked / "models" / "abstractvision"
    return Path.home() / ".cache" / "abstractvision" / "models"


def _token_from_env() -> Optional[str]:
    # Hugging Face tooling recognizes a few env vars; prefer HF_TOKEN for consistency with our docs.
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(k)
        if v and str(v).strip():
            return str(v).strip()
    return None


def _require_hf_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install Diffusers/Transformers (or huggingface_hub) first.\n"
            "Example: pip install -U huggingface_hub"
        ) from e
    return hf_hub_download, snapshot_download


def _safe_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _iter_sets() -> List[ModelSetSpec]:
    # Notes on sources:
    # - Flux.2-dev 8-bit: use pre-converted GGUF (not gated) + Comfy-Org VAE + Unsloth Mistral GGUF.
    # - Klein / Qwen: use official Diffusers repos (snapshots) so they can be loaded via Diffusers backend.
    return [
        ModelSetSpec(
            name="sd15_diffusers",
            description="Stable Diffusion 1.5 (Diffusers snapshot; default REPL model).",
            snapshots=[
                HfSnapshotSpec(repo_id="runwayml/stable-diffusion-v1-5"),
            ],
        ),
        ModelSetSpec(
            name="flux2_dev_8bit_gguf",
            description="FLUX.2-dev 8-bit (Q8_0) GGUF + VAE + Mistral LLM (for stable-diffusion.cpp backend).",
            files=[
                HfFileSpec(
                    repo_id="city96/FLUX.2-dev-gguf",
                    filename="flux2-dev-Q8_0.gguf",
                ),
                HfFileSpec(
                    repo_id="Comfy-Org/flux2-dev",
                    filename="split_files/vae/flux2-vae.safetensors",
                    rename_to="flux2_ae.safetensors",
                ),
                HfFileSpec(
                    repo_id="unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                    filename="Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
                ),
            ],
        ),
        ModelSetSpec(
            name="flux2_klein_4b_gguf",
            description="FLUX.2-klein-4B GGUF + VAE + Qwen3-4B LLM (for stable-diffusion.cpp backend; avoids diffusers@main).",
            files=[
                HfFileSpec(
                    repo_id="leejet/FLUX.2-klein-4B-GGUF",
                    filename="flux-2-klein-4b-Q8_0.gguf",
                ),
                HfFileSpec(
                    repo_id="Comfy-Org/flux2-dev",
                    filename="split_files/vae/flux2-vae.safetensors",
                    rename_to="flux2_ae.safetensors",
                ),
                HfFileSpec(
                    repo_id="unsloth/Qwen3-4B-GGUF",
                    filename="Qwen3-4B-Q4_K_M.gguf",
                ),
            ],
        ),
        ModelSetSpec(
            name="flux2_klein_9b_gguf",
            description="FLUX.2-klein-9B GGUF + VAE + Qwen3-8B LLM (for stable-diffusion.cpp backend; avoids gated Diffusers snapshot).",
            files=[
                HfFileSpec(
                    repo_id="leejet/FLUX.2-klein-9B-GGUF",
                    filename="flux-2-klein-9b-Q8_0.gguf",
                ),
                HfFileSpec(
                    repo_id="Comfy-Org/flux2-dev",
                    filename="split_files/vae/flux2-vae.safetensors",
                    rename_to="flux2_ae.safetensors",
                ),
                HfFileSpec(
                    repo_id="unsloth/Qwen3-8B-GGUF",
                    filename="Qwen3-8B-Q4_K_M.gguf",
                ),
            ],
        ),
        ModelSetSpec(
            name="flux2_klein_4b_diffusers",
            description="FLUX.2-klein-4B (Diffusers snapshot; requires diffusers@main for Flux2KleinPipeline).",
            snapshots=[
                HfSnapshotSpec(repo_id="black-forest-labs/FLUX.2-klein-4B"),
            ],
        ),
        ModelSetSpec(
            name="flux2_klein_9b_diffusers",
            description="FLUX.2-klein-9B (Diffusers snapshot; gated; requires diffusers@main + HF_TOKEN).",
            snapshots=[
                HfSnapshotSpec(repo_id="black-forest-labs/FLUX.2-klein-9B"),
            ],
        ),
        ModelSetSpec(
            name="qwen_image_diffusers",
            description="Qwen Image (Diffusers snapshot).",
            snapshots=[
                HfSnapshotSpec(repo_id="Qwen/Qwen-Image"),
            ],
        ),
        ModelSetSpec(
            name="qwen_image_2512_diffusers",
            description="Qwen Image 2512 (Diffusers snapshot).",
            snapshots=[
                HfSnapshotSpec(repo_id="Qwen/Qwen-Image-2512"),
            ],
        ),
    ]


def _resolve_selected(available: Sequence[ModelSetSpec], names: Sequence[str], *, all_sets: bool) -> List[ModelSetSpec]:
    by_name = {s.name: s for s in available}
    if all_sets:
        return list(available)
    out: List[ModelSetSpec] = []
    for n in names:
        key = str(n or "").strip()
        if not key:
            continue
        if key not in by_name:
            raise SystemExit(f"Unknown model set: {key!r}. Use --list to see valid names.")
        out.append(by_name[key])
    if not out:
        raise SystemExit("No model sets selected. Use --all or --set ... (or --list).")
    return out


def _plan_lines(sets: Sequence[ModelSetSpec], out_dir: Path) -> List[str]:
    lines: List[str] = []
    lines.append(f"Output directory: {out_dir}")
    lines.append("")
    for s in sets:
        lines.append(f"- {s.name}: {s.description}")
        for f in s.files:
            dst_name = f.rename_to or Path(f.filename).name
            lines.append(f"  - file: {f.repo_id}:{f.filename} -> {out_dir / s.name / dst_name}")
        for snap in s.snapshots:
            local_name = snap.local_dir_name or _safe_dir_name(snap.repo_id)
            lines.append(f"  - snapshot: {snap.repo_id} -> {out_dir / s.name / local_name}")
        lines.append("")
    return lines


def _sd_cli_hint() -> str:
    """High-signal guidance for selecting CLI vs python mode.

    We intentionally keep this lightweight and avoid hardcoding release asset URLs.
    """

    found = shutil.which("sd-cli")
    if found:
        return f"sd-cli: found at {found!r} (CLI mode will be used)"
    if sys.platform == "darwin":
        return (
            "sd-cli: not found in PATH (python bindings fallback may be CPU-only). "
            "For Metal acceleration on macOS, download `sd-cli` from stable-diffusion.cpp releases and pass its path "
            "as the last arg to `/backend sdcpp ...` (or add it to PATH)."
        )
    return (
        "sd-cli: not found in PATH (python bindings fallback will be used if installed). "
        "If you want GPU acceleration (e.g. CUDA), install an appropriate `sd-cli` build and add it to PATH."
    )


def _download_file(spec: HfFileSpec, *, out_dir: Path, token: Optional[str]) -> Path:
    hf_hub_download, _snapshot_download = _require_hf_hub()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            repo_type=spec.repo_type,
            revision=spec.revision,
            token=token,
            local_dir=str(out_dir),
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            raise SystemExit(
                f"Failed to download {spec.repo_id}:{spec.filename} (likely gated).\n"
                "Fix: accept the model terms on Hugging Face in your browser and set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) "
                "in your environment."
            ) from e
        raise

    p = Path(path)
    if spec.rename_to:
        target = out_dir / str(spec.rename_to)
        if target != p:
            target.parent.mkdir(parents=True, exist_ok=True)
            p.replace(target)
            return target
    return p


def _download_snapshot(spec: HfSnapshotSpec, *, out_dir: Path, token: Optional[str]) -> Path:
    _hf_hub_download, snapshot_download = _require_hf_hub()
    out_dir.mkdir(parents=True, exist_ok=True)
    local_name = spec.local_dir_name or _safe_dir_name(spec.repo_id)
    local_dir = out_dir / local_name
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type=spec.repo_type,
            revision=spec.revision,
            allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
            ignore_patterns=list(spec.ignore_patterns) if spec.ignore_patterns else None,
            token=token,
            local_dir=str(local_dir),
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            raise SystemExit(
                f"Failed to download snapshot for {spec.repo_id!r} (likely gated).\n"
                "Fix: accept the model terms on Hugging Face in your browser and set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) "
                "in your environment."
            ) from e
        raise
    return local_dir


def _post_instructions(selected_sets: Sequence[ModelSetSpec], out_dir: Path) -> List[str]:
    # Keep this short and high-signal: show commands for the user to test quickly.
    lines: List[str] = []
    selected = {s.name for s in selected_sets}

    if any(n.endswith("_gguf") for n in selected):
        lines.extend(["", "stable-diffusion.cpp runtime hint:", f"  {_sd_cli_hint()}"])

    if "sd15_diffusers" in selected:
        local_dir = out_dir / "sd15_diffusers" / "runwayml__stable-diffusion-v1-5"
        lines.extend(
            [
                "",
                "Stable Diffusion 1.5 quick test (Diffusers backend):",
                "  abstractvision repl",
                f"  /backend diffusers {local_dir} auto",
                '  /t2i "a watercolor painting of a lighthouse" --width 512 --height 512 --steps 20 --open',
            ]
        )

    if "flux2_klein_4b_gguf" in selected:
        base = out_dir / "flux2_klein_4b_gguf"
        diffusion = base / "flux-2-klein-4b-Q8_0.gguf"
        vae = base / "flux2_ae.safetensors"
        llm = base / "Qwen3-4B-Q4_K_M.gguf"
        lines.extend(
            [
                "",
                "FLUX.2-klein-4B (GGUF) quick test (stable-diffusion.cpp backend):",
                "  abstractvision repl",
                "  /backend sdcpp " + " ".join([str(diffusion), str(vae), str(llm)]) + "  # optional last arg: /path/to/sd-cli",
                '  /t2i "A cat holding a sign that says hello world" --steps 4 --guidance-scale 1.0 --sampling-method euler --diffusion-fa --offload-to-cpu --open',
            ]
        )

    if "flux2_dev_8bit_gguf" in selected:
        base = out_dir / "flux2_dev_8bit_gguf"
        diffusion = base / "flux2-dev-Q8_0.gguf"
        vae = base / "flux2_ae.safetensors"
        llm = base / "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf"
        lines.extend(
            [
                "",
                "FLUX.2-dev (GGUF; heavier) quick test (stable-diffusion.cpp backend):",
                "  abstractvision repl",
                "  /backend sdcpp "
                + " ".join([str(diffusion), str(vae), str(llm)])
                + "  # optional last arg: /path/to/sd-cli (recommended for Metal/CUDA acceleration)",
                '  /t2i "a minimalist product photo of a matte black espresso machine, studio lighting" --steps 10 --guidance-scale 1.0 --sampling-method euler --diffusion-fa --offload-to-cpu --open',
            ]
        )

    if "flux2_klein_9b_gguf" in selected:
        base = out_dir / "flux2_klein_9b_gguf"
        diffusion = base / "flux-2-klein-9b-Q8_0.gguf"
        vae = base / "flux2_ae.safetensors"
        llm = base / "Qwen3-8B-Q4_K_M.gguf"
        lines.extend(
            [
                "",
                "FLUX.2-klein-9B (GGUF) quick test (stable-diffusion.cpp backend):",
                "  abstractvision repl",
                "  /backend sdcpp " + " ".join([str(diffusion), str(vae), str(llm)]) + "  # optional last arg: /path/to/sd-cli",
                '  /t2i "A cat holding a sign that says hello world" --steps 4 --guidance-scale 1.0 --sampling-method euler --diffusion-fa --offload-to-cpu --open',
            ]
        )

    if any(n in selected for n in ("flux2_klein_4b_diffusers", "flux2_klein_9b_diffusers")):
        lines.extend(
            [
                "",
                "FLUX.2 klein (Diffusers) quick test:",
                "  # Requires diffusers@main for Flux2KleinPipeline:",
                "  #   pip install -U \"abstractvision[huggingface-dev]\"",
                "  #   pip install -U \"git+https://github.com/huggingface/diffusers@main\"",
                "  abstractvision repl",
                "  /backend diffusers black-forest-labs/FLUX.2-klein-4B mps float16",
                '  /t2i "A cat holding a sign that says hello world" --steps 4 --guidance-scale 1.0 --width 1024 --height 1024 --open',
            ]
        )

    if any(n in selected for n in ("qwen_image_diffusers", "qwen_image_2512_diffusers")):
        lines.extend(
            [
                "",
                "Qwen Image (Diffusers) quick test (macOS MPS):",
                "  abstractvision repl",
                "  /backend diffusers Qwen/Qwen-Image-2512 mps float16",
                '  /t2i "a poster with the word ABSTRACT rendered perfectly in bold typography" --width 512 --height 512 --steps 10 --guidance-scale 2.5 --open',
            ]
        )

    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    available = _iter_sets()

    p = argparse.ArgumentParser(description="Download heavyweight model sets for AbstractVision testing.")
    p.add_argument("--out-dir", default=str(_default_out_dir()), help="Where to store downloaded weights.")
    p.add_argument("--token", default=None, help="HF token (optional). Prefer setting HF_TOKEN env var instead.")
    p.add_argument("--set", action="append", default=[], help="Model set name to download (repeatable).")
    p.add_argument("--all", action="store_true", help="Download all known model sets.")
    p.add_argument("--list", action="store_true", help="List available model sets and exit.")
    p.add_argument("--plan", action="store_true", help="Print the download plan and exit (no downloads).")
    args = p.parse_args(list(argv) if argv is not None else None)

    if bool(args.list):
        for s in available:
            print(f"{s.name}\n  {s.description}\n")
        return 0

    out_dir = Path(str(args.out_dir)).expanduser()
    selected = _resolve_selected(available, list(args.set or []), all_sets=bool(args.all))

    plan = _plan_lines(selected, out_dir)
    if bool(args.plan):
        print("\n".join(plan))
        return 0

    token = str(args.token).strip() if args.token else _token_from_env()

    # Execute downloads.
    print("\n".join(plan))
    for s in selected:
        set_dir = out_dir / s.name
        set_dir.mkdir(parents=True, exist_ok=True)
        for f in s.files:
            _download_file(f, out_dir=set_dir, token=token)
        for snap in s.snapshots:
            _download_snapshot(snap, out_dir=set_dir, token=token)

    for line in _post_instructions(selected, out_dir):
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
