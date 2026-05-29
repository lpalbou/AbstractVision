from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .artifacts import LocalAssetStore, is_artifact_ref
from .backends import (
    HuggingFaceDiffusersBackendConfig,
    HuggingFaceDiffusersVisionBackend,
    MFluxBackendConfig,
    MFluxVisionBackend,
    OpenAICompatibleBackendConfig,
    OpenAICompatibleVisionBackend,
    StableDiffusionCppBackendConfig,
    StableDiffusionCppVisionBackend,
)
from .model_capabilities import VisionModelCapabilitiesRegistry
from .model_downloads import (
    catalog_target_scope,
    default_model_target,
    download_hf_repo_snapshot,
    download_model_preset,
    find_model_preset,
    format_model_preset_rows,
    HuggingFaceAccessError,
    local_model_profile,
    looks_like_hf_repo_id,
    MacOSGGUFUnsupportedError,
    model_presets,
    normalize_model_engine,
    normalize_model_target,
    resolve_hf_token,
    resolve_model_target_and_engine,
    resolve_sdcpp_model_selection,
)
from .model_cache import default_hf_cache_root, default_legacy_model_root, ensure_hf_repo_snapshot
from .types import ImageEditRequest, ImageGenerationRequest, VideoProgressEvent
from .vision_manager import VisionManager

DEFAULT_REPL_BACKEND = ""
DEFAULT_DIFFUSERS_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_DIFFUSERS_DEVICE = "auto"
DEFAULT_T2I_WIDTH = 512
DEFAULT_T2I_HEIGHT = 512
DEFAULT_T2I_STEPS = 10
DEFAULT_I2I_STEPS = 15


class _CliVideoProgress:
    def __init__(self, *, enabled: bool = True, stream: Any = None) -> None:
        self.enabled = bool(enabled)
        self.stream = stream if stream is not None else sys.stderr
        self._wrote = False

    def __call__(self, event: VideoProgressEvent) -> None:
        if not self.enabled:
            return
        total_frames = event.total_frames
        if total_frames is not None and total_frames > 0:
            frame_part = f"{event.frame}/{total_frames} frames"
        else:
            frame_part = f"{event.frame} frames"
        if event.progress is not None:
            progress_part = f" ({max(0.0, min(1.0, float(event.progress))) * 100:5.1f}%)"
        else:
            progress_part = ""
        step_part = ""
        if event.step is not None:
            if event.total_steps is not None and event.total_steps > 0:
                step_part = f" step {event.step}/{event.total_steps}"
            else:
                step_part = f" step {event.step}"
        message = f"Generating video: {frame_part}{progress_part} {event.phase}{step_part}"
        print("\r" + message, end="", file=self.stream, flush=True)
        self._wrote = True
        if str(event.phase or "").lower() == "complete":
            self.close()

    def close(self) -> None:
        if self.enabled and self._wrote:
            print(file=self.stream, flush=True)
            self._wrote = False


def _generic_mlx_backend_error() -> str:
    return (
        "AbstractVision does not have a generic MLX image/video backend yet. "
        "Use `--target mlx` to browse MLX artifacts and `--provider mlx-gen` "
        "(or `mlx-gen/<exact-model-id>`) for MLX-Gen-compatible MLX image/video models."
    )


def _normalize_cli_provider(value: Any) -> str:
    provider = str(value or "").strip().lower().replace("_", "-")
    if provider in {"mflux", "m-flux", "mlxgen", "mlx-gen"}:
        return "mlx-gen"
    return provider


def _normalize_catalog_task_filter(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    return {
        "t2i": "text_to_image",
        "i2i": "image_to_image",
        "image_edit": "image_to_image",
        "t2v": "text_to_video",
        "i2v": "image_to_video",
        "video": "text_to_video",
        "video_generation": "text_to_video",
    }.get(raw, raw)


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _env_bool(key: str, default: bool = False) -> bool:
    v = _env(key)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _runtime_supported_tasks_for_catalog_preset(
    preset: Any,
    *,
    model_id: str,
    registry_tasks: Sequence[str],
) -> List[str]:
    tasks = [str(task_name) for task_name in registry_tasks]
    engine = normalize_model_engine(getattr(preset, "engine", None))
    try:
        if engine == "diffusers":
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(
                    model_id=str(model_id),
                    device="cpu",
                    allow_download=False,
                )
            )
        elif engine in {"mflux", "mlx-gen"}:
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model=str(getattr(preset, "key", model_id)))
            )
        elif engine == "stable-diffusion.cpp":
            backend = StableDiffusionCppVisionBackend(
                config=StableDiffusionCppBackendConfig(
                    sd_cli_path="sd-cli",
                    model=str(getattr(preset, "key", model_id)),
                    capabilities_model_id=str(model_id),
                )
            )
        else:
            return tasks
        allowed = {str(task_name) for task_name in backend.get_capabilities().supported_tasks or []}
        return [task_name for task_name in tasks if task_name in allowed]
    except Exception:
        return tasks


def _default_repl_backend() -> str:
    explicit = _env("ABSTRACTVISION_PROVIDER") or _env("ABSTRACTVISION_BACKEND")
    if explicit:
        return str(explicit)
    if _env("OPENAI_BASE_URL"):
        return "openai"
    return DEFAULT_REPL_BACKEND


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def _open_file(path: Path) -> None:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    if sys.platform == "darwin":
        subprocess.run(["open", str(p)], check=False)
        return
    if sys.platform.startswith("win"):
        # Best-effort. `start` is a shell built-in.
        subprocess.run(["cmd", "/c", "start", "", str(p)], check=False)
        return
    subprocess.run(["xdg-open", str(p)], check=False)


def _interactive_hf_token_retry(
    error: HuggingFaceAccessError,
    *,
    current_token: Optional[str] = None,
    json_mode: bool = False,
) -> Optional[str]:
    if json_mode or not sys.stdin.isatty():
        return None
    print(error, file=sys.stderr)
    if current_token:
        prompt = (
            "Paste a different Hugging Face token to retry now (input hidden, blank to abort): "
        )
    else:
        prompt = "Paste a Hugging Face token to retry now (input hidden, blank to abort): "
    try:
        entered = str(getpass.getpass(prompt)).strip()
    except (EOFError, KeyboardInterrupt):
        entered = ""
    return entered or None


def _build_openai_backend_from_args(args: argparse.Namespace) -> OpenAICompatibleVisionBackend:
    base_url = str(args.base_url or "").strip()
    if not base_url:
        raise SystemExit("Missing --base-url (or $OPENAI_BASE_URL).")
    cfg = OpenAICompatibleBackendConfig(
        base_url=base_url,
        api_key=str(args.api_key) if args.api_key else None,
        model_id=str(args.model_id) if args.model_id else None,
        timeout_s=float(args.timeout_s),
        models_path=str(getattr(args, "models_path", None) or "/models"),
        image_generations_path=str(args.images_generations_path),
        image_edits_path=str(args.images_edits_path),
        text_to_video_path=str(args.text_to_video_path) if args.text_to_video_path else None,
        image_to_video_path=str(args.image_to_video_path) if args.image_to_video_path else None,
        image_to_video_mode=str(args.image_to_video_mode),
    )
    return OpenAICompatibleVisionBackend(config=cfg)


def _first_nonempty(*values: Any) -> Optional[str]:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _split_provider_prefix(model: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    s = str(model or "").strip()
    if not s or "/" not in s:
        return None, s or None
    head, tail = s.split("/", 1)
    head = head.strip().lower().replace("_", "-")
    if head in {
        "openai",
        "openai-compatible",
        "openai_compatible",
        "proxy",
        "diffusers",
        "huggingface",
        "hf",
        "mlx",
        "mlx-gen",
        "mlxgen",
        "mflux",
        "m-flux",
        "sdcpp",
        "stable-diffusion.cpp",
        "stable-diffusion-cpp",
        "stable_diffusion_cpp",
    }:
        return _normalize_cli_provider(head), tail.strip() or None
    return None, s or None


def _resolve_cached_diffusers_model_id(model_id: str) -> str:
    """Resolve a curated diffusers preset to the HF cache when possible."""

    candidate = str(model_id or "").strip()
    if not candidate:
        return candidate
    try:
        preset = find_model_preset(
            candidate, target="diffusers", engine="diffusers", require_8bit=False
        )
    except Exception:
        try:
            preset = find_model_preset(
                candidate, target="gguf", engine="diffusers", require_8bit=True
            )
        except MacOSGGUFUnsupportedError:
            raise
        except Exception:
            return candidate

    legacy_root = default_legacy_model_root()
    legacy_dir = legacy_root / preset.local_dir_name
    try:
        ensure_hf_repo_snapshot(
            preset.repo_id,
            source_dir=legacy_dir,
            cache_dir=str(default_hf_cache_root()),
            cleanup_source=True,
        )
    except Exception:
        pass
    return str(preset.repo_id).strip() or candidate


def _build_manager_from_args(args: argparse.Namespace) -> VisionManager:
    store = LocalAssetStore(args.store_dir) if args.store_dir else LocalAssetStore()
    provider_kind = _normalize_cli_provider(
        getattr(args, "provider", None)
        or getattr(args, "backend", None)
        or _env("ABSTRACTVISION_PROVIDER")
        or _env("ABSTRACTVISION_BACKEND", "openai")
        or "openai"
    )

    model_value = _first_nonempty(
        getattr(args, "model", None),
        getattr(args, "model_id", None),
        getattr(args, "mflux_model", None),
        _env("ABSTRACTVISION_MODEL"),
        _env("ABSTRACTVISION_MODEL_ID"),
        _env("ABSTRACTVISION_MFLUX_MODEL"),
    )
    prefix_provider, unprefixed_model = _split_provider_prefix(model_value)
    if prefix_provider == "mlx":
        raise SystemExit(_generic_mlx_backend_error())
    if prefix_provider and not str(getattr(args, "provider", "") or "").strip():
        provider_kind = prefix_provider
    if unprefixed_model is not None:
        model_value = unprefixed_model

    if provider_kind == "mlx":
        raise SystemExit(_generic_mlx_backend_error())
    if provider_kind == "mlx-gen":
        backend = MFluxVisionBackend(
            config=MFluxBackendConfig(
                model=str(model_value) if model_value else None,
                base_model=str(getattr(args, "mflux_base_model", None) or "") or None,
                model_dir=str(getattr(args, "mflux_model_dir", None) or "") or None,
                allow_download=bool(getattr(args, "mflux_allow_download", False)),
            )
        )
    elif provider_kind in {"openai", "openai-compatible", "openai_compatible", "proxy"}:
        if model_value and not getattr(args, "model_id", None):
            setattr(args, "model_id", model_value)
        backend = _build_openai_backend_from_args(args)
    elif provider_kind in {"diffusers", "huggingface", "hf", "hf-diffusers"}:
        model_id = str(model_value or DEFAULT_DIFFUSERS_MODEL_ID).strip()
        if not model_id:
            model_id = DEFAULT_DIFFUSERS_MODEL_ID
        try:
            model_id = _resolve_cached_diffusers_model_id(model_id)
        except MacOSGGUFUnsupportedError as e:
            raise SystemExit(str(e)) from e
        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(
                model_id=model_id,
                device=str(
                    getattr(args, "diffusers_device", DEFAULT_DIFFUSERS_DEVICE)
                    or DEFAULT_DIFFUSERS_DEVICE
                ),
                torch_dtype=str(getattr(args, "diffusers_torch_dtype", None) or "") or None,
                allow_download=bool(getattr(args, "diffusers_allow_download", False)),
                auto_retry_fp32=bool(getattr(args, "diffusers_auto_retry_fp32", True)),
            )
        )
    elif provider_kind in {
        "sdcpp",
        "stable-diffusion.cpp",
        "stable-diffusion-cpp",
        "stable_diffusion_cpp",
    }:
        sdcpp_model = str(model_value or getattr(args, "sdcpp_model", None) or "") or None
        sdcpp_diffusion_model = str(getattr(args, "sdcpp_diffusion_model", None) or "") or None
        sdcpp_vae = str(getattr(args, "sdcpp_vae", None) or "") or None
        sdcpp_llm = str(getattr(args, "sdcpp_llm", None) or "") or None
        sdcpp_llm_vision = str(getattr(args, "sdcpp_llm_vision", None) or "") or None
        resolved_sdcpp = None
        if sdcpp_model and not any((sdcpp_diffusion_model, sdcpp_vae, sdcpp_llm, sdcpp_llm_vision)):
            candidate_path = Path(str(sdcpp_model)).expanduser()
            if not candidate_path.exists():
                try:
                    resolved_sdcpp = resolve_sdcpp_model_selection(
                        str(sdcpp_model), allow_download=False
                    )
                except MacOSGGUFUnsupportedError as e:
                    raise SystemExit(str(e)) from e
                except ValueError:
                    resolved_sdcpp = None
                except RuntimeError as e:
                    raise SystemExit(str(e)) from e
        backend = StableDiffusionCppVisionBackend(
            config=StableDiffusionCppBackendConfig(
                sd_cli_path=str(getattr(args, "sdcpp_bin", None) or "sd-cli"),
                model=(resolved_sdcpp.model if resolved_sdcpp is not None else sdcpp_model),
                capabilities_model_id=(
                    resolved_sdcpp.capabilities_model_id if resolved_sdcpp is not None else None
                ),
                diffusion_model=(
                    resolved_sdcpp.diffusion_model
                    if resolved_sdcpp is not None
                    else sdcpp_diffusion_model or None
                ),
                vae=resolved_sdcpp.vae if resolved_sdcpp is not None else sdcpp_vae or None,
                llm=resolved_sdcpp.llm if resolved_sdcpp is not None else sdcpp_llm or None,
                llm_vision=(
                    resolved_sdcpp.llm_vision
                    if resolved_sdcpp is not None
                    else sdcpp_llm_vision or None
                ),
                extra_args=tuple(
                    shlex.split(str(getattr(args, "sdcpp_extra_args", "") or ""))
                    if getattr(args, "sdcpp_extra_args", None)
                    else ()
                ),
                timeout_s=float(getattr(args, "timeout_s", 60.0 * 60.0)),
                cwd=None,
            )
        )
    else:
        raise SystemExit(
            "Unknown provider/backend. Supported one-shot providers: openai, openai-compatible, diffusers, sdcpp, mlx-gen."
        )
    reg = VisionModelCapabilitiesRegistry()

    cap_model_id = (
        str(args.capabilities_model_id) if getattr(args, "capabilities_model_id", None) else None
    )
    if cap_model_id and cap_model_id not in set(reg.list_models()):
        raise SystemExit(
            f"--capabilities-model-id '{cap_model_id}' is not present in the registry. "
            "Use `abstractvision models` to list known ids, or omit this flag to disable gating."
        )

    return VisionManager(
        backend=backend, store=store, model_id=cap_model_id, registry=reg if cap_model_id else None
    )


def _cmd_models(_: argparse.Namespace) -> int:
    reg = VisionModelCapabilitiesRegistry()
    for mid in reg.list_models():
        print(mid)
    return 0


def _cmd_tasks(_: argparse.Namespace) -> int:
    reg = VisionModelCapabilitiesRegistry()
    for t in reg.list_tasks():
        task_spec = reg.get_task(t)
        desc = task_spec.get("description")
        maturity = str(task_spec.get("maturity") or "").strip().lower()
        maturity_suffix = " (experimental)" if maturity == "experimental" else ""
        if isinstance(desc, str) and desc.strip():
            print(f"{t}{maturity_suffix}: {desc.strip()}")
        else:
            print(f"{t}{maturity_suffix}")
    return 0


def _cmd_show_model(args: argparse.Namespace) -> int:
    reg = VisionModelCapabilitiesRegistry()
    spec = reg.get(str(args.model_id))
    print(spec.model_id)
    print(f"provider: {spec.provider}")
    print(f"license: {spec.license}")
    if spec.notes:
        print(f"notes: {spec.notes}")
    if spec.downloads:
        print("downloads:")
        for dl in spec.downloads:
            bits = str(dl.bits) if dl.bits is not None else "n/a"
            line = f"  - {dl.key}  engine={dl.engine}  target={dl.target}  bits={bits}  repo={dl.repo_id}"
            if dl.source:
                line += f"  source={dl.source}"
            print(line)
            if dl.notes:
                print(f"      notes: {dl.notes}")
    print("tasks:")
    for task_name, ts in sorted(spec.tasks.items()):
        print(f"  - {task_name}")
        if ts.requires:
            print(f"      requires: {json.dumps(ts.requires, sort_keys=True)}")
        if ts.params:
            required = sorted(
                [
                    k
                    for k, v in ts.params.items()
                    if isinstance(v, dict) and v.get("required") is True
                ]
            )
            optional = sorted(
                [
                    k
                    for k, v in ts.params.items()
                    if isinstance(v, dict) and v.get("required") is False
                ]
            )
            if required:
                print(f"      required params: {', '.join(required)}")
            if optional:
                print(f"      optional params: {', '.join(optional)}")
    return 0


def _cmd_provider_models(args: argparse.Namespace) -> int:
    if bool(getattr(args, "openai", False)):
        base_url = (
            str(args.base_url).strip()
            if getattr(args, "base_url", None)
            else str(_env("OPENAI_BASE_URL", "https://api.openai.com/v1") or "").strip()
        )
    else:
        base_url = str(getattr(args, "base_url", None) or _env("OPENAI_BASE_URL") or "").strip()
    if not base_url:
        raise SystemExit("Missing --base-url (or use --openai for https://api.openai.com/v1).")

    api_key = getattr(args, "api_key", None)
    if api_key is None:
        api_key = _env("OPENAI_API_KEY")
    backend = OpenAICompatibleVisionBackend(
        config=OpenAICompatibleBackendConfig(
            base_url=base_url,
            api_key=str(api_key) if api_key else None,
            timeout_s=float(getattr(args, "timeout_s", 300.0)),
            models_path=str(getattr(args, "models_path", None) or "/models"),
        )
    )
    models = backend.list_provider_models(task=getattr(args, "task", None))
    if bool(getattr(args, "json", False)):
        _print_json([asdict(m) for m in models])
    else:
        for m in models:
            print(m.id)
    return 0


def _cmd_model_presets(args: argparse.Namespace) -> int:
    target = str(getattr(args, "target", "auto") or "auto")
    engine = str(getattr(args, "engine", "auto") or "auto")
    all_flag = bool(getattr(args, "all", False))
    include_non_8bit = all_flag
    include_all_targets = bool(getattr(args, "all_targets", False))
    try:
        resolved_target, resolved_engine = resolve_model_target_and_engine(
            target=target, engine=engine
        )
        selected_targets = catalog_target_scope(
            target=target, engine=engine, include_all_targets=include_all_targets
        )
        presets = model_presets(
            target=target,
            engine=engine,
            include_non_8bit=include_non_8bit,
            include_all_targets=include_all_targets,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    # Diffusers artifacts are typically full snapshots; treat them as opt-in when
    # listing, but don't force an 8-bit-only view that would otherwise be empty.
    if not include_non_8bit and resolved_target in {"diffusers", "hf-snapshot"}:
        include_non_8bit = True
        presets = model_presets(
            target=target,
            engine=engine,
            include_non_8bit=include_non_8bit,
            include_all_targets=include_all_targets,
        )
    if bool(getattr(args, "json", False)):
        _print_json([p.to_dict() for p in presets])
        return 0

    selected_target = "all" if include_all_targets else ",".join(selected_targets)
    selected_engine = resolved_engine or "any"
    if (
        not include_all_targets
        and str(target).strip().lower() in {"", "auto", "default"}
        and resolved_engine is None
    ):
        print(f"platform: {local_model_profile()}")
    print(f"target: {selected_target} (auto default: {default_model_target()})")
    print(f"provider/engine: {selected_engine}")
    if all_flag:
        print("policy: showing all presets (including explicit non-8-bit fallbacks)")
    elif resolved_target == "mlx" or resolved_engine == "mlx-gen":
        print(
            "policy: MLX-Gen models are exact published repo ids; q4 is recommended first, q8 is quality-focused; video fallbacks are listed when no q4/q8 artifact exists"
        )
    elif resolved_target == "diffusers":
        print("policy: Diffusers target includes full snapshots (16-bit/FP) by default")
    elif resolved_target == "hf-snapshot":
        print("policy: HF snapshot targets include full snapshots by default")
    else:
        print(
            "policy: quantized presets only by default; pass --all to show explicit non-quantized fallbacks"
        )
    print("tip: `abstractvision catalog` joins presets with the capability registry (tasks)")
    print(
        "tip: `download org/name` downloads arbitrary Hugging Face repos (not shown here) into the HF cache"
    )
    print()
    for line in format_model_preset_rows(presets):
        print(line)
    return 0


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    if not rows:
        return [" ".join(headers)]
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    out = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    for row in rows:
        out.append(fmt.format(*[str(x) for x in row]))
    return out


def _cmd_model_catalog(args: argparse.Namespace) -> int:
    """List capability models that have curated download presets."""

    task = _normalize_catalog_task_filter(getattr(args, "task", "") or "")
    target = str(getattr(args, "target", "auto") or "auto")
    engine = str(getattr(args, "engine", "auto") or "auto")
    include_non_8bit = bool(getattr(args, "all", False))
    include_all_targets = bool(getattr(args, "all_targets", False))
    try:
        resolved_target, resolved_engine = resolve_model_target_and_engine(
            target=target, engine=engine
        )
        selected_targets = catalog_target_scope(
            target=target, engine=engine, include_all_targets=include_all_targets
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    reg = VisionModelCapabilitiesRegistry()
    try:
        presets = model_presets(
            target=target,
            engine=engine,
            include_non_8bit=True,
            include_all_targets=include_all_targets,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if not include_non_8bit:
        # Recommended view: include quantized presets, plus a best-effort
        # fallback for any key/target/engine group that lacks one.
        grouped: Dict[tuple[str, str, str], List[Any]] = {}
        for preset in presets:
            grouped.setdefault((preset.key, preset.target, preset.engine), []).append(preset)
        filtered = []
        for group_presets in grouped.values():
            quantized = [
                p
                for p in group_presets
                if p.quantization_bits == 8
                or (
                    p.target == "mlx"
                    and p.engine in {"mflux", "mlx-gen"}
                    and p.quantization_bits in {2, 4, 8}
                )
            ]
            if quantized:
                filtered.extend(sorted(quantized, key=lambda p: (p.source_priority, p.repo_id)))
            else:
                filtered.append(
                    sorted(group_presets, key=lambda p: (p.source_priority, p.repo_id))[0]
                )
        presets = sorted(filtered, key=lambda p: (p.key, p.source_priority, p.repo_id))

    rows: List[Sequence[Any]] = []
    for preset in presets:
        registry_model_id = str(preset.upstream_repo_id or preset.repo_id)
        model_id = (
            str(preset.repo_id)
            if preset.target == "mlx"
            and preset.engine == "mlx-gen"
            and preset.source == "abstractframework-mlx-gen"
            else registry_model_id
        )
        try:
            spec = reg.get(registry_model_id)
            supported_tasks = _runtime_supported_tasks_for_catalog_preset(
                preset,
                model_id=registry_model_id,
                registry_tasks=sorted(spec.tasks.keys()),
            )
        except Exception:
            # #FALLBACK: allow listing presets even if the capability registry is missing an entry.
            supported_tasks = []
        if task and task not in supported_tasks:
            continue
        if not supported_tasks:
            continue
        tasks = ",".join(supported_tasks)
        rows.append(
            (
                model_id,
                preset.target,
                preset.engine,
                str(preset.quantization_bits) if preset.quantization_bits is not None else "n/a",
                preset.repo_id,
                preset.source,
                tasks,
            )
        )

    if bool(getattr(args, "json", False)):
        out: List[Dict[str, Any]] = []
        json_ids = {
            (
                str(p.repo_id)
                if p.target == "mlx"
                and p.engine == "mlx-gen"
                and p.source == "abstractframework-mlx-gen"
                else str(p.upstream_repo_id or p.repo_id)
            )
            for p in presets
        }
        for model_id in sorted(json_ids):
            matching_presets = [
                p
                for p in presets
                if (
                    str(p.repo_id)
                    if p.target == "mlx"
                    and p.engine == "mlx-gen"
                    and p.source == "abstractframework-mlx-gen"
                    else str(p.upstream_repo_id or p.repo_id)
                )
                == model_id
            ]
            registry_model_id = (
                str((matching_presets[0].upstream_repo_id or matching_presets[0].repo_id))
                if matching_presets
                else model_id
            )
            spec = reg.get(registry_model_id) if registry_model_id in reg.list_models() else None
            matching = [
                {
                    **p.to_dict(),
                    "tasks": _runtime_supported_tasks_for_catalog_preset(
                        p,
                        model_id=model_id,
                        registry_tasks=sorted(spec.tasks.keys()) if spec else [],
                    ),
                }
                for p in matching_presets
            ]
            if not matching:
                continue
            runtime_tasks: List[str] = []
            for item in matching:
                runtime_tasks.extend(str(task_name) for task_name in item.get("tasks") or [])
            runtime_tasks = sorted(set(runtime_tasks))
            if task and task not in runtime_tasks:
                continue
            if not runtime_tasks:
                continue
            out.append(
                {
                    "model_id": model_id,
                    "provider": spec.provider if spec else "unknown",
                    "license": spec.license if spec else "unknown",
                    "notes": spec.notes if spec else "",
                    "tasks": runtime_tasks,
                    "task_specs": (
                        {
                            task_name: {
                                "inputs": list(task_spec.inputs),
                                "outputs": list(task_spec.outputs),
                                "params": dict(task_spec.params),
                                "requires": (
                                    dict(task_spec.requires)
                                    if isinstance(task_spec.requires, dict)
                                    else None
                                ),
                            }
                            for task_name, task_spec in sorted(spec.tasks.items())
                            if task_name in runtime_tasks
                        }
                        if spec
                        else {}
                    ),
                    "downloads": matching,
                }
            )
        _print_json(out)
        return 0

    selected_target = "all" if include_all_targets else ",".join(selected_targets)
    selected_engine = resolved_engine or "any"
    print(f"task: {task or 'any'}")
    if (
        not include_all_targets
        and str(target).strip().lower() in {"", "auto", "default"}
        and resolved_engine is None
    ):
        print(f"platform: {local_model_profile()}")
    print(f"target: {selected_target} (auto default: {default_model_target()})")
    print(f"provider/engine: {selected_engine}")
    print(
        "policy: lists exact published model ids; MLX-Gen q4/q8 and vetted pre-packed low-bit artifacts are separate models; video fallbacks appear when no quantized artifact exists (pass --all for full list)"
    )
    print("tip: `abstractvision model-presets --all-targets --all` shows the raw preset table")
    print()
    for line in _format_table(
        ("model_id", "target", "engine", "bits", "repo", "source", "tasks"),
        rows,
    ):
        print(line)
    return 0


def _cmd_download_model(args: argparse.Namespace) -> int:
    target = str(getattr(args, "target", "auto") or "auto")
    engine = str(getattr(args, "engine", "auto") or "auto")
    raw_target = str(target or "auto").strip().lower()
    raw_engine = str(engine or "auto").strip().lower().replace("_", "-")
    raw_names = getattr(args, "names", None)
    if raw_names is None:
        raw_names = [getattr(args, "name", "")]
    names = [str(n).strip() for n in raw_names if str(n or "").strip()]

    def _maybe_engine_prefix(value: str) -> Optional[str]:
        candidate = normalize_model_engine(value)
        if candidate in {"mflux", "mlx-gen", "mlx", "diffusers", "stable-diffusion.cpp"}:
            return candidate
        return None

    try:
        normalized_engine = normalize_model_engine(engine)
        if len(names) >= 2:
            prefix_engine = _maybe_engine_prefix(names[0])
            if prefix_engine is not None and (
                normalized_engine is None or normalized_engine == prefix_engine
            ):
                engine = prefix_engine
                normalized_engine = prefix_engine
                names = names[1:]
        if len(names) == 1:
            solo_engine = _maybe_engine_prefix(names[0])
            if solo_engine is not None and (
                normalized_engine is None or normalized_engine == solo_engine
            ):
                raise SystemExit(
                    f"Missing model name after engine prefix {names[0]!r}. "
                    f"List presets with: abstractvision model-presets --provider {names[0]}"
                )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if not names:
        raise SystemExit(
            "Missing model name. Use `abstractvision model-presets` to list curated presets."
        )

    try:
        selected_target, _selected_engine = resolve_model_target_and_engine(
            target=target, engine=engine
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    model_dir = Path(str(args.model_dir)).expanduser() if getattr(args, "model_dir", None) else None
    results: List[Dict[str, Any]] = []
    json_mode = bool(getattr(args, "json", False))
    cli_token = resolve_hf_token(str(getattr(args, "token", None) or "") or None)
    allow_non_8bit = bool(getattr(args, "allow_non_8bit", False))
    require_8bit = not allow_non_8bit
    if raw_target in {"diffusers", "hf-snapshot"} or raw_engine in {"diffusers", "transformers"}:
        # Diffusers targets are full pipeline snapshots; do not force an 8-bit-only policy.
        require_8bit = False

    for idx, name in enumerate(names):
        if not json_mode and len(names) > 1:
            if idx:
                print()
            print(f"name: {name}")
        try:
            preset = find_model_preset(
                name,
                target=target,
                engine=engine,
                require_8bit=require_8bit,
            )
        except ValueError as e:
            mlx_gen_non_quantized_ok = raw_engine in {
                "mflux",
                "m-flux",
                "mlx-gen",
                "mlxgen",
                "mlx_gen",
            } or raw_target in {"mlx", "apple", "mac", "macos", "osx", "metal"}
            if require_8bit and mlx_gen_non_quantized_ok:
                try:
                    preset = find_model_preset(
                        name,
                        target=target,
                        engine=engine,
                        require_8bit=False,
                    )
                except ValueError:
                    preset = None
                if preset is not None:
                    # Exact non-quantized MLX-Gen/runtime snapshots such as Wan
                    # and FIBO are valid curated models even though no q4/q8
                    # artifact exists for them.
                    pass
                else:
                    preset = None
            else:
                preset = None
            if preset is not None:
                pass
            elif looks_like_hf_repo_id(name):
                if not json_mode:
                    print(f"selected_repo_id: {name}")
                    print("download: huggingface_hub snapshot (HF cache)")
                    print(
                        "#FALLBACK: repo id is not a curated preset; downloading full snapshot to the HF cache."
                    )
                try:
                    path = download_hf_repo_snapshot(
                        name,
                        token=cli_token,
                        cache_dir=str(default_hf_cache_root()),
                        max_workers=int(getattr(args, "max_workers", 4) or 4),
                    )
                except HuggingFaceAccessError as e2:
                    retry_token = _interactive_hf_token_retry(
                        e2, current_token=cli_token, json_mode=json_mode
                    )
                    if retry_token is None:
                        raise SystemExit(str(e2)) from e2
                    try:
                        path = download_hf_repo_snapshot(
                            name,
                            token=retry_token,
                            cache_dir=str(default_hf_cache_root()),
                            max_workers=int(getattr(args, "max_workers", 4) or 4),
                        )
                        cli_token = retry_token
                    except (HuggingFaceAccessError, RuntimeError) as e3:
                        raise SystemExit(str(e3)) from e3
                except RuntimeError as e2:
                    raise SystemExit(str(e2)) from e2

                item = {
                    "repo_id": name,
                    "snapshot_dir": str(path),
                    "source": "huggingface_hub_cache",
                }
                results.append(item)
                if not json_mode:
                    print(f"snapshot_dir: {path}")
                continue
            else:
                raise SystemExit(str(e)) from e

        if not json_mode:
            print(f"selected: {preset.repo_id}")
            print(f"target: {selected_target}")
            print(
                f"artifact: {preset.target}; bits: {preset.quantization_bits or 'n/a'}; provider/engine: {preset.engine}"
            )
            if preset.upstream_repo_id and preset.source != "official":
                print(
                    f"#FALLBACK: upstream source is {preset.upstream_repo_id}; using {preset.source} artifact for {selected_target}."
                )
            if preset.notes:
                note = str(preset.notes)
                if (
                    preset.target == "diffusers"
                    and preset.engine == "diffusers"
                    and "full Diffusers snapshot (not 8-bit)" in note
                ):
                    note = "Official curated Diffusers snapshot (16-bit)."
                print(note)

        if preset.engine == "stable-diffusion.cpp":
            # Ensure the runtime exists (auto-installs `sd-cli` on Apple Silicon by default).
            try:
                from abstractvision.backends.stable_diffusion_cpp import _require_sd_cli  # type: ignore

                _require_sd_cli(os.environ.get("ABSTRACTVISION_SDCPP_BIN", "sd-cli") or "sd-cli")
            except Exception as e:
                raise SystemExit(str(e)) from e

        try:
            path = download_model_preset(
                preset,
                model_dir=model_dir,
                token=cli_token,
                max_workers=int(getattr(args, "max_workers", 4) or 4),
            )
        except HuggingFaceAccessError as e:
            retry_token = _interactive_hf_token_retry(
                e, current_token=cli_token, json_mode=json_mode
            )
            if retry_token is None:
                raise SystemExit(str(e)) from e
            try:
                path = download_model_preset(
                    preset,
                    model_dir=model_dir,
                    token=retry_token,
                    max_workers=int(getattr(args, "max_workers", 4) or 4),
                )
                cli_token = retry_token
            except (HuggingFaceAccessError, RuntimeError) as e2:
                raise SystemExit(str(e2)) from e2
        except RuntimeError as e:
            raise SystemExit(str(e)) from e

        item = preset.to_dict()
        item["snapshot_dir"] = str(path)
        if preset.engine == "stable-diffusion.cpp":
            resolved = resolve_sdcpp_model_selection(
                preset.key,
                allow_download=True,
                token=cli_token,
                max_workers=int(getattr(args, "max_workers", 4) or 4),
            )
            item.update(
                {
                    "resolved_model": resolved.model,
                    "resolved_diffusion_model": resolved.diffusion_model,
                    "resolved_vae": resolved.vae,
                    "resolved_llm": resolved.llm,
                    "resolved_llm_vision": resolved.llm_vision,
                }
            )
        results.append(item)
        if not json_mode:
            print(f"snapshot_dir: {path}")
            if preset.engine == "stable-diffusion.cpp":
                if item.get("resolved_model"):
                    print(f"resolved_model: {item['resolved_model']}")
                if item.get("resolved_diffusion_model"):
                    print(f"resolved_diffusion_model: {item['resolved_diffusion_model']}")
                if item.get("resolved_vae"):
                    print(f"resolved_vae: {item['resolved_vae']}")
                if item.get("resolved_llm"):
                    print(f"resolved_llm: {item['resolved_llm']}")
                if item.get("resolved_llm_vision"):
                    print(f"resolved_llm_vision: {item['resolved_llm_vision']}")

    if json_mode:
        _print_json(results[0] if len(results) == 1 else results)
    return 0


def _cmd_t2i(args: argparse.Namespace) -> int:
    vm = _build_manager_from_args(args)
    request = _resolve_t2i_request(
        vm,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    out = vm.generate_image(
        request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        steps=request.steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
        extra=dict(request.extra or {}),
    )
    _print_json(out)
    if isinstance(vm.store, LocalAssetStore) and isinstance(out, dict) and is_artifact_ref(out):
        p = vm.store.get_content_path(out["$artifact"])
        if p is not None:
            print(str(p))
            if args.open:
                _open_file(p)
    return 0


def _resolve_t2i_request(
    vm: Any,
    *,
    prompt: str,
    negative_prompt: Optional[str],
    width: Optional[int],
    height: Optional[int],
    steps: Optional[int],
    guidance_scale: Optional[float],
    seed: Optional[int],
    extra: Optional[Dict[str, Any]] = None,
) -> ImageGenerationRequest:
    request = ImageGenerationRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        extra=dict(extra or {}),
    )
    backend = getattr(vm, "backend", None)
    normalize = getattr(backend, "normalize_image_generation_request", None)
    if callable(normalize):
        try:
            request = normalize(request)
        except Exception:
            pass
    if request.width is None:
        request = replace(request, width=DEFAULT_T2I_WIDTH)
    if request.height is None:
        request = replace(request, height=DEFAULT_T2I_HEIGHT)
    if request.steps is None:
        request = replace(request, steps=DEFAULT_T2I_STEPS)
    return request


def _resolve_i2i_steps(
    vm: Any,
    *,
    prompt: str,
    image: bytes,
    mask: Optional[bytes],
    negative_prompt: Optional[str],
    guidance_scale: Optional[float],
    seed: Optional[int],
    requested_steps: Optional[int],
) -> int:
    if requested_steps is not None:
        return int(requested_steps)
    backend = getattr(vm, "backend", None)
    normalize = getattr(backend, "normalize_image_edit_request", None)
    if callable(normalize):
        try:
            normalized = normalize(
                ImageEditRequest(
                    prompt=prompt,
                    image=image,
                    mask=mask,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    steps=None,
                    guidance_scale=guidance_scale,
                    extra={},
                )
            )
            if getattr(normalized, "steps", None) is not None:
                return int(normalized.steps)
        except Exception:
            pass
    return DEFAULT_I2I_STEPS


def _cmd_i2i(args: argparse.Namespace) -> int:
    vm = _build_manager_from_args(args)
    image_bytes = Path(args.image).expanduser().read_bytes()
    mask_bytes = Path(args.mask).expanduser().read_bytes() if args.mask else None
    steps = _resolve_i2i_steps(
        vm,
        prompt=args.prompt,
        image=image_bytes,
        mask=mask_bytes,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        requested_steps=args.steps,
    )
    extra: Dict[str, Any] = {}
    if getattr(args, "strength", None) is not None:
        extra["strength"] = float(args.strength)
    out = vm.edit_image(
        args.prompt,
        image=image_bytes,
        mask=mask_bytes,
        negative_prompt=args.negative_prompt,
        steps=steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        extra=extra,
    )
    _print_json(out)
    if isinstance(vm.store, LocalAssetStore) and isinstance(out, dict) and is_artifact_ref(out):
        p = vm.store.get_content_path(out["$artifact"])
        if p is not None:
            print(str(p))
            if args.open:
                _open_file(p)
    return 0


def _cmd_t2v(args: argparse.Namespace) -> int:
    vm = _build_manager_from_args(args)
    extra: Dict[str, Any] = {}
    if getattr(args, "max_sequence_length", None) is not None:
        extra["max_sequence_length"] = int(args.max_sequence_length)
    progress = _CliVideoProgress(enabled=bool(getattr(args, "progress", True)))
    if progress.enabled:
        extra["on_progress"] = progress
    try:
        out = vm.generate_video(
            args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            fps=args.fps,
            num_frames=args.num_frames,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            extra=extra or None,
        )
    finally:
        progress.close()
    _print_json(out)
    if isinstance(vm.store, LocalAssetStore) and isinstance(out, dict) and is_artifact_ref(out):
        p = vm.store.get_content_path(out["$artifact"])
        if p is not None:
            print(str(p))
            if args.open:
                _open_file(p)
    return 0


def _cmd_i2v(args: argparse.Namespace) -> int:
    vm = _build_manager_from_args(args)
    image_bytes = Path(args.image).expanduser().read_bytes()
    extra: Dict[str, Any] = {}
    if getattr(args, "max_sequence_length", None) is not None:
        extra["max_sequence_length"] = int(args.max_sequence_length)
    progress = _CliVideoProgress(enabled=bool(getattr(args, "progress", True)))
    if progress.enabled:
        extra["on_progress"] = progress
    try:
        out = vm.image_to_video(
            image_bytes,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            fps=args.fps,
            num_frames=args.num_frames,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            extra=extra or None,
        )
    finally:
        progress.close()
    _print_json(out)
    if isinstance(vm.store, LocalAssetStore) and isinstance(out, dict) and is_artifact_ref(out):
        p = vm.store.get_content_path(out["$artifact"])
        if p is not None:
            print(str(p))
            if args.open:
                _open_file(p)
    return 0


@dataclass
class _ReplState:
    backend_kind: str = field(default_factory=_default_repl_backend)
    base_url: Optional[str] = field(default_factory=lambda: _env("OPENAI_BASE_URL"))
    api_key: Optional[str] = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    model_id: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_MODEL_ID"))
    capabilities_model_id: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_CAPABILITIES_MODEL_ID")
    )
    store_dir: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_STORE_DIR"))
    timeout_s: float = field(
        default_factory=lambda: float(_env("ABSTRACTVISION_TIMEOUT_S", "3600") or "3600")
    )

    images_generations_path: str = field(
        default_factory=lambda: _env(
            "ABSTRACTVISION_IMAGES_GENERATIONS_PATH", "/images/generations"
        )
        or "/images/generations"
    )
    images_edits_path: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_IMAGES_EDITS_PATH", "/images/edits")
        or "/images/edits"
    )
    models_path: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_MODELS_PATH", "/models") or "/models"
    )
    text_to_video_path: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_TEXT_TO_VIDEO_PATH")
    )
    image_to_video_path: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_IMAGE_TO_VIDEO_PATH")
    )
    image_to_video_mode: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_IMAGE_TO_VIDEO_MODE", "multipart")
        or "multipart"
    )

    diffusers_device: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_DIFFUSERS_DEVICE", DEFAULT_DIFFUSERS_DEVICE)
        or DEFAULT_DIFFUSERS_DEVICE
    )
    diffusers_torch_dtype: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE")
    )
    diffusers_allow_download: bool = field(
        default_factory=lambda: _env_bool("ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD", False)
    )
    diffusers_auto_retry_fp32: bool = field(
        default_factory=lambda: _env_bool("ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32", True)
    )

    sdcpp_bin: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_SDCPP_BIN", "sd-cli") or "sd-cli"
    )
    sdcpp_model: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_SDCPP_MODEL"))
    sdcpp_diffusion_model: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_SDCPP_DIFFUSION_MODEL")
    )
    sdcpp_vae: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_SDCPP_VAE"))
    sdcpp_llm: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_SDCPP_LLM"))
    sdcpp_llm_vision: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_SDCPP_LLM_VISION")
    )
    sdcpp_extra_args: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_SDCPP_EXTRA_ARGS")
    )
    mflux_model: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_MFLUX_MODEL"))
    mflux_base_model: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_MFLUX_BASE_MODEL")
    )
    mflux_model_dir: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_MODEL_DIR"))
    mflux_allow_download: bool = field(
        default_factory=lambda: _env_bool("ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD", False)
    )

    defaults: Dict[str, Any] = None
    _cached_backend_key: Optional[Tuple[Any, ...]] = None
    _cached_backend: Any = None
    _cached_store_dir: Optional[str] = None
    _cached_store: Optional[LocalAssetStore] = None

    def __post_init__(self) -> None:
        if self.model_id is None and str(self.backend_kind or "").strip().lower() in {
            "diffusers",
            "huggingface",
            "hf",
            "hf-diffusers",
        }:
            self.model_id = DEFAULT_DIFFUSERS_MODEL_ID
        if self.mflux_model is None and _normalize_cli_provider(self.backend_kind) == "mlx-gen":
            self.mflux_model = self.model_id
        if self.defaults is None:
            self.defaults = {
                "t2i": {
                    "width": None,
                    "height": None,
                    "steps": None,
                    "guidance_scale": None,
                    "seed": None,
                    "negative_prompt": None,
                },
                "i2i": {
                    "steps": None,
                    "guidance_scale": None,
                    "seed": None,
                    "negative_prompt": None,
                },
                "t2v": {
                    "width": None,
                    "height": None,
                    "fps": None,
                    "num_frames": None,
                    "steps": None,
                    "guidance_scale": None,
                    "seed": None,
                    "negative_prompt": None,
                },
                "i2v": {
                    "width": None,
                    "height": None,
                    "fps": None,
                    "num_frames": None,
                    "steps": None,
                    "guidance_scale": None,
                    "seed": None,
                    "negative_prompt": None,
                },
            }


def _repl_help() -> str:
    return (
        "Commands:\n"
        "  /help                       Show this help\n"
        "  /exit                       Quit (aliases: /quit, /q)\n"
        "  /models                     List known capability model ids\n"
        "  /catalog [task]             List downloadable models from the registry + preset catalog\n"
        "  /provider-models            List models from the configured OpenAI-compatible provider\n"
        "  /tasks                      List known task keys\n"
        "  /show-model <id>            Show a model's tasks + params\n"
        "  /config                     Show current backend/store config\n"
        "  CLI: abstractvision model-presets lists curated cache-backed downloads\n"
        "\n"
        "Backends:\n"
        "  No backend is selected by default unless ABSTRACTVISION_BACKEND or OPENAI_BASE_URL is set.\n"
        "  /backend openai <base_url> [api_key] [model_id]\n"
        "  /backend diffusers <model_id_or_path> [device] [torch_dtype]\n"
        "      default model: runwayml/stable-diffusion-v1-5\n"
        "      cache-only by default; set ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1 to permit downloads\n"
        "  /backend sdcpp <model_key|model.gguf|model.safetensors> [sd_cli_path]\n"
        "  /backend sdcpp <diffusion_model.gguf> <vae.safetensors> <llm.gguf> [sd_cli_path]\n"
        "      use a cached model key for curated Qwen/FLUX bundles; component mode remains available for explicit wiring\n"
        "  /backend mlx-gen <preset_or_local_path> [base_model]\n"
        "      Apple Silicon MLX-Gen engine for exact MLX-Gen image/video model ids (requires abstractvision[mlx-gen])\n"
        "\n"
        "Defaults and output:\n"
        "  /cap-model <id|off>         Set capability-gating model id (from registry) or 'off'\n"
        "  /store <dir|default>        Set local store dir\n"
        "  /set <k> <v>                Set a default param (width, height, steps, seed, guidance_scale, negative_prompt)\n"
        "  /unset <k>                  Unset a default param\n"
        "  /defaults                   Show current defaults\n"
        "  /open <artifact_id>         Open a locally stored artifact (LocalAssetStore only)\n"
        "\n"
        "Generation:\n"
        "  /t2i <prompt...> [--width N --height N --steps N --seed N --guidance-scale F --negative-prompt ...] [--open]\n"
        "  /i2i --image path <prompt...> [--mask path --steps N --seed N --guidance-scale F --negative-prompt ...] [--open]\n"
        "  /t2v <prompt...> [--width N --height N --fps N --num-frames N --max-sequence-length N --steps N --seed N --guidance-scale F --negative-prompt ...] [--open] [--no-progress]\n"
        "  /i2v --image path [prompt...] [--width N --height N --fps N --num-frames N --max-sequence-length N --steps N --seed N --guidance-scale F --negative-prompt ...] [--open] [--no-progress]\n"
        "      extra flags are forwarded through request.extra\n"
        "\n"
        "Quick examples:\n"
        "  # Local model download policy: q4 MLX-Gen on macOS, q8 available for quality, cache-backed by default\n"
        "  abstractvision model-presets\n"
        "  abstractvision download AbstractFramework/flux.2-klein-4b-4bit --provider mlx-gen\n"
        "  /backend mlx-gen AbstractFramework/flux.2-klein-4b-4bit\n"
        '  /t2i "a product photo of a matte black espresso machine" --steps 4 --guidance-scale 1.0 --open\n'
        "  abstractvision download briaai/FIBO --provider mlx-gen\n"
        "  /backend mlx-gen briaai/FIBO\n"
        '  /t2i "a studio product photo of a white ceramic mug" --steps 50 --guidance-scale 4.0 --open\n'
        "  abstractvision download prism-ml/bonsai-image-ternary-4B-mlx-2bit --provider mlx-gen\n"
        "  /backend mlx-gen prism-ml/bonsai-image-ternary-4B-mlx-2bit\n"
        '  /t2i "a bonsai tree in a quiet ceramic studio" --steps 4 --guidance-scale 1.0 --open\n'
        "  abstractvision download Wan-AI/Wan2.2-TI2V-5B-Diffusers --provider mlx-gen\n"
        "  /backend mlx-gen Wan-AI/Wan2.2-TI2V-5B-Diffusers\n"
        '  /t2v "a red fox walking through a snowy forest" --num-frames 121 --steps 50 --fps 24 --max-sequence-length 256 --open\n'
        '  /i2v --image ./first-frame.png "slow camera push-in" --num-frames 121 --steps 50 --fps 24 --max-sequence-length 256 --open\n'
        "\n"
        "  # Local Diffusers path: Stable Diffusion 1.5 (requires abstractvision[diffusers])\n"
        "  /backend diffusers runwayml/stable-diffusion-v1-5 auto\n"
        '  /t2i "a watercolor painting of a lighthouse" --width 512 --height 512 --steps 10 --open\n'
        "\n"
        "  # Modern small FLUX path: FLUX.2-klein-4B (requires Diffusers main today)\n"
        "  /backend diffusers black-forest-labs/FLUX.2-klein-4B mps float16\n"
        '  /t2i "a product photo of a matte black espresso machine" --steps 4 --guidance-scale 1.0 --open\n'
        "\n"
        "  # stable-diffusion.cpp single-model path, preferably with an sd-cli binary for GPU acceleration\n"
        "  /backend sdcpp /path/to/sd-v1-5.gguf /path/to/sd-cli\n"
        '  /t2i "a watercolor painting of a lighthouse" --width 512 --height 512 --steps 10 --open\n'
        "\n"
        "  # Curated stable-diffusion.cpp bundle path for FLUX/Qwen-class models\n"
        "  abstractvision download flux2-klein-base-4b --provider sdcpp\n"
        "  /backend sdcpp flux2-klein-base-4b /path/to/sd-cli\n"
        '  /t2i "a product photo of a matte black espresso machine" --steps 4 --guidance-scale 1.0 --open\n'
        "\n"
        "Tip: typing plain text runs /t2i with that prompt.\n"
    )


def _parse_flags_and_rest(tokens: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    flags: Dict[str, Any] = {}
    rest: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            rest.append(t)
            i += 1
            continue
        key = t[2:].replace("-", "_")
        if i + 1 >= len(tokens):
            flags[key] = True
            i += 1
            continue
        val = tokens[i + 1]
        if val.startswith("--"):
            flags[key] = True
            i += 1
            continue
        flags[key] = val
        i += 2
    return flags, rest


def _parse_flag_args(tokens: List[str]) -> Dict[str, Any]:
    flags, _ = _parse_flags_and_rest(tokens)
    return flags


def _coerce_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float):
        return v
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _coerce_scalar(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (bool, int, float)):
        return v
    s = str(v).strip()
    if not s:
        return None
    low = s.lower()
    if low in {"1", "true", "yes", "on"}:
        return True
    if low in {"0", "false", "no", "off"}:
        return False
    try:
        return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        return s


def _video_progress_enabled_from_flags(flags: Dict[str, Any]) -> bool:
    if bool(_coerce_scalar(flags.get("no_progress"))):
        return False
    if "progress" in flags:
        value = _coerce_scalar(flags.get("progress"))
        if value is not None:
            return bool(value)
    return True


def _build_openai_backend_from_state(state: _ReplState) -> OpenAICompatibleVisionBackend:
    backend_kind = _normalize_cli_provider(state.backend_kind)
    if backend_kind in {"openai-compatible", "openai_compatible", "proxy"}:
        backend_kind = "openai"
    if backend_kind != "openai":
        raise ValueError("Provider model listing requires an OpenAI-compatible backend.")
    base_url = str(state.base_url or "").strip()
    if not base_url:
        raise ValueError(
            "Backend is not configured. Use: /backend openai <base_url> [api_key] [model_id]"
        )
    return OpenAICompatibleVisionBackend(
        config=OpenAICompatibleBackendConfig(
            base_url=base_url,
            api_key=str(state.api_key) if state.api_key else None,
            model_id=str(state.model_id) if state.model_id else None,
            timeout_s=float(state.timeout_s),
            models_path=str(state.models_path),
            image_generations_path=str(state.images_generations_path),
            image_edits_path=str(state.images_edits_path),
            text_to_video_path=str(state.text_to_video_path) if state.text_to_video_path else None,
            image_to_video_path=(
                str(state.image_to_video_path) if state.image_to_video_path else None
            ),
            image_to_video_mode=str(state.image_to_video_mode),
        )
    )


def _build_manager_from_state(state: _ReplState) -> VisionManager:
    if state._cached_store is not None and state._cached_store_dir == state.store_dir:
        store = state._cached_store
    else:
        store = LocalAssetStore(state.store_dir) if state.store_dir else LocalAssetStore()
        state._cached_store = store
        state._cached_store_dir = state.store_dir

    backend_kind = str(state.backend_kind or "").strip().lower()
    if backend_kind in {"openai-compatible", "openai_compatible", "proxy"}:
        backend_kind = "openai"
    elif backend_kind in {"huggingface", "hf", "hf-diffusers"}:
        backend_kind = "diffusers"
    elif backend_kind in {
        "sd-cpp",
        "stable-diffusion.cpp",
        "stable_diffusion_cpp",
        "stable-diffusion-cpp",
    }:
        backend_kind = "sdcpp"
    backend_key: Tuple[Any, ...]
    if backend_kind == "openai":
        base_url = str(state.base_url or "").strip()
        if not base_url:
            raise ValueError(
                "Backend is not configured. Use: /backend openai <base_url> [api_key] [model_id]"
            )
        backend_key = (
            "openai",
            base_url,
            state.api_key,
            state.model_id,
            state.timeout_s,
            state.models_path,
            state.images_generations_path,
            state.images_edits_path,
            state.text_to_video_path,
            state.image_to_video_path,
            state.image_to_video_mode,
        )
        if state._cached_backend is not None and state._cached_backend_key == backend_key:
            backend = state._cached_backend
        else:
            cfg = OpenAICompatibleBackendConfig(
                base_url=base_url,
                api_key=str(state.api_key) if state.api_key else None,
                model_id=str(state.model_id) if state.model_id else None,
                timeout_s=float(state.timeout_s),
                models_path=str(state.models_path),
                image_generations_path=str(state.images_generations_path),
                image_edits_path=str(state.images_edits_path),
                text_to_video_path=(
                    str(state.text_to_video_path) if state.text_to_video_path else None
                ),
                image_to_video_path=(
                    str(state.image_to_video_path) if state.image_to_video_path else None
                ),
                image_to_video_mode=str(state.image_to_video_mode),
            )
            backend = OpenAICompatibleVisionBackend(config=cfg)
            state._cached_backend = backend
            state._cached_backend_key = backend_key
    elif backend_kind == "diffusers":
        model_id = str(state.model_id or "").strip()
        if not model_id:
            raise ValueError(
                "Diffusers backend is not configured. Use: /backend diffusers <model_id_or_path> [device]"
            )
        model_id = _resolve_cached_diffusers_model_id(model_id)
        backend_key = (
            "diffusers",
            model_id,
            str(state.diffusers_device),
            str(state.diffusers_torch_dtype) if state.diffusers_torch_dtype else None,
            bool(state.diffusers_allow_download),
            bool(state.diffusers_auto_retry_fp32),
        )
        if state._cached_backend is not None and state._cached_backend_key == backend_key:
            backend = state._cached_backend
        else:
            cfg = HuggingFaceDiffusersBackendConfig(
                model_id=model_id,
                device=str(state.diffusers_device),
                torch_dtype=(
                    str(state.diffusers_torch_dtype) if state.diffusers_torch_dtype else None
                ),
                allow_download=bool(state.diffusers_allow_download),
                auto_retry_fp32=bool(state.diffusers_auto_retry_fp32),
            )
            backend = HuggingFaceDiffusersVisionBackend(config=cfg)
            state._cached_backend = backend
            state._cached_backend_key = backend_key
    elif backend_kind in {
        "sdcpp",
        "stable-diffusion.cpp",
        "stable_diffusion_cpp",
        "stable-diffusion-cpp",
    }:
        sdcpp_model = str(state.sdcpp_model) if state.sdcpp_model else None
        sdcpp_diffusion_model = (
            str(state.sdcpp_diffusion_model) if state.sdcpp_diffusion_model else None
        )
        sdcpp_vae = str(state.sdcpp_vae) if state.sdcpp_vae else None
        sdcpp_llm = str(state.sdcpp_llm) if state.sdcpp_llm else None
        sdcpp_llm_vision = str(state.sdcpp_llm_vision) if state.sdcpp_llm_vision else None
        resolved_sdcpp = None
        if sdcpp_model and not any((sdcpp_diffusion_model, sdcpp_vae, sdcpp_llm, sdcpp_llm_vision)):
            candidate_path = Path(str(sdcpp_model)).expanduser()
            if not candidate_path.exists():
                try:
                    resolved_sdcpp = resolve_sdcpp_model_selection(
                        str(sdcpp_model), allow_download=False
                    )
                except ValueError:
                    resolved_sdcpp = None
                except RuntimeError as e:
                    raise ValueError(str(e)) from e
        backend_key = (
            "sdcpp",
            str(state.sdcpp_bin),
            resolved_sdcpp.model if resolved_sdcpp is not None else sdcpp_model,
            resolved_sdcpp.diffusion_model if resolved_sdcpp is not None else sdcpp_diffusion_model,
            resolved_sdcpp.vae if resolved_sdcpp is not None else sdcpp_vae,
            resolved_sdcpp.llm if resolved_sdcpp is not None else sdcpp_llm,
            resolved_sdcpp.llm_vision if resolved_sdcpp is not None else sdcpp_llm_vision,
            str(state.sdcpp_extra_args) if state.sdcpp_extra_args else None,
        )
        if state._cached_backend is not None and state._cached_backend_key == backend_key:
            backend = state._cached_backend
        else:
            cfg = StableDiffusionCppBackendConfig(
                sd_cli_path=str(state.sdcpp_bin),
                model=resolved_sdcpp.model if resolved_sdcpp is not None else sdcpp_model,
                capabilities_model_id=(
                    resolved_sdcpp.capabilities_model_id if resolved_sdcpp is not None else None
                ),
                diffusion_model=(
                    resolved_sdcpp.diffusion_model
                    if resolved_sdcpp is not None
                    else sdcpp_diffusion_model
                ),
                vae=resolved_sdcpp.vae if resolved_sdcpp is not None else sdcpp_vae,
                llm=resolved_sdcpp.llm if resolved_sdcpp is not None else sdcpp_llm,
                llm_vision=(
                    resolved_sdcpp.llm_vision if resolved_sdcpp is not None else sdcpp_llm_vision
                ),
                extra_args=(
                    shlex.split(str(state.sdcpp_extra_args)) if state.sdcpp_extra_args else ()
                ),
            )
            backend = StableDiffusionCppVisionBackend(config=cfg)
            state._cached_backend = backend
            state._cached_backend_key = backend_key
    elif backend_kind == "mlx-gen":
        backend_key = (
            "mlx-gen",
            str(state.mflux_model) if state.mflux_model else None,
            str(state.mflux_base_model) if state.mflux_base_model else None,
            str(state.mflux_model_dir) if state.mflux_model_dir else None,
            bool(state.mflux_allow_download),
        )
        if state._cached_backend is not None and state._cached_backend_key == backend_key:
            backend = state._cached_backend
        else:
            cfg = MFluxBackendConfig(
                model=str(state.mflux_model) if state.mflux_model else None,
                base_model=str(state.mflux_base_model) if state.mflux_base_model else None,
                model_dir=str(state.mflux_model_dir) if state.mflux_model_dir else None,
                allow_download=bool(state.mflux_allow_download),
            )
            backend = MFluxVisionBackend(config=cfg)
            state._cached_backend = backend
            state._cached_backend_key = backend_key
    elif not backend_kind:
        raise ValueError(
            "Backend is not configured. Use /backend openai <base_url>, "
            "/backend mlx-gen <preset_or_path>, /backend diffusers <model_id_or_path>, or /backend sdcpp <model_key_or_path>."
        )
    else:
        raise ValueError(
            f"Unknown backend kind: {backend_kind!r} (expected 'openai', 'mlx-gen', 'diffusers', or 'sdcpp')"
        )

    reg = VisionModelCapabilitiesRegistry()
    cap_id = str(state.capabilities_model_id) if state.capabilities_model_id else None
    if cap_id and cap_id not in set(reg.list_models()):
        raise ValueError(f"capability model id not in registry: {cap_id!r}")
    return VisionManager(
        backend=backend, store=store, model_id=cap_id, registry=reg if cap_id else None
    )


def _cmd_repl(_: argparse.Namespace) -> int:
    reg = VisionModelCapabilitiesRegistry()
    state = _ReplState()

    print("AbstractVision CLI")
    print(f"- registry schema_version: {reg.schema_version()}")
    print("Type /help for commands.\n")

    while True:
        try:
            line = input("abstractvision> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if not line.startswith("/"):
            line = "/t2i " + line

        try:
            tokens = shlex.split(line)
        except ValueError as e:
            print(f"Parse error: {e}")
            continue
        if not tokens:
            continue
        cmd = tokens[0].lstrip("/").strip().lower()
        args = tokens[1:]

        try:
            if cmd in {"exit", "quit", "q"}:
                return 0
            if cmd == "help":
                print(_repl_help())
                continue
            if cmd == "models":
                for mid in reg.list_models():
                    print(mid)
                continue
            if cmd in {"catalog", "model-catalog"}:
                task = str(args[0]).strip() if args else ""
                _cmd_model_catalog(
                    argparse.Namespace(
                        task=task,
                        target="auto",
                        engine="auto",
                        all=False,
                        all_targets=True,
                        json=False,
                    )
                )
                continue
            if cmd in {"provider-models", "openai-models", "remote-models"}:
                flags = _parse_flag_args(args)
                models = _build_openai_backend_from_state(state).list_provider_models(
                    task=flags.get("task")
                )
                if bool(flags.get("json")):
                    _print_json([asdict(m) for m in models])
                else:
                    for m in models:
                        print(m.id)
                continue
            if cmd == "tasks":
                for t in reg.list_tasks():
                    desc = reg.get_task(t).get("description")
                    if isinstance(desc, str) and desc.strip():
                        print(f"{t}: {desc.strip()}")
                    else:
                        print(t)
                continue
            if cmd == "show-model":
                if not args:
                    print("Usage: /show-model <model_id>")
                    continue
                _cmd_show_model(argparse.Namespace(model_id=" ".join(args)))
                continue
            if cmd == "config":
                out: Dict[str, Any] = {
                    "backend_kind": state.backend_kind,
                    "base_url": state.base_url,
                    "model_id": state.model_id,
                    "capabilities_model_id": state.capabilities_model_id,
                    "store_dir": state.store_dir,
                    "timeout_s": state.timeout_s,
                    "models_path": state.models_path,
                    "images_generations_path": state.images_generations_path,
                    "images_edits_path": state.images_edits_path,
                    "text_to_video_path": state.text_to_video_path,
                    "image_to_video_path": state.image_to_video_path,
                    "image_to_video_mode": state.image_to_video_mode,
                    "diffusers_device": state.diffusers_device,
                    "diffusers_torch_dtype": state.diffusers_torch_dtype,
                    "diffusers_allow_download": state.diffusers_allow_download,
                    "diffusers_auto_retry_fp32": state.diffusers_auto_retry_fp32,
                    "sdcpp_bin": state.sdcpp_bin,
                    "sdcpp_model": state.sdcpp_model,
                    "sdcpp_diffusion_model": state.sdcpp_diffusion_model,
                    "sdcpp_vae": state.sdcpp_vae,
                    "sdcpp_llm": state.sdcpp_llm,
                    "sdcpp_llm_vision": state.sdcpp_llm_vision,
                    "sdcpp_extra_args": state.sdcpp_extra_args,
                    "mflux_model": state.mflux_model,
                    "mflux_base_model": state.mflux_base_model,
                    "mflux_model_dir": state.mflux_model_dir,
                    "mflux_allow_download": state.mflux_allow_download,
                    "defaults": state.defaults,
                }
                _print_json(out)
                continue
            if cmd == "backend":
                if not args:
                    print(
                        "Usage: /backend openai <base_url> [api_key] [model_id]  OR  "
                        "/backend diffusers <model_id_or_path> [device] [torch_dtype]  OR  "
                        "/backend sdcpp <model_key|model.gguf|model.safetensors> [sd_cli_path]  OR  "
                        "/backend sdcpp <diffusion_model.gguf> <vae.safetensors> <llm.gguf> [sd_cli_path]  OR  "
                        "/backend mlx-gen <preset_or_local_path> [base_model]"
                    )
                    continue
                kind = _normalize_cli_provider(args[0])
                if kind == "openai":
                    if len(args) < 2:
                        print("Usage: /backend openai <base_url> [api_key] [model_id]")
                        continue
                    state.backend_kind = "openai"
                    state.base_url = args[1]
                    state.api_key = args[2] if len(args) >= 3 else state.api_key
                    state.model_id = args[3] if len(args) >= 4 else None
                    print("ok")
                    continue
                if kind == "diffusers":
                    if len(args) < 2:
                        print("Usage: /backend diffusers <model_id_or_path> [device] [torch_dtype]")
                        continue
                    state.backend_kind = "diffusers"
                    state.model_id = args[1]
                    # Allow: /backend diffusers <model> [device] [torch_dtype]
                    # And also: /backend diffusers <model> <torch_dtype>  (keeps existing device)
                    dtype_tokens = {
                        "auto",
                        "float16",
                        "fp16",
                        "bfloat16",
                        "bf16",
                        "float32",
                        "fp32",
                    }
                    if len(args) >= 3 and str(args[2]).strip().lower() in dtype_tokens:
                        state.diffusers_torch_dtype = str(args[2]).strip()
                    else:
                        state.diffusers_device = (
                            args[2] if len(args) >= 3 else state.diffusers_device
                        )
                        state.diffusers_torch_dtype = (
                            str(args[3]).strip() if len(args) >= 4 else state.diffusers_torch_dtype
                        )
                    state.model_id = _resolve_cached_diffusers_model_id(str(state.model_id))
                    print("ok")
                    continue
                if kind == "sdcpp":
                    if len(args) < 2:
                        print(
                            "Usage: /backend sdcpp <model_key|model.gguf|model.safetensors> [sd_cli_path]  OR  "
                            "/backend sdcpp <diffusion_model.gguf> <vae.safetensors> <llm.gguf> [sd_cli_path]"
                        )
                        continue
                    state.backend_kind = "sdcpp"
                    state.model_id = None
                    if len(args) <= 3:
                        state.sdcpp_model = args[1]
                        state.sdcpp_diffusion_model = None
                        state.sdcpp_vae = None
                        state.sdcpp_llm = None
                        state.sdcpp_llm_vision = None
                        state.sdcpp_bin = args[2] if len(args) == 3 else state.sdcpp_bin
                    else:
                        state.sdcpp_model = None
                        state.sdcpp_diffusion_model = args[1]
                        state.sdcpp_vae = args[2]
                        state.sdcpp_llm = args[3]
                        state.sdcpp_bin = args[4] if len(args) >= 5 else state.sdcpp_bin
                    print("ok")
                    continue
                if kind == "mlx-gen":
                    if len(args) < 2:
                        print("Usage: /backend mlx-gen <preset_or_local_path> [base_model]")
                        continue
                    if str(args[1] or "").strip().lower().startswith("mlx/"):
                        print(_generic_mlx_backend_error())
                        continue
                    state.backend_kind = "mlx-gen"
                    state.mflux_model = args[1]
                    state.mflux_base_model = args[2] if len(args) >= 3 else None
                    state.model_id = args[1]
                    print("ok")
                    continue
                if kind == "mlx":
                    print(_generic_mlx_backend_error())
                    continue
                print("Unknown backend kind. Use: openai | mlx-gen | diffusers | sdcpp")
                continue
            if cmd == "cap-model":
                if not args:
                    print("Usage: /cap-model <model_id|off>")
                    continue
                if args[0].lower() == "off":
                    state.capabilities_model_id = None
                    print("ok (capability gating disabled)")
                    continue
                mid = " ".join(args).strip()
                if mid not in set(reg.list_models()):
                    print("Unknown model id (use /models).")
                    continue
                state.capabilities_model_id = mid
                print("ok")
                continue
            if cmd == "store":
                if not args:
                    print("Usage: /store <dir|default>")
                    continue
                if args[0].lower() == "default":
                    state.store_dir = None
                else:
                    state.store_dir = str(Path(args[0]).expanduser())
                print("ok")
                continue
            if cmd == "set":
                if len(args) < 2:
                    print("Usage: /set <key> <value>")
                    continue
                key = args[0].replace("-", "_")
                value = " ".join(args[1:])
                updated = False
                for group in state.defaults.keys():
                    if key in state.defaults.get(group, {}):
                        state.defaults[group][key] = value
                        updated = True
                if not updated:
                    for group in state.defaults.keys():
                        state.defaults.setdefault(group, {})[key] = value
                print("ok")
                continue
            if cmd == "unset":
                if not args:
                    print("Usage: /unset <key>")
                    continue
                key = args[0].replace("-", "_")
                for group in state.defaults.keys():
                    if key in state.defaults.get(group, {}):
                        state.defaults[group][key] = None
                print("ok")
                continue
            if cmd == "defaults":
                _print_json(state.defaults)
                continue
            if cmd == "open":
                if not args:
                    print("Usage: /open <artifact_id>")
                    continue
                store = LocalAssetStore(state.store_dir) if state.store_dir else LocalAssetStore()
                p = store.get_content_path(args[0])
                if p is None:
                    print("Not found in local store.")
                    continue
                print(str(p))
                _open_file(p)
                continue
            if cmd == "t2i":
                if not args:
                    print("Usage: /t2i <prompt...> [--width ...]")
                    continue
                flags, rest = _parse_flags_and_rest(args)
                prompt = " ".join(rest).strip()
                if not prompt:
                    print("Missing prompt.")
                    continue

                vm = _build_manager_from_state(state)
                d = dict(state.defaults.get("t2i", {}))
                d.update(flags)
                extra = {
                    k: _coerce_scalar(v)
                    for k, v in d.items()
                    if k
                    not in {
                        "width",
                        "height",
                        "steps",
                        "guidance_scale",
                        "seed",
                        "negative_prompt",
                        "open",
                    }
                    and v is not None
                }
                request = _resolve_t2i_request(
                    vm,
                    prompt=prompt,
                    negative_prompt=d.get("negative_prompt"),
                    width=_coerce_int(d.get("width")),
                    height=_coerce_int(d.get("height")),
                    steps=_coerce_int(d.get("steps")),
                    guidance_scale=_coerce_float(d.get("guidance_scale")),
                    seed=_coerce_int(d.get("seed")),
                    extra=extra,
                )
                out = vm.generate_image(
                    request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    steps=request.steps,
                    guidance_scale=request.guidance_scale,
                    seed=request.seed,
                    extra=extra,
                )
                _print_json(out)
                if (
                    isinstance(vm.store, LocalAssetStore)
                    and isinstance(out, dict)
                    and is_artifact_ref(out)
                ):
                    p = vm.store.get_content_path(out["$artifact"])
                    if p is not None:
                        print(str(p))
                        if bool(flags.get("open")):
                            _open_file(p)
                continue
            if cmd == "i2i":
                if not args:
                    print("Usage: /i2i --image path <prompt...> [--mask path ...]")
                    continue
                flags, rest = _parse_flags_and_rest(args)
                image_path = flags.get("image")
                if not image_path:
                    print("Missing --image path")
                    continue
                mask_path = flags.get("mask")
                prompt = " ".join(rest).strip()
                if not prompt:
                    print("Missing prompt.")
                    continue

                vm = _build_manager_from_state(state)
                d = dict(state.defaults.get("i2i", {}))
                d.update(flags)
                extra = {
                    k: _coerce_scalar(v)
                    for k, v in d.items()
                    if k
                    not in {
                        "image",
                        "mask",
                        "steps",
                        "guidance_scale",
                        "seed",
                        "negative_prompt",
                        "open",
                    }
                    and v is not None
                }
                img = Path(str(image_path)).expanduser().read_bytes()
                mask = Path(str(mask_path)).expanduser().read_bytes() if mask_path else None
                steps = _resolve_i2i_steps(
                    vm,
                    prompt=prompt,
                    image=img,
                    mask=mask,
                    negative_prompt=d.get("negative_prompt"),
                    guidance_scale=_coerce_float(d.get("guidance_scale")),
                    seed=_coerce_int(d.get("seed")),
                    requested_steps=_coerce_int(d.get("steps")),
                )
                out = vm.edit_image(
                    prompt,
                    image=img,
                    mask=mask,
                    negative_prompt=d.get("negative_prompt"),
                    steps=steps,
                    guidance_scale=_coerce_float(d.get("guidance_scale")),
                    seed=_coerce_int(d.get("seed")),
                    extra=extra,
                )
                _print_json(out)
                if (
                    isinstance(vm.store, LocalAssetStore)
                    and isinstance(out, dict)
                    and is_artifact_ref(out)
                ):
                    p = vm.store.get_content_path(out["$artifact"])
                    if p is not None:
                        print(str(p))
                        if bool(flags.get("open")):
                            _open_file(p)
                continue
            if cmd == "t2v":
                if not args:
                    print(
                        "Usage: /t2v <prompt...> [--width ... --height ... --fps ... --num-frames ... --max-sequence-length ...]"
                    )
                    continue
                flags, rest = _parse_flags_and_rest(args)
                prompt = " ".join(rest).strip()
                if not prompt:
                    print("Missing prompt.")
                    continue

                vm = _build_manager_from_state(state)
                d = dict(state.defaults.get("t2v", {}))
                d.update(flags)
                if d.get("num_frames") is None and d.get("frames") is not None:
                    d["num_frames"] = d.get("frames")
                extra = {
                    k: _coerce_scalar(v)
                    for k, v in d.items()
                    if k
                    not in {
                        "width",
                        "height",
                        "fps",
                        "frames",
                        "num_frames",
                        "steps",
                        "guidance_scale",
                        "seed",
                        "negative_prompt",
                        "open",
                        "progress",
                        "no_progress",
                    }
                    and v is not None
                }
                progress = _CliVideoProgress(enabled=_video_progress_enabled_from_flags(d))
                if progress.enabled:
                    extra["on_progress"] = progress
                try:
                    out = vm.generate_video(
                        prompt,
                        negative_prompt=d.get("negative_prompt"),
                        width=_coerce_int(d.get("width")),
                        height=_coerce_int(d.get("height")),
                        fps=_coerce_int(d.get("fps")),
                        num_frames=_coerce_int(d.get("num_frames")),
                        steps=_coerce_int(d.get("steps")),
                        guidance_scale=_coerce_float(d.get("guidance_scale")),
                        seed=_coerce_int(d.get("seed")),
                        extra=extra,
                    )
                finally:
                    progress.close()
                _print_json(out)
                if (
                    isinstance(vm.store, LocalAssetStore)
                    and isinstance(out, dict)
                    and is_artifact_ref(out)
                ):
                    p = vm.store.get_content_path(out["$artifact"])
                    if p is not None:
                        print(str(p))
                        if bool(flags.get("open")):
                            _open_file(p)
                continue
            if cmd == "i2v":
                if not args:
                    print(
                        "Usage: /i2v --image path [prompt...] [--width ... --height ... --fps ... --num-frames ... --max-sequence-length ...]"
                    )
                    continue
                flags, rest = _parse_flags_and_rest(args)
                image_path = flags.get("image")
                if not image_path:
                    print("Missing --image path")
                    continue
                prompt = " ".join(rest).strip() or None

                vm = _build_manager_from_state(state)
                d = dict(state.defaults.get("i2v", {}))
                d.update(flags)
                if d.get("num_frames") is None and d.get("frames") is not None:
                    d["num_frames"] = d.get("frames")
                extra = {
                    k: _coerce_scalar(v)
                    for k, v in d.items()
                    if k
                    not in {
                        "image",
                        "width",
                        "height",
                        "fps",
                        "frames",
                        "num_frames",
                        "steps",
                        "guidance_scale",
                        "seed",
                        "negative_prompt",
                        "open",
                        "progress",
                        "no_progress",
                    }
                    and v is not None
                }
                img = Path(str(image_path)).expanduser().read_bytes()
                progress = _CliVideoProgress(enabled=_video_progress_enabled_from_flags(d))
                if progress.enabled:
                    extra["on_progress"] = progress
                try:
                    out = vm.image_to_video(
                        img,
                        prompt=prompt,
                        negative_prompt=d.get("negative_prompt"),
                        width=_coerce_int(d.get("width")),
                        height=_coerce_int(d.get("height")),
                        fps=_coerce_int(d.get("fps")),
                        num_frames=_coerce_int(d.get("num_frames")),
                        steps=_coerce_int(d.get("steps")),
                        guidance_scale=_coerce_float(d.get("guidance_scale")),
                        seed=_coerce_int(d.get("seed")),
                        extra=extra,
                    )
                finally:
                    progress.close()
                _print_json(out)
                if (
                    isinstance(vm.store, LocalAssetStore)
                    and isinstance(out, dict)
                    and is_artifact_ref(out)
                ):
                    p = vm.store.get_content_path(out["$artifact"])
                    if p is not None:
                        print(str(p))
                        if bool(flags.get("open")):
                            _open_file(p)
                continue

            print("Unknown command. Type /help.")
        except Exception as e:
            print(f"Error: {e}")


def _cmd_playground(args: argparse.Namespace) -> int:
    from .playground_server import PlaygroundServerConfig, run_playground_server

    cfg = PlaygroundServerConfig(host=str(args.host), port=int(args.port))
    return run_playground_server(cfg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="abstractvision", description="AbstractVision CLI (capabilities + generation)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="List known model ids (from capability registry).").set_defaults(
        _fn=_cmd_models
    )
    pm = sub.add_parser(
        "provider-models",
        aliases=["openai-models", "remote-models"],
        help="List models from an OpenAI/OpenAI-compatible provider catalog.",
    )
    pm.add_argument(
        "--openai",
        action="store_true",
        help="Use the official OpenAI API base URL (or OPENAI_BASE_URL if set).",
    )
    pm.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (e.g. http://localhost:1234/v1).",
    )
    pm.add_argument("--api-key", default=None, help="API key (Bearer).")
    pm.add_argument(
        "--models-path",
        default=_env("ABSTRACTVISION_MODELS_PATH", "/models"),
        help="Path for the provider model catalog (default: /models).",
    )
    pm.add_argument(
        "--task",
        default=None,
        help="Best-effort task filter (e.g. text_to_image, image_to_image).",
    )
    pm.add_argument("--timeout-s", type=float, default=30.0, help="HTTP timeout seconds.")
    pm.add_argument(
        "--json", action="store_true", help="Print full provider model entries as JSON."
    )
    pm.set_defaults(_fn=_cmd_provider_models)
    sub.add_parser("tasks", help="List known task keys (from capability registry).").set_defaults(
        _fn=_cmd_tasks
    )

    presets = sub.add_parser(
        "model-presets",
        aliases=["vision-models"],
        help="List curated cache-backed vision model download presets (quantized by default where available).",
    )
    presets.add_argument(
        "--target",
        default="auto",
        choices=["auto", "mlx", "gguf", "fp8", "diffusers", "hf-snapshot", "macos", "gpu"],
        help=(
            "Runtime artifact target (default: auto; Apple Silicon prefers mlx, CUDA hosts prefer fp8, others prefer diffusers). "
            "If left as auto, an explicit --provider/--engine will infer the matching target."
        ),
    )
    presets.add_argument(
        "--provider",
        "--engine",
        "--backend",
        dest="engine",
        default="auto",
        choices=[
            "auto",
            "any",
            "mlx-gen",
            "mlxgen",
            "mflux",
            "m-flux",
            "mlx",
            "gguf",
            "sdcpp",
            "stable-diffusion.cpp",
            "diffusers",
            "transformers",
        ],
        help="Runtime provider/engine filter (default: any for the selected target). Use --provider mlx-gen for the Apple Silicon runtime; use --target mlx to browse MLX artifacts.",
    )
    presets.add_argument(
        "--all", action="store_true", help="Also show explicit non-quantized fallbacks."
    )
    presets.add_argument(
        "--all-targets", action="store_true", help="Show presets for every target."
    )
    presets.add_argument("--json", action="store_true", help="Print full preset metadata as JSON.")
    presets.set_defaults(_fn=_cmd_model_presets)

    catalog = sub.add_parser(
        "catalog",
        aliases=["model-catalog", "download-catalog"],
        help="List downloadable models (capability registry joined with curated presets).",
    )
    catalog.add_argument(
        "--task",
        default="",
        help="Optional task filter (e.g. text_to_image, image_to_image).",
    )
    catalog.add_argument(
        "--target",
        default="auto",
        choices=["auto", "mlx", "gguf", "fp8", "diffusers", "hf-snapshot", "macos", "gpu"],
        help=(
            "Runtime artifact target (default: auto; Apple Silicon prefers mlx, CUDA hosts prefer fp8, others prefer diffusers). "
            "If left as auto, an explicit --provider/--engine will infer the matching target."
        ),
    )
    catalog.add_argument(
        "--provider",
        "--engine",
        "--backend",
        dest="engine",
        default="auto",
        choices=[
            "auto",
            "any",
            "mlx-gen",
            "mlxgen",
            "mflux",
            "m-flux",
            "mlx",
            "gguf",
            "sdcpp",
            "stable-diffusion.cpp",
            "diffusers",
            "transformers",
        ],
        help="Runtime provider/engine filter (default: any for the selected target). Use --provider mlx-gen for the Apple Silicon runtime; use --target mlx to browse MLX artifacts.",
    )
    catalog.add_argument(
        "--all", action="store_true", help="Also include explicit non-quantized fallbacks."
    )
    catalog.add_argument(
        "--all-targets", action="store_true", help="Show downloadable presets for every target."
    )
    catalog.add_argument("--json", action="store_true", help="Print the catalog as JSON.")
    catalog.set_defaults(_fn=_cmd_model_catalog)

    dl = sub.add_parser(
        "download",
        aliases=["download-model", "download-vision-model"],
        help="Download a curated vision model preset (quantized by default where available) or a Hugging Face repo snapshot.",
    )
    dl.add_argument(
        "names",
        nargs="+",
        help=(
            "Exact model id or Hugging Face repo id (e.g. AbstractFramework/flux.2-klein-4b-4bit). "
            "You can pass multiple names to download them in sequence. "
            "Shorthand: `download mlx-gen AbstractFramework/flux.2-klein-4b-4bit` (provider/engine prefix)."
        ),
    )
    dl.add_argument(
        "--target",
        default="auto",
        choices=["auto", "mlx", "gguf", "fp8", "diffusers", "hf-snapshot", "macos", "gpu"],
        help=(
            "Runtime artifact target (default: auto; Apple Silicon prefers mlx, CUDA hosts prefer fp8, others prefer diffusers). "
            "If left as auto, an explicit --provider/--engine will infer the matching target."
        ),
    )
    dl.add_argument(
        "--provider",
        "--engine",
        "--backend",
        dest="engine",
        default="auto",
        choices=[
            "auto",
            "any",
            "mlx-gen",
            "mlxgen",
            "mflux",
            "m-flux",
            "mlx",
            "gguf",
            "sdcpp",
            "stable-diffusion.cpp",
            "diffusers",
            "transformers",
        ],
        help="Runtime provider/engine filter (default: any for the selected target). Use --provider mlx-gen for the Apple Silicon runtime; use --target mlx to browse MLX artifacts.",
    )
    dl.add_argument(
        "--model-dir",
        default=_env("ABSTRACTVISION_MODEL_DIR"),
        help="Legacy preset root to import from (downloads now land in the Hugging Face cache).",
    )
    dl.add_argument(
        "--token",
        default=_env("HUGGINGFACE_HUB_TOKEN") or _env("HF_TOKEN"),
        help="Hugging Face token, if needed.",
    )
    dl.add_argument(
        "--max-workers", type=int, default=4, help="Hugging Face download workers (default: 4)."
    )
    dl.add_argument(
        "--allow-non-8bit",
        action="store_true",
        help="Permit explicit fallback presets when no quantized artifact is curated.",
    )
    dl.add_argument(
        "--json", action="store_true", help="Print selected preset + snapshot path as JSON."
    )
    dl.set_defaults(_fn=_cmd_download_model)

    sm = sub.add_parser("show-model", help="Show a model's supported tasks and params.")
    sm.add_argument("model_id")
    sm.set_defaults(_fn=_cmd_show_model)

    repl = sub.add_parser(
        "cli",
        aliases=["repl"],
        help="Interactive CLI for testing capabilities, downloads, and generation.",
    )
    repl.set_defaults(_fn=_cmd_repl)

    playground = sub.add_parser(
        "playground",
        aliases=["serve"],
        help="Run the self-contained local web playground and API server.",
    )
    playground.add_argument(
        "--host", default="127.0.0.1", help="Host/interface to bind (default: 127.0.0.1)."
    )
    playground.add_argument("--port", type=int, default=8091, help="Port to bind (default: 8091).")
    playground.set_defaults(_fn=_cmd_playground)

    def _add_provider_flags(ap: argparse.ArgumentParser) -> None:
        ap.add_argument(
            "--provider",
            "--backend",
            dest="provider",
            default=None,
            help=(
                "Provider/backend: openai, openai-compatible, diffusers, sdcpp, or mlx-gen. "
                "Use --target mlx for MLX artifacts; generic provider 'mlx' is not supported."
            ),
        )
        ap.add_argument(
            "--model",
            "--model-id",
            dest="model",
            default=None,
            help="Provider model id, preset key, local path, or repo id (provider-specific).",
        )
        # Backward-compatible alias (kept visible so older docs still work).
        ap.add_argument(
            "--mflux-model",
            dest="model",
            default=None,
            help="Compatibility alias for --model when using --provider mlx-gen (preset, local path, or repo id).",
        )

        # OpenAI/OpenAI-compatible provider config.
        ap.add_argument(
            "--base-url",
            default=_env("OPENAI_BASE_URL"),
            help="OpenAI-compatible base URL (e.g. http://localhost:1234/v1).",
        )
        ap.add_argument("--api-key", default=_env("OPENAI_API_KEY"), help="API key (Bearer).")

        # Local Diffusers provider config.
        ap.add_argument(
            "--diffusers-device",
            default=_env("ABSTRACTVISION_DIFFUSERS_DEVICE", DEFAULT_DIFFUSERS_DEVICE),
            help="Diffusers device: cpu|cuda|mps|auto (default: auto).",
        )
        ap.add_argument(
            "--diffusers-torch-dtype",
            default=_env("ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE"),
            help="Diffusers torch dtype (e.g. float16).",
        )
        ap.add_argument(
            "--diffusers-allow-download",
            action="store_true",
            default=_env_bool("ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD", False),
            help="Allow Diffusers to download missing model files (default: cache-only).",
        )
        ap.add_argument(
            "--diffusers-auto-retry-fp32",
            dest="diffusers_auto_retry_fp32",
            action="store_true",
            default=_env_bool("ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32", True),
            help="Enable Diffusers fp32 retry fallback on dtype/device failures (default).",
        )
        ap.add_argument(
            "--diffusers-no-auto-retry-fp32",
            dest="diffusers_auto_retry_fp32",
            action="store_false",
            help="Disable Diffusers fp32 retry fallback on dtype/device failures.",
        )

        # stable-diffusion.cpp provider config.
        ap.add_argument(
            "--sdcpp-bin",
            default=_env("ABSTRACTVISION_SDCPP_BIN", "sd-cli"),
            help="Path to sd-cli (default: sd-cli).",
        )
        ap.add_argument(
            "--sdcpp-model",
            default=_env("ABSTRACTVISION_SDCPP_MODEL"),
            help="stable-diffusion.cpp model key or single-model path (overrides --model).",
        )
        ap.add_argument(
            "--sdcpp-diffusion-model",
            default=_env("ABSTRACTVISION_SDCPP_DIFFUSION_MODEL"),
            help="stable-diffusion.cpp diffusion_model path (GGUF).",
        )
        ap.add_argument(
            "--sdcpp-vae",
            default=_env("ABSTRACTVISION_SDCPP_VAE"),
            help="stable-diffusion.cpp VAE path (safetensors).",
        )
        ap.add_argument(
            "--sdcpp-llm",
            default=_env("ABSTRACTVISION_SDCPP_LLM"),
            help="stable-diffusion.cpp LLM path (safetensors or GGUF).",
        )
        ap.add_argument(
            "--sdcpp-llm-vision",
            default=_env("ABSTRACTVISION_SDCPP_LLM_VISION"),
            help="stable-diffusion.cpp vision LLM path (GGUF; optional, used by some Qwen variants).",
        )
        ap.add_argument(
            "--sdcpp-extra-args",
            default=_env("ABSTRACTVISION_SDCPP_EXTRA_ARGS"),
            help="Extra args forwarded to sd-cli / bindings (quoted string).",
        )

        # MLX-Gen provider config (env var names preserve MFLUX compatibility).
        ap.add_argument(
            "--mflux-base-model",
            default=_env("ABSTRACTVISION_MFLUX_BASE_MODEL"),
            help="Optional MLX-Gen base family for local paths or custom repos: flux2-klein-4b, flux2-klein-9b, flux2-klein-base-4b, flux2-klein-base-9b, bonsai-image-ternary, z-image, z-image-turbo, qwen-image/qwen-image-2512, qwen-image-edit-2511, ernie-image-turbo, fibo, fibo-lite, fibo-edit, fibo-edit-rmbg, or wan2.2-ti2v-5b.",
        )
        ap.add_argument(
            "--mflux-model-dir",
            "--model-dir",
            dest="mflux_model_dir",
            default=_env("ABSTRACTVISION_MODEL_DIR"),
            help="Legacy MLX/MFLUX preset root imported into the Hugging Face cache when older installs are migrated.",
        )
        ap.add_argument(
            "--mflux-allow-download",
            action="store_true",
            default=_env_bool("ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD", False),
            help="Allow MLX-Gen to resolve/download non-local model ids (repo ids).",
        )

        ap.add_argument(
            "--capabilities-model-id",
            default=_env("ABSTRACTVISION_CAPABILITIES_MODEL_ID"),
            help="Optional: enforce support using a registry model id.",
        )
        ap.add_argument(
            "--timeout-s",
            type=float,
            default=float(_env("ABSTRACTVISION_TIMEOUT_S", "3600") or "3600"),
            help="Timeout seconds for HTTP calls and local runtimes (default: 3600).",
        )
        ap.add_argument(
            "--store-dir",
            default=_env("ABSTRACTVISION_STORE_DIR"),
            help="Local asset store dir (default: ~/.abstractvision/assets).",
        )
        ap.add_argument(
            "--images-generations-path",
            default=_env("ABSTRACTVISION_IMAGES_GENERATIONS_PATH", "/images/generations"),
            help="Path for image generations.",
        )
        ap.add_argument(
            "--images-edits-path",
            default=_env("ABSTRACTVISION_IMAGES_EDITS_PATH", "/images/edits"),
            help="Path for image edits.",
        )
        ap.add_argument(
            "--text-to-video-path",
            default=_env("ABSTRACTVISION_TEXT_TO_VIDEO_PATH"),
            help="Optional path for text-to-video.",
        )
        ap.add_argument(
            "--image-to-video-path",
            default=_env("ABSTRACTVISION_IMAGE_TO_VIDEO_PATH"),
            help="Optional path for image-to-video.",
        )
        ap.add_argument(
            "--image-to-video-mode",
            default=_env("ABSTRACTVISION_IMAGE_TO_VIDEO_MODE", "multipart"),
            help="image_to_video mode: multipart|json_b64.",
        )

    t2i = sub.add_parser(
        "t2i", help="One-shot text-to-image (stores output and prints artifact ref + path)."
    )
    _add_provider_flags(t2i)
    t2i.add_argument("prompt")
    t2i.add_argument("--negative-prompt", default=None)
    t2i.add_argument("--width", type=int, default=None)
    t2i.add_argument("--height", type=int, default=None)
    t2i.add_argument("--steps", type=int, default=None)
    t2i.add_argument("--guidance-scale", type=float, default=None, dest="guidance_scale")
    t2i.add_argument("--seed", type=int, default=None)
    t2i.add_argument("--open", action="store_true", help="Open the output file (best-effort).")
    t2i.set_defaults(_fn=_cmd_t2i)

    i2i = sub.add_parser(
        "i2i", help="One-shot image-to-image edit (stores output and prints artifact ref + path)."
    )
    _add_provider_flags(i2i)
    i2i.add_argument("--image", required=True, help="Input image file path.")
    i2i.add_argument("--mask", default=None, help="Optional mask file path.")
    i2i.add_argument("prompt")
    i2i.add_argument("--negative-prompt", default=None)
    i2i.add_argument("--steps", type=int, default=None)
    i2i.add_argument("--guidance-scale", type=float, default=None, dest="guidance_scale")
    i2i.add_argument("--seed", type=int, default=None)
    i2i.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Edit strength (img2img noising/unnoising; backend-dependent).",
    )
    i2i.add_argument("--open", action="store_true", help="Open the output file (best-effort).")
    i2i.set_defaults(_fn=_cmd_i2i)

    t2v = sub.add_parser(
        "t2v", help="One-shot text-to-video (stores output and prints artifact ref + path)."
    )
    _add_provider_flags(t2v)
    t2v.add_argument("prompt")
    t2v.add_argument("--negative-prompt", default=None)
    t2v.add_argument("--width", type=int, default=None)
    t2v.add_argument("--height", type=int, default=None)
    t2v.add_argument("--fps", type=int, default=None)
    t2v.add_argument("--num-frames", "--frames", type=int, default=None, dest="num_frames")
    t2v.add_argument("--max-sequence-length", type=int, default=None)
    t2v.add_argument("--steps", type=int, default=None)
    t2v.add_argument("--guidance-scale", type=float, default=None, dest="guidance_scale")
    t2v.add_argument("--seed", type=int, default=None)
    t2v.add_argument("--open", action="store_true", help="Open the output file (best-effort).")
    t2v.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Show video generation progress on stderr.",
    )
    t2v.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable video generation progress output.",
    )
    t2v.set_defaults(_fn=_cmd_t2v)

    i2v = sub.add_parser(
        "i2v",
        help="One-shot image-to-video (experimental; stores output and prints artifact ref + path).",
    )
    _add_provider_flags(i2v)
    i2v.add_argument("--image", required=True, help="Input image file path.")
    i2v.add_argument("prompt", nargs="?", default=None, help="Optional guidance prompt.")
    i2v.add_argument("--negative-prompt", default=None)
    i2v.add_argument("--width", type=int, default=None)
    i2v.add_argument("--height", type=int, default=None)
    i2v.add_argument("--fps", type=int, default=None)
    i2v.add_argument("--num-frames", "--frames", type=int, default=None, dest="num_frames")
    i2v.add_argument("--max-sequence-length", type=int, default=None)
    i2v.add_argument("--steps", type=int, default=None)
    i2v.add_argument("--guidance-scale", type=float, default=None, dest="guidance_scale")
    i2v.add_argument("--seed", type=int, default=None)
    i2v.add_argument("--open", action="store_true", help="Open the output file (best-effort).")
    i2v.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Show video generation progress on stderr.",
    )
    i2v.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable video generation progress output.",
    )
    i2v.set_defaults(_fn=_cmd_i2v)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    fn = getattr(args, "_fn", None)
    if not callable(fn):
        raise SystemExit(2)
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
