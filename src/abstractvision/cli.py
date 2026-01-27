from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .artifacts import LocalAssetStore, is_artifact_ref
from .backends import (
    HuggingFaceDiffusersBackendConfig,
    HuggingFaceDiffusersVisionBackend,
    OpenAICompatibleBackendConfig,
    OpenAICompatibleVisionBackend,
    StableDiffusionCppBackendConfig,
    StableDiffusionCppVisionBackend,
)
from .model_capabilities import VisionModelCapabilitiesRegistry
from .vision_manager import VisionManager


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


def _build_openai_backend_from_args(args: argparse.Namespace) -> OpenAICompatibleVisionBackend:
    base_url = str(args.base_url or "").strip()
    if not base_url:
        raise SystemExit("Missing --base-url (or $ABSTRACTVISION_BASE_URL).")
    cfg = OpenAICompatibleBackendConfig(
        base_url=base_url,
        api_key=str(args.api_key) if args.api_key else None,
        model_id=str(args.model_id) if args.model_id else None,
        timeout_s=float(args.timeout_s),
        image_generations_path=str(args.images_generations_path),
        image_edits_path=str(args.images_edits_path),
        text_to_video_path=str(args.text_to_video_path) if args.text_to_video_path else None,
        image_to_video_path=str(args.image_to_video_path) if args.image_to_video_path else None,
        image_to_video_mode=str(args.image_to_video_mode),
    )
    return OpenAICompatibleVisionBackend(config=cfg)


def _build_manager_from_args(args: argparse.Namespace) -> VisionManager:
    store = LocalAssetStore(args.store_dir) if args.store_dir else LocalAssetStore()
    backend = _build_openai_backend_from_args(args)
    reg = VisionModelCapabilitiesRegistry()

    cap_model_id = str(args.capabilities_model_id) if getattr(args, "capabilities_model_id", None) else None
    if cap_model_id and cap_model_id not in set(reg.list_models()):
        raise SystemExit(
            f"--capabilities-model-id '{cap_model_id}' is not present in the registry. "
            "Use `abstractvision models` to list known ids, or omit this flag to disable gating."
        )

    return VisionManager(backend=backend, store=store, model_id=cap_model_id, registry=reg if cap_model_id else None)


def _cmd_models(_: argparse.Namespace) -> int:
    reg = VisionModelCapabilitiesRegistry()
    for mid in reg.list_models():
        print(mid)
    return 0


def _cmd_tasks(_: argparse.Namespace) -> int:
    reg = VisionModelCapabilitiesRegistry()
    for t in reg.list_tasks():
        desc = reg.get_task(t).get("description")
        if isinstance(desc, str) and desc.strip():
            print(f"{t}: {desc.strip()}")
        else:
            print(t)
    return 0


def _cmd_show_model(args: argparse.Namespace) -> int:
    reg = VisionModelCapabilitiesRegistry()
    spec = reg.get(str(args.model_id))
    print(spec.model_id)
    print(f"provider: {spec.provider}")
    print(f"license: {spec.license}")
    if spec.notes:
        print(f"notes: {spec.notes}")
    print("tasks:")
    for task_name, ts in sorted(spec.tasks.items()):
        print(f"  - {task_name}")
        if ts.requires:
            print(f"      requires: {json.dumps(ts.requires, sort_keys=True)}")
        if ts.params:
            required = sorted([k for k, v in ts.params.items() if isinstance(v, dict) and v.get("required") is True])
            optional = sorted([k for k, v in ts.params.items() if isinstance(v, dict) and v.get("required") is False])
            if required:
                print(f"      required params: {', '.join(required)}")
            if optional:
                print(f"      optional params: {', '.join(optional)}")
    return 0


def _cmd_t2i(args: argparse.Namespace) -> int:
    vm = _build_manager_from_args(args)
    out = vm.generate_image(
        args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    _print_json(out)
    if isinstance(vm.store, LocalAssetStore) and isinstance(out, dict) and is_artifact_ref(out):
        p = vm.store.get_content_path(out["$artifact"])
        if p is not None:
            print(str(p))
            if args.open:
                _open_file(p)
    return 0


def _cmd_i2i(args: argparse.Namespace) -> int:
    vm = _build_manager_from_args(args)
    image_bytes = Path(args.image).expanduser().read_bytes()
    mask_bytes = Path(args.mask).expanduser().read_bytes() if args.mask else None
    out = vm.edit_image(
        args.prompt,
        image=image_bytes,
        mask=mask_bytes,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
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
    backend_kind: str = _env("ABSTRACTVISION_BACKEND", "openai") or "openai"
    base_url: Optional[str] = _env("ABSTRACTVISION_BASE_URL")
    api_key: Optional[str] = _env("ABSTRACTVISION_API_KEY")
    model_id: Optional[str] = _env("ABSTRACTVISION_MODEL_ID")
    capabilities_model_id: Optional[str] = _env("ABSTRACTVISION_CAPABILITIES_MODEL_ID")
    store_dir: Optional[str] = _env("ABSTRACTVISION_STORE_DIR")
    timeout_s: float = float(_env("ABSTRACTVISION_TIMEOUT_S", "300") or "300")

    images_generations_path: str = _env("ABSTRACTVISION_IMAGES_GENERATIONS_PATH", "/images/generations") or "/images/generations"
    images_edits_path: str = _env("ABSTRACTVISION_IMAGES_EDITS_PATH", "/images/edits") or "/images/edits"
    text_to_video_path: Optional[str] = _env("ABSTRACTVISION_TEXT_TO_VIDEO_PATH")
    image_to_video_path: Optional[str] = _env("ABSTRACTVISION_IMAGE_TO_VIDEO_PATH")
    image_to_video_mode: str = _env("ABSTRACTVISION_IMAGE_TO_VIDEO_MODE", "multipart") or "multipart"

    diffusers_device: str = _env("ABSTRACTVISION_DIFFUSERS_DEVICE", "cpu") or "cpu"
    diffusers_torch_dtype: Optional[str] = _env("ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE")
    diffusers_allow_download: bool = _env_bool("ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD", False)
    diffusers_auto_retry_fp32: bool = _env_bool("ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32", False)

    sdcpp_bin: str = _env("ABSTRACTVISION_SDCPP_BIN", "sd-cli") or "sd-cli"
    sdcpp_model: Optional[str] = _env("ABSTRACTVISION_SDCPP_MODEL")
    sdcpp_diffusion_model: Optional[str] = _env("ABSTRACTVISION_SDCPP_DIFFUSION_MODEL")
    sdcpp_vae: Optional[str] = _env("ABSTRACTVISION_SDCPP_VAE")
    sdcpp_llm: Optional[str] = _env("ABSTRACTVISION_SDCPP_LLM")
    sdcpp_llm_vision: Optional[str] = _env("ABSTRACTVISION_SDCPP_LLM_VISION")
    sdcpp_extra_args: Optional[str] = _env("ABSTRACTVISION_SDCPP_EXTRA_ARGS")

    defaults: Dict[str, Any] = None
    _cached_backend_key: Optional[Tuple[Any, ...]] = None
    _cached_backend: Any = None
    _cached_store_dir: Optional[str] = None
    _cached_store: Optional[LocalAssetStore] = None

    def __post_init__(self) -> None:
        if self.defaults is None:
            self.defaults = {
                "t2i": {"width": None, "height": None, "steps": None, "guidance_scale": None, "seed": None, "negative_prompt": None},
                "i2i": {"steps": None, "guidance_scale": None, "seed": None, "negative_prompt": None},
            }


def _repl_help() -> str:
    return (
        "Commands:\n"
        "  /help                     Show this help\n"
        "  /exit                     Quit\n"
        "  /models                   List known capability model ids\n"
        "  /tasks                    List known task keys\n"
        "  /show-model <id>          Show a model's tasks + params\n"
        "  /config                   Show current backend/store config\n"
        "  /backend openai <base_url> [api_key] [model_id]\n"
        "  /backend diffusers <model_id_or_path> [device] [torch_dtype]\n"
        "  /backend sdcpp <diffusion_model.gguf> <vae.safetensors> <llm.gguf> [sd_cli_path]\n"
        "                           (Qwen Image: requires diffusion-model + vae + llm)\n"
        "  /cap-model <id|off>       Set capability-gating model id (from registry) or 'off'\n"
        "  /store <dir|default>      Set local store dir\n"
        "  /set <k> <v>              Set default param (k like width/height/steps/seed/guidance_scale/negative_prompt)\n"
        "  /unset <k>                Unset default param\n"
        "  /defaults                 Show current defaults\n"
        "  /t2i <prompt...> [--width N --height N --steps N --seed N --guidance-scale F --negative ...] [--open]\n"
        "                           (extra flags are forwarded via request.extra)\n"
        "  /i2i --image path <prompt...> [--mask path --steps N --seed N --guidance-scale F --negative ...] [--open]\n"
        "                           (extra flags are forwarded via request.extra)\n"
        "  /open <artifact_id>       Open a locally stored artifact (LocalAssetStore only)\n"
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


def _build_manager_from_state(state: _ReplState) -> VisionManager:
    if state._cached_store is not None and state._cached_store_dir == state.store_dir:
        store = state._cached_store
    else:
        store = LocalAssetStore(state.store_dir) if state.store_dir else LocalAssetStore()
        state._cached_store = store
        state._cached_store_dir = state.store_dir

    backend_kind = str(state.backend_kind or "").strip().lower() or "openai"
    backend_key: Tuple[Any, ...]
    if backend_kind == "openai":
        base_url = str(state.base_url or "").strip()
        if not base_url:
            raise ValueError("Backend is not configured. Use: /backend openai <base_url> [api_key] [model_id]")
        backend_key = (
            "openai",
            base_url,
            state.api_key,
            state.model_id,
            state.timeout_s,
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
                image_generations_path=str(state.images_generations_path),
                image_edits_path=str(state.images_edits_path),
                text_to_video_path=str(state.text_to_video_path) if state.text_to_video_path else None,
                image_to_video_path=str(state.image_to_video_path) if state.image_to_video_path else None,
                image_to_video_mode=str(state.image_to_video_mode),
            )
            backend = OpenAICompatibleVisionBackend(config=cfg)
            state._cached_backend = backend
            state._cached_backend_key = backend_key
    elif backend_kind == "diffusers":
        model_id = str(state.model_id or "").strip()
        if not model_id:
            raise ValueError("Diffusers backend is not configured. Use: /backend diffusers <model_id_or_path> [device]")
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
                torch_dtype=str(state.diffusers_torch_dtype) if state.diffusers_torch_dtype else None,
                allow_download=bool(state.diffusers_allow_download),
                auto_retry_fp32=bool(state.diffusers_auto_retry_fp32),
            )
            backend = HuggingFaceDiffusersVisionBackend(config=cfg)
            state._cached_backend = backend
            state._cached_backend_key = backend_key
    elif backend_kind in {"sdcpp", "stable-diffusion.cpp", "stable_diffusion_cpp", "stable-diffusion-cpp"}:
        backend_key = (
            "sdcpp",
            str(state.sdcpp_bin),
            str(state.sdcpp_model) if state.sdcpp_model else None,
            str(state.sdcpp_diffusion_model) if state.sdcpp_diffusion_model else None,
            str(state.sdcpp_vae) if state.sdcpp_vae else None,
            str(state.sdcpp_llm) if state.sdcpp_llm else None,
            str(state.sdcpp_llm_vision) if state.sdcpp_llm_vision else None,
            str(state.sdcpp_extra_args) if state.sdcpp_extra_args else None,
        )
        if state._cached_backend is not None and state._cached_backend_key == backend_key:
            backend = state._cached_backend
        else:
            cfg = StableDiffusionCppBackendConfig(
                sd_cli_path=str(state.sdcpp_bin),
                model=str(state.sdcpp_model) if state.sdcpp_model else None,
                diffusion_model=str(state.sdcpp_diffusion_model) if state.sdcpp_diffusion_model else None,
                vae=str(state.sdcpp_vae) if state.sdcpp_vae else None,
                llm=str(state.sdcpp_llm) if state.sdcpp_llm else None,
                llm_vision=str(state.sdcpp_llm_vision) if state.sdcpp_llm_vision else None,
                extra_args=shlex.split(str(state.sdcpp_extra_args)) if state.sdcpp_extra_args else (),
            )
            backend = StableDiffusionCppVisionBackend(config=cfg)
            state._cached_backend = backend
            state._cached_backend_key = backend_key
    else:
        raise ValueError(f"Unknown backend kind: {backend_kind!r} (expected 'openai', 'diffusers', or 'sdcpp')")

    reg = VisionModelCapabilitiesRegistry()
    cap_id = str(state.capabilities_model_id) if state.capabilities_model_id else None
    if cap_id and cap_id not in set(reg.list_models()):
        raise ValueError(f"capability model id not in registry: {cap_id!r}")
    return VisionManager(backend=backend, store=store, model_id=cap_id, registry=reg if cap_id else None)


def _cmd_repl(_: argparse.Namespace) -> int:
    reg = VisionModelCapabilitiesRegistry()
    state = _ReplState()

    print("AbstractVision REPL")
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
            if cmd in {"exit", "quit"}:
                return 0
            if cmd == "help":
                print(_repl_help())
                continue
            if cmd == "models":
                for mid in reg.list_models():
                    print(mid)
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
                    "images_generations_path": state.images_generations_path,
                    "images_edits_path": state.images_edits_path,
                    "text_to_video_path": state.text_to_video_path,
                    "image_to_video_path": state.image_to_video_path,
                    "image_to_video_mode": state.image_to_video_mode,
                    "diffusers_device": state.diffusers_device,
                    "diffusers_torch_dtype": state.diffusers_torch_dtype,
                    "diffusers_allow_download": state.diffusers_allow_download,
                    "sdcpp_bin": state.sdcpp_bin,
                    "sdcpp_model": state.sdcpp_model,
                    "sdcpp_diffusion_model": state.sdcpp_diffusion_model,
                    "sdcpp_vae": state.sdcpp_vae,
                    "sdcpp_llm": state.sdcpp_llm,
                    "sdcpp_llm_vision": state.sdcpp_llm_vision,
                    "sdcpp_extra_args": state.sdcpp_extra_args,
                    "defaults": state.defaults,
                }
                _print_json(out)
                continue
            if cmd == "backend":
                if not args:
                    print(
                        "Usage: /backend openai <base_url> [api_key] [model_id]  OR  "
                        "/backend diffusers <model_id_or_path> [device] [torch_dtype]  OR  "
                        "/backend sdcpp <diffusion_model.gguf> <vae.safetensors> <llm.gguf> [sd_cli_path]"
                    )
                    continue
                kind = str(args[0]).strip().lower()
                if kind == "openai":
                    if len(args) < 2:
                        print("Usage: /backend openai <base_url> [api_key] [model_id]")
                        continue
                    state.backend_kind = "openai"
                    state.base_url = args[1]
                    state.api_key = args[2] if len(args) >= 3 else state.api_key
                    state.model_id = args[3] if len(args) >= 4 else state.model_id
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
                    dtype_tokens = {"auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"}
                    if len(args) >= 3 and str(args[2]).strip().lower() in dtype_tokens:
                        state.diffusers_torch_dtype = str(args[2]).strip()
                    else:
                        state.diffusers_device = args[2] if len(args) >= 3 else state.diffusers_device
                        state.diffusers_torch_dtype = str(args[3]).strip() if len(args) >= 4 else state.diffusers_torch_dtype
                    print("ok")
                    continue
                if kind == "sdcpp":
                    if len(args) < 4:
                        print("Usage: /backend sdcpp <diffusion_model.gguf> <vae.safetensors> <llm.gguf> [sd_cli_path]")
                        continue
                    state.backend_kind = "sdcpp"
                    state.sdcpp_diffusion_model = args[1]
                    state.sdcpp_vae = args[2]
                    state.sdcpp_llm = args[3]
                    state.sdcpp_bin = args[4] if len(args) >= 5 else state.sdcpp_bin
                    print("ok")
                    continue
                print("Unknown backend kind. Use: openai | diffusers | sdcpp")
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
                for group in ("t2i", "i2i"):
                    if key in state.defaults.get(group, {}):
                        state.defaults[group][key] = value
                        updated = True
                if not updated:
                    for group in ("t2i", "i2i"):
                        state.defaults.setdefault(group, {})[key] = value
                print("ok")
                continue
            if cmd == "unset":
                if not args:
                    print("Usage: /unset <key>")
                    continue
                key = args[0].replace("-", "_")
                for group in ("t2i", "i2i"):
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
                    if k not in {"width", "height", "steps", "guidance_scale", "seed", "negative_prompt", "open"} and v is not None
                }
                out = vm.generate_image(
                    prompt,
                    negative_prompt=d.get("negative_prompt"),
                    width=_coerce_int(d.get("width")),
                    height=_coerce_int(d.get("height")),
                    steps=_coerce_int(d.get("steps")),
                    guidance_scale=_coerce_float(d.get("guidance_scale")),
                    seed=_coerce_int(d.get("seed")),
                    extra=extra,
                )
                _print_json(out)
                if isinstance(vm.store, LocalAssetStore) and isinstance(out, dict) and is_artifact_ref(out):
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
                    if k not in {"image", "mask", "steps", "guidance_scale", "seed", "negative_prompt", "open"} and v is not None
                }
                img = Path(str(image_path)).expanduser().read_bytes()
                mask = Path(str(mask_path)).expanduser().read_bytes() if mask_path else None
                out = vm.edit_image(
                    prompt,
                    image=img,
                    mask=mask,
                    negative_prompt=d.get("negative_prompt"),
                    steps=_coerce_int(d.get("steps")),
                    guidance_scale=_coerce_float(d.get("guidance_scale")),
                    seed=_coerce_int(d.get("seed")),
                    extra=extra,
                )
                _print_json(out)
                if isinstance(vm.store, LocalAssetStore) and isinstance(out, dict) and is_artifact_ref(out):
                    p = vm.store.get_content_path(out["$artifact"])
                    if p is not None:
                        print(str(p))
                        if bool(flags.get("open")):
                            _open_file(p)
                continue

            print("Unknown command. Type /help.")
        except Exception as e:
            print(f"Error: {e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="abstractvision", description="AbstractVision CLI (capabilities + generation).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="List known model ids (from capability registry).").set_defaults(_fn=_cmd_models)
    sub.add_parser("tasks", help="List known task keys (from capability registry).").set_defaults(_fn=_cmd_tasks)

    sm = sub.add_parser("show-model", help="Show a model's supported tasks and params.")
    sm.add_argument("model_id")
    sm.set_defaults(_fn=_cmd_show_model)

    repl = sub.add_parser("repl", help="Interactive REPL for testing capabilities and generation.")
    repl.set_defaults(_fn=_cmd_repl)

    def _add_backend_flags(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--base-url", default=_env("ABSTRACTVISION_BASE_URL"), help="OpenAI-compatible base URL (e.g. http://localhost:1234/v1).")
        ap.add_argument("--api-key", default=_env("ABSTRACTVISION_API_KEY"), help="API key (Bearer).")
        ap.add_argument("--model-id", default=_env("ABSTRACTVISION_MODEL_ID"), help="Remote model id/name.")
        ap.add_argument("--capabilities-model-id", default=_env("ABSTRACTVISION_CAPABILITIES_MODEL_ID"), help="Optional: enforce support using a registry model id.")
        ap.add_argument("--timeout-s", type=float, default=float(_env("ABSTRACTVISION_TIMEOUT_S", "300") or "300"), help="HTTP timeout seconds (default: 300).")
        ap.add_argument("--store-dir", default=_env("ABSTRACTVISION_STORE_DIR"), help="Local asset store dir (default: ~/.abstractvision/assets).")
        ap.add_argument("--images-generations-path", default=_env("ABSTRACTVISION_IMAGES_GENERATIONS_PATH", "/images/generations"), help="Path for image generations.")
        ap.add_argument("--images-edits-path", default=_env("ABSTRACTVISION_IMAGES_EDITS_PATH", "/images/edits"), help="Path for image edits.")
        ap.add_argument("--text-to-video-path", default=_env("ABSTRACTVISION_TEXT_TO_VIDEO_PATH"), help="Optional path for text-to-video.")
        ap.add_argument("--image-to-video-path", default=_env("ABSTRACTVISION_IMAGE_TO_VIDEO_PATH"), help="Optional path for image-to-video.")
        ap.add_argument("--image-to-video-mode", default=_env("ABSTRACTVISION_IMAGE_TO_VIDEO_MODE", "multipart"), help="image_to_video mode: multipart|json_b64.")

    t2i = sub.add_parser("t2i", help="One-shot text-to-image (stores output and prints artifact ref + path).")
    _add_backend_flags(t2i)
    t2i.add_argument("prompt")
    t2i.add_argument("--negative-prompt", default=None)
    t2i.add_argument("--width", type=int, default=None)
    t2i.add_argument("--height", type=int, default=None)
    t2i.add_argument("--steps", type=int, default=None)
    t2i.add_argument("--guidance-scale", type=float, default=None, dest="guidance_scale")
    t2i.add_argument("--seed", type=int, default=None)
    t2i.add_argument("--open", action="store_true", help="Open the output file (best-effort).")
    t2i.set_defaults(_fn=_cmd_t2i)

    i2i = sub.add_parser("i2i", help="One-shot image-to-image edit (stores output and prints artifact ref + path).")
    _add_backend_flags(i2i)
    i2i.add_argument("--image", required=True, help="Input image file path.")
    i2i.add_argument("--mask", default=None, help="Optional mask file path.")
    i2i.add_argument("prompt")
    i2i.add_argument("--negative-prompt", default=None)
    i2i.add_argument("--steps", type=int, default=None)
    i2i.add_argument("--guidance-scale", type=float, default=None, dest="guidance_scale")
    i2i.add_argument("--seed", type=int, default=None)
    i2i.add_argument("--open", action="store_true", help="Open the output file (best-effort).")
    i2i.set_defaults(_fn=_cmd_i2i)

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
