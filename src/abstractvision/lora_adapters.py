from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .adapter_capabilities import VisionAdapterCapabilitiesRegistry
from .types import LoRAAdapterSpec

LORA_EXTRA_KEYS = ("loras", "loras_json", "lora", "lora_json")


def _trimmed_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_scale(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _looks_like_probably_file_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return text.startswith(("./", "../", "/", "~")) or text.endswith(
        (".safetensors", ".bin", ".pt", ".ckpt")
    )


def _norm_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


@lru_cache(maxsize=1)
def _adapter_capability_registry() -> Optional[VisionAdapterCapabilitiesRegistry]:
    try:
        return VisionAdapterCapabilitiesRegistry()
    except Exception:
        return None


def _adapter_source_parts(spec: LoRAAdapterSpec) -> Tuple[str, str]:
    source = str(spec.source or "").strip()
    if source.startswith("hf:"):
        source = source[3:].strip()
    relative_path = ""
    if source and ":" in source:
        repo_id, rel = source.split(":", 1)
        if "/" in repo_id and not repo_id.startswith(("./", "../", "/", "~")):
            source = str(repo_id).strip()
            relative_path = str(rel).strip()
    if not relative_path:
        suffix_parts = [
            str(part).strip()
            for part in (spec.subfolder, spec.weight_name)
            if str(part or "").strip()
        ]
        if suffix_parts and source and not _looks_like_probably_file_path(source):
            relative_path = "/".join(suffix_parts)
    return source, relative_path


def _detect_step_count(*values: Any) -> Optional[int]:
    text = " ".join(_norm_token(value) for value in values if _norm_token(value))
    if not text:
        return None
    for pattern in (
        r"(?:^|[^0-9])([48])\s*steps?(?:$|[^0-9])",
        r"(?:^|[^0-9])nfe[- ]?([48])(?:$|[^0-9])",
    ):
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def _matches_model_aliases(profile: Dict[str, Any], model: Any) -> bool:
    model_s = _norm_token(model)
    if not model_s:
        return True
    aliases = tuple(profile.get("model_aliases") or ())
    if aliases and model_s in aliases:
        return True
    compatible_base = _norm_token(profile.get("compatible_base_model"))
    if compatible_base and model_s == compatible_base:
        return True
    return False


def known_lora_adapter_profile(value: Any) -> Optional[Dict[str, Any]]:
    spec = _coerce_one_adapter(value)
    if spec is None:
        return None
    repo_id, relative_path = _adapter_source_parts(spec)
    steps = _detect_step_count(repo_id, relative_path)
    registry = _adapter_capability_registry()
    if registry is None:
        return None
    profile = registry.match_profile(repo_id=repo_id, relative_path=relative_path)
    if profile is None:
        return None
    return profile.to_profile_dict(detected_value=steps)


def recommended_lora_request_overrides(
    lora_adapters: Any = None,
    *,
    extra: Any = None,
    task: Optional[str] = None,
    model: Any = None,
) -> Dict[str, Any]:
    adapters = resolve_request_lora_adapters(lora_adapters, extra=extra)
    recommendations: List[Dict[str, Any]] = []
    task_s = str(task or "").strip()
    for spec in adapters:
        profile = known_lora_adapter_profile(spec)
        if not isinstance(profile, dict):
            continue
        if str(profile.get("artifact_role") or "adapter") != "adapter":
            continue
        tasks = tuple(str(item).strip() for item in profile.get("compatible_tasks") or ())
        if task_s and tasks and task_s not in tasks:
            continue
        if model is not None and not _matches_model_aliases(profile, model):
            continue
        recommended = profile.get("recommended_parameters")
        if isinstance(recommended, dict) and recommended:
            recommendations.append(recommended)
    if not recommendations:
        return {}

    merged: Dict[str, Any] = {}
    keys = {
        str(key).strip()
        for item in recommendations
        for key in item.keys()
        if str(key).strip()
    }
    for key in sorted(keys):
        values = []
        for item in recommendations:
            if key not in item:
                continue
            value = item.get(key)
            if value not in values:
                values.append(value)
        if len(values) == 1:
            merged[key] = values[0]
    return merged


def _coerce_one_adapter(value: Any) -> Optional[LoRAAdapterSpec]:
    if isinstance(value, LoRAAdapterSpec):
        return value
    if isinstance(value, str):
        source = _trimmed_or_none(value)
        return LoRAAdapterSpec(source=source) if source else None
    if not isinstance(value, dict):
        return None
    source = _trimmed_or_none(value.get("source"))
    if not source:
        return None
    return LoRAAdapterSpec(
        source=source,
        scale=_coerce_scale(value.get("scale")),
        weight_name=_trimmed_or_none(value.get("weight_name")),
        subfolder=_trimmed_or_none(value.get("subfolder")),
        adapter_name=_trimmed_or_none(value.get("adapter_name")),
        target_role=_trimmed_or_none(value.get("target_role")),
    )


def _coerce_adapter_sequence(value: Any) -> Tuple[LoRAAdapterSpec, ...]:
    if value is None:
        return ()
    if isinstance(value, (LoRAAdapterSpec, dict, str)):
        item = _coerce_one_adapter(value)
        return (item,) if item is not None else ()
    if not isinstance(value, Sequence):
        return ()
    out: List[LoRAAdapterSpec] = []
    for item in value:
        spec = _coerce_one_adapter(item)
        if spec is not None:
            out.append(spec)
    return tuple(out)


def parse_legacy_lora_extra(extra: Any) -> Tuple[LoRAAdapterSpec, ...]:
    if not isinstance(extra, dict) or not extra:
        return ()
    raw: Any = None
    for key in LORA_EXTRA_KEYS:
        if extra.get(key) is not None:
            raw = extra.get(key)
            break
    if raw is None:
        return ()
    items: Any = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return ()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                items = json.loads(stripped)
            except Exception:
                items = stripped
        if isinstance(items, str):
            items = [part.strip() for part in items.split(",") if part.strip()]
    if isinstance(items, dict):
        items = [items]
    return _coerce_adapter_sequence(items)


def resolve_request_lora_adapters(
    lora_adapters: Any = None,
    *,
    extra: Any = None,
) -> Tuple[LoRAAdapterSpec, ...]:
    typed = _coerce_adapter_sequence(lora_adapters)
    if typed:
        return typed
    return parse_legacy_lora_extra(extra)


def serialize_lora_adapter(spec: LoRAAdapterSpec) -> dict[str, Any]:
    out = {"source": spec.source}
    if spec.scale is not None:
        out["scale"] = float(spec.scale)
    if spec.weight_name:
        out["weight_name"] = spec.weight_name
    if spec.subfolder:
        out["subfolder"] = spec.subfolder
    if spec.adapter_name:
        out["adapter_name"] = spec.adapter_name
    if spec.target_role:
        out["target_role"] = spec.target_role
    return out


def serialize_lora_adapters(adapters: Iterable[LoRAAdapterSpec]) -> List[dict[str, Any]]:
    return [serialize_lora_adapter(spec) for spec in adapters]


def lora_adapter_signature(adapters: Sequence[LoRAAdapterSpec]) -> Optional[str]:
    if not adapters:
        return None
    parts: List[str] = []
    for spec in adapters:
        parts.append(
            "|".join(
                [
                    str(spec.source or ""),
                    str(spec.subfolder or ""),
                    str(spec.weight_name or ""),
                    str(spec.adapter_name or ""),
                    str(spec.target_role or ""),
                    str(spec.scale if spec.scale is not None else 1.0),
                ]
            )
        )
    return hashlib.md5("::".join(parts).encode("utf-8")).hexdigest()[:12]


def resolved_adapter_name(spec: LoRAAdapterSpec) -> str:
    if spec.adapter_name:
        return spec.adapter_name
    key = "|".join(
        [
            str(spec.source or ""),
            str(spec.subfolder or ""),
            str(spec.weight_name or ""),
            str(spec.target_role or ""),
        ]
    )
    return "lora_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def adapter_source_for_mlx_gen(spec: LoRAAdapterSpec) -> str:
    source = str(spec.source or "").strip()
    if not source:
        raise ValueError("LoRA adapter source is required.")
    if _looks_like_probably_file_path(source) and (spec.weight_name or spec.subfolder):
        raise ValueError(
            "MLX-Gen LoRA adapters accept either a direct file path/repo:file handle or a repo id "
            "plus weight_name/subfolder, not both."
        )
    suffix_parts = [part for part in (spec.subfolder, spec.weight_name) if part]
    if not suffix_parts:
        return source
    return f"{source}:{'/'.join(suffix_parts)}"


def lora_scales_for_backend(adapters: Sequence[LoRAAdapterSpec]) -> Optional[List[float]]:
    if not adapters:
        return None
    out = [float(spec.scale) if spec.scale is not None else 1.0 for spec in adapters]
    return out


def lora_target_roles_for_backend(adapters: Sequence[LoRAAdapterSpec]) -> Optional[List[str]]:
    roles = [str(spec.target_role) for spec in adapters if spec.target_role]
    return roles or None
