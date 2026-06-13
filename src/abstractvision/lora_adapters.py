from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, List, Optional, Sequence, Tuple

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
