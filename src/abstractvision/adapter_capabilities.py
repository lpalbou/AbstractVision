from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .capability_assets import load_json_asset


ADAPTER_CAPABILITIES_ENV_VAR = "ABSTRACTVISION_ADAPTER_CAPABILITIES_PATH"


def _norm_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


@dataclass(frozen=True)
class AdapterRelativePathMatchSpec:
    contains_all: Tuple[str, ...] = ()
    contains_any: Tuple[str, ...] = ()
    contains_none: Tuple[str, ...] = ()
    starts_with: Optional[str] = None
    ends_with: Optional[str] = None
    equals: Optional[str] = None

    def matches(self, relative_path: str) -> bool:
        text = _norm_token(relative_path)
        if self.equals and text != _norm_token(self.equals):
            return False
        if self.starts_with and not text.startswith(_norm_token(self.starts_with)):
            return False
        if self.ends_with and not text.endswith(_norm_token(self.ends_with)):
            return False
        if self.contains_all and any(_norm_token(token) not in text for token in self.contains_all):
            return False
        if self.contains_any and not any(_norm_token(token) in text for token in self.contains_any):
            return False
        if self.contains_none and any(_norm_token(token) in text for token in self.contains_none):
            return False
        return True

    def specificity_rank(self) -> int:
        return (
            (20 if self.equals else 0)
            + (10 if self.starts_with else 0)
            + (10 if self.ends_with else 0)
            + len(self.contains_all) * 4
            + len(self.contains_any) * 2
            + len(self.contains_none) * 2
        )


@dataclass(frozen=True)
class AdapterDetectedParameterSpec:
    name: str
    default: Optional[int] = None


@dataclass(frozen=True)
class VisionAdapterProfileSpec:
    id: str
    repo_id: str
    kind: str
    family: str
    artifact_role: str = "adapter"
    compatible_base_model: Optional[str] = None
    base_models: Tuple[str, ...] = ()
    compatible_tasks: Tuple[str, ...] = ()
    model_aliases: Tuple[str, ...] = ()
    recommended_parameters: Dict[str, Any] = field(default_factory=dict)
    compatibility_notes: Tuple[str, ...] = ()
    documentation_urls: Tuple[str, ...] = ()
    recommended_local_quantization_bits: Tuple[int, ...] = ()
    notes: Optional[str] = None
    relative_path_match: Optional[AdapterRelativePathMatchSpec] = None
    detected_parameter: Optional[AdapterDetectedParameterSpec] = None

    def matches(self, *, repo_id: str, relative_path: str) -> bool:
        if _norm_token(repo_id) != _norm_token(self.repo_id):
            return False
        if self.relative_path_match is None:
            return True
        return self.relative_path_match.matches(relative_path)

    def specificity_rank(self) -> int:
        return self.relative_path_match.specificity_rank() if self.relative_path_match else 0

    def to_profile_dict(self, *, detected_value: Optional[int] = None) -> Dict[str, Any]:
        recommended = dict(self.recommended_parameters or {})
        if self.detected_parameter is not None:
            value = detected_value
            if value is None:
                value = self.detected_parameter.default
            if value is not None:
                recommended[str(self.detected_parameter.name)] = int(value)
        out: Dict[str, Any] = {
            "artifact_role": str(self.artifact_role),
            "kind": str(self.kind),
            "family": str(self.family),
            "compatible_base_model": (
                str(self.compatible_base_model).strip()
                if self.compatible_base_model
                else None
            ),
            "base_models": tuple(str(item).strip() for item in self.base_models if str(item).strip()),
            "compatible_tasks": tuple(
                str(item).strip() for item in self.compatible_tasks if str(item).strip()
            ),
            "model_aliases": tuple(
                _norm_token(item) for item in self.model_aliases if _norm_token(item)
            ),
            "recommended_parameters": recommended,
            "compatibility_notes": tuple(
                str(item).strip() for item in self.compatibility_notes if str(item).strip()
            ),
            "documentation_urls": tuple(
                str(item).strip() for item in self.documentation_urls if str(item).strip()
            ),
            "recommended_local_quantization_bits": tuple(
                int(item) for item in self.recommended_local_quantization_bits
            ),
        }
        if self.notes:
            out["notes"] = str(self.notes).strip()
        return out


class VisionAdapterCapabilitiesRegistry:
    """Loads `assets/vision_adapter_capabilities.json` and resolves adapter hints."""

    DEFAULT_ASSET_PATH = "assets/vision_adapter_capabilities.json"

    def __init__(self, *, asset_path: Optional[str] = None):
        self._asset_path = asset_path or self.DEFAULT_ASSET_PATH
        self._schema_version: str = ""
        self._profiles: Tuple[VisionAdapterProfileSpec, ...] = ()
        self._load()

    def _load(self) -> None:
        data = load_json_asset(
            package="abstractvision",
            default_asset_path=self.DEFAULT_ASSET_PATH,
            asset_path=self._asset_path,
            env_var=ADAPTER_CAPABILITIES_ENV_VAR,
            label="Adapter capability asset",
        )
        validate_adapter_capabilities_json(data)

        self._schema_version = str(data.get("schema_version") or "")
        profiles_raw = data.get("profiles", [])
        parsed: List[VisionAdapterProfileSpec] = []
        for item in profiles_raw:
            relative_match_raw = item.get("relative_path_match")
            relative_match = (
                AdapterRelativePathMatchSpec(
                    contains_all=tuple(
                        str(value).strip()
                        for value in relative_match_raw.get("contains_all", [])
                        if str(value).strip()
                    ),
                    contains_any=tuple(
                        str(value).strip()
                        for value in relative_match_raw.get("contains_any", [])
                        if str(value).strip()
                    ),
                    contains_none=tuple(
                        str(value).strip()
                        for value in relative_match_raw.get("contains_none", [])
                        if str(value).strip()
                    ),
                    starts_with=(
                        str(relative_match_raw.get("starts_with") or "").strip() or None
                    ),
                    ends_with=(
                        str(relative_match_raw.get("ends_with") or "").strip() or None
                    ),
                    equals=str(relative_match_raw.get("equals") or "").strip() or None,
                )
                if isinstance(relative_match_raw, dict)
                else None
            )
            detected_raw = item.get("detected_parameter")
            detected_parameter = (
                AdapterDetectedParameterSpec(
                    name=str(detected_raw.get("name") or "").strip(),
                    default=(
                        int(detected_raw.get("default"))
                        if detected_raw.get("default") not in (None, "")
                        else None
                    ),
                )
                if isinstance(detected_raw, dict)
                and str(detected_raw.get("name") or "").strip()
                else None
            )
            parsed.append(
                VisionAdapterProfileSpec(
                    id=str(item.get("id") or "").strip(),
                    repo_id=str(item.get("repo_id") or "").strip(),
                    kind=str(item.get("kind") or "").strip(),
                    family=str(item.get("family") or "").strip(),
                    artifact_role=str(item.get("artifact_role") or "adapter").strip(),
                    compatible_base_model=(
                        str(item.get("compatible_base_model") or "").strip() or None
                    ),
                    base_models=tuple(
                        str(value).strip()
                        for value in item.get("base_models", [])
                        if str(value).strip()
                    ),
                    compatible_tasks=tuple(
                        str(value).strip()
                        for value in item.get("compatible_tasks", [])
                        if str(value).strip()
                    ),
                    model_aliases=tuple(
                        str(value).strip()
                        for value in item.get("model_aliases", [])
                        if str(value).strip()
                    ),
                    recommended_parameters=dict(item.get("recommended_parameters", {})),
                    compatibility_notes=tuple(
                        str(value).strip()
                        for value in item.get("compatibility_notes", [])
                        if str(value).strip()
                    ),
                    documentation_urls=tuple(
                        str(value).strip()
                        for value in item.get("documentation_urls", [])
                        if str(value).strip()
                    ),
                    recommended_local_quantization_bits=tuple(
                        int(value)
                        for value in item.get("recommended_local_quantization_bits", [])
                        if str(value).strip()
                    ),
                    notes=str(item.get("notes") or "").strip() or None,
                    relative_path_match=relative_match,
                    detected_parameter=detected_parameter,
                )
            )
        self._profiles = tuple(parsed)

    def schema_version(self) -> str:
        return self._schema_version

    def iter_profiles(self) -> Iterable[VisionAdapterProfileSpec]:
        return iter(self._profiles)

    def match_profile(
        self,
        *,
        repo_id: str,
        relative_path: str,
    ) -> Optional[VisionAdapterProfileSpec]:
        matches = [
            profile
            for profile in self._profiles
            if profile.matches(repo_id=repo_id, relative_path=relative_path)
        ]
        if not matches:
            return None
        matches.sort(key=lambda profile: profile.specificity_rank(), reverse=True)
        return matches[0]


_PathPart = Union[str, int]


def _fmt_path(parts: Sequence[_PathPart]) -> str:
    out: List[str] = []
    for part in parts:
        if isinstance(part, int):
            out.append(f"[{part}]")
        else:
            if not out:
                out.append(str(part))
            else:
                out.append(f"[{part!r}]")
    return "".join(out) if out else "<root>"


def validate_adapter_capabilities_json(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Invalid adapter capability asset: top-level JSON must be an object.")

    schema_version = data.get("schema_version")
    if schema_version is None:
        raise ValueError("Invalid adapter capability asset: missing required key 'schema_version'.")
    if not isinstance(schema_version, (str, int, float)):
        raise ValueError(
            "Invalid adapter capability asset: 'schema_version' must be a string or number."
        )

    profiles = data.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("Invalid adapter capability asset: 'profiles' must be a list.")

    def _err(path: Sequence[_PathPart], msg: str) -> None:
        raise ValueError(f"Invalid adapter capability asset at {_fmt_path(path)}: {msg}")

    def _expect_str(value: Any, path: Sequence[_PathPart]) -> str:
        if not isinstance(value, str) or not value.strip():
            _err(path, "expected non-empty string")
        return value

    def _expect_list_of_str(value: Any, path: Sequence[_PathPart]) -> None:
        if not isinstance(value, list):
            _err(path, "expected list of strings")
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                _err([*path, index], "expected non-empty string")

    def _expect_dict(value: Any, path: Sequence[_PathPart]) -> Dict[str, Any]:
        if not isinstance(value, dict):
            _err(path, "expected object")
        return value

    allowed_roles = {"adapter", "adapter_repo", "full_model"}
    for index, item in enumerate(profiles):
        path = ["profiles", index]
        item_dict = _expect_dict(item, path)
        _expect_str(item_dict.get("id"), [*path, "id"])
        _expect_str(item_dict.get("repo_id"), [*path, "repo_id"])
        _expect_str(item_dict.get("kind"), [*path, "kind"])
        _expect_str(item_dict.get("family"), [*path, "family"])

        role = item_dict.get("artifact_role", "adapter")
        if not isinstance(role, str) or role not in allowed_roles:
            _err([*path, "artifact_role"], f"expected one of {sorted(allowed_roles)}")

        for key in (
            "base_models",
            "compatible_tasks",
            "model_aliases",
            "compatibility_notes",
            "documentation_urls",
        ):
            if key in item_dict:
                _expect_list_of_str(item_dict.get(key), [*path, key])

        quantization_bits = item_dict.get("recommended_local_quantization_bits")
        if quantization_bits is not None:
            if not isinstance(quantization_bits, list):
                _err([*path, "recommended_local_quantization_bits"], "expected list of integers")
            for q_index, value in enumerate(quantization_bits):
                try:
                    int(value)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid adapter capability asset at {_fmt_path([*path, 'recommended_local_quantization_bits', q_index])}: "
                        "expected integer"
                    ) from exc

        recommended_parameters = item_dict.get("recommended_parameters")
        if recommended_parameters is not None and not isinstance(recommended_parameters, dict):
            _err([*path, "recommended_parameters"], "expected object")

        relative_match = item_dict.get("relative_path_match")
        if relative_match is not None:
            relative_match_dict = _expect_dict(relative_match, [*path, "relative_path_match"])
            for key in ("contains_all", "contains_any", "contains_none"):
                if key in relative_match_dict:
                    _expect_list_of_str(
                        relative_match_dict.get(key),
                        [*path, "relative_path_match", key],
                    )
            for key in ("starts_with", "ends_with", "equals"):
                if key in relative_match_dict:
                    _expect_str(
                        relative_match_dict.get(key),
                        [*path, "relative_path_match", key],
                    )

        detected_parameter = item_dict.get("detected_parameter")
        if detected_parameter is not None:
            detected_parameter_dict = _expect_dict(
                detected_parameter, [*path, "detected_parameter"]
            )
            _expect_str(detected_parameter_dict.get("name"), [*path, "detected_parameter", "name"])
            if (
                "default" in detected_parameter_dict
                and detected_parameter_dict.get("default") not in (None, "")
            ):
                try:
                    int(detected_parameter_dict.get("default"))
                except Exception as exc:
                    raise ValueError(
                        f"Invalid adapter capability asset at {_fmt_path([*path, 'detected_parameter', 'default'])}: "
                        "expected integer"
                    ) from exc
