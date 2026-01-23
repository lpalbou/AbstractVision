from __future__ import annotations

import json
import pkgutil
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .errors import CapabilityNotSupportedError, UnknownModelError


@dataclass(frozen=True)
class VisionTaskSpec:
    """Declarative spec for a single task supported by a model."""

    task: str
    inputs: List[str]
    outputs: List[str]
    params: Dict[str, Any]
    requires: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class VisionModelSpec:
    model_id: str
    provider: str
    license: str
    tasks: Dict[str, VisionTaskSpec]
    notes: str = ""


class VisionModelCapabilitiesRegistry:
    """Loads `assets/vision_model_capabilities.json` and answers capability questions."""

    DEFAULT_ASSET_PATH = "assets/vision_model_capabilities.json"

    def __init__(self, *, asset_path: Optional[str] = None):
        self._asset_path = asset_path or self.DEFAULT_ASSET_PATH
        self._schema_version: str = ""
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._models: Dict[str, VisionModelSpec] = {}
        self._load()

    def _load(self) -> None:
        raw = pkgutil.get_data("abstractvision", self._asset_path)
        if raw is None:
            raise RuntimeError(f"Capability asset not found: abstractvision/{self._asset_path}")
        data = json.loads(raw.decode("utf-8"))
        validate_capabilities_json(data)

        self._schema_version = str(data.get("schema_version") or "")
        tasks_raw = data.get("tasks", {})
        self._tasks = tasks_raw if isinstance(tasks_raw, dict) else {}

        models = data.get("models", {})
        if not isinstance(models, dict):
            raise ValueError("Invalid capability asset: `models` must be an object keyed by model_id.")

        parsed: Dict[str, VisionModelSpec] = {}
        for model_id, spec in models.items():
            provider = str(spec.get("provider", "unknown"))
            license_name = str(spec.get("license", "unknown"))
            notes = str(spec.get("notes", "")) if spec.get("notes") is not None else ""

            tasks_raw = spec.get("tasks", {})
            if not isinstance(tasks_raw, dict):
                raise ValueError(f"Invalid tasks for model {model_id}: must be an object keyed by task.")

            tasks: Dict[str, VisionTaskSpec] = {}
            for task_name, t in tasks_raw.items():
                inputs = list(t.get("inputs", []))
                outputs = list(t.get("outputs", []))
                params = dict(t.get("params", {}))
                requires = t.get("requires")
                tasks[task_name] = VisionTaskSpec(
                    task=str(task_name),
                    inputs=[str(x) for x in inputs],
                    outputs=[str(x) for x in outputs],
                    params=params,
                    requires=requires if isinstance(requires, dict) else None,
                )

            parsed[str(model_id)] = VisionModelSpec(
                model_id=str(model_id),
                provider=provider,
                license=license_name,
                tasks=tasks,
                notes=notes,
            )

        self._models = parsed

    def list_models(self) -> List[str]:
        return sorted(self._models.keys())

    def schema_version(self) -> str:
        return self._schema_version

    def list_tasks(self) -> List[str]:
        return sorted([str(k) for k in self._tasks.keys() if isinstance(k, str) and k.strip()])

    def get_task(self, task: str) -> Dict[str, Any]:
        task = str(task or "")
        out = self._tasks.get(task)
        if isinstance(out, dict):
            return out
        return {}

    def get(self, model_id: str) -> VisionModelSpec:
        try:
            return self._models[model_id]
        except KeyError as e:
            raise UnknownModelError(f"Unknown vision model id: {model_id}") from e

    def supports(self, model_id: str, task: str) -> bool:
        try:
            return task in self.get(model_id).tasks
        except UnknownModelError:
            return False

    def require_support(self, model_id: str, task: str) -> None:
        if not self.supports(model_id, task):
            raise CapabilityNotSupportedError(f"Model '{model_id}' does not support task '{task}'.")

    def models_for_task(self, task: str) -> List[str]:
        out: List[str] = []
        for mid, spec in self._models.items():
            if task in spec.tasks:
                out.append(mid)
        return sorted(out)

    def iter_task_specs(self, model_id: str) -> Iterable[VisionTaskSpec]:
        return self.get(model_id).tasks.values()


_PathPart = Union[str, int]


def _fmt_path(parts: Sequence[_PathPart]) -> str:
    out: List[str] = []
    for p in parts:
        if isinstance(p, int):
            out.append(f"[{p}]")
        else:
            if not out:
                out.append(str(p))
            else:
                out.append(f"[{p!r}]")
    return "".join(out) if out else "<root>"


def validate_capabilities_json(data: Any) -> None:
    """Validate the `vision_model_capabilities.json` schema (dependency-light).

    This is intentionally a "soft schema": it enforces required structure and
    internal reference integrity, while allowing additive fields.
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid capability asset: top-level JSON must be an object.")

    schema_version = data.get("schema_version")
    if schema_version is None:
        raise ValueError("Invalid capability asset: missing required key 'schema_version'.")
    if not isinstance(schema_version, (str, int, float)):
        raise ValueError("Invalid capability asset: 'schema_version' must be a string or number.")

    tasks = data.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("Invalid capability asset: 'tasks' must be an object keyed by task name.")
    for task_name, task_spec in tasks.items():
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError(f"Invalid capability asset: task key must be a non-empty string (got {task_name!r}).")
        if not isinstance(task_spec, dict):
            raise ValueError(f"Invalid capability asset: tasks[{task_name!r}] must be an object.")
        desc = task_spec.get("description")
        if desc is not None and not isinstance(desc, str):
            raise ValueError(f"Invalid capability asset: tasks[{task_name!r}]['description'] must be a string.")

    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError("Invalid capability asset: 'models' must be an object keyed by model_id.")

    def _err(path: Sequence[_PathPart], msg: str) -> None:
        raise ValueError(f"Invalid capability asset at {_fmt_path(path)}: {msg}")

    def _expect_dict(value: Any, path: Sequence[_PathPart]) -> Dict[str, Any]:
        if not isinstance(value, dict):
            _err(path, "expected object")
        return value

    def _expect_str(value: Any, path: Sequence[_PathPart]) -> str:
        if not isinstance(value, str) or not value.strip():
            _err(path, "expected non-empty string")
        return value

    def _expect_list_of_str(value: Any, path: Sequence[_PathPart]) -> List[str]:
        if not isinstance(value, list):
            _err(path, "expected list of strings")
        out: List[str] = []
        for i, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                _err([*path, i], "expected non-empty string")
            out.append(item)
        return out

    for model_id, model_spec in models.items():
        if not isinstance(model_id, str) or not model_id.strip():
            _err(["models"], f"model key must be a non-empty string (got {model_id!r})")
        model_path: List[_PathPart] = ["models", model_id]
        m = _expect_dict(model_spec, model_path)

        provider = m.get("provider")
        if provider is not None:
            _expect_str(provider, [*model_path, "provider"])
        else:
            _err([*model_path, "provider"], "missing required key")

        license_name = m.get("license")
        if license_name is not None:
            _expect_str(license_name, [*model_path, "license"])
        else:
            _err([*model_path, "license"], "missing required key")

        tasks_raw = m.get("tasks")
        if tasks_raw is None:
            _err([*model_path, "tasks"], "missing required key")
        tmap = _expect_dict(tasks_raw, [*model_path, "tasks"])
        for task_name, task_spec in tmap.items():
            _expect_str(task_name, [*model_path, "tasks", task_name])
            if task_name not in tasks:
                _err([*model_path, "tasks", task_name], "unknown task (not present in top-level 'tasks')")

            tpath: List[_PathPart] = [*model_path, "tasks", task_name]
            t = _expect_dict(task_spec, tpath)
            _expect_list_of_str(t.get("inputs", []), [*tpath, "inputs"])
            _expect_list_of_str(t.get("outputs", []), [*tpath, "outputs"])

            params = t.get("params")
            if params is None:
                _err([*tpath, "params"], "missing required key")
            pmap = _expect_dict(params, [*tpath, "params"])
            for pname, pspec in pmap.items():
                _expect_str(pname, [*tpath, "params", pname])
                pobj = _expect_dict(pspec, [*tpath, "params", pname])
                required = pobj.get("required")
                if not isinstance(required, bool):
                    _err([*tpath, "params", pname, "required"], "expected boolean")

            requires = t.get("requires")
            if requires is not None:
                robj = _expect_dict(requires, [*tpath, "requires"])
                base_model_id = robj.get("base_model_id")
                if base_model_id is not None:
                    _expect_str(base_model_id, [*tpath, "requires", "base_model_id"])
                    if base_model_id not in models:
                        _err([*tpath, "requires", "base_model_id"], f"unknown model id: {base_model_id!r}")
