## Task 0025: Cloud image models and their aliases in the model registry, one name lookup for every caller

**Date**: 2026-10-27  
**Status**: Planned  
**Priority**: P2  

---

## Main goals

- `assets/vision_model_capabilities.json` knows the cloud image models (OpenAI `gpt-image-*`, `dall-e-2`, `dall-e-3`, and the image models OpenAI-compatible relays commonly serve) with the tasks each serves, and the other names each model is published under (aliases).
- One public lookup, e.g. `abstractvision.model_tasks(model_id) -> {task: ...}` (name and aliases, routing prefixes such as `openai-compatible/` ignored), is the only place a caller learns what a model name can do.

## Secondary goals

- AbstractVision's own OpenAI-compatible backend and AbstractGateway's endpoint catalog use that lookup instead of their own name prefixes.

---

## Context / problem

Operator, 2026-10-27: the image models served by an endpoint a person configured must appear in their list of vision models, recognised faithfully. Today the recognition of a cloud model by its name is written twice as prefix rules: `backends/openai_compatible.py` (`_model_family`: `gpt-image-` / `dall-e-`) and AbstractGateway `endpoint_vision_catalog.py` (`known_image_model_tasks`). A model with another name (a relay's `flux-schnell`, `qwen-image`, a new OpenAI image model) is missed until both are edited, and the two can disagree. The registry already describes 68 local models with their tasks (`tasks.text_to_image`, ...), but no cloud model and no alias, so it cannot answer for a relay's names.

AbstractCore's `model_capabilities.json` is not the place: it describes text models (286 entries, `capability_routes` text in, text out); image generation is AbstractVision's capability.

---

## Current code reality (2026-10-27)

- `src/abstractvision/assets/vision_model_capabilities.json`: `schema_version`, `tasks`, `models` (68, Hugging Face ids, provider `huggingface`), per model `tasks.<task>` with inputs, outputs and params; no `aliases` field, no OpenAI model.
- `src/abstractvision/backends/openai_compatible.py:99-102`: `_model_family` by prefix; used for request shape (`image[]` field, edits) at 250, 496-531.
- AbstractGateway `src/abstractgateway/endpoint_vision_catalog.py`: `known_image_model_tasks` by prefix (`gpt-image`, `chatgpt-image`, `dall-e-2`, `dall-e`); `apply_endpoint_evidence` already prefers an endpoint's own per-model metadata and falls back to this name rule.

---

## Constraints

- Additive schema change (`aliases: [..]` per model; cloud models with `provider: openai`), schema file updated, old readers unaffected.
- A name the registry does not know stays unknown (no guess); an endpoint's own metadata still wins over the registry (AbstractGateway rule).
- No release of AbstractCore needed.

---

## Research, options, and references

- **Option A**: aliases in the existing registry + `model_tasks()` lookup; callers migrate. One source of truth, data not code.
- **Option B**: keep prefix rules, add more prefixes. Rejected: two copies, every new model is a code change in two packages.

---

## Decision

**Chosen approach**: Option A.

---

## Dependencies

- Consumer: AbstractGateway backlog 0941 (use the lookup in the endpoint vision catalog).

---

## Implementation plan

- Add `aliases` and the cloud image models (with their tasks) to the registry and its schema.
- Add `model_tasks(model_id)` (alias resolution, prefix stripping, case-insensitive) and export it.
- Replace `_model_family`'s recognition by the lookup where it decides capability (keep request-shape logic keyed by family).
- Tests: known names, aliases, prefixes, unknown name, schema validation.

---

## Success criteria

- `model_tasks("openai-compatible/gpt-image-1")` and an alias of a local model answer their tasks; an unknown name answers nothing.
- No capability decision by name prefix remains in AbstractVision.

---

## Test plan

- `pytest -q`

---

## Report (fill only when completed)

### Summary

### Validation
