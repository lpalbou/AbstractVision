## Task 001: MVP — end-to-end generative vision (tools + artifact outputs + at least 1 backend)

**Date**: 2026-01-23  
**Status**: Planned  
**Priority**: P0  

---

## Main goals

- Deliver the “golden path” user experience for generative vision inside the Abstract ecosystem:
  - an agent/workflow can request image/video generation,
  - the output is stored durably,
  - the caller receives a small JSON artifact ref (not bytes).
- Support at least one practical backend for real usage (remote first):
  - T2I (`text_to_image`) and I2I (`image_to_image`)
  - (optional if endpoint supports) T2V/I2V

## Secondary goals

- Keep `abstractvision` a separate package from `abstractcore` (LLM + analysis) while integrating cleanly via tools.
- Ensure everything is intuitive for third parties: stable tool schemas, stable output refs, dependency-light base install.

---

## Context / problem

AbstractCore already provides multimodal *analysis* (vision input to LLMs). Generative vision (image/video *outputs*) is a different problem:
- large binary outputs,
- long-running jobs,
- different endpoints/runtime stacks (diffusers, vendor APIs, OpenAI-compatible servers),
- and a stronger need for artifact-based durability and UI integration.

We want generative vision to feel native to the framework without bloating AbstractCore’s contracts.

---

## Constraints

- Base install must remain dependency-light (no torch/diffusers/httpx required just to import core types).
- No implicit model downloads in the “golden path”.
- Outputs must be artifact refs to support durable runs/tool outputs/third-party UIs.

---

## Research, options, and references

- **Option A**: Add generative image/video directly into AbstractCore provider APIs.
  - Pros: one package.
  - Cons: expands AbstractCore surface area and deps; mixes LLM IO and generative media job semantics.
- **Option B (chosen)**: `abstractvision` owns generative media; integrate into AbstractCore via tools + artifact refs.
  - Pros: clean separation; pluggable backends; minimal impact on AbstractCore; best for third-party integration.

Seed model references (for later backends; do not hardcode in the public contract):
- Qwen image gen: `https://huggingface.co/Qwen/Qwen-Image-2512`
- Qwen image edit: `https://huggingface.co/Qwen/Qwen-Image-Edit-2511`
- Wan video T2V: `https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B`
- HunyuanVideo: `https://huggingface.co/tencent/HunyuanVideo-1.5`

---

## Decision

**Chosen approach**:
- Make the MVP “end-to-end path” explicitly **tool + artifact ref** driven:
  - AbstractCore calls `abstractvision` via tools (Task 011),
  - `abstractvision` stores outputs and returns refs (Task 008),
  - one backend is implemented for real usage (Task 006 remote-first).

**Why**:
- This is the most intuitive and scalable integration surface for end users and third parties.

---

## Dependencies

- **Backlog tasks**:
  - Completed: `docs/backlog/completed/003_hf_model_landscape_and_capability_registry.md`
  - Completed: `docs/backlog/completed/005_core_api_tasks_and_abstractions.md`
  - Planned: `docs/backlog/planned/004_capability_schema_and_validation.md`
  - Planned: `docs/backlog/planned/006_openai_compatible_backend_for_image_and_video.md`
  - Planned: `docs/backlog/planned/008_asset_outputs_and_third_party_integration.md`
  - Planned: `docs/backlog/planned/010_readme_and_examples_for_model_selection.md`
  - Planned: `docs/backlog/planned/011_abstractcore_tool_integration_and_artifact_refs.md`

---

## Implementation plan

- Lock the output contract (artifact refs) and storage path (Task 008).
- Ship official AbstractCore tool wrappers (Task 011).
- Implement one real backend with mocked unit tests (Task 006).
- Document the golden path with copy/paste examples (Task 010).
- Ensure CI/unit tests cover the capability registry and output shapes (Task 009).

---

## Success criteria

- A user can install `abstractvision` and:
  - use it standalone with a local store, or
  - register the official tools in an AbstractCore app and receive artifact refs.
- Tool outputs are small JSON objects containing `"$artifact"` and `content_type`.

---

## Test plan

- `python -m unittest discover -s abstractvision/tests -p "test_*.py" -q`

---

## Report (fill only when completed)

### Summary

TBD

### Validation

- Tests: TBD

