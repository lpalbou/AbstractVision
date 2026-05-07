## Task 014: Lightweight OpenAI-compatible packaging for plugin hosts

**Date**: 2026-05-07  
**Status**: Completed  
**Priority**: P1  

---

## Main goals

- Make `pip install abstractvision` a lightweight install that supports shared contracts, the capability registry, artifact refs, the OpenAI-compatible HTTP backend, CLI glue, and AbstractCore plugin registration without installing local inference runtimes.
- Keep local Diffusers and stable-diffusion.cpp generation available through explicit extras.
- Enable AbstractCore to publish a media/plugin server image with AbstractVision installed while staying remote-first and avoiding implicit Torch, Diffusers, CUDA-adjacent wheels, or local model stacks.

## Secondary goals

- Preserve the public `VisionManager` API and backend interfaces.
- Preserve the AbstractCore capability plugin entry point:
  - `abstractvision = "abstractvision.integrations.abstractcore_plugin:register"`
- Keep `import abstractvision`, backend export lookup, CLI import, and AbstractCore plugin discovery import-light.
- Improve local-backend install hints so users get a clear action when they select a backend whose extra is not installed.
- Update docs and release notes so the install behavior change is explicit.

---

## Context / problem

AbstractCore wants two server image profiles:

- `abstractcore-server`: a lightweight OpenAI-compatible gateway for remote providers.
- A media-capable server image: AbstractCore plus AbstractVoice and AbstractVision plugin support.

The media/plugin image should be useful for remote OpenAI-compatible image/video endpoints without pulling local vision inference stacks. AbstractVision already has the runtime surface for this:

- OpenAI-compatible HTTP backend: `src/abstractvision/backends/openai_compatible.py`
- AbstractCore capability plugin: `src/abstractvision/integrations/abstractcore_plugin.py`
- AbstractCore plugin default: OpenAI-compatible when configured through `ABSTRACTVISION_BASE_URL`
- Lazy backend exports: `src/abstractvision/backends/__init__.py`

The current package metadata contradicts that deployment target. `pyproject.toml` installs local Diffusers dependencies in the base package:

- `diffusers`
- `torch`
- `transformers`
- `accelerate`
- `safetensors`
- `sentencepiece`
- `protobuf`
- `einops`
- `peft`
- `Pillow`

Python extras can add dependencies, but they cannot remove base dependencies. As long as these packages stay in `[project].dependencies`, `abstractvision[openai-compatible]` is only an intent marker, not a lightweight install profile.

The source import boundary is already mostly correct. Local probes showed that `import abstractvision`, importing the AbstractCore plugin module, and importing the CLI do not import `torch`, `diffusers`, `transformers`, `PIL`, or `stable_diffusion_cpp`. The remaining work is packaging semantics, runtime defaults, error messages, CI, docs, and release hygiene.

This task intentionally supersedes the current internal backlog principle that says the default install includes Diffusers. Update `docs/backlog/README.md` as part of implementation so future planning uses the new install boundary.

---

## Constraints

- Do not rewrite the generation abstraction.
- Do not remove local Diffusers or stable-diffusion.cpp support.
- Do not make AbstractVision depend on AbstractCore.
- Do not include CUDA, model weights, or implicit model downloads in Python package metadata.
- Do not import Torch, Diffusers, Transformers, PIL, stable-diffusion.cpp bindings, or model code during package import or plugin discovery.
- Explicit local-backend selection must not silently fall back to OpenAI-compatible HTTP.
- Local model downloads remain disabled by default; users must pre-download models or opt in explicitly.
- Generated artifacts must remain representable as compact refs/metadata, not only inlined media bytes.

---

## Research, options, and references

This section combines the proposed backlog item, local repository inspection, and three focused sub-agent reviews.

- **Option A: Keep base install local-first and document custom Docker images**
  - Pros:
    - No behavior change for users who expect `pip install abstractvision` to run local Diffusers immediately.
    - Matches the current README and backlog README wording.
  - Cons:
    - AbstractCore cannot depend on AbstractVision in a lightweight media/plugin image.
    - `abstractvision[openai-compatible]` remains a no-op marker that cannot remove Torch/Diffusers.
    - Remote-only users still download local inference dependencies.
  - Assessment:
    - Rejected for this task because it blocks the AbstractCore image goal.

- **Option B: Make base install lightweight and move local runtimes behind extras**
  - Pros:
    - Clean single-distribution semantics.
    - Base package contains the API contracts, registry, artifact helpers, stdlib OpenAI-compatible backend, CLI, and plugin registration.
    - AbstractCore can install AbstractVision without local generation weight.
    - Local model execution becomes an explicit deployment choice.
  - Cons:
    - `pip install abstractvision` changes meaning.
    - REPL/playground local-first defaults and docs must change.
    - Downstream users that relied on base local generation must update install targets.
  - Assessment:
    - Chosen approach.

- **Option C: Create a separate lightweight distribution**
  - Examples:
    - `abstractvision-core`
    - `abstractvision-remote`
  - Pros:
    - Preserves current `abstractvision` local-first behavior.
    - Gives AbstractCore a lightweight dependency target.
  - Cons:
    - Adds package/distribution complexity.
    - Risks import-path confusion, split entry points, duplicate docs, and harder support.
  - Assessment:
    - Rejected unless Option B proves too disruptive after release feedback.

- **Option D: Use `[dependency-groups]` for local runtimes**
  - Pros:
    - Useful for source-tree development workflows and grouping local test/doc/lint dependencies.
  - Cons:
    - Dependency groups are not built package metadata and are not the right user-facing install interface for PyPI wheels.
    - Plugin hosts need wheel metadata extras, not source-tree-only groups.
  - Assessment:
    - Do not use dependency groups for runtime install profiles. Keep runtime profiles in `[project.optional-dependencies]`.

References and key findings:

- `pyproject.toml` specification: `[project].dependencies` maps to core package requirements, and `[project.optional-dependencies]` maps to extras metadata.
  - https://packaging.python.org/en/latest/specifications/pyproject-toml/#dependencies-optional-dependencies
- PyPA dependency-groups specification: dependency groups are not included in built package metadata and are intended for internal development/source-tree workflows.
  - https://packaging.python.org/en/latest/specifications/dependency-groups/
- pip dependency-groups guide: pip supports installing dependency groups from a `pyproject.toml`, but this is a separate interface from wheel extras.
  - https://pip.pypa.io/en/stable/user_guide/#dependency-groups
- Current packaging source of truth:
  - `pyproject.toml`
- Historical packaging context:
  - `docs/backlog/completed/012_packaging_extras_and_release_hygiene.md`
- Proposed draft superseded by this planned task:
  - `docs/backlog/proposed/2026-05-07_lightweight_openai_compatible_packaging.md`

---

## Decision

**Chosen approach**: Option B. Make the base distribution lightweight and move local inference runtimes into explicit extras.

Install profiles:

- `abstractvision`
  - Lightweight core, capability registry, artifact helpers, OpenAI-compatible HTTP backend, CLI, and AbstractCore plugin entry point.
  - No Torch, Diffusers, Transformers, stable-diffusion.cpp python bindings, or local inference packages.
- `abstractvision[openai-compatible]`
  - Empty or very small compatibility/intent extra.
  - Use for generic OpenAI-shaped endpoints, including local or third-party `/v1` image servers.
  - Keep because it makes Dockerfiles and AbstractCore dependency declarations readable.
- `abstractvision[openai]`
  - Empty or very small official-OpenAI intent extra.
  - Use when the target provider is OpenAI itself, even though the current implementation still uses the stdlib OpenAI-compatible HTTP backend and does not require the OpenAI SDK.
  - Keep distinct from `openai-compatible` so docs and dependency declarations can communicate provider intent without adding runtime weight.
- `abstractvision[diffusers]`
  - Canonical local HuggingFace/Diffusers runtime extra.
  - Include the full dependency set currently required by base Diffusers support:
    - `diffusers`
    - `torch`
    - `transformers`
    - `accelerate`
    - `safetensors`
    - `sentencepiece`
    - `protobuf`
    - `einops`
    - `peft`
    - `Pillow`
- `abstractvision[huggingface]`
  - Backward-compatible alias for `diffusers`.
  - Keep because current docs and older callers may request it.
- `abstractvision[sdcpp]`
  - stable-diffusion.cpp python binding fallback.
  - Include `stable-diffusion-cpp-python` and `Pillow` if python-binding edit support depends on PIL after base `Pillow` is removed.
  - Continue to support external `sd-cli` without requiring the python binding.
- `abstractvision[local]`
  - Convenience bundle for `diffusers` plus `sdcpp`.
- `abstractvision[all]`
  - Convenience bundle for all runtime backends:
    - `openai`
    - `openai-compatible`
    - `diffusers`
    - `sdcpp`
  - Do not include contributor-only extras such as `test`, `docs`, or `dev`.
- `abstractvision[diffusers-dev]`
  - Canonical development-compatible Diffusers stack for newer/unreleased pipelines.
  - Keep `huggingface-dev` as a compatibility alias.
- `abstractvision[test]`
  - Test dependencies for local contributors.
  - Keep base tests light; move Torch-specific tests into a separate CI path or keep `torch` here only for contributors running the whole suite locally.
- `abstractvision[docs]`
  - Documentation tooling.
- `abstractvision[dev]`
  - Test/docs/build/lint/pre-commit tooling.
  - Include whatever is needed for the contributor workflow, but do not use `dev` as a runtime profile in user docs.
- `abstractvision[abstractcore]`
  - Empty compatibility marker. AbstractCore is supplied by the host application.

Runtime defaults:

- AbstractCore plugin:
  - Keep remote-first default. It already defaults to OpenAI-compatible HTTP and imports local backend modules only when explicitly selected.
- One-shot CLI commands:
  - Keep `abstractvision t2i` and `abstractvision i2i` OpenAI-compatible only.
  - Clarify help/docs that these commands require `--base-url` or `ABSTRACTVISION_BASE_URL`.
- REPL:
  - Prefer remote-first when `ABSTRACTVISION_BASE_URL` is configured.
  - If no backend is configured, start unconfigured and require explicit `/backend openai ...`, `/backend diffusers ...`, or `/backend sdcpp ...`.
  - Do not silently select Diffusers in a lightweight base install.
- Playground:
  - Prefer OpenAI-compatible when `ABSTRACTVISION_BASE_URL` is configured.
  - Otherwise list only cached/configured local models and make local selection explicit.
  - Keep raw Hugging Face model id normalization for local mode, but do not make it the unconfigured default.

**Why**:

- It is the only single-distribution approach that lets AbstractCore include AbstractVision without local inference weight.
- It aligns package metadata with runtime behavior: remote/OpenAI-compatible operation is available from base; local generation is explicit.
- It preserves local backend support and the public API while reducing install surprises for plugin hosts, Docker images, CI, and serverless/remote-only deployments.
- It keeps existing historical names (`huggingface`, `huggingface-dev`) as aliases while introducing clearer canonical names (`diffusers`, `diffusers-dev`).

---

## Dependencies

- **Backlog tasks**:
  - Completed: `docs/backlog/completed/006_openai_compatible_backend_for_image_and_video.md`
  - Completed: `docs/backlog/completed/007_local_hf_backend_strategy_diffusers.md`
  - Completed: `docs/backlog/completed/009_test_matrix_and_ci_for_capabilities.md`
  - Completed: `docs/backlog/completed/011_abstractcore_tool_integration_and_artifact_refs.md`
  - Completed: `docs/backlog/completed/012_packaging_extras_and_release_hygiene.md`
  - Proposed source draft: `docs/backlog/proposed/2026-05-07_lightweight_openai_compatible_packaging.md`
- **Source areas**:
  - `pyproject.toml`
  - `src/abstractvision/__init__.py`
  - `src/abstractvision/backends/__init__.py`
  - `src/abstractvision/backends/huggingface_diffusers.py`
  - `src/abstractvision/backends/stable_diffusion_cpp.py`
  - `src/abstractvision/integrations/abstractcore_plugin.py`
  - `src/abstractvision/cli.py`
  - `src/abstractvision/playground_server.py`
- **Tests**:
  - `tests/test_packaging_metadata.py`
  - `tests/test_abstractcore_plugin.py`
  - `tests/test_openai_compatible_backend.py`
  - `tests/test_cli_smoke.py`
  - `tests/test_playground_server.py`
  - `tests/test_huggingface_diffusers_backend.py`
- **CI/release**:
  - `.github/workflows/ci.yml`
  - `.github/workflows/release.yml`
  - `CHANGELOG.md`
  - `scripts/generate_llms_full.py`

---

## Implementation plan

### 1. Packaging metadata

- Move all local Diffusers runtime dependencies from `[project].dependencies` into a canonical `diffusers` extra.
- Leave `[project].dependencies` empty unless a small shared runtime dependency is genuinely required by base OpenAI-compatible operation.
- Add/complete extras:
  - `openai-compatible = []`
  - `openai = []`
  - `diffusers = [...]`
  - `huggingface = [...]` as an alias for `diffusers`
  - `sdcpp = ["stable-diffusion-cpp-python>=0.4.2", "Pillow>=9.0"]` if PIL is required by python-binding edit paths
  - `local = diffusers + sdcpp`
  - `all = openai + openai-compatible + diffusers + sdcpp`
  - `diffusers-dev = [...]`
  - `huggingface-dev = [...]` as an alias for `diffusers-dev`
  - `abstractcore = []`
  - `test`, `docs`, and `dev` contributor extras
- Ensure `diffusers`, `huggingface`, and `local` include dependencies currently masked by base:
  - `sentencepiece`
  - `protobuf`
  - `einops`
  - `peft`
- Avoid adding extras whose names differ only by `-`, `_`, or `.` because packaging tools normalize extra names.

### 2. Optional dependency errors

- Update Diffusers lazy import errors to recommend:
  - `pip install "abstractvision[diffusers]"`
  - or `pip install "abstractvision[local]"`
- Keep CUDA-specific PyTorch guidance separate from the general missing-extra hint.
- Update `stable_diffusion_cpp.py` messages to mention both valid paths:
  - install/configure an external `sd-cli`
  - or `pip install "abstractvision[sdcpp]"`
- Preserve typed errors such as `OptionalDependencyMissingError` so callers can handle configuration failures.

### 3. Runtime defaults and behavior

- Keep AbstractCore plugin behavior remote-first:
  - default backend remains OpenAI-compatible
  - local backends remain explicit through owner config or `ABSTRACTVISION_BACKEND`
- Keep one-shot CLI commands remote-first:
  - `abstractvision t2i`
  - `abstractvision i2i`
- Change REPL initialization:
  - if `ABSTRACTVISION_BACKEND` is set, honor it
  - else if `ABSTRACTVISION_BASE_URL` is set, default to `openai`
  - else start with no active backend and require explicit `/backend ...`
- Change playground initialization:
  - if `ABSTRACTVISION_BACKEND` is set, honor it
  - else if `ABSTRACTVISION_BASE_URL` is set, default to OpenAI-compatible
  - else avoid implicit Diffusers preload/default model
  - list cached/configured local models only when they are actually available or downloads are explicitly enabled
- When a user explicitly selects `diffusers` without the extra installed:
  - accept the configuration
  - fail at load/generation with actionable `OptionalDependencyMissingError`
  - do not auto-install, auto-download, or silently fall back to OpenAI
- When a user explicitly selects `sdcpp` without `sd-cli` or `abstractvision[sdcpp]`:
  - fail with a clear message listing both supported install paths
  - do not fall back to Diffusers
- In playground HTTP handlers, map `AbstractVisionError` and `OptionalDependencyMissingError` to client/configuration errors instead of generic 500 responses.

### 4. Tests and CI

- Expand `tests/test_packaging_metadata.py`:
  - base dependencies exclude:
    - `torch`
    - `diffusers`
    - `transformers`
    - `accelerate`
    - `safetensors`
    - `sentencepiece`
    - `protobuf`
    - `einops`
    - `peft`
    - `Pillow`
    - `stable-diffusion-cpp-python`
  - `diffusers`, `huggingface`, and `local` include the complete Diffusers stack
  - `sdcpp` and `local` include `stable-diffusion-cpp-python`
  - `all` includes all runtime backend dependencies without test/docs/dev tooling
  - `openai`, `openai-compatible`, and `abstractcore` remain valid extras
  - AbstractCore entry point remains present
- Add subprocess import-light coverage:
  - block or detect imports of `torch`, `diffusers`, `transformers`, `PIL`, and `stable_diffusion_cpp`
  - import `abstractvision`
  - import the AbstractCore plugin module
  - load entry points through `importlib.metadata`
  - assert heavy modules are absent from `sys.modules`
- Add clean wheel smoke coverage:
  - build a wheel
  - install base into a temporary clean virtual environment
  - import `abstractvision`
  - import `abstractvision.backends.openai_compatible`
  - inspect entry points
  - verify no local runtime packages were installed
- Split CI jobs:
  - base matrix across supported Python versions without Torch/Diffusers
  - local-backend job for Torch/Diffusers-specific tests, likely narrower than the full Python matrix
- Expand OpenAI-compatible backend tests:
  - URL response download
  - authorization header
  - custom image endpoint paths
  - capabilities with configured video paths
  - `generate_video`
  - `image_to_video` in `multipart` mode
  - `image_to_video` in `json_b64` mode
  - invalid response shapes
  - provider HTTP errors

### 5. Documentation and release hygiene

- Update current user-facing docs:
  - `README.md`
  - `docs/getting-started.md`
  - `docs/faq.md`
  - `docs/reference/configuration.md`
  - `docs/reference/backends.md`
  - `docs/reference/abstractcore-integration.md`
- Update internal docs:
  - `docs/backlog/README.md`
  - possibly add a note to `docs/backlog/completed/012_packaging_extras_and_release_hygiene.md` or reference this task as the newer packaging source of truth
- Update source docstrings:
  - `src/abstractvision/__init__.py`
- Update acknowledgments/release files if wording says runtime dependencies are declared without distinguishing base dependencies from optional extras:
  - `ACKNOWLEDGMENTS.md`
  - `CHANGELOG.md`
- Regenerate LLM docs after docs/metadata changes:
  - `python scripts/generate_llms_full.py`
- Treat this as a behavior-changing minor release:
  - bump `src/abstractvision/__init__.py` from `0.2.6` to `0.3.0`
  - add a clear `CHANGELOG.md` entry
  - mention that upgraded existing environments may retain previously installed Torch/Diffusers wheels and that clean envs/Docker layers are the right validation path

---

## Success criteria

- `pip install abstractvision` installs no local vision inference runtimes.
- Base install can:
  - import `abstractvision`
  - import the OpenAI-compatible backend
  - load the capability registry
  - register the AbstractCore plugin entry point
  - run OpenAI-compatible one-shot CLI commands when `base_url` is configured
- `pip install "abstractvision[openai-compatible]"` remains valid and lightweight.
- `pip install "abstractvision[openai]"` remains valid and lightweight for official OpenAI provider intent.
- `pip install "abstractvision[diffusers]"` enables local Diffusers image generation where the platform supports the required wheels.
- `pip install "abstractvision[huggingface]"` remains a compatibility alias for local Diffusers runtime dependencies.
- `pip install "abstractvision[sdcpp]"` enables stable-diffusion.cpp python binding fallback where supported.
- `pip install "abstractvision[local]"` installs both local runtime families.
- `pip install "abstractvision[all]"` installs all runtime backend dependencies without contributor tooling.
- AbstractCore can include AbstractVision in a remote-first media/plugin image without pulling Torch/Diffusers.
- Explicit local backend selection without its runtime extra fails with a typed, actionable install hint.
- No import-light regression: base import and plugin discovery do not import heavy modules.
- README, docs, backlog principles, changelog, and generated `llms-full.txt` reflect the new install model.

---

## Test plan

- Unit suite:
  - `PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -q`
- Packaging metadata:
  - assert base dependency exclusions
  - assert extras contents
  - assert entry point metadata
- Import-light subprocess:
  - `python -c "import abstractvision; import abstractvision.integrations.abstractcore_plugin; import importlib.metadata as m; list(m.entry_points(group='abstractcore.capabilities_plugins'))"`
  - assert heavy modules are not imported
- Clean wheel smoke:
  - `python -m build`
  - create a temporary venv
  - install the built wheel without extras
  - import package/backend/plugin
  - inspect installed metadata
- OpenAI-compatible backend unit tests with mocked HTTP:
  - image generation
  - image edits
  - URL response handling
  - auth and custom endpoint paths
  - optional video endpoints
  - invalid/provider error responses
- CLI smoke:
  - `abstractvision models`
  - `abstractvision tasks`
  - `abstractvision t2i --base-url ...` with mocked/local compatible endpoint where possible
  - REPL startup without configured backend does not attempt Diffusers
- Playground smoke:
  - remote-configured startup lists configured OpenAI-compatible model
  - unconfigured lightweight startup does not preload Diffusers
  - missing local runtime maps to a configuration/client error with install hint
- Optional live smoke, gated by env vars:
  - OpenAI-compatible image endpoint
  - local Diffusers generation
  - stable-diffusion.cpp CLI or python binding

---

## Report

### Summary

- Moved the base package to a lightweight install with no mandatory runtime dependencies and shifted local runtimes into explicit extras:
  - `diffusers` for local Hugging Face/Diffusers execution
  - `huggingface` as the backward-compatible Diffusers alias
  - `sdcpp` for stable-diffusion.cpp python binding support
  - `local` and `all` as convenience runtime bundles
  - `openai`, `openai-compatible`, and `abstractcore` as valid lightweight intent/compatibility markers
- Updated REPL and playground defaults so a clean base install starts unconfigured unless `ABSTRACTVISION_BASE_URL` is present, in which case it defaults to the OpenAI-compatible backend.
- Preserved explicit local backend selection while improving missing-extra errors for Diffusers and stable-diffusion.cpp.
- Hardened OpenAI-compatible backend HTTP error handling and expanded coverage for auth headers, custom paths, URL downloads, video generation, image-to-video modes, malformed responses, and provider errors.
- Expanded packaging/import-light tests and split CI/release jobs into lightweight base validation and narrower local-Diffusers validation.
- Updated README, getting-started, reference docs, FAQ, AbstractCore integration docs, backlog principles, acknowledgments, changelog, and generated LLM docs for the new install model.
- Excluded proposed backlog drafts from source distributions so local working drafts cannot leak into release artifacts.
- Bumped the package version to `0.3.0` because this changes the meaning of `pip install abstractvision` for clean environments.

### Validation

- Focused tests: `PYTHONPATH=src python -m unittest tests.test_packaging_metadata tests.test_abstractcore_plugin tests.test_cli_smoke tests.test_playground_server tests.test_openai_compatible_backend -q` passed, 38 tests.
- Full unit suite: `PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -q` passed, 73 tests.
- Build and metadata: `python -m build` produced the `0.3.0` sdist and wheel, and `python -m twine check dist/*` passed.
- Clean base wheel smoke: installed the built wheel into a fresh virtual environment and confirmed import-light package/backend/plugin imports, AbstractCore entry point discovery, no base `Requires-Dist`, and no installed heavy local runtime packages.
- Wheel metadata inspection confirmed no base dependencies and the expected runtime extras.
- Docs: `mkdocs build -q` passed. MkDocs Material emitted its upstream MkDocs 2.0 compatibility warning.
