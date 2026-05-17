# Completed: Vision Install Profiles And Pending OpenAI Defaults

## Metadata
- Created: 2026-05-08
- Status: Completed
- Completed: 2026-05-08

## Context

AbstractVision is a capability package. It should own image/video backend selection and provider
defaults. AbstractCore discovers it as a capability plugin; Gateway composes and reports readiness.

## Current Code Reality

- Base dependencies are empty.
- OpenAI/OpenAI-compatible HTTP backend is stdlib-based.
- Local engines are explicit extras: `diffusers`, `sdcpp`, `huggingface`, `local`, and `all`.
- Pending changes switch the AbstractCore plugin default from legacy
  `abstractvision:openai-compatible` to `abstractvision:openai`, add OpenAI base URL/default model
  handling, and keep the compatible backend id as an alias.
- A separate AbstractFlow/Gateway note had captured a narrower deployment issue, but
  Gateway-specific backlog belongs with Gateway/Core. This item resolves the
  package-owned Vision defaults and docs.
- Vision also has a package-owned playground server. Treat it as a local/dev surface, not the
  production Gateway/Core serving boundary.
- OpenAI model ids are not discovered or selected dynamically. AbstractVision exposes an
  explicit provider catalog abstraction for OpenAI-compatible `GET /models` inspection, but
  the plugin uses the static `gpt-image-1` default for the official OpenAI path unless a
  model id is configured.

## Problem

Vision should support Gateway/Core remote-light installs by default, but it should not absorb
Gateway deployment config or invent platform-wide `apple`/`gpu` profiles that belong to aggregate
packages.

Vision's normal framework role is an outbound capability backend. It should consume provider
credentials, base URLs, model ids, endpoint paths, cache paths, and local backend settings. It
should not inherit Gateway/Core bearer tokens or browser origin policy unless the package grows an
intentional production server.

There are pending caveats:

- the legacy backend id alias should either preserve old compatible-endpoint behavior or be clearly
  documented as an alias to env-driven backend selection;
- docs must consistently require `ABSTRACTVISION_BACKEND=openai-compatible` plus
  `OPENAI_BASE_URL` for compatible endpoints;
- OpenAI default model docs must avoid claiming a conservative default is the latest model.

## Proposed Direction

Keep Vision package profiles local to Vision:

- `abstractvision`: light package, contracts, registry, stdlib HTTP backend, plugin entry point.
- `abstractvision[openai]`: provider-intent marker, currently no extra deps.
- `abstractvision[openai-compatible]`: compatible-endpoint marker, currently no extra deps.
- `abstractvision[diffusers]`: local Torch/Diffusers stack.
- `abstractvision[huggingface]`: compatibility alias for Diffusers.
- `abstractvision[sdcpp]`: stable-diffusion.cpp python binding support.
- `abstractvision[local]`: local backends.
- `abstractvision[all]`: all Vision backends, not contributor tooling.
- `abstractvision[apple]` / `abstractvision[all-apple]`: native macOS profile aliases for the
  full local Vision stack.
- `abstractvision[gpu]`: Diffusers/Torch GPU stack.
- `abstractvision[all-gpu]`: full GPU-relevant local Vision stack.

Gateway/Core/root should still own the higher-level deployment aggregation; Vision profile aliases
only describe Vision-owned backend dependencies.

## Pending Changes Guidance

Keep:

- plugin default moving to official OpenAI semantics;
- fallback to `OPENAI_API_KEY`;
- explicit compatible endpoint mode;
- plugin tests covering OpenAI default and explicit compatible config;
- docs that state Gateway should not mutate Vision env at request time.
- docs that distinguish production Core/Gateway routes from the local Vision playground server.

Revise before merge:

- ensure `abstractvision:openai-compatible` alias behavior is intentional and tested;
- document `OPENAI_BASE_URL`, `OPENAI_IMAGE_MODEL`, and `OPENAI_IMAGE_MODEL_ID` if supported;
- update compatible endpoint examples to include `ABSTRACTVISION_BACKEND=openai-compatible`;
- avoid "latest OpenAI image model" wording unless verified at release time.
- document that the playground server currently needs separate local/dev security treatment and
  should not be recommended as an authenticated production serving surface.

Do not do:

- no Gateway-specific provider defaults in Vision beyond plugin config hooks;
- no local model engine in base install;
- no platform-wide `apple`/`gpu` no-op extras just for symmetry.

## Promotion Criteria

Promote when the current plugin-default implementation is reviewed and the alias/model-default
caveats are resolved.

## Validation Ideas

- Plugin registration test for both backend ids.
- Default OpenAI path test using `OPENAI_API_KEY`.
- Explicit compatible endpoint test.
- Missing-key and missing-base-url tests with actionable errors.
- Import-light test proving base install does not import Torch/Diffusers/sdcpp.
- Gateway capability discovery test for installed-but-unconfigured Vision.

## Completion Report

### Date

2026-05-08

### Summary

- Kept the `abstractvision:openai` AbstractCore plugin default for official OpenAI semantics.
- Preserved compatible-endpoint behavior for the legacy `abstractvision:openai-compatible`
  backend id and for `OPENAI_BASE_URL`-only setups.
- Added standard OpenAI aliases for the plugin default path: `OPENAI_BASE_URL`,
  `OPENAI_API_KEY`, `OPENAI_IMAGE_MODEL_ID`, and `OPENAI_IMAGE_MODEL`.
- Confirmed that these aliases configure a model id only; they do not trigger automatic
  OpenAI-compatible `GET /models` discovery, vision-capability inference, or latest-model
  selection.
- Kept Vision package install profiles local to Vision; no `apple` / `gpu` extras were added.
- Tightened docs so compatible endpoints use `ABSTRACTVISION_BACKEND=openai-compatible`
  plus `OPENAI_BASE_URL`, and so the playground is described as local/dev only.
- Added an explicit provider catalog abstraction for OpenAI/OpenAI-compatible `GET /models`
  inspection without changing automatic model selection behavior.

### Files and Symbols Touched

- `src/abstractvision/integrations/abstractcore_plugin.py`
  - `_AbstractVisionCapability`
  - `register`
- `src/abstractvision/types.py`
  - `ProviderModelInfo`
- `src/abstractvision/backends/base_backend.py`
  - `VisionBackend.list_provider_models`
- `src/abstractvision/backends/openai_compatible.py`
  - `OpenAICompatibleVisionBackend.list_provider_models`
- `src/abstractvision/vision_manager.py`
  - `VisionManager.list_provider_models`
- `src/abstractvision/cli.py`
  - `abstractvision provider-models`
- `tests/test_abstractcore_plugin.py`
- `tests/test_openai_compatible_backend.py`
- `tests/test_cli_smoke.py`
- `tests/test_manager_capability_checks.py`
- `README.md`
- `docs/architecture.md`
- `docs/reference/abstractcore-integration.md`
- `docs/reference/configuration.md`
- `docs/getting-started.md`
- `playground/README.md`

### Validation

- `PYTHONPATH=src python -m unittest tests.test_abstractcore_plugin -q` passed, 13 tests.
- `PYTHONPATH=src python -m unittest tests.test_packaging_metadata tests.test_abstractcore_plugin tests.test_cli_smoke tests.test_playground_server tests.test_openai_compatible_backend -q` passed, 44 tests.
- `PYTHONPATH=src python -m unittest tests.test_cli_smoke.TestCliSmoke.test_provider_models_openai_uses_default_catalog tests.test_openai_compatible_backend.TestOpenAICompatibleVisionBackend.test_list_provider_models_default_openai_catalog_live -q` passed, covering default OpenAI CLI catalog selection and the live OpenAI provider catalog path.
- `PYTHONPATH=src python -m unittest tests.test_openai_compatible_backend.TestOpenAICompatibleVisionBackend.test_list_provider_models_default_openai_catalog_live -q` passed with `OPENAI_API_KEY` present; the default OpenAI `/models` catalog returned image models through `VisionManager.list_provider_models(task="text_to_image")`.
- `env -u OPENAI_API_KEY PYTHONPATH=src python -m unittest tests.test_openai_compatible_backend.TestOpenAICompatibleVisionBackend.test_list_provider_models_default_openai_catalog_live -q` passed with `skipped=1`.
- `PYTHONPATH=src python -m unittest tests.test_openai_compatible_backend tests.test_manager_capability_checks tests.test_cli_smoke -q` passed, 28 tests.
- `PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -q` passed, 86 tests after the live provider catalog regression was added.
- `python -m ruff check --ignore UP src/abstractvision/integrations/abstractcore_plugin.py tests/test_abstractcore_plugin.py` passed.
- `python -m ruff check --ignore UP src/abstractvision/types.py src/abstractvision/backends/base_backend.py src/abstractvision/backends/openai_compatible.py src/abstractvision/vision_manager.py src/abstractvision/cli.py tests/test_openai_compatible_backend.py tests/test_cli_smoke.py tests/test_manager_capability_checks.py` passed.
- `mkdocs build -q` passed. MkDocs Material emitted its upstream MkDocs 2.0 compatibility warning.

### Residual Risks

- Gateway capability discovery still needs a downstream Gateway/Core-side test because this
  package should not own Gateway readiness policy.
- Provider catalog listing is explicit and best-effort. It can query OpenAI/OpenAI-compatible
  `GET /models`, but newer models should still be selected explicitly through `vision_model_id`,
  `ABSTRACTVISION_MODEL_ID`, `OPENAI_IMAGE_MODEL_ID`, or `OPENAI_IMAGE_MODEL`.
