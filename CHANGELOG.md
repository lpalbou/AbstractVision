# Changelog

## 0.3.11 - 2026-05-22

- Hugging Face cache integrity: treat repo-level `.incomplete` blobs as an incomplete snapshot, so interrupted Diffusers downloads no longer masquerade as a usable cached model.
- Qwen Image Edit downloads: `abstractvision download qwen-image-edit-2511 --provider diffusers` now resumes the real 16-bit snapshot instead of incorrectly short-circuiting on a partial cache.

## 0.3.10 - 2026-05-22

- Catalog/handles: make the curated Qwen Image Edit aliases consistent across the registry, downloads, and CLI catalog surfaces (`qwen-image-edit`, `qwen-image-edit-2509`, `qwen-image-edit-2511`), and re-verify all packaged Hugging Face repo ids in `vision_model_capabilities.json`.
- Runtime truth: quarantine unreliable local runtime task paths until they are re-validated. Local Diffusers `GLM-Image` is now hidden/disabled, local MFLUX is surfaced for `text_to_image` only, and local Diffusers `text_to_video` is marked experimental and disabled from the normal local surfaces.
- Playground/catalog UX: keep the per-task playground tabs with per-tab load/unload controls, but filter task selectors through backend-supported runtime truth so disabled local tasks do not surface. Update the `download` command messaging so curated 16-bit Diffusers snapshots are reported honestly instead of looking like a failed 8-bit fallback.
- Docs/backlog/tests: add the planned quarantine follow-up item, refresh the core docs around current local support policy, expand regression coverage for the runtime blacklists and catalog filtering, and regenerate `llms-full.txt`.

## 0.3.9 - 2026-05-21

- Local Diffusers video: implement the first in-process `text_to_video` path through `zai-org/CogVideoX-2b` / `THUDM/CogVideoX-2b`, including registry-backed request normalization, preload warmup, unload support, MPS-safe explicit FP16 device moves, and MP4 artifact outputs packaged through `ffmpeg`.
- Interactive surfaces: add one-shot `abstractvision t2v`, REPL `/t2v`, and playground `Text → Video` generation with backend/task-aware model discovery, progress reporting, and catalog surfacing for local video-capable models.
- AbstractCore integration: route plugin `t2v` / `i2v` calls through the same explicit backend binding path used by image generation, extend residency task aliases for video, and track request-warm local `text_to_video` models in the loaded-model inventory.
- Catalog/tests/docs: add curated capability/download metadata for `zai-org/CogVideoX-2b`, keep local image-edit surfaces intact, expand backend/CLI/playground/plugin regression coverage, refresh README/reference/FAQ/getting-started docs, and regenerate `llms-full.txt`.

## 0.3.8 - 2026-05-20

- Model registry/catalog: expand the curated `vision_model_capabilities.json` inventory with more current official or reputable vision model families and cross-platform download variants, tighten provider provenance, remove adapter-only repos from the standalone model catalog, and expose richer model notes / `task_specs` through the CLI catalog surfaces.
- Apple/MLX semantics: treat `mlx` as a download target rather than a runnable backend, fail fast on generic `mlx` provider/model routing across CLI, playground, and AbstractCore, and limit shipped MFLUX presets/docs to the families the backend actually supports today.
- GGUF / local-runtime routing: improve stable-diffusion.cpp bundle resolution for curated multi-artifact models and legacy Qwen GGUF repo ids, resolve curated `sdcpp` model keys consistently in both one-shot CLI and REPL paths, and make AbstractCore request-scoped `sdcpp` routing explicit while canonicalizing MFLUX aliases for backend reuse.
- Docs/policy/tests: add the ADR set and planned backlog items for curated catalog ownership, adapter-aware model graphs, and the long-term native MLX direction; refresh README/reference/FAQ/agent docs; regenerate `llms-full.txt`; and expand regression coverage for catalog aliases, GLM defaults, MFLUX routing, and local backend bundle selection.

## 0.3.7 - 2026-05-19

- AbstractCore plugin: add explicit local model residency control for in-process `diffusers`, `mflux`, and `sdcpp` backends through `load_resident_model(...)`, `list_loaded_models(...)`, `list_resident_models(...)`, and `unload_resident_model(...)`, with `load_model(...)` / `unload_model(...)` compatibility aliases for Core route adapters.
- Residency semantics: preserve explicit resident models across model switches while keeping the previous unload-on-switch behavior for non-resident request-warm backends, and report stable process-local `load_id` / `backend_kind` / `resident` metadata instead of relying on backend private fields.
- Safety/robustness: reject OpenAI/OpenAI-compatible HTTP backends for residency control even on localhost, normalize provider/task filters consistently, reject ambiguous unload requests, and keep unload behavior deterministic for injected local backends.
- Tests/docs: expand AbstractCore plugin coverage for local preload/list/unload flows, request-warm loaded-state reporting, task-aware loaded-model filters, injected backend residency, and switch-survival behavior; document the new local residency control surface and regenerate `llms-full.txt`.

## 0.3.6 - 2026-05-18

- Model registry/downloads: expand `vision_model_capabilities.json` and curated preset coverage for current Hugging Face text-to-image and image-to-image models, including FLUX.1/2, Qwen Image, ERNIE Image, GLM Image, Z-Image, SDXL, SD Turbo, and SD3.5. The registry now remains the packaged source of truth for both downloadable artifacts and task capability metadata.
- Cache/catalog: keep curated downloads in the Hugging Face cache by default, migrate older `~/models/<preset>` trees into that cache on first use, and align Diffusers/MFLUX cache discovery across CLI, REPL, playground, and AbstractCore so cached local models surface consistently.
- API-level normalization: move model-specific parameter normalization into shared backend hooks used by `VisionManager` and the playground server. Model constraints such as MFLUX distilled `guidance_scale=1.0`, unsupported negative prompts, and GLM `steps/guidance_scale/32-multiple dimensions` now apply consistently through the CLI/REPL, playground, and AbstractCore integration.
- Playground/API: expose structured per-model `task_specs` from `/v1/vision/models`, only enable the edit surface for models that advertise `image_to_image`, switch models by auto-loading the new selection, and truncate `b64_json` in logs instead of dumping the full base64 payload.
- AbstractCore/provider catalogs: keep remote provider/model listing robust when internet or API credentials are unavailable, and keep local provider/model catalogs aligned with the same cache-backed discovery used by the interactive surfaces.
- CLI/docs: keep `abstractvision cli` as the canonical interactive command (with `repl` as a legacy alias), expand model catalog/docs coverage, refresh backend/configuration/AbstractCore/playground references, and regenerate `llms.txt` / `llms-full.txt` for release.

## 0.3.5 - 2026-05-13

- AbstractCore plugin: accept runtime output metadata such as `provider` and `size` without rejecting image generation requests, and forward per-call image model selectors to OpenAI-compatible backends.


## 0.3.4 - 2026-05-09

- Packaging: constrain the optional `stable-diffusion-cpp-python` binding to
  `<0.4.6` for `sdcpp`, `local`, `apple`, `all`, `all-apple`, and `all-gpu`
  extras. Version `0.4.6` currently ships an sdist that fails native Linux
  builds because required vendored libwebm CMake files are missing.

## 0.3.3 - 2026-05-08

- Packaging: add shared native install profiles. `abstractvision[apple]` and
  `abstractvision[all-apple]` install the full local macOS vision stack;
  `abstractvision[gpu]` installs the Diffusers/Torch GPU stack; and
  `abstractvision[all-gpu]` installs Diffusers plus stable-diffusion.cpp
  bindings for full local GPU-capable deployments.
- Docs/tests: document and verify the Apple/GPU profile aliases while keeping
  the base package lightweight and remote/OpenAI-compatible by default.

## 0.3.2 - 2026-05-08

- AbstractCore plugin: switch the default backend id to `abstractvision:openai` with official OpenAI defaults, while keeping `abstractvision:openai-compatible` registered as a legacy-compatible backend id.
- OpenAI configuration: use standard `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_IMAGE_MODEL_ID`, and `OPENAI_IMAGE_MODEL` aliases for the plugin default path; keep compatible endpoints explicit with `ABSTRACTVISION_BACKEND=openai-compatible` and `OPENAI_BASE_URL`.
- Provider catalogs: add OpenAI/OpenAI-compatible `/models` discovery through `VisionManager.list_provider_models(...)`, `abstractvision provider-models`, REPL `/provider-models`, and `llm.vision.list_provider_models(...)` in the AbstractCore plugin.
- API/contracts: add `ProviderModelInfo`, provider catalog backend hooks, task filtering for OpenAI image model catalogs, and JSON-safe bounded provider metadata at the AbstractCore plugin boundary.
- Docs/agent context: refresh AbstractCore integration, backend/configuration/API docs, playground production-boundary notes, completed backlog reports, `llms.txt`, and regenerated `llms-full.txt`.
- Tests: add coverage for OpenAI defaults, legacy compatible aliases, provider catalog parsing/filtering, plugin catalog exposure, CLI/REPL catalog commands, manager delegation, and a credential-gated live OpenAI catalog check.

## 0.3.1 - 2026-05-07

- CI/release: move GitHub Actions checkout/setup-python/artifact/release actions to Node 24-compatible major versions, removing the Node 20 deprecation warnings from release runs.
- Tests: strengthen packaging metadata coverage so Diffusers aliases and `local`/`all` runtime bundles cannot drift from their intended dependency sets.
- Docs: separate contributor-only extras from runtime install profiles and explicitly mark `dev` as unsuitable for application/runtime dependency declarations.

## 0.3.0 - 2026-05-07

- Packaging: make the base install lightweight. `pip install abstractvision` no longer installs Torch, Diffusers, Transformers, Pillow, or local inference runtimes by default.
- Extras: add canonical runtime profiles `abstractvision[openai]`, `abstractvision[openai-compatible]`, `abstractvision[diffusers]`, `abstractvision[sdcpp]`, `abstractvision[local]`, and `abstractvision[all]`; keep `huggingface`/`huggingface-dev` compatibility aliases.
- Runtime defaults: keep AbstractCore plugin and one-shot CLI remote-first, and stop the REPL/playground from silently selecting Diffusers when no backend is configured.
- Errors/tests/CI: improve local-backend missing-extra hints, add stronger packaging/import-light/OpenAI-compatible coverage, and split CI into lightweight base and local Diffusers paths.
- Docs: update install guidance, backend references, and internal backlog policy for explicit local runtime extras.

## 0.2.6 - 2026-05-06

- Docs: refresh install extras, AbstractCore integration, playground ownership, OpenAI-compatible request-shape notes, and backend/config references so the public docs match the current code.
- Agent docs: update `llms.txt`, include playground endpoint docs in the generated `llms-full.txt` bundle, and regenerate the AI-ready documentation from current sources.
- Contributing: clarify that `abstractvision[abstractcore]` is a compatibility marker and that AbstractCore is supplied by the host application.

## 0.2.5 - 2026-05-06

- Packaging: keep the default Diffusers backend installable while moving `stable-diffusion-cpp-python` out of the base dependency set and into the explicit `sdcpp`/`local` extras. This keeps AbstractCore plugin installs from failing on platforms where stable-diffusion.cpp bindings need a local native build.
- AbstractCore plugin: restore the OpenAI-compatible HTTP backend as the default while keeping local `diffusers` and `sdcpp` backends explicit through config/env.
- OpenAI-compatible backend: shape requests correctly for real OpenAI GPT image models while preserving local OpenAI-compatible extensions for unknown model ids.
- Playground: capture the active backend at job submission time so background jobs do not accidentally run on a newly selected model.
- Docs/tests: clarify the default install shape, document when to install `abstractvision[sdcpp]`, and add metadata/OpenAI/playground coverage so these paths do not regress.

## 0.2.4 - 2026-05-06

- Playground: add a self-contained `abstractvision playground` command that serves both the web UI and `/v1/vision/*` API locally, so playground testing no longer depends on an AbstractCore server.
- Playground: package the HTML asset in the wheel, default the UI to the serving origin, and avoid stale persisted API URLs that could keep calling an older AbstractCore endpoint.
- Playground model loading: accept raw Hugging Face ids such as `runwayml/stable-diffusion-v1-5` directly, while still accepting explicit backend prefixes like `diffusers/...`, `sdcpp/...`, and `openai-compatible/...`.
- Packaging/CI: keep AbstractCore out of AbstractVision dependency metadata and test workflows; AbstractCore remains an optional host integration loaded lazily when present.
- Docs/tests: refresh playground docs around the self-contained local server and add coverage for cached model listing, raw model loading, playground jobs, and tool integration without installing AbstractCore.

## 0.2.3 - 2026-05-06

- AbstractCore plugin: support local Diffusers and stable-diffusion.cpp backends through `llm.vision`, not only OpenAI-compatible HTTP. The default plugin path now matches the REPL default: local Diffusers with `runwayml/stable-diffusion-v1-5`, cache-only unless `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1` is set.
- AbstractCore plugin: keep OpenAI-compatible usage available with `ABSTRACTVISION_BACKEND=openai` plus `OPENAI_BASE_URL`, and preserve artifact-store behavior for generated media outputs.

## 0.2.2 - 2026-05-06

- Release automation: add GitHub Actions CI/release workflows, issue templates, pre-commit config, MkDocs config, PyPI trusted publishing, GitHub Releases, and release-time docs deployment to `gh-pages`.
- Packaging: support Python 3.9-3.13, modernize license metadata, include package data explicitly, and add test/docs/dev extras.
- Defaults: make the REPL default to local Diffusers with `runwayml/stable-diffusion-v1-5`, `ABSTRACTVISION_DIFFUSERS_DEVICE=auto`, and cache-only/offline runtime downloads disabled by default.
- Diffusers backend: keep `allow_download=False` as the default, force Hugging Face offline/cache-only env during loads/calls, disable implicit HF token use offline, and load from cached snapshot paths when present.
- Diffusers backend: ignore unknown REPL `extra` flags that the pipeline `__call__` does not accept, avoiding `unexpected keyword argument` crashes.
- Diffusers backend: add better fp16 variant fallback behavior, MPS dtype/invalid-output retry handling, LoRA/Rapid-AIO offline handling, and clearer missing-local-model errors.
- Capability registry: add `black-forest-labs/FLUX.2-klein-9B` and normalize FLUX license id to `flux-non-commercial-license`.
- Docs: refresh quickstarts around Stable Diffusion 1.5 first, add clearer macOS Metal/NVIDIA CUDA/CPU guidance, expand stable-diffusion.cpp notes, and point users to cache-only local workflows.
- Tooling: add `scripts/download_model_sets.py` for explicit heavyweight model downloads (Stable Diffusion 1.5, FLUX 2 GGUF/Diffusers, and Qwen Image snapshots).
- Cleanup: remove the misspelled duplicate `ACKNOWLEDMENTS.md`.

## 0.2.1

- Documentation refresh for public release:
  - add `docs/api.md` and strengthen cross-linking between README and docs
  - add `CONTRIBUTING.md`, `SECURITY.md`, and `ACKNOWLEDGMENTS.md`
  - add `llms.txt` and generated `llms-full.txt` for agent-oriented context
  - clarify playground/server endpoint expectations (`/v1/vision/*`)

## 0.2.0

- Add stable-diffusion.cpp (`sd-cli`) backend for local GGUF diffusion models.
- REPL: forward unknown `--flags` as backend `extra` parameters.
- Add a tiny web playground (`playground/vision_playground.html`) for testing via AbstractCore Server vision endpoints (`/v1/vision/*`).

## 0.1.0

- Initial MVP: capability registry + schema validation.
- Artifact-first outputs via `LocalAssetStore` and runtime adapter.
- OpenAI-compatible HTTP backend for image generation/editing (optional video endpoints via config).
- Local Diffusers backend for images (opt-in deps).
- AbstractCore tool integration (`make_vision_tools`) with artifact refs.
- CLI/REPL for interactive manual testing.
