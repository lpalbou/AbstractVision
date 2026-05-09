# Changelog

## Unreleased

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
- OpenAI configuration: add standard `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_IMAGE_MODEL_ID`, and `OPENAI_IMAGE_MODEL` aliases for the plugin default path; keep compatible endpoints explicit with `ABSTRACTVISION_BACKEND=openai-compatible` and `ABSTRACTVISION_BASE_URL`.
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
- AbstractCore plugin: keep OpenAI-compatible usage available with `ABSTRACTVISION_BACKEND=openai` plus `ABSTRACTVISION_BASE_URL`, and preserve artifact-store behavior for generated media outputs.

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
