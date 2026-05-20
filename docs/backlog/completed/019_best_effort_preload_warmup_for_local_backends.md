# Completed: Improve best-effort preload warmup for local image backends

## Metadata
- Created: 2026-05-20
- Status: Completed
- Completed: 2026-05-20
- Priority: P1

## Main goals

- Make `preload()` execute a representative first-inference warmup for in-process local backends where that work can persist.
- Keep the public semantics simple: `preload()` remains a best-effort eager warmup, not a new explicit "true warm" contract.

## Secondary goals

- Avoid paying the extra warmup cost more than once for the same loaded model or pipeline state.
- Add deterministic tests that prove `preload()` shifts real generation work earlier for supported engines.

## Context / problem

Current local warmup is uneven across engines:

- `src/abstractvision/backends/mflux.py`: `preload()` only reaches `_ensure_model_impl()`, while the real first-request path still runs later in `_generate_impl()`.
- `src/abstractvision/backends/huggingface_diffusers.py`: `preload()` only loads the common `t2i` pipeline and does not execute a representative pipeline call.
- `src/abstractvision/backends/stable_diffusion_cpp.py`: `preload()` only constructs the python-binding model eagerly; CLI mode shells out a fresh subprocess per request, so no persistent package-local warmup exists there.

This means AbstractVision is already decent at "load resident weights", but it is not yet as good as it could be at "make the next real request as warm as possible" for the in-process local engines.

Expected first-matching-request upside from code inspection:

- `mflux`: likely the largest win, roughly `20-50%` on the first matching request when compile/init work is moved into `preload()`.
- `diffusers`: smaller but still useful, roughly `5-15%` typical for long-lived workers.
- `stable-diffusion.cpp` python mode: modest, roughly `5-15%`.

These are engineering estimates, not measured guarantees. CI/unit tests should confirm that the real generation path is exercised during `preload()`, but they will not prove exact wall-clock gains on real weights.

## Constraints

- Keep the public backend contract stable; do not add a new user-facing warmup API for this task.
- Keep `preload()` best-effort and backend-owned.
- Do not pretend CLI mode or remote HTTP backends can be persistently warmed when the process model does not support it.
- Keep tests lightweight and deterministic with fakes/mocks rather than real model downloads.
- Preserve existing generation behavior and current routing/residency semantics.

## Research, options, and references

- **Option A**: keep load-only `preload()` behavior.
  - Trade-off: zero implementation risk, but it leaves the known MFLUX gap and smaller Diffusers/sdcpp gaps untouched.
- **Option B**: introduce a separate public warmup API or explicit "warmness" metadata first.
  - Trade-off: more semantic clarity, but the user goal here is practical warmup improvement, not a larger public API change.
- **Option C**: improve existing `preload()` so supported in-process backends perform one representative discarded generation and remember they are already warmed for the current loaded state.
  - Trade-off: slightly higher preload cost, but it keeps the public surface simple and moves more cold-start work out of the first user request.

## Decision

**Chosen approach**: implement Option C.

**Why**:

- `preload()` already exists specifically to do eager preparation work.
- The strongest gap is behavioral, not semantic: the code is not yet pushing enough real first-inference work into preload.
- A backend-local, idempotent warmup step is the smallest change that can improve the actual user-visible latency profile.

## Dependencies

- **Completed backlog**:
  - `docs/backlog/completed/007_local_hf_backend_strategy_diffusers.md`
  - `docs/backlog/completed/013_stable_diffusion_cpp_gguf_backend.md`
  - `docs/backlog/completed/018_capability_residency_hooks.md`
- **Key code paths**:
  - `src/abstractvision/backends/base_backend.py`
  - `src/abstractvision/backends/mflux.py`
  - `src/abstractvision/backends/huggingface_diffusers.py`
  - `src/abstractvision/backends/stable_diffusion_cpp.py`
  - `tests/test_mflux_backend.py`
  - `tests/test_huggingface_diffusers_backend.py`
  - `tests/test_stable_diffusion_cpp_backend.py`

## Implementation plan

- Add backend-local warmup bookkeeping so repeated `preload()` calls do not rerun the same warmup unnecessarily.
- For `mflux`, after model construction, run one minimal valid representative generation on the runtime thread and discard the output.
- For `diffusers`, keep the current `t2i` focus but run one normalized representative pipeline call and discard the output.
- For `stable-diffusion.cpp` python mode, start by testing representative warmup, but keep the backend free to stop at eager model construction if measurement shows full hidden generation is not worth the cost; leave CLI mode unchanged.
- Add tests that prove `preload()` executes the real generate path once and that the first subsequent user request reuses the already-warmed state.

## Success criteria

- `preload()` on `mflux` executes a representative generate path, not just model construction.
- `preload()` on `diffusers` executes a representative `t2i` pipeline call, not just pipeline loading.
- `preload()` on `stable-diffusion.cpp` python mode performs the most cost-effective eager preparation that survives into the next request; CLI mode remains load-only/no-op.
- Repeated `preload()` calls for the same loaded state do not rerun warmup.
- Existing generation/edit behavior remains intact and the relevant backend tests pass.

## Test plan

- `PYTHONPATH=src python -m unittest tests.test_playground_server -q`
- `PYTHONPATH=src python -m unittest tests.test_mflux_backend -q`
- `PYTHONPATH=src python -m unittest tests.test_huggingface_diffusers_backend -q`
- `PYTHONPATH=src python -m unittest tests.test_stable_diffusion_cpp_backend -q`
- `PYTHONPATH=src python -m unittest tests.test_playground_server tests.test_mflux_backend tests.test_huggingface_diffusers_backend tests.test_stable_diffusion_cpp_backend -q`

## Completion Report

Date: 2026-05-20

### Summary

- `mflux` `preload()` now executes one real deterministic text-to-image warmup on the existing runtime thread and remembers warm state by `_model_key`, so repeated preload calls do not rerun the same work.
- `huggingface_diffusers` `preload()` still targets the common `t2i` pipeline, but it now runs a real warmup inference once per loaded pipeline object and serializes preload/generate/edit behind a backend-local `RLock`.
- `stable_diffusion_cpp` python mode was initially implemented with a real warmup generate during `preload()`, but follow-up measurement showed that eager model construction captures the useful gain more cleanly. The final behavior is load-only `preload()` for python mode; CLI mode remains unchanged because it has no persistent in-process state to warm.
- `PlaygroundState.load_model()` now preloads the replacement backend before swapping active state, so a warmup failure no longer drops the previously active model.

### Files And Symbols Touched

- `src/abstractvision/backends/mflux.py`
  - `MFluxVisionBackend.preload()`
  - `MFluxVisionBackend._preload_impl()`
  - `MFluxVisionBackend._warmup_request()`
  - `MFluxVisionBackend._generate_impl()`
- `src/abstractvision/backends/huggingface_diffusers.py`
  - `HuggingFaceDiffusersVisionBackend.preload()`
  - `HuggingFaceDiffusersVisionBackend.unload()`
  - `HuggingFaceDiffusersVisionBackend.generate_image_with_progress()`
  - `HuggingFaceDiffusersVisionBackend.edit_image_with_progress()`
  - `HuggingFaceDiffusersVisionBackend._set_pipeline()`
  - `HuggingFaceDiffusersVisionBackend._warmup_generation_request()`
- `src/abstractvision/backends/stable_diffusion_cpp.py`
  - `StableDiffusionCppVisionBackend.preload()`
  - `StableDiffusionCppVisionBackend._generate_image_python()`
  - `StableDiffusionCppVisionBackend.generate_image_with_progress()`
- `src/abstractvision/playground_server.py`
  - `PlaygroundState.load_model()`
- `tests/test_mflux_backend.py`
- `tests/test_huggingface_diffusers_backend.py`
- `tests/test_stable_diffusion_cpp_backend.py`
- `tests/test_playground_server.py`

### Validation

- `git diff --check`
- `PYTHONPATH=src python -m unittest tests.test_playground_server -q`
- `PYTHONPATH=src python -m unittest tests.test_mflux_backend -q`
- `PYTHONPATH=src python -m unittest tests.test_huggingface_diffusers_backend -q`
- `PYTHONPATH=src python -m unittest tests.test_stable_diffusion_cpp_backend -q`
- `PYTHONPATH=src python -m unittest tests.test_playground_server tests.test_mflux_backend tests.test_huggingface_diffusers_backend tests.test_stable_diffusion_cpp_backend -q`
  - Result: `Ran 61 tests in 5.524s` and `OK`
- After installing the missing `mflux` runtime for empirical measurement:
  - `PYTHONPATH=src python -m unittest tests.test_playground_server tests.test_mflux_backend tests.test_huggingface_diffusers_backend tests.test_stable_diffusion_cpp_backend -q`
  - Result: `Ran 61 tests in 6.462s` and `OK`

### Measured Benchmark Examples

Measured on this machine:

- `macOS arm64` with `MPS`
- local cached model weights
- fresh subprocess per sample to preserve honest cold-start behavior
- two cold runs and two preload runs per benchmark

Benchmarked configurations:

- `mflux`: `flux2-klein-4b` with `default_width=512`, `default_height=512`, request `seed=1`
- `diffusers`: `black-forest-labs/FLUX.2-klein-4B` on `device='mps'`, request `steps=1`, `seed=1`

Measured medians:

- `mflux`
  - cold first request: `6.793s`
  - preload call: `6.341s`
  - first request after preload: `3.964s`
  - first-request latency improvement after preload: `41.6%`
- `diffusers`
  - cold first request: `16.836s`
  - preload call: `16.618s`
  - first request after preload: `6.787s`
  - first-request latency improvement after preload: `59.7%`
- `stable-diffusion.cpp` python mode
  - model stack: `leejet/FLUX.2-klein-base-4B-GGUF` (`Q8_0`) + official `black-forest-labs/FLUX.2-klein-base-4B` VAE + `unsloth/Qwen3-4B-GGUF` (`Q4_K_M`)
  - backend/runtime: `stable-diffusion-cpp-python 0.4.5`, CPU backend on this machine
  - benchmark shape: `256x256`, `steps=1`, Euler, fresh subprocesses
  - final shipped `preload()` median: `2.432s`
  - cold first request median: `27.321s`
  - first request after `preload()` median: `24.753s`
  - first-request latency improvement after `preload()`: `9.4%`
  - deeper comparison showed that this gain comes from eager model construction; an experimental hidden full-warmup variant was worse operationally at about `49.511s` preload and `25.723s` first request after preload

Interpretation:

- `preload()` does not reduce total `preload + first request` wall time; it shifts expensive one-time work out of the first user-visible request.
- `mflux` now lands in the expected “large win” class and materially validates the implementation direction.
- `diffusers` improved more than originally estimated on this machine and model, which suggests this backend had more first-inference one-time work than the earlier code-only estimate implied.
- `stable-diffusion.cpp` python mode now has measured validation too, but the gain is modest and comes almost entirely from eager model construction. A hidden full warmup generation made `preload()` much more expensive without improving the next request enough to justify it.

### Outcome Against Estimates

- `mflux`: the implementation now warms the real in-process generate path using backend-default dimensions and model-default steps/guidance, which is the strongest practical move available in the current architecture.
- `diffusers`: the implementation now warms one real `t2i` pipeline call using backend/model defaults, which should reduce first-request latency for long-lived workers without expanding the public API.
- `stable-diffusion.cpp` python mode: follow-up measurement showed the useful benefit is eager model construction, not hidden representative generation. The final implementation keeps load-only `preload()` for python mode and still shows a measured `9.4%` first-request reduction on a real FLUX.2 component stack; CLI mode remains correctly untouched.

### Residual Risks And Limits

- The unit tests prove the current backend-specific preload behavior and that repeated preload calls remain safe. Real benchmark examples now exist for `mflux`, `diffusers`, and `stable-diffusion.cpp` python mode on this machine.
- `huggingface_diffusers` warmup intentionally targets only the common `t2i` path. It does not separately warm `i2i`, `inpaint`, LoRA-specific, or Rapid-AIO-specific variants during preload.
- Warmup remains best-effort. It improves the first matching request, but it is not a promise that every later request shape or backend-specific variant is fully hot.
- The `stable-diffusion.cpp` number above is for python-binding CPU mode. CLI mode still has no meaningful persistent warm state in the current architecture, so the benchmark does not change the earlier conclusion that CLI warmup remains effectively unsupported.
- The `stable-diffusion.cpp` benchmark also used an official Black Forest Labs FLUX.2 VAE side artifact rather than a Comfy-hosted VAE, which confirms that this FLUX.2 component stack does not require a Comfy-specific VAE source.
- Installing `mflux` for measurement upgraded the local Python environment's `numpy` and `pillow` versions to satisfy `mflux` runtime requirements. AbstractVision's tested slice still passed afterward, but the shared interpreter now reports dependency conflicts with `digital-article-backend`.

### Post-Completion Insights

- Once `preload()` does real inference work, caller rollback behavior matters. The playground fix was necessary and suggests other future preload callers should be reviewed with the same assumption.
- Public "true warmness" semantics are still not required for this task. The current implementation is valuable even while `preload()` remains an intentionally best-effort contract.
