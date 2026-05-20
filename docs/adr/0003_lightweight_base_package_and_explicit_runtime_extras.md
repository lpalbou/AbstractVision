# ADR 0003: Keep the base package lightweight and make local runtimes explicit

Status: Accepted.

## Context

`abstractvision` serves very different environments: remote OpenAI-compatible hosts, local CLI
users, playground users, and AbstractCore plugin hosts. Those environments should not all pay the
dependency and platform cost of local Torch, Diffusers, GGUF bindings, Apple-only MLX stacks, or
model download helpers.

The current code already reflects this pressure:

- base package dependencies are empty in [`pyproject.toml`](../../pyproject.toml);
- local runtimes live behind optional extras such as `diffusers`, `sdcpp`, `mflux`, and `models`;
- heavy backend modules are imported lazily;
- tests already guard import-light behavior for plugin registration and package import.

Without a durable rule, future work can quietly reintroduce heavy default dependencies, eager local
runtime imports, or package startup behavior that breaks lightweight remote deployments.

## Decision

AbstractVision keeps the base install lightweight and treats local inference stacks as explicit
operator choices.

The rules are:

1. `pip install abstractvision` must remain usable for the shared API, registry, artifacts, CLI,
   and OpenAI-compatible HTTP backend without installing local inference runtimes.
2. Heavy local runtimes and helpers stay behind explicit extras:
   - `diffusers` for Torch/Diffusers
   - `sdcpp` for `stable-diffusion-cpp-python`
   - `mflux` for Apple Silicon MLX/MFLUX
   - `models` for Hugging Face download helpers
3. Backend modules that depend on heavy optional packages must stay lazily imported.
4. AbstractCore integration must remain lazy. AbstractVision must not require AbstractCore to be
   installed just to import the base package.
5. New optional runtime families must follow the same pattern unless the user explicitly accepts a
   heavier base package design in a future ADR.

## Consequences

### Positive

- Remote-only and plugin-host installs stay small and portable.
- Local runtime failures stay opt-in instead of breaking every install.
- Packaging intent remains easy to explain in docs and tests.

### Negative

- Contributors must keep optional-import boundaries clean.
- Docs and errors must keep explaining which extra a user needs for a given local backend.

### Neutral

- Convenience extras such as `local`, `all`, and platform bundles may still exist.
- This ADR does not prevent stronger local-runtime integrations; it only requires them to stay
  explicit.

## Enforcement

- Reviewers should reject new unconditional heavy dependencies in `[project].dependencies`.
- Reviewers should reject eager imports of Diffusers, Torch, `stable_diffusion_cpp`, `mflux`, or
  similar local runtime modules from the base package import path.
- Contributor docs must continue to describe runtime extras explicitly.
- New local backends must document their extra and missing-dependency hint paths in the same change.

## Validation

- Keep import-light tests passing, especially:
  - package import without heavy backend imports
  - AbstractCore plugin registration without heavy runtime imports
- Keep packaging metadata aligned with the documented extras.
- Verify docs still describe the lightweight base install and explicit extras accurately.

## Backlog links

- [docs/backlog/completed/012_packaging_extras_and_release_hygiene.md](../backlog/completed/012_packaging_extras_and_release_hygiene.md)
- [docs/backlog/completed/014_lightweight_openai_compatible_packaging.md](../backlog/completed/014_lightweight_openai_compatible_packaging.md)
- [docs/backlog/completed/015_vision_install_profiles_and_pending_defaults.md](../backlog/completed/015_vision_install_profiles_and_pending_defaults.md)

## Related

- [README.md](../../README.md)
- [docs/getting-started.md](../getting-started.md)
- [docs/reference/backends.md](../reference/backends.md)
- [pyproject.toml](../../pyproject.toml)
- [tests/test_abstractcore_plugin.py](../../tests/test_abstractcore_plugin.py)
