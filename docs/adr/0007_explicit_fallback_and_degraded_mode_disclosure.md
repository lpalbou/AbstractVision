# ADR 0007: Disclose fallbacks and degraded modes explicitly

Status: Accepted.

## Context

AbstractVision has several intentional fallback or degraded-mode paths:

- curated non-8-bit fallback presets when no 8-bit artifact exists;
- community runtime-native artifacts when no official equivalent exists;
- Diffusers dtype retry behavior;
- `sdcpp` python-binding fallback when `sd-cli` is absent;
- explicit repo-id snapshot fallback outside the curated preset set;
- best-effort preload behavior whose limits differ by backend.

These paths are sometimes the right engineering choice, but they become dangerous when they are
silent. Silent fallback changes trust: the user may think they are using a curated 8-bit path, a
GPU runtime, a specific source artifact, or a truly warmed backend when they are not.

The repo already uses visible `#FALLBACK` markers and many precise error messages. That should be a
policy, not just a habit.

## Decision

AbstractVision must surface fallbacks and degraded modes explicitly whenever they materially change
provenance, quality, performance, or runtime behavior.

The rules are:

1. User-facing fallback catalog entries must carry explicit provenance and limitation notes.
2. Runtime fallback that changes material behavior must be documented and, where practical,
   surfaced in logs, output, or error messages.
3. Correctness-critical unsupported states must fail closed with a clear error instead of silently
   picking a different behavior.
4. Best-effort features must be described with their real limits. Do not market “loaded” as
   “warmed” or “supported in the registry” as “implemented by the current backend.”
5. Searchable markers such as `#FALLBACK` are acceptable and encouraged for intentional user-facing
   fallback sites in code and docs.

## Consequences

### Positive

- Operator trust improves because important fallback behavior is visible.
- Reviewers and future contributors can audit fallback sites more easily.
- Performance and compatibility claims stay more honest.

### Negative

- Some messages and docs become more verbose because they need to explain degraded behavior.
- Fallback paths need deliberate wording and maintenance.

### Neutral

- This ADR does not forbid all fallback behavior.
- It does forbid silent fallback that hides a material contract change.

## Enforcement

- Reviewers should reject silent fallback on correctness-critical paths.
- New fallback sites that materially affect the user-facing contract should carry a visible marker,
  message, or note.
- Catalog and docs should continue to label non-8-bit, gated, or community-source fallback entries
  clearly.

## Validation

- Keep tests passing for explicit failure on unsupported backend/task combinations and missing local
  component files.
- Verify representative catalog output still labels fallback entries clearly.
- When a new degraded mode is added, add or update at least one test or doc example that proves the
  degraded behavior is surfaced.

## Backlog links

- [docs/backlog/completed/014_lightweight_openai_compatible_packaging.md](../backlog/completed/014_lightweight_openai_compatible_packaging.md)
- [docs/backlog/completed/015_vision_install_profiles_and_pending_defaults.md](../backlog/completed/015_vision_install_profiles_and_pending_defaults.md)
- [docs/backlog/completed/019_best_effort_preload_warmup_for_local_backends.md](../backlog/completed/019_best_effort_preload_warmup_for_local_backends.md)

## Related

- [README.md](../../README.md)
- [docs/getting-started.md](../getting-started.md)
- [src/abstractvision/model_downloads.py](../../src/abstractvision/model_downloads.py)
- [src/abstractvision/backends/huggingface_diffusers.py](../../src/abstractvision/backends/huggingface_diffusers.py)
- [src/abstractvision/backends/stable_diffusion_cpp.py](../../src/abstractvision/backends/stable_diffusion_cpp.py)
