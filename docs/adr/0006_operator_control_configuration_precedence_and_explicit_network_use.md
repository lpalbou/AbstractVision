# ADR 0006: Keep runtime selection explicit and operator-controlled

Status: Accepted.

## Context

AbstractVision can be driven through Python, CLI/REPL, playground, and AbstractCore. Those surfaces
all combine defaults, environment variables, explicit arguments, and backend-specific behavior. The
package also has multiple paths that could silently change operator intent if left unchecked:

- runtime backend selection;
- provider model ids;
- cache-only versus download-enabled local behavior;
- provider catalog inspection versus actual model selection;
- curated bundle resolution for component-based local models.

The current code already has good instincts here:

- local Diffusers and MFLUX default to cache-only behavior;
- provider catalogs are inspection-only;
- AbstractCore owner config is read before env vars;
- curated `sdcpp` keys resolve cached companion files and fail clearly when they are missing.

That behavior needs to become durable policy.

## Decision

AbstractVision keeps runtime-affecting choices explicit and operator-controlled.

The rules are:

1. Caller-supplied config for a surface takes precedence over ambient environment variables, and
   ambient environment variables take precedence over baked defaults.
2. In the AbstractCore plugin, `owner.config` is authoritative over `ABSTRACTVISION_*` and related
   environment aliases.
3. Provider model catalogs are explicit inspection only. AbstractVision must not silently select or
   switch image models based on `/models` responses.
4. Local runtime downloads are opt-in. Cache-backed or pre-downloaded behavior remains the default
   unless the operator explicitly enables downloads or runs `download-model`.
5. Curated model keys are allowed for local flows when the package can resolve them to explicit
   cached artifacts. Missing required artifacts must fail early with an actionable message.
6. Legacy aliases and compatibility shims may exist, but they must remain documented and auditable.
7. No surface should silently change backend, network, or download behavior in a way that surprises
   an operator who did not explicitly ask for it.

## Consequences

### Positive

- Operators can reason about backend selection and network use more reliably.
- Cache-only local workflows stay predictable.
- Curated local flows become simpler without hiding what files are actually being used.

### Negative

- Some convenience shortcuts remain intentionally unavailable if they would hide important operator
  choices.
- Config docs and tests must keep pace with precedence rules.

### Neutral

- This ADR still allows smart defaults, as long as those defaults do not silently perform material
  new work such as downloading weights or switching to a different provider model.

## Enforcement

- Reviewers should reject new automatic model-selection behavior driven only by provider catalogs.
- Reviewers should reject runtime downloads enabled by default for local backends unless a future
  ADR changes that policy.
- Config precedence changes must update the docs and tests in the same pass.
- Error messages for missing curated local components must stay actionable and point back to the
  supported download or configuration path.

## Validation

- Keep CLI, playground, and AbstractCore tests passing for config precedence and cache-only local
  behavior.
- Verify provider model listing remains inspection-only.
- Verify curated local key resolution either resolves explicit cached files or fails with a precise
  remediation hint.

## Backlog links

- [docs/backlog/completed/014_lightweight_openai_compatible_packaging.md](../backlog/completed/014_lightweight_openai_compatible_packaging.md)
- [docs/backlog/completed/015_vision_install_profiles_and_pending_defaults.md](../backlog/completed/015_vision_install_profiles_and_pending_defaults.md)
- [docs/backlog/completed/016_abstractcore_plugin_catalog_discovery_surface.md](../backlog/completed/016_abstractcore_plugin_catalog_discovery_surface.md)
- [docs/backlog/completed/018_capability_residency_hooks.md](../backlog/completed/018_capability_residency_hooks.md)

## Related

- [docs/reference/configuration.md](../reference/configuration.md)
- [docs/reference/abstractcore-integration.md](../reference/abstractcore-integration.md)
- [src/abstractvision/cli.py](../../src/abstractvision/cli.py)
- [src/abstractvision/playground_server.py](../../src/abstractvision/playground_server.py)
- [src/abstractvision/integrations/abstractcore_plugin.py](../../src/abstractvision/integrations/abstractcore_plugin.py)
