# ADR 0009: Keep docs, backlog, and ADRs code-first

Status: Accepted.

## Context

This repo already relies heavily on durable written artifacts: README/docs, a structured backlog,
and now ADRs. Those artifacts are useful only if they follow the shipped code instead of replacing
it as the source of truth.

The repo’s recent workflow and the user’s own process expectations make the problem concrete:

- backlog items can stay in `proposed/` after code is already shipped;
- docs can lag behind runtime behavior or newly added helpers;
- architectural rules can disappear into chat or one-off implementation notes unless they are
  promoted into ADRs;
- future work becomes slower when contributors trust stale prose instead of inspecting code first.

## Decision

AbstractVision treats code as the operational source of truth and uses docs, backlog, and ADRs for
different supporting roles.

The rules are:

1. Code is the operational source of truth for shipped behavior.
2. ADRs record durable policy that should constrain future work.
3. `docs/backlog/` records planning state, implementation history, and completion reporting.
4. User-facing docs explain shipped behavior; they are not substitutes for code inspection.
5. Before writing or revising consequential backlog items, ADRs, or docs, contributors should
   inspect the relevant code first.
6. When code, docs, backlog, and ADRs drift, the drift must be fixed explicitly or recorded
   explicitly. Do not rely on chat history or assumed intent.

## Consequences

### Positive

- Repo knowledge becomes easier to trust and maintain.
- Durable process rules and task history stop getting mixed together.
- Completion reporting becomes easier to audit.

### Negative

- Contributors must do more upfront inspection before writing planning or policy text.
- Hygiene work remains necessary after implementation, especially when backlog state changes.

### Neutral

- This ADR does not prevent forward-looking backlog work; it only forbids treating backlog prose as
  the source of truth for what is already shipped.

## Enforcement

- Reviewers should reject backlog or doc claims that obviously conflict with the shipped code.
- When a completed backlog item establishes or changes a durable rule, the same pass should either
  update an ADR or state why no ADR is needed.
- Moving backlog items between `proposed`, `planned`, and `completed` should keep filenames,
  numbering, and completion notes consistent with the code reality.

## Validation

- Audit backlog moves and completion reports for code references and explicit completion status.
- Audit user-facing docs for stale examples after behavioral changes.
- Keep ADR links, backlog links, and docs cross-links current when related policy changes.

## Backlog links

- [docs/backlog/completed/018_capability_residency_hooks.md](../backlog/completed/018_capability_residency_hooks.md)
- [docs/backlog/completed/019_best_effort_preload_warmup_for_local_backends.md](../backlog/completed/019_best_effort_preload_warmup_for_local_backends.md)

## Related

- [docs/README.md](../README.md)
- [docs/backlog/](../backlog/)
- [docs/adr/README.md](README.md)
