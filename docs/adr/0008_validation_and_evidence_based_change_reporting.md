# ADR 0008: Require validation and evidence-based change reporting

Status: Accepted.

## Context

AbstractVision spans remote HTTP backends, multiple local runtime families, curated download
surfaces, and platform-specific behavior. Broad claims such as “supported,” “faster,” or “done”
are easy to overstate if they are not tied to specific tests or measurements.

Recent work in this repo has already shown the risk:

- warmup improvements needed real benchmark details before the result was credible;
- backend support statements need to distinguish code-level support from real-model validation;
- packaging and config changes often affect many surfaces at once.

The repo needs a durable reporting bar so future implementation notes, changelog entries, and
completion reports stay decision-grade.

## Decision

Meaningful behavior changes in AbstractVision must be accompanied by explicit validation and
evidence-based reporting.

The rules are:

1. Every meaningful change must state what was validated and what remains unverified.
2. New backend, catalog, config, fallback, or compatibility behavior should get reproducible tests
   when feasible, not only prose.
3. Performance claims must include enough context to evaluate them:
   model, backend/runtime, key parameters, platform, and whether the numbers are measured or
   estimated.
4. Completion reports and changelog notes must distinguish shipped behavior from future ideas or
   open gaps.
5. When a claim cannot be fully validated on the current machine, the remaining gap must be stated
   explicitly instead of being quietly assumed closed.

## Consequences

### Positive

- Performance and support claims become more trustworthy.
- Future contributors can see what level of evidence already exists.
- Regression risk drops because important behavior is tied to concrete checks.

### Negative

- Some tasks take longer because validation and reporting are part of completion.
- Contributors must be precise about what they did not verify.

### Neutral

- Validation remains proportional to risk and surface area; this ADR does not require full
  end-to-end hardware coverage for every small change.

## Enforcement

- Reviewers should ask for explicit validation notes on meaningful changes.
- Performance write-ups that omit workload and platform details should be treated as incomplete.
- Backlog completion reports should keep measured facts separate from inference and recommendation.

## Validation

- Keep focused automated tests for new behavior when feasible.
- Audit backlog completion reports and changelog entries for explicit validation notes.
- For performance-sensitive changes, retain or add reproducible benchmark commands in the related
  backlog item or docs.

## Backlog links

- [docs/backlog/completed/009_test_matrix_and_ci_for_capabilities.md](../backlog/completed/009_test_matrix_and_ci_for_capabilities.md)
- [docs/backlog/completed/019_best_effort_preload_warmup_for_local_backends.md](../backlog/completed/019_best_effort_preload_warmup_for_local_backends.md)

## Related

- [CHANGELOG.md](../../CHANGELOG.md)
- [docs/backlog/](../backlog/)
- [tests/](../../tests/)
