## Backlog (how we work)

This folder is the **single source of truth** for planned and completed work in AbstractVision.

### Principles

- **Backlog items are self-contained**: each task includes the outcome of research, key constraints, design choices, and the reasons for those choices (with references).
- **No test-driven special casing**: implementation must be general-purpose logic, not tailored to tests.
- **Keep the public contract stable**: preserve the integrator-facing API (`VisionManager.generate_image/edit_image/generate_angles/generate_video/image_to_video/...`).
- **Artifact-first outputs**: generated images/videos must be representable as small JSON objects (refs), not inlined bytes, to support tool calling + workflows + third-party integrations.
- **Batteries-included install, but lazy imports**: the default install includes local backends, but importing `abstractvision` should not eagerly import heavy runtime stacks unless needed (avoid model loads/weight downloads at import time; defer heavy backend imports until backend construction or first use).
- **Prefer permissive licensing**: only adopt MIT/Apache/BSD-compatible components. If none exist, document feasibility and create a backlog item.

Note: some older completed backlog items mention “heavy deps behind extras”. Treat those as historical context; current packaging is “batteries included”, but we still want lazy imports and explicit model download/load semantics.

---

## Folder layout

- `docs/backlog/planned/`: work to do next
- `docs/backlog/completed/`: finished tasks with a completion report

---

## Naming convention

Create new tasks as:

- `docs/backlog/planned/{NNN}_{short_description}.md`

Rules:
- **NNN**: zero-padded integer (e.g. `017`)
- **short_description**: snake_case, concise

Note: older tasks may use legacy names; do not rename unless a dedicated cleanup task exists.

---

## Task template (required sections)

Every backlog item must include:

- **Title**
- **Date**
- **Status**: Planned / Completed
- **Priority**: P0/P1/P2
- **Main goal(s)** and **secondary goal(s)**
- **Context / problem statement**
- **Constraints**
- **Research & references** (links + key findings)
- **Decision** (what we choose + why)
- **Dependencies** (ADRs and other tasks)
- **Implementation plan** (small steps, minimal surface area)
- **Success criteria**
- **Test plan**
- **Report** (for completed tasks only)

See `docs/backlog/template.md`.

---

## Work process (development cycle)

0. **Think, design, and plan** with long-term consequences in mind; prefer the cleanest, simplest, most efficient approach.
1. **Write the task** in `docs/backlog/planned/{NNN}_{short_description}.md` (self-contained, with research and decisions).
2. **Implement** the task.
3. **Test** and fix issues.
4. **Only when all tests pass**, move the task to `docs/backlog/completed/` and add the completion report at the end.
5. **Check recurrent tasks** (see `docs/recurrent/`) before claiming completion.
6. After completion, **bump semantic version** and add a **CHANGELOG** entry (this is a recurrent task).
