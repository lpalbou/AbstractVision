## Task 013: Interactive CLI/REPL for manual testing (capabilities + generation + params)

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P1  

---

## Main goals

- Provide a fast, dependency-light interactive UI (“REPL”) to test AbstractVision features:
  - inspect available models and tasks (from the capabilities registry)
  - configure backend + model id + output store
  - run T2I / I2I generation with common parameters
  - get a stored artifact ref result + file path you can open

## Secondary goals

- Keep it cross-platform and safe-by-default:
  - no implicit downloads
  - no implicit network calls unless the user configures a remote backend

---

## Context / problem

Unit tests validate contracts, but they don’t provide a good “human loop” to quickly:
- tweak prompts and parameters (width/height/steps/seed),
- inspect capability mismatches,
- iterate on backend wiring,
- confirm artifacts are stored correctly.

We want a minimal interactive surface that third parties can also use for integration debugging.

---

## Constraints

- Standard library only (no UI dependencies).
- Must remain optional / non-invasive: should not be imported at `abstractvision` import time.

---

## Decision

**Chosen approach**:
- Add a small CLI with subcommands plus a REPL mode:
  - `abstractvision models|tasks|show-model ...`
  - `abstractvision repl` (interactive)
- In REPL:
  - commands start with `/` (consistent with other framework CLIs)
  - generation commands call `VisionManager` and store outputs via `LocalAssetStore`
  - provide a best-effort “open file” helper for locally stored artifacts

---

## Dependencies

- Completed: `docs/backlog/completed/008_asset_outputs_and_third_party_integration.md`
- Completed: `docs/backlog/completed/006_openai_compatible_backend_for_image_and_video.md` (remote backend to exercise)

---

## Implementation plan

- Add `abstractvision/src/abstractvision/cli.py`:
  - argparse subcommands + REPL loop using `input()` + `shlex.split()`
  - inspect commands (models/tasks/show)
  - generation commands for T2I and I2I (video optional later)
- Add an entrypoint in packaging (`setup.py`) for `abstractvision` command.
- Add docs snippet in `README.md` (Task 010) showing how to use the REPL.

---

## Success criteria

- A user can run a REPL and generate/store an image via a configured backend.
- The tool prints a stable artifact ref and the local file path for easy viewing.

---

## Test plan

- Unit tests:
  - basic CLI command parsing smoke (no network)
  - registry inspection commands
- Manual smoke:
  - start REPL, configure backend, run `/t2i ...`, open resulting file

---

## Report (fill only when completed)

### Summary

- Added a dependency-light CLI + REPL for quickly testing capability registry lookups and exercising generation flows against a configured backend.
- Outputs are stored as artifact refs via `LocalAssetStore` and the CLI prints both the ref JSON and the local content path for easy viewing.
- Added a `python -m abstractvision` entrypoint and a `console_scripts` entry (`abstractvision`) for installed usage.

### Validation

- Tests: `python -m unittest discover -s abstractvision/tests -p "test_*.py" -q`
- Added CLI smoke coverage in `abstractvision/tests/test_cli_smoke.py`.

### How to test (interactive UI)

1) Install the package (editable is fine for local dev):
- `pip install -e abstractvision`

2) Start the interactive REPL:
- `abstractvision repl`

3) In the REPL, configure an OpenAI-compatible backend and (optionally) capability gating:
- `/backend openai http://localhost:1234/v1 <api_key_optional> <remote_model_id_optional>`
- `/cap-model zai-org/GLM-Image` (or `/cap-model off`)

4) Set defaults once, then iterate quickly:
- `/set width 1024`
- `/set height 1024`
- `/set steps 30`
- `/set guidance_scale 5.0`
- `/set seed 42`

5) Generate and open results:
- `/t2i "a cinematic photo of a red fox in snow" --open`
- `/i2i --image ./input.png "make it watercolor" --open`
- `/open <artifact_id>` (re-open any previously stored output)

Notes:
- The REPL stores outputs in `~/.abstractvision/assets` by default. Use `/store <dir>` to change it.
- For one-shot (non-REPL) usage: `abstractvision t2i ...` and `abstractvision i2i ...`.
