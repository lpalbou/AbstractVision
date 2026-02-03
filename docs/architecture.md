# AbstractVision Architecture

This document explains how `abstractvision` fits into the AbstractFramework ecosystem and how it integrates with `abstractcore` in both **library mode** and **framework mode**.

For the framework-level architecture overview, see `docs/architecture.md`.

## What AbstractVision is (and is not)

`abstractvision` is the **generative vision** library:
- image generation/editing (T2I/I2I)
- optionally video generation (T2V/I2V) when a backend supports it

It is **not** the owner of “LLM image/video input attachments”.
That input-modality handling lives in `abstractcore` (media pipeline + policies).

## Integration surface: AbstractCore capability plugin

`abstractvision` registers an AbstractCore capability plugin entry point that exposes:

- `core.vision` (VisionCapability)

This keeps `abstractcore` dependency-light while enabling deterministic vision APIs when the plugin is installed (ADR-0028).

Implementation:
- Plugin entry: `abstractvision/src/abstractvision/integrations/abstractcore_plugin.py`

At a high level:
```text
AbstractCore (library)  ──loads plugin──►  core.vision (capability)
                                            │
                                            ▼
                                     VisionManager
                                            │
                                            ▼
                                  Vision backend (HTTP/local)
```

### Backend model (v0)
The AbstractCore plugin currently exposes an **OpenAI-compatible HTTP backend**.

Rationale:
- stable interop shape (OpenAI `/v1/images/*` style)
- works well for “framework mode” where the gateway/runtime may run the backend elsewhere

Configuration is env/config-driven (see the plugin file for exact keys).

## Library mode vs framework mode

### Library mode
Users can use:
- `abstractvision` directly (Python)
- or `abstractcore` with `core.vision` available (plugin installed)

Outputs may be returned as:
- raw bytes (e.g. PNG/WEBP/MP4)
- or small dict payloads (when the backend returns an artifact ref shape)

No durability guarantees exist unless the caller provides a store adapter.

### Framework mode (durable)
When an `artifact_store` is provided (typically via runtime/gateway):
- large outputs are stored durably as artifacts
- flows and thin clients refer to them by artifact id (`{"$artifact":"..."}`)

See:
- `docs/adr/0024-attachment-placeholders-and-compaction-invariants.md`
- `docs/backlog/completed/611-abstractgateway-voice-audio-capabilities-durable-wiring-v0.md`

## Relationship to AbstractCore Server vision endpoints

AbstractCore Server can expose OpenAI-compatible image endpoints:
- `POST /v1/images/generations`
- `POST /v1/images/edits`

These endpoints **delegate to AbstractVision internally** (when installed in the same environment).

This is an HTTP interoperability surface for non-Python clients; it does not replace the artifact-first durability contract in runtime/gateway mode.

See:
- `abstractcore/abstractcore/server/README.md` (Image Generation section)

## Related docs

- `docs/adr/0028-capabilities-plugins-and-library-framework-modes.md`
- `docs/guide/capability-vision.md`
- `docs/architecture.md`

