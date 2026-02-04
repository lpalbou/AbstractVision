# Artifacts (artifact refs + stores)

AbstractVision supports “artifact-first” outputs: return a small JSON dict that points to a stored blob instead of inlining bytes.

Code pointers:
- Store interface + helpers: `src/abstractvision/artifacts.py`
- Orchestration logic: `VisionManager._maybe_store()` in `src/abstractvision/vision_manager.py`

See also:
- Getting started (REPL stores outputs by default): `docs/getting-started.md`

## Output shapes

`VisionManager` returns:

- **Without a store**: `GeneratedAsset` (`src/abstractvision/types.py`)
  - contains bytes (`data`), `mime_type`, and best-effort metadata
- **With a store**: an artifact ref dict (via `MediaStore.store_bytes(...)`)
  - minimum shape: `{"$artifact": "<id>"}` (`is_artifact_ref()` checks this)
  - common fields: `content_type`, `sha256`, `filename`, `size_bytes`, `metadata`

## LocalAssetStore (standalone mode)

`LocalAssetStore` stores files under `~/.abstractvision/assets` by default (`src/abstractvision/artifacts.py`):

- Blob: `~/.abstractvision/assets/<artifact_id>.<ext>`
- Metadata: `~/.abstractvision/assets/<artifact_id>.meta.json`

Minimal usage:

```python
from abstractvision import LocalAssetStore, VisionManager
from abstractvision.backends import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend

store = LocalAssetStore()
backend = OpenAICompatibleVisionBackend(config=OpenAICompatibleBackendConfig(base_url="http://localhost:1234/v1"))
vm = VisionManager(backend=backend, store=store)

ref = vm.generate_image("a watercolor painting of a lighthouse")
blob = store.load_bytes(ref["$artifact"])  # type: ignore[index]
```

## RuntimeArtifactStoreAdapter (framework mode)

`RuntimeArtifactStoreAdapter` is a duck-typed adapter for an external artifact store (e.g. AbstractRuntime / AbstractCore runtime),
so AbstractVision can depend on an artifact store **without** a hard dependency (`src/abstractvision/artifacts.py`).

