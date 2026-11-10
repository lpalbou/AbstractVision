## Task 031: `abstractvision t2i` defaults to the remote provider while `abstractvision download` defaults to the local 8-bit artifact

**Date**: 2026-11-10  
**Status**: Proposed  
**Priority**: P2  

---

## Main goals

- A user who ran `abstractvision download <model>` with no flags (which picks the local artifact: on CUDA
  `target: gguf; bits: 8; provider/engine: stable-diffusion.cpp`) can run `abstractvision t2i <prompt> --model <model>`
  with no provider flag and get a local generation from that artifact.

## Secondary goals

- The error for a remote provider without a base URL names the local alternative that is already downloaded.

## Evidence (0.3.35, Ubuntu 24.04, Quadro RTX 5000, 2026-11-10)

```
$ abstractvision download black-forest-labs/FLUX.2-klein-4B
target: gguf
artifact: gguf; bits: 8; provider/engine: stable-diffusion.cpp
resolved_diffusion_model: .../models--leejet--FLUX.2-klein-4B-GGUF/snapshots/.../flux-2-klein-4b-Q8_0.gguf

$ abstractvision t2i "A red ceramic mug on a white table, studio light" --model black-forest-labs/FLUX.2-klein-4B --width 768 --height 768 --steps 4 --seed 7
Missing --base-url (or $OPENAI_BASE_URL).
(exit 1, 0.19 s)

$ abstractvision t2i "A red ceramic mug on a white table, studio light" --provider sdcpp --model black-forest-labs/FLUX.2-klein-4B --width 768 --height 768 --steps 4 --seed 7
<png written>  wall 31.55 s first (model load), 10.05 s on the second run; 7.6 GB of VRAM, GPU at 100 %
```

## Proposed change

- `t2i`/`i2i` with no `--provider`: when the model id resolves to a local artifact already present (the same
  resolution `download` uses: `local_model_profile()` and the catalog's auto target order), pick that local engine;
  fall back to the remote provider only when nothing local is present, with a message that says so.
- Keep `--provider` explicit as the override.

## Acceptance

- The two commands above, in that order, produce an image without `--provider`.
- A test with a stubbed local artifact asserts the provider chosen; a test without one asserts the remote error
  names `abstractvision download <model>`.
