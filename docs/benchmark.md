# Platform check (0.3.35, 2026-11-10)

AbstractVision 0.3.35 was installed from scratch on three machines and asked to generate images at
each model's catalog defaults. "Warm" is the second generation with the model resident in the same
process. Times are wall clock for the generation call; the load time is given separately.

## The three machines

| | Linux, processor only | Linux NVIDIA | Apple silicon laptop |
|---|---|---|---|
| Hardware | 6 cores, 11 GB RAM (about 3 GB free), no GPU, Python 3.14 | Quadro RTX 5000 16 GB (Turing, compute capability 7.5), 4 vCPU, 26 GB RAM | Apple silicon, Metal (MLX) |
| Install | `torch torchvision` from the CPU index, then `abstractvision[gpu]` (1.4 GB) | `abstractvision[gpu]` (5.7 GB, no compiler) | `abstractvision[apple]` |
| Device and dtype picked | `cpu`, float32 | `cuda`, float16 (the card has no native bfloat16) | Metal, MLX quantized weights |

## Images

| Model, size, steps | Linux, processor only | Linux NVIDIA | Apple silicon (MLX-Gen) |
|---|---|---|---|
| FLUX.2 [klein] 4B, 1024², 4 steps (Diffusers, model CPU offload) | not run (needs about 24 GB of host RAM) | load 20 s; 20.1 s, 18.0 s warm; 8.6 GB of GPU memory | see the 9B row |
| FLUX.2 [klein] 9B 8-bit, 1024², 4 steps (MLX-Gen) | | | 41 s first (includes the load), 33 s warm |
| FLUX.2 [klein] 4B GGUF Q8, 512², 4 steps (stable-diffusion.cpp) | | 7.5 s, 6.5 s warm; 6.6 GB | |
| SDXL base, 1024² | | load 7 to 9 s; 21.0 s, 20.7 s warm; 12.7 GB | |
| SDXL Turbo / SD Turbo, 512², 1 step | | 1.0 / 0.5 s warm; 0.7 / 0.2 s warm | |
| Stable Diffusion 1.5 / 1.4, 512² | | load 5.5 s; 4.7 s, 4.1 s warm; 3.9 GB | |
| PixArt-Sigma, 1024² | | 12.7 s, 11.0 s warm; 14.0 GB | |
| Kolors, 1024² (offload) | | 45.1 s, 34.8 s warm; 12.0 GB | |
| Lumina-2, 1024² | | 82.8 s, 82.6 s warm; 14.2 GB | |
| Z-Image Turbo GGUF, 512², 9 steps | | 14.1 s, 12.7 s warm; 9.8 GB | |
| Qwen-Image-2512 GGUF Q8, 512², 20 steps | | 150 s; 14.3 GB | |
| Qwen-Image-Edit-2511 GGUF, 768², 15 steps, cfg 2.5 | | 290 s; 14.3 GB | |
| tiny-sd (a 400 MB test model), 384² / 512², 12 to 15 steps | 50.7 s, 51.6 s warm | 1.3 s, 0.8 s warm | |

What the table says: on the NVIDIA card the Diffusers models load once and stay resident, the
second generation reuses them, and `unload` returns the GPU memory (the process keeps only the CUDA
context); the GGUF path through `sd-cli` reloads the model on every call. Models trained in
bfloat16 (Z-Image, OmniGen) give black images in float16 on a card without native bfloat16: run them
in float32 there, or use their GGUF variant. The processor path works with the same API and is for
small models or patience. Gated repositories (FLUX.1, FLUX.2 [klein] 9B and dev, Stable Diffusion
3) need a Hugging Face token and were not part of this check; Wan 2.2 video is disabled on the
Diffusers backend and refuses with a clear message.

Methodology: clean virtual environments, models downloaded by the package, `nvidia-smi` sampled
every 0.25 s during the GPU runs, output images checked for pixel variance. The per-model raw
results of the NVIDIA run are kept with the release notes of 0.3.35.
