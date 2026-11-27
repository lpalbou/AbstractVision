# Changelog

## 0.3.38 - 2026-11-27

This release adds typed image-edit options on MLX-Gen (latent image-to-image, FLUX.2 masked edits,
multi-reference edits, outpaint and reframe), Z-Image edits, a control image on Qwen-Image, video-to-video
on Wan 2.2 T2V-A14B and Wan 2.1 VACE 1.3B, SeedVR2 video restoration (`video_upscale`), cached source
repository ids, and `preload(task=..., warm_up=...)`. The MLX-Gen backend requires
`mlx-gen>=0.40.2,<0.41.0` and, on macOS, MLX 0.32.4 or later. Requests that set none of the new fields run
as in 0.3.37, except for the changes listed under Changed.

### New

- **MLX-Gen image routes chosen by the engine's planner.** Every `image_to_image` request on FLUX.2, Z-Image and
  base Qwen-Image, and a request on any MLX-Gen model that sets `image_strength`, is routed by mlx-gen's own planner
  (`resolve_generation_plan`, the one `mlxgen generate` uses) against the model's capability rows; the capability id
  it selects decides the MLX-Gen class and call (`abstractvision.backends.mlx_gen_routes`). Other requests run as in
  0.3.37. Everything is checked before any weights load, in `normalize_image_edit_request` as in `edit_image`: two
  options that contradict each other raise `ValueError`; an option no row of the model applies, and adapters on a
  row that takes none, raise `CapabilityNotSupportedError` naming the field (`supported=()`); a combination no row
  serves raises it with the planner's message (`field=None`). A route new in this release writes the run's files
  into one temporary directory, removed whatever the outcome; no metadata names it.
- **Latent image-to-image: `ImageEditRequest.image_strength`**, in `(0, 1]` (the plugin's `i2i(image_strength=...)`).
  On MLX-Gen, `extra["image_strength"]` is the same option. FLUX.2 klein runs it with `Flux2Klein` (the base models
  pass the negative prompt, which their latent row takes); base Qwen-Image (`AbstractFramework/qwen-image-8bit`, the
  Qwen-Image-2512 packages) now runs `image_to_image` with `QwenImage`, at `image_strength=0.4` when the request
  names none; ERNIE-Image-Turbo runs it as its 0.3.37 latent edit. `list_provider_models()` publishes
  `parameter_defaults["image_strength"] = 0.4` and its range for every model with a latent route, and the catalog
  names the parameter `image_strength` on FLUX.2 klein, Qwen-Image, Z-Image and ERNIE (ERNIE's `strength` is
  described as the 0.3.37 keyword below).
  `extra["strength"]` (the CLI's `--strength`, the plugin's `strength=`) is the 0.3.37 cross-backend keyword: on
  every route 0.3.37 had it keeps exactly its 0.3.37 meaning, and on the latent routes new in this release it is
  read as the run's strength with the same clamp to `[0, 1]`. It never selects a route and is not read on a masked
  route, and combined with `image_strength` it raises `ValueError`. A task this release adds to a model
  (`image_to_image` on base Qwen-Image and Z-Image, `video_to_video` on Wan) is published and accepted only where the
  engine's rows carry a route for it; without rows the model has its 0.3.37 tasks.

- **FLUX.2 masked edits.** An edit with a `mask` on FLUX.2 klein runs `Flux2KleinInpaint` (MLX-Gen's
  `flux2.inpaint`): the mask is decoded before any weights load (`ValueError` naming `mask` when it is not a
  picture), the bytes are passed unchanged, an unset `guidance_scale` stays unset so the class applies its default,
  and the base models take a negative prompt. A mask with `image_strength` raises `ValueError`; adapters on a row
  that takes none are refused (`field="lora_adapters"`). The instance is built for masked edits: a plain edit after
  it builds `Flux2KleinEdit` again, after releasing it.

- **Multi-reference edits: `ImageEditRequest.reference_images`**, the images after `image` in order (bytes or file
  paths; the plugin's `i2i` and `i2i_batch` resolve file paths and artifact refs to bytes). On MLX-Gen,
  `extra["reference_images"]` is the same option. FLUX.2 klein and Qwen-Image-Edit 2509/2511 run it on their
  multi-reference route (the 0.3.37 edit call with the list after the source); no maximum count is published or
  enforced. A model without that route refuses it (`field="reference_images"`, `supported=()`), as does a
  combination with a mask or a strength (`field=None`): the planner counts every picture, those of the 0.3.37
  `extra` keys included, so a reference is never dropped. An item that is not image bytes or a path, or a picture
  that fails to decode, raises `ValueError`. On a model the engine publishes no rows for, references run their
  0.3.37 edit branch. The pictures a routed request carries (a mask, references) are decoded once
  its route is known, still before any weights load.

- **Outpaint and reframe: `ImageEditRequest.outpaint_padding`, `outpaint_fill`, `outpaint_fill_color`,
  `reframe_padding`.** They run MLX-Gen's own canvas session (`mflux.outpaint`: canvas, fill policy, pass plan),
  its workspace in the run's directory: FLUX.2 klein outpaint with `Flux2KleinOutpaint` and reframe (distilled)
  with `Flux2KleinEdit`, Qwen-Image-Edit both with `QwenImageEdit`. The session is built before any weights load
  and refuses a box the engine cannot apply (`ValueError` naming the field); a fill the row does not list, a fill or
  color on Qwen's fixed fill, a color without `outpaint_fill="solid"`, a fill without an outpaint and
  `extra["width"]`/`["height"]` with either are refused naming the field. A two-pass outpaint counts its progress
  across both passes and is refused in low-RAM mode (`field="memory_mode"`, `supported=("default",)`). Results carry
  `outpaint_notice` and `outpaint_warnings`.

- **Video-to-video: `VideoToVideoRequest`, `VisionBackend.video_to_video`, `VisionManager.video_to_video` and the
  plugin's `v2v`**, a new task `video_to_video` (catalog, residency names `video_to_video`, `video-to-video`, `v2v`).
  On MLX-Gen, Wan 2.2 T2V-A14B runs it with `Wan2_2_TI2V`: `video_strength` in `(0, 1]` (unset: the engine's 0.8),
  an optional one-image `mask`, the A14B defaults and memory plan of text-to-video, and a negative prompt only when
  set; as on the engine CLI, `guidance_2` takes the model's 3.0 only when the request sets neither guidance (a
  `guidance_scale` set alone leaves it to the engine, which then uses the same value). The source is a file read in place or bytes written once to the run's directory, and is probed before any
  weights load (a clip too short for `num_frames` at `fps` is refused with `field="num_frames"`); `extra` takes the
  progress callbacks, `max_sequence_length` and `vae_tiling`. The other backends refuse the task.

- **Wan 2.1 VACE 1.3B (`Wan-AI/Wan2.1-VACE-1.3B-diffusers`)**, a new catalog entry and MLX-Gen model
  (`wan2.1-vace-1.3b`, run by `WanVace`) for `video_to_video`: `reference_images` (any number, in order),
  `masked_region_mode` (`generate` or `repaint`, with a mask) and `conditioning_scale` (default 1.0); it refuses
  `video_strength` and `guidance_2` by name, and Wan 2.2 T2V-A14B refuses the three VACE fields. Defaults 832x480,
  81 frames at 16 fps, 30 steps, guidance 5.0; `memory_mode="low"` releases the denoiser before the decode and drops
  the model. A VACE preload loads `WanVace` and runs no warm-up generation; in the default mode it reads the
  transformer and the VAE ahead, as the other Wan preloads do (Changed 23).

- **Z-Image edits.** `image_to_image` on Z-Image and Z-Image-Turbo runs `ZImage` latent (at `image_strength=0.4`
  when the request names none; Turbo keeps its forced guidance 0.0 and passes a negative prompt, which its row takes), and a `mask` on
  Z-Image-Turbo runs its masked edit (the mask instead of the strength). Base Z-Image refuses a mask
  (`field="mask"`), and both refuse references, outpaint and reframe by name.

- **Control image on Qwen-Image (`AbstractFramework/qwen-image-8bit`).** `control_image` (and `control_strength`,
  unset: the engine CLI's 0.4) runs `QwenImageControlNet` with the ControlNet its row names
  (`InstantX/Qwen-Image-ControlNet-Union`). Its side weights are looked up in the Hugging Face cache before any
  weights load and never downloaded during a generation; when missing, the request is refused
  (`field="control_image"`) naming the repository, the file and `abstractvision download
  InstantX/Qwen-Image-ControlNet-Union`. Any other model refuses `control_image`; a `control_strength` without it is
  refused, and one that is not a finite number raises `ValueError`. A plain generation after a control run builds
  `QwenImage` again.

- **`preload(task=...)` on MLX-Gen.** The MLX-Gen backend's `preload` takes `task`, the task the host is about to
  run, as a hint: a model that has it is loaded as a request of that task loads it (on FLUX.2 klein,
  `task="image_to_image"` loads `Flux2KleinEdit`; a latent, masked or outpaint request rebuilds its own class); a
  model without it, or `task=None`, preloads as before. It never refuses. The AbstractCore plugin's
  `load_resident_model` passes the payload's `task` to a backend whose `preload` takes a `task` (or `**kwargs`).

- **FLUX.2 klein 9B consistency LoRA profile.** `vision_adapter_capabilities.json` describes
  `dx8152/Flux2-Klein-9B-Consistency` (file `Flux2-Klein-9B-consistency-V2.safetensors`): an image-to-image LoRA that
  improves the consistency of klein 9B edits, no trigger word, apache-2.0, validated by MLX-Gen on
  `AbstractFramework/flux.2-klein-9b-8bit`'s edit route; no parameter recommendation (the model card gives none).
  `list_provider_adapters` (and `abstractvision adapters`) carries its notes and link once it is cached.

- **`preload(warm_up=False)` on MLX-Gen.** The warm-up is optional: `warm_up=False` resolves, checks and builds the
  model (a bad model still fails at the preload) and runs and reads nothing ahead; the default, `True`, warms as
  Changed 23 says, and `low` / `lowest` are the same either way. `preload_warmed()` says whether the held model was
  warmed; the plugin passes the load payload's `warm_up` to a `preload` that takes it, and its loaded-model records
  carry `warmed` (`None` from a backend that does not say).

- **SeedVR2 video restoration (`video_upscale`).** `VideoUpscaleRequest(video, resolution, softness, seed,
  start_seconds, max_frames, drop_audio, quantize, ...)`, `VisionManager.upscale_video(video, **kwargs)` and the plugin's
  `upscale_video(video, **kwargs)` restore a clip on SeedVR2 3B and 7B (MLX-Gen). `video` is the clip's bytes or a
  local file read in place (a `str` or `Path` in the plugin). Before any weights load, MLX-Gen's own plan
  (`SeedVR2VideoRestorePlan.for_video`, the one `mlxgen upscale --video-path` uses) probes the clip and decides the
  scale, the chunking and the memory profile; a scale above 1x is refused (`field="resolution"`), a clip beyond the
  safe memory profile or below its temporal quality floor is refused (`field=None`), each with the engine's message.
  The plan counts the weights as loaded when this backend already holds the instance the run reuses. The run
  applies the plan as the CLI does (MLX-Gen's low-RAM saver keeping the transformer, `restore_video_to_path` with
  the planned chunks) in `memory_mode` `default` and `low` alike, under one MLX cache limit from before the load to
  the end (the plan's 8 GB, never above a limit the host set), and drops the model at its end. The result is an MP4
  whatever the source's container (the source's audio copied by MLX-Gen, which fails the run after the restoration when it
  cannot; `drop_audio=True` gives a silent result, no copy attempted), progress is per chunk, a cancel removes the partial output. Unset values are the engine's: 1x, softness 0.0, from the clip's start to its
  end. `preload(task="video_upscale")` reads the transformer and the VAE the run keeps.
- **MLX-Gen 0.40.2.** Every extra that installs MLX-Gen (`mlx-gen`, `mflux`, `apple`, `all`, `all-apple`)
  requires `mlx-gen>=0.40.2,<0.41.0` (0.3.37 required `>=0.40.1`) and, on macOS, `mlx>=0.32.4` as before.
  Upgrade with `pip install -U "abstractvision[mlx-gen]"`. `video_upscale` runs on the SeedVR2 video
  restoration plan that 0.40.2 adds. 0.40.2 also changes, with no change to AbstractVision's calls: the Wan
  prompt encoding (Changed 27); SeedVR2 video restoration, which needs a fraction of 0.40.1's memory with
  identical output and restores frames up to about 1.4 megapixels on its safe memory profile's 48 GB budget
  (1280x720 fits), a larger frame being refused before the model loads (`field=None`) with a message naming the
  largest size that fits; and FLUX.2 klein, which keeps one copy of its transformer weights whatever the number
  of canvas sizes a process runs.

### Changed

Against 0.3.37 (the entries are numbered so that others can refer to them):

1. MLX-Gen refuses an `image_strength`, typed or `extra["image_strength"]`, outside `(0, 1]`
   (`CapabilityNotSupportedError`, `field="image_strength"`); 0.3.37 clamped `extra["image_strength"]` to `[0, 1]`.
2. MLX-Gen raises `ValueError` for an `extra["image_strength"]` that is not a finite number; 0.3.37 dropped it.
3. `extra["image_strength"]` on FLUX.2 (a model the engine publishes rows for) runs the latent route
   (`Flux2Klein`); 0.3.37 ignored it and ran `Flux2KleinEdit`.
4. `extra["image_strength"]` on Qwen-Image-Edit and FIBO-Edit is refused (`field="image_strength"`, `supported=()`);
   0.3.37 ignored it.
5. `extra["image_strength"]` on FIBO and Fibo-lite is refused before the model loads; 0.3.37 loaded it, then the
   engine raised.
6. `extra["reference_images"]` on a model whose rows have no multi-reference route is refused
   (`field="reference_images"`): on plain Qwen-Image-Edit (not 2509 or 2511), where 0.3.37 passed the references to
   the edit, and on ERNIE-Image-Turbo, FIBO and Fibo-lite, where 0.3.37 ran the edit and ignored them. A local
   ERNIE-Image-Turbo, FIBO or Fibo-lite folder the engine publishes no rows for refuses them too; a FLUX.2 or
   Qwen-Image-Edit folder without rows runs them as 0.3.37 did.
7. `extra["reference_images"]` with an adapter on a route that takes none (`AbstractFramework/qwen-image-edit-2509-8bit`)
   is refused (`field="lora_adapters"`); 0.3.37 ran it.
8. A typed `image_strength` or `reference_images` with the same `extra` key of another value raises `ValueError`, as
   does `image_strength` with `extra["strength"]` and `reference_images` with `extra["images"]`,
   `["additional_images"]` or `["image_paths"]`.
9. The plugin keywords `image_strength=`, `reference_images=`, `outpaint_padding=`, `outpaint_fill=`,
   `outpaint_fill_color=` and `reframe_padding=` are refused with `field=` on `t2i`, `t2v`, `i2v` and
   `upscale_image` on every backend, where 0.3.37 put them in `extra` and nothing read them; on `i2i` they are typed
   fields, which Diffusers, OpenAI-compatible and stable-diffusion.cpp refuse by name (0.3.37 put them in `extra`,
   which the OpenAI-compatible backend forwarded to the remote). An unset value (None, an empty list, a blank
   string) is never refused. Passed as `extra={...}`, and `strength=`, they travel unchanged.
10. `i2i(control_image=...)` and `i2i(control_strength=...)` are refused (`field=`); 0.3.37 put them in `extra`, where
    nothing read them.
11. `image_to_image` on base Qwen-Image and on Z-Image runs latent, at `image_strength=0.4` unless set; 0.3.37
    refused the task.
12. A `mask` on FLUX.2 klein runs `Flux2KleinInpaint`; 0.3.37 refused it (`field="mask"`) after loading the model.
13. `control_image` on `AbstractFramework/qwen-image-8bit` runs `QwenImageControlNet`; 0.3.37 refused it
    (`field="control_image"`).
14. Published answers: `get_capabilities().supports_mask` is true on FLUX.2 klein and Z-Image-Turbo, and
    `supports_control_image` on `AbstractFramework/qwen-image-8bit`;
    `supported_tasks` and `list_provider_models(task="image_to_image")` include base Qwen-Image and Z-Image, and
    `supported_tasks` gains `video_to_video` on Wan 2.2 T2V-A14B and Wan 2.1 VACE 1.3B. The catalog lists
    `image_to_image` on `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512`, `Tongyi-MAI/Z-Image` and
    `Tongyi-MAI/Z-Image-Turbo` (for the MLX-Gen backend), `image_strength` on FLUX.2 klein, ERNIE-Image-Turbo and
    those four, `outpaint_padding`, `outpaint_fill` and `outpaint_fill_color` on FLUX.2 klein, `reframe_padding` on
    the distilled FLUX.2 klein 4B and 9B (not on klein Base), `outpaint_padding` and `reframe_padding` on
    Qwen-Image-Edit, 2509 and 2511 (no fill fields),
    `control_image` and `control_strength` on `Qwen/Qwen-Image`'s `text_to_image`, and the new task
    `video_to_video`. `reference_images` is no longer listed for `Qwen/Qwen-Image-Edit`.
    `list_provider_models()`' `parameter_constraints["negative_prompt"]` follows the engine's rows as the runs do
    (Changed 21): FLUX.2 klein Base and Z-Image-Turbo no longer publish `{"supported": false}` (a task whose routes
    disagree publishes `supported: true` with a description naming it); a model without rows keeps 0.3.37's answer.
15. A typed request option that is an empty list or tuple counts as unset (`option_is_set`). No 0.3.37 option is a
    sequence, so no 0.3.37 request changes.
16. stable-diffusion.cpp, which lists every catalog task of a model, lists `image_to_image` (and `supports_mask`)
    with `capabilities_model_id` set to `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512`, `Tongyi-MAI/Z-Image` or
    `Tongyi-MAI/Z-Image-Turbo`, and `video_to_video` with
    `Wan-AI/Wan2.2-T2V-A14B`.
17. On a model the engine publishes no rows for (a local folder whose name it cannot place), `extra["image_strength"]`
    is refused (`field="image_strength"`) on every family but ERNIE-Image-Turbo, where it runs 0.3.37's latent edit;
    0.3.37 ignored it on the other families.
18. `metadata["mlx_gen"]` no longer carries `controlnet_image_path`, which 0.3.37 returned (always `None` then): it
    is left out like the other temporary file paths of a run.
19. In `memory_mode="default"` the MLX cache is capped as the engine's CLI caps it (total RAM / 8, between 1 and 8
    GiB, once per process; a limit the host set at or below half of physical memory is left alone, a higher one is
    replaced by the default cap) for every MLX-Gen model; 0.3.37 left Wan models uncapped (the Wan loader does not
    apply the cap itself).
20. Some requests 0.3.37 refused are now refused before the model loads instead of after, or with a different
    `field` or exception type: a mask with references on FLUX.2; `extra["reference_images"]` on FIBO-Edit;
    references with `extra["images"]` on FIBO-Edit; a mask on a FLUX.2 folder the engine cannot place; a mask with
    `extra["reference_images"]` on ERNIE-Image-Turbo, FIBO, Fibo-lite and Qwen-Image-Edit 2509 and 2511 (now refused
    before the load); `supported` on FIBO's and Fibo-lite's refusals goes from `None` to `()`. Nothing that ran is
    affected.
21. A negative prompt follows the row of the route a request runs (`supports_negative_prompt`) on every MLX-Gen
    image route of a model the engine publishes rows for. It is now passed on FLUX.2 klein Base (4B and 9B) on
    text-to-image, edit and multi-reference (0.3.37 dropped it; the latent, masked and outpaint routes already
    passed it), and on Z-Image-Turbo on text-to-image, latent and masked edits (0.3.37 dropped it; at its forced
    guidance 0.0 the engine runs no negative branch). FLUX.2 klein Base with a negative prompt at `guidance_scale`
    1.0 or below now raises the engine's `ValueError` after the model loads. A route whose row takes none still
    drops it silently; a model without rows keeps 0.3.37's answer.
22. `load_resident_model({"task": "image_to_image", ...})` on an MLX-Gen model that has both text-to-image and
    image-to-image (FLUX.2 klein, ERNIE-Image-Turbo, FIBO, Fibo-lite, and base Qwen-Image and Z-Image, which gain
    image-to-image in this release) loads the model for image-to-image (on FLUX.2 klein `Flux2KleinEdit`) and warms
    it with one image-to-image run (Changed 23) where 0.3.37 loaded the text-to-image class and ran its text-to-image
    warm-up. A payload without `task`, or with
    `text_to_image`, loads as in 0.3.37; other backends are unchanged.
23. A default-mode MLX-Gen `preload` warms what a run of its task reads with a small ordinary request of that
    route. 0.3.37 ran one text-to-image generation at the backend's default canvas (1024x1024 unless
    `default_width` / `default_height` were set) and the model's default steps, and only built the object for every
    other task. Now an image task runs one warm-up of the task on its plain route, its output discarded, at 256x256
    (352 on base Qwen-Image, FIBO and Fibo-lite, 384 on FIBO-Edit, 512 on ERNIE-Image-Turbo and Bonsai: the smallest
    side MLX-Gen itself runs those families at) and 2 steps (the fewest MLX-Gen's flow-match scheduler takes); a
    backend whose `default_width` / `default_height` were set keeps that canvas, as in 0.3.37. Image-to-image preloads (Qwen-Image-Edit, FIBO-Edit,
    and the image models given `task="image_to_image"`: `Flux2KleinEdit` on FLUX.2 klein) and SeedVR2 preloads (a
    64x64 picture at 1x) run one too. Video preloads read the first denoiser and, where the run encodes an input
    (image-to-video, video-to-video), the VAE (Wan TI2V-5B, T2V-A14B, I2V-A14B, VACE). The preload is short and the
    weights are resident when it returns; the first request reads none. MiniMax-H3 and the `low` and `lowest` modes
    are unchanged.
24. SeedVR2's published answers gain `video_upscale`: `get_capabilities().supported_tasks` and
    `list_provider_models()` on SeedVR2 3B and 7B, and the catalog (a top-level `video_upscale` task, with its params
    on `ByteDance-Seed/SeedVR2-3B` and `-7B`). The catalog's `min_runtime_version` of Wan T2V-A14B and VACE
    `video_to_video` and of MiniMax-H3 is 0.40.2, the package floor. The image upscale is unchanged.
25. With mlx-gen 0.40.1 installed, every MLX-Gen request now raises `OptionalDependencyMissingError` naming
    `mlx-gen>=0.40.2,<0.41.0`, as any version outside the pins already did.
26. On MLX-Gen, a source repository id that maps to several prepared packages (the four FLUX.2 klein repos,
    `Tongyi-MAI/Z-Image` and `Z-Image-Turbo`, the five Qwen-Image repos, `baidu/ERNIE-Image-Turbo`) runs its own
    cached snapshot as released (16-bit, unquantized, as `mlxgen generate --model <id>` does), even when a
    prepared package of it is cached too; 0.3.37 refused the id as ambiguous whatever was cached. Cached means a
    snapshot that can load as the model: `model_index.json` and readable `transformer/*.safetensors`, no partial
    download. It is a source, not a prepared package: it publishes the engine's load-time `quantize` widths and
    accepts `quantize`. Through the AbstractCore plugin the id reaches the backend unchanged (0.3.37 rewrote
    `Qwen/Qwen-Image-2512` to `qwen-image`, which it still does when that source is not cached). A source needs
    several times a prepared package's memory. Not cached, or cached without those files, the id is refused as
    ambiguous as before. The `Qwen/Qwen-Image-2512` source snapshot given as a folder, which 0.3.37 answered as a
    prepared package (no `quantize` widths, `quantize` refused), is a source too.
27. With the required mlx-gen 0.40.2, Wan videos on MLX-Gen (Wan 2.2 text-, image- and video-to-video, Wan 2.1
    VACE) for a given prompt and seed differ slightly from those of 0.3.37 with mlx-gen 0.40.1: the same scene and
    composition, with small differences in details, because the engine's prompt encoding is now exact (float32).
    A new prompt is encoded in seconds instead of more than a minute. See MLX-Gen's
    [Wan prompt encoding](https://github.com/lpalbou/mlx-gen/blob/main/docs/wan-text-encoder.md).
28. On MLX-Gen, a local model is identified by its own folder: the folder's name, or the repository id of a
    Hugging Face cache snapshot (`models--org--name/snapshots/<revision>`), and its own card; a parent directory
    never decides. This sets its base model when `base_model` is not given, its stored width
    (`metadata["quantization_bits"]` of a request without `quantize`, and the SeedVR2 local-folder match), the
    routes MLX-Gen publishes for it, and whether discovery skips it as a partial or incompatible download.
    0.3.37 read the whole path, so a model under a parent named after another model or width
    (`~/models/qwen-image-edit_q4/flux2-klein-4b`, `models_q4`, a temporary folder whose random name holds `_q1`)
    could be identified, routed or refused as that other model, or get a width it does not have, and a parent
    named `*.partial`, `*-partial` or `*.incompatible` hid every model under it. A folder whose own name and card
    place no model needs `base_model`.

## 0.3.37 - 2026-11-26

This release adds MiniMax-H3 (video with a synchronized stereo soundtrack), a `memory_mode` setting on every
generation request, downloads planned against the cache with a disk-space check, model listings that show access,
cache state and memory per mode, and what an application embedding `VisionManager` needs: cancellation, load-time quantization, the
engine's route rows and metadata. It requires MLX-Gen 0.40.1. Requests that set none of the new fields run as before,
except for the changes listed below.

### New

- **MLX-Gen 0.40.1.** Every extra that installs MLX-Gen (`mlx-gen`, `mflux`, `apple`, `all`, `all-apple`) requires
  `mlx-gen>=0.40.1,<0.41.0` (0.3.36 required `>=0.38.0,<0.39.0`). It brings MiniMax-H3, saves images and videos with
  Pillow 13, and publishes capability schema 17. The calls AbstractVision makes for FLUX.2, Z-Image, Qwen, ERNIE,
  FIBO, Bonsai, SeedVR2 and Wan are unchanged. On macOS the same extras require `mlx>=0.32.4`: MLX 0.31.0 to 0.32.3
  can end the process with a Metal "invalid resource" error during a MiniMax-H3 decode when the MLX cache is small,
  and MLX 0.32.4 fixed it. In those versions the scalar a convolution pads its channels with is not kept alive until
  the GPU runs the convolution: in the 3D convolution path (0.31.0 to 0.32.3), which the MiniMax-H3 video decode
  runs, and in a 2D convolution path that pads input channels (0.32.1 to 0.32.3); 0.32.4 keeps it alive in both.
  Upgrade with `pip install -U "abstractvision[mlx-gen]"`. An upgrade without the extra
  keeps the installed mlx-gen and MLX, so the MLX-Gen backend refuses an mlx-gen outside `>=0.40.1,<0.41.0`, and on
  macOS an MLX older than 0.32.4: the first request or `preload()` raises `OptionalDependencyMissingError` naming the
  installed version and that command, before the model is looked for; a pre-release of a floor is below it.
- **MiniMax-H3: video with a synchronized stereo soundtrack**, `text_to_video` and first-frame `image_to_video`,
  through the MLX-Gen backend. The catalog entry `MiniMaxAI/MiniMax-H3` lists the packages
  `AbstractFramework/minimax-h3-8bit` (recommended), `AbstractFramework/minimax-h3-bf16` and
  `AbstractFramework/minimax-h3-4bit`, and the source repository. `base_model` picks the schedule:
  `minimax-h3-turbo-544p` (8 steps, 960x544), `minimax-h3-turbo` (8 steps, 1344x768) or `minimax-h3` (50 steps).
  - Video requests take the typed fields `soundscape`, `music`, `generate_audio` and `audio_shift`; `flow_shift` is
    the video shift. The prompt is passed to MLX-Gen unchanged, and values left unset keep the schedule's defaults.
  - Every value is checked before any weights load, against MLX-Gen's capability row (frame grid `17n+5` at 24 fps,
    canvas multiple) and its own MiniMax-H3 rules (aspect window 1:4 to 4:1, prompt composition, adapters), and a
    refusal about one field names it: what the route cannot apply or a value outside what it accepts raises
    `CapabilityNotSupportedError` (negative prompt, guidance, another fps, an unknown `extra` key, an adapter on
    another role, a canvas off the grid, a frame count that rounds outside 124 to 345: a count from 108 to 345 is
    rounded up to the next `17n+5` count, so 125 runs as 141); a value of the wrong type, or two fields that contradict
    each other, raises `ValueError`. A first frame is decoded first. Wan and the other local backends refuse the
    soundtrack fields.
  - Results are the MP4 with its AAC soundtrack, `GeneratedAsset.audio` (`GeneratedAudioTrack`: sample rate,
    channels, duration, codec, whether muxed, and the track as WAV bytes when it could not be muxed),
    `GeneratedAsset.audio_summary()` (the one shape the store, the plugin, the tools and HTTP report), and MLX-Gen's
    metadata under `metadata["mlx_gen"]` (the composed prompt, audio facts, memory peaks). With a store,
    `VisionManager` stores a soundtrack that could not be muxed as a WAV artifact of its own and records the summary
    as `audio` on the video's ref and metadata; without a store, the AbstractCore plugin returns the MP4 bytes and
    logs that the soundtrack is not in them.
  - `memory_mode="low"` frees the transformer before the decode, and the next request, a batch item included, loads
    the model again: with the 8-bit package, a 640x352, 124-frame clip peaked at 24.7 GiB alone, in sequence and
    in a batch, and building the instance again took 1.3 to 3.1 s per clip. `"lowest"` adds transformer block streaming (10.6 GiB on the
    same clip; the next request also loads the model again) and is refused with load-time `quantize` and with an
    adapter that trains the AdaLN projections.
  - The AbstractFramework packages are gated on Hugging Face, and the MiniMax H3 Community License's grant excludes
    the European Union, the United Kingdom, the Republic of Korea and the United States of America. AbstractVision
    never downloads these weights or the Turbo adapters during generation, even with `allow_download`; download them
    explicitly after accepting the licence (`abstractvision download MiniMaxAI/MiniMax-H3 --base-model
    minimax-h3-turbo-544p`).
  - CLI: `--soundscape`, `--music`, `--no-audio` and `--audio-shift` on `t2v` and `i2v`.
- **`memory_mode` on every generation request**: `default`, `low` or `lowest`, as a request field, a
  `VisionManager` keyword (batches included), `--memory-mode` on `t2i`, `i2i`, `upscale`, `t2v` and `i2v`, an
  AbstractCore plugin keyword, `extra["memory_mode"]` and a `make_vision_tools(..., memory_mode=...)` argument.
  `ABSTRACTVISION_MEMORY_MODE` sets a process default for requests that name none.
  - MLX-Gen: `low` is MLX-Gen's low-RAM mode as each route's `mlxgen` command applies `--low-ram`, with a 1 GB MLX
    cache from before the model loads. Image and edit routes free the text encoders and the transformer, Wan routes
    clear the cache every step and release the denoisers before the decode, SeedVR2 sets the cache limit only, and
    MiniMax-H3 frees the transformer before the decode. `lowest` adds transformer block streaming; it is accepted
    for MiniMax-H3 and refused on every other current catalog model.
  - Diffusers on CUDA: `low` and `lowest` run with model and sequential CPU offload; a request only raises the
    configured `cpu_offload`, never lowers it, and every result records the mode in `metadata["memory_mode"]`.
    stable-diffusion.cpp: `low` offloads the model parameters to the CPU, and its results record the mode too.
    OpenAI-compatible: the modes the remote lists for the model (see below), else `default` only.
  - A mode named on a request that the backend cannot honour raises `CapabilityNotSupportedError` before any model
    loads or any request is sent, in every backend; `VisionManager`, the plugin and the tools pass the value through
    (`make_vision_tools` refuses only a value that is not a memory mode). A process default the backend cannot
    honour runs as the nearest less aggressive mode the backend lists (`lowest` → `low` → `default`), with one
    warning per process.
  - Image, edit, Wan and MiniMax-H3 models run in `low` (MiniMax-H3 in `lowest` too) free part of the model, which
    is dropped when the run ends, whatever its outcome; the next request loads it again, and the AbstractCore plugin
    no longer lists it as loaded. A batch is a series of such requests: each item loads the model and drops it,
    so every item peaks as one `low` run and under 1.3 GiB stays held between items. Measured per item of a `low`
    batch: FLUX.2 [klein] 9B 8-bit at 1024x1024, 4 steps, 13.2 to 15.4 GiB and 0.5 to 2.5 s to build the instance;
    Wan 2.2 T2V-A14B 8-bit at 480x240, 17 frames, 4 steps, 29.0 to 29.5 GiB and 1.4 to 2.7 s (the weights are then
    read during the run). SeedVR2's `low` is the cache limit alone and keeps its model.
  - `low` and `lowest` hold MLX-Gen's 1 GB MLX cache limit from before the model loads to the end of the run, never
    above a limit already in force; overlapping runs share it, and the limit found before is restored when the last
    one ends. `default` runs set no limit: MLX-Gen applies its own (RAM/8, 1 to 8 GiB) and keeps one the host set.
  - Discovery: `get_capabilities().memory_modes`, `parameter_constraints["memory_mode"]`
    on each MLX-Gen catalog entry, and `memory_modes` per provider in the plugin's `available_providers()` (what
    that provider's backend lists for its configured model and device, without loading a model or sending a
    request). Diffusers with `device="auto"` lists `default` on macOS (no CUDA) without importing torch; elsewhere
    it asks torch, importing it once.
- **Downloads.**
  - `abstractvision download <model id>` selects one package. The host's target order picks the target (preferring
    an 8-bit package unless `--allow-non-8bit`; `--provider` or `--target` restricts it); within that target, a single
    package is selected, and several select the one the catalog marks as recommended (MiniMax-H3 and SeedVR2: the
    8-bit package; Qwen-Image-2512 and FIBO: the 4-bit package); with none recommended, the command
    stops and lists them, whatever `--bits`, `--provider` or `--allow-non-8bit` narrowed them to. `--bits 2|4|8|16`
    selects among the packages of that width only, in every target of the host, or stops saying none has it and
    listing the widths there are. A repository named exactly with `--allow-non-8bit`, or with its MLX-Gen engine named
    (`--provider mlx-gen`), selects that repository (when `--bits` is not given or is its width) on every host (`abstractvision download MiniMaxAI/MiniMax-H3
    --allow-non-8bit` fetches the MiniMax-H3 source repository, which the model id alone does not).
  - Every download is planned before anything is written, for every repository it fetches from: the package, a
    stable-diffusion.cpp bundle's companions (the text encoder and VAE a GGUF diffusion model needs), and with
    `--base-model <schedule>` the adapter that MLX-Gen schedule loads (the MiniMax-H3 Turbo LoRA). The files each
    selects are read from the Hugging Face listing (metadata only; a Diffusers pipeline's `model_index.json` is read
    into memory), and each is either complete in the cache or to fetch. The command prints the access (`open` or
    `gated`), the size and file count, how many files are complete and how many are to fetch, one line per companion
    or adapter, one total, and the free space of the cache disk. The files to fetch must fit there at their full
    size, less what their partial `.incomplete` files already hold (huggingface_hub writes each file into that same
    file, continuing it over plain HTTP and rewriting it from the start over Xet), a rule the `disk:` line states;
    otherwise the command stops (`InsufficientDiskSpaceError`) and writes nothing. It then fetches exactly the files
    to fetch, from the commit the listing named, and records that commit as the repository's `main`: a package whose
    files are all in the cache is reported as present and nothing is fetched, and one missing files (for example a
    Wan 2.2 A14B package downloaded by 0.3.36, without `transformer_2/`) gets them.
  - A `--base-model` that is not a schedule of the package, or an adapter that cannot be named because mlx-gen
    cannot be imported, stops the command before anything is fetched, naming what is missing. The adapter is fetched
    into huggingface_hub's cache, where mlx-gen finds it. Without `--base-model`, `download` lists the schedules
    whose adapter is absent, with the command that fetches it.
  - Ctrl-C ends the command with exit code 130 and says that the files that completed stay in the cache and that
    the same command continues with the rest. A download that ends with files missing gives their count and one of
    them.
  - Without the Hub listing, `download` fetches nothing and says why (offline mode, the Hub unreachable, an HTTP
    error, huggingface_hub or a module it imports not installed, or a listing without file sizes, which huggingface.co
    always gives but a mirror set with `HF_ENDPOINT` may not), and what to do. It reports what the cache holds as it
    is, not compared with the Hub; when a local look sees a gap or a repository is missing, it stops. Generation with
    `allow_download` that finds the model cached makes no Hub request, as before.
  - `--dry-run` prints the same plan, the commit of each repository the fetch would take included; it writes
    nothing and exits 0, and when the download would not fit, its last line says so.
  - A Hugging Face `owner/name` the catalog does not list downloads the whole repository through the same plan,
    size and disk check, with its access from the listing.
  - The catalog marks gated repositories (`gated: true`). `download` says so before it starts, and a gated
    repository that refuses the account ends with `HuggingFaceAccessError` naming the repository page and the
    licence to accept. For a repository not marked gated, a 401 says what the Hub's answer means: the repository
    does not exist, is private, or is gated, and the Hub does not say which.
  - Python: `select_download_preset(name, *, target, engine, bits, allow_non_8bit)` (the selection `download`
    makes), `download_command(repo_id, *, engine, base_model)` (the command that selects exactly that repository on
    this host), `find_model_preset(..., bits=, download_rule=)`, `plan_model_preset_download(preset, *, base_model, model_dir, token,
    all_files)` (a `DownloadPlan` of `DownloadPart`s, each with its `HubDownloadPlan`), `download_model_preset(...,
    base_model=, report=)`, `repository_preset(repo_id)`, `abstractvision.model_inventory.preset_status(preset)` and
    `local_cache_state(preset)`, and `abstractvision.backends.mflux.resolve_mlx_gen_schedule_adapter(base_model,
    download=False)`.
- **Model listings.** `catalog` marks the recommended package with `*` in `bits` and adds the `access`, `cache`
  and `memory_modes` columns. `show-model` adds, per download, its access, cache state with size on disk, the memory
  modes this host can run and, for MLX-Gen packages with published measurements, the peak memory of each mode and
  the smallest memory it was tested on (read from the installed mlx-gen), plus the schedules and the adapter each
  loads. `models --json` and `show-model --json` print the same as JSON. The peaks come from
  `abstractvision.model_inventory.memory_peaks(repo_id)`, also the AbstractCore plugin's `memory_peaks(model)`: one
  row per measured run of that package's stored width, with its `task` and `memory_mode`; a run names no schedule,
  and mlx-gen publishes the same runs on every MiniMax-H3 schedule. The cache state is a local look, offline:
  `cached` is a snapshot with no gap it can see (no partial file, link without content or missing weight file an
  index names, every `model_index.json` component whose folder the package's patterns name present, at least one
  weight file), `incomplete` names the gap, `absent` is nothing in the cache: the same three words in the `catalog`
  table, `show-model` and `--json`. Only `download` compares a snapshot with
  the Hub, so a file that nothing local refers to, such as one an interrupted download never started, can be
  missing from a snapshot shown as `cached`.
- **OpenAI-compatible backend with AbstractCore's server.** A remote that lists `memory_modes` for the model in
  `GET {base_url}/vision/models` (the entry whose `routed_model` is the model sent, `extra["model"]` included, which
  such a remote receives as given, or `defaults[<task>]` when the request names none) receives `memory_mode`, `quantize` and the soundtrack fields it advertises; a
  request with `cancel_event` runs there as a remote job, and setting the event cancels it (AbstractCore's server
  refuses such a job while another generation runs, which the backend raises as a `RuntimeError` saying so); a
  video response's `audio` fills `GeneratedAsset.audio`, and `metadata["memory_mode"]` holds the mode the
  remote reports it ran. Such a remote receives the request's other `extra` keys nested (an `extra` object in JSON,
  an `extra_json` form field). Any other remote refuses these fields with `CapabilityNotSupportedError` before
  anything is sent, and receives `extra` at the top level as before. A request with `cancel_event` to the OpenAI
  API, to a remote whose discovery route answers HTTP 404, 405 or 501 (an answer, kept), or to a vision-route remote
  for a model it does not list, runs as a plain request, the event checked only before sending; while the remote
  has not answered (its discovery request failed), such a request raises the `RuntimeError` of a failed request.
- **For applications embedding AbstractVision.**
  - `cancel_event` on every request (and AbstractCore plugin keyword): MLX-Gen stops between denoising steps and
    raises the new `GenerationCancelledError`; Diffusers and stable-diffusion.cpp check it before a run starts
    (`get_capabilities().cancellation`). What stays loaded after a cancelled MLX-Gen run is listed per model family
    in the API guide; the error carries no traceback of the run, so a caller that keeps it holds none of its memory.
  - Load-time `quantize` (bits) on generation requests, as `--quantize` and as an AbstractCore plugin keyword,
    validated against the route (`get_capabilities().quantize_bits`) and refused on prepared packages and on the
    other local backends. MLX-Gen and a remote that advertises it read it by one integer rule: a bool, a non-integral
    number or a string that is not a decimal integer raises `ValueError` naming `quantize`, and the remote receives
    the width as an integer (an upscale request and a SeedVR2 preload read `False` and the other unquantized
    spellings as an unquantized load, see Changed).
  - `VisionManager.describe_routes(model=..., task=...)` returns MLX-Gen's capability rows as JSON, so a host can
    build its controls and memory planning without importing MLX-Gen.
  - MLX-Gen results carry the engine's metadata under `metadata["mlx_gen"]` on every route, upscaling included.
  - On MiniMax-H3, weights or a run that MLX-Gen's preflight finds cannot fit in memory raise
    `GpuOutOfMemoryError` naming the `memory_mode` it used and the lower ones to try (MLX-Gen 0.40.1 has this
    preflight for MiniMax-H3 only). On Wan, a `MemoryError` raised while the model loads or runs is raised as the
    same `GpuOutOfMemoryError`.
  - Importing AbstractVision and constructing a `VisionManager` write no files, set no environment variables and
    import neither AbstractCore nor an engine.
  - `extra["memory_mode"]` is the memory mode on every backend; `extra["quantize"]` and the soundtrack keys are the
    options only on a backend that applies them (MLX-Gen, a remote that advertises them) and pass through `extra`
    unchanged elsewhere. A field and an `extra` value that disagree raise `ValueError`.
  - `CapabilityNotSupportedError.field` and `.supported` name the refused request field and the values accepted
    there, when a refusal is about one field (None otherwise); the message is unchanged. On MLX-Gen a refusal
    about one field names it (`memory_mode`, `quantize` with the accepted widths, `width`, `num_frames`, `steps`, the
    shifts, `fps`, `negative_prompt`, guidance, `lora_adapters`, `control_image`, `mask`); a refusal about two fields
    together (`memory_mode="lowest"` with load-time `quantize`, or with an adapter that trains the AdaLN projections)
    leaves both None.
  - `preload(memory_mode=None, quantize=None)` on every backend: the values are checked as a request's are, before
    anything loads, and the model loads as a request carrying them loads it, so that request reuses it; a backend
    that takes no load-time quantize refuses it. The AbstractCore plugin's `load_resident_model(...)` passes a
    request's `memory_mode` and `quantize`. On MLX-Gen the load runs under that request's cache hold; in a mode
    whose runs drop the model (`low`, `lowest`, and every mode on Wan TI2V-5B, which releases its denoiser before
    the decode), the preloaded instance serves one request and no warm-up runs.
    Only a text-to-image model kept by its mode runs one warm-up generation, in that mode; an edit-only model is
    loaded for edits.
  - MLX-Gen records `memory_mode` (with `low_ram` and `stream_blocks`) in every result's metadata, `"default"`
    included.
  - MLX-Gen checks a request once, on its own thread: `VisionManager` still calls the backend's `normalize_*`
    (any backend may rely on it), and the backend's entry point does not check the request it returned again.
    A task the configured model has no route for (an image task on MiniMax-H3) is refused before anything is read.
  - `VisionBackend.has_loaded_model()` says whether this process holds a backend's model weights now (MLX-Gen,
    Diffusers and stable-diffusion.cpp's Python binding answer; the default, `False`, is the answer of a backend
    that loads nothing here). The AbstractCore plugin lists and pins loaded models by it, asking it when called, and
    a load whose backend answers `False` returns `state: "not_loaded"` with a `reason`.

### Changed

- `download` of a catalog model with a single package (for example `stabilityai/stable-diffusion-xl-base-1.0`,
  `briaai/Fibo-Edit`, the alias `stable-diffusion`) selects that package without flags and fetches the files it
  selects (for a Diffusers pipeline, those the backend loads), where 0.3.36 fetched the whole repository of an
  `owner/name` and refused any other name without `--allow-non-8bit` (which selected the same package).
- `download` without flags selects the package 0.3.36 selected for every name it selected one for, on every host:
  where several packages of one target remain, the catalog now marks that package `recommended` (Qwen-Image-2512's
  4-bit MLX-Gen package, GGUF and Diffusers pipeline, FIBO's 4-bit package, SeedVR2's 8-bit packages), so `catalog`
  shows the `*` on these rows too. A model id for which 0.3.36 fetched the whole upstream repository unfiltered and
  that has several packages in the chosen target and none recommended (FLUX.2 [klein], Qwen-Image, Qwen-Image-Edit,
  Z-Image, ERNIE-Image-Turbo, the Wan 2.2 models, HunyuanVideo-1.5) stops and lists them; so does
  `tencent/HunyuanVideo-1.5` with `--provider diffusers` or `--allow-non-8bit`, where 0.3.36 took the first of its
  eight pipelines by name. Name a package, or, for a source repository, name it with `--allow-non-8bit`. On Apple
  silicon, `Qwen/Qwen-Image-2512`, `briaai/FIBO`, `ByteDance-Seed/SeedVR2-3B` and `-7B`, for which 0.3.36 fetched
  the whole upstream repository, select the recommended package; `--allow-non-8bit` fetches the repository itself.
- `download <owner/name> --allow-non-8bit` selects that repository on every host, where 0.3.36 did so on Apple
  silicon and picked the host's first target or package elsewhere (on an NVIDIA host, the GGUF package of
  `black-forest-labs/FLUX.2-klein-4B` rather than that Diffusers pipeline; on NVIDIA and processor-only hosts, the
  SeedVR2 8-bit and FIBO 4-bit packages rather than `ByteDance-Seed/SeedVR2-3B`, `-7B` and `briaai/FIBO`).
- `download <owner/name> --provider <engine>` (or `--target`) for a model with no package of that engine stops and
  lists the packages it has, where 0.3.36 ignored the provider and fetched the whole repository. With
  `--provider mlx-gen`, a model id that has several MLX-Gen packages stops and lists them, or selects the
  recommended one (`Qwen/Qwen-Image-2512` is the 4-bit package), where 0.3.36 fetched the whole upstream
  repository.
- A name that is neither in the catalog nor an `owner/name` points to `abstractvision catalog` instead of listing
  every known name.
- The one-shot commands' local-default notice gives the `download` command that selects the package that engine
  loads on this host (for Diffusers, `abstractvision download black-forest-labs/FLUX.2-klein-4B --allow-non-8bit`:
  the model id alone selects the GGUF package on NVIDIA and processor-only hosts and stops on Apple silicon), and no
  longer calls that default 8-bit.
- `--allow-non-8bit` is no longer needed for a model with a single package; it still lets the host's target order
  choose a target without an 8-bit package.
- `download` prints the package's own target and engine (a GGUF package says `gguf` and `stable-diffusion.cpp`).
- A legacy `~/models/<preset>` tree (or one under `--model-dir`) is still moved into the cache as before, and
  `download` then says it was imported and not compared with the Hub, and fetches nothing on top of it.
- `make_vision_tools`: a parameter the call leaves unset and the catalog lists without a default stays unset, so the
  engine applies its own default. Models whose catalog lists `steps` without a default no longer run at the tool's
  10 steps (15 for edits).
- AbstractCore plugin: `guidance_2` on `t2i` or `i2i` raises `CapabilityNotSupportedError` (it raised `TypeError`), as
  do the soundtrack options on an image call. Every other unknown keyword stays a backend `extra` parameter.
- MLX-Gen result metadata names the model and adapters by repo id (a Hugging Face cache snapshot) or by file or
  directory name, never by local path: every field of the engine's metadata that holds a path (`model`,
  `lora_paths`, `lora_application_reports`, the video file checked by `video_health`, and the `original_*` fields an
  upscale copies from its input), in `metadata["mlx_gen"]` and its top-level copies, and `requested_lora_adapters`.
  The fields naming the files written for a run (the source image or first frame, an edit's image list and mask, the
  clip and its audio sidecar) are left out.
  The refusals of a prepared package given `quantize`, of a model path that does not exist and of an adapter
  `lowest` cannot stream name the model or adapter the same way.
- MLX-Gen SeedVR2 upscaling checks `quantize` as the other families do: a prepared package refuses it (on
  `upscale_image` and `preload`), and the source accepts MLX-Gen's widths, refusing any other integer with
  `CapabilityNotSupportedError` (`field="quantize"`, it raised `ValueError`) and a non-integral number with
  `ValueError` (it was truncated). An upscale request still accepts `"none"`, `""`, `"0"`, `0`, `"false"`, `False`,
  `"off"` and `"no"` (in any case) and now loads unquantized for them on every path, as a SeedVR2
  `preload(quantize=...)` does: in 0.3.36 a direct backend call did, while `VisionManager` and the plugin loaded the
  8-bit default. `extra={"quantize": ""}` is now an unset option.
- MLX-Gen video requests are type-checked: a width, height, frame count, fps, step count or Wan
  `extra["max_sequence_length"]` that is not an integer, a guidance scale, `guidance_2` or shift (a field, or Wan's
  `extra["flow_shift"]`) that is not a finite number (NaN, infinity), or a seed outside `[0, 2**64 - 1]`, raises
  `ValueError` naming the field instead of being truncated or passed on; an image-to-video first frame that cannot
  be decoded raises `ValueError` naming `image`.
- MLX-Gen Wan results carry, under `metadata["mlx_gen"]`, the metadata MLX-Gen's save writes beside the clip (as
  MiniMax-H3 results do), which adds what only the save knows (video health, the process's peak memory).
- MLX-Gen refuses `control_image`, and `control_strength` given alone, with `CapabilityNotSupportedError` (`field`
  names it) before anything loads, and `get_capabilities().supports_control_image` is `False`: MLX-Gen runs structured control through its
  Qwen ControlNet variant and a ControlNet sidecar repository, which AbstractVision does not load. The catalog no
  longer lists `control_image` and `control_strength` for Qwen Image.
- An exception raised in a progress callback is logged once and no longer ends the run; stop a run with
  `cancel_event`.
- OpenAI-compatible backend: `list_provider_models()` and a request that sets a memory, sound or `quantize` field
  or a `cancel_event` (or runs under `ABSTRACTVISION_MEMORY_MODE`) also query `GET {base_url}/vision/models` (10 s
  timeout, answer reused for 30 s; never for the OpenAI API); `get_capabilities()` reads
  the last answer without a request, and a request that sets none of them, carries no `extra` and names its model
  without an `openai/` or `openai-compatible/` prefix makes the same HTTP calls as before. A discovery request that fails (timeout, connection error, an HTTP 5xx other than 501) is not kept: the
  previous answer stays in use, and without one a request that names a memory mode other than `default`,
  `quantize` or a soundtrack field is refused with the reason, and one with a `cancel_event` raises the
  `RuntimeError` of a failed request. A discovery answered with HTTP 501, like 404 and 405, is an answer. A remote HTTP 422 with the error code `capability_not_supported`, and
  a remote job that fails with that `error_code`, raise `CapabilityNotSupportedError` with the remote's message,
  `field` and `supported` instead of `RuntimeError`.
- OpenAI-compatible backend: `list_provider_models()` on a remote that serves the vision routes (AbstractCore's
  server) lists its `GET {base_url}/vision/models` catalog, each model by the `routed_model` a request sends, with
  its memory modes and audio support; it read that server's `GET /models`, which lists text models. Other remotes
  are listed from `GET /models` as before.
- OpenAI-compatible backend: `list_provider_models()` against a remote that answers `GET /models` with HTTP 404, 405
  or 501 raises `CapabilityNotSupportedError` (the remote has no model catalog) instead of `RuntimeError`. The
  AbstractCore plugin's `list_provider_models()` raises it as it is when no backend has a catalog, as it does when
  the selected backend cannot list models, instead of `AbstractVisionError`; a backend without one is skipped when
  another answers.
- OpenAI-compatible backend: a request that carries `extra` (keys other than `model`), or names its model with an
  `openai/` or `openai-compatible/` prefix, first asks the remote with `GET {base_url}/vision/models` whether it
  serves the vision routes (never the OpenAI API). The answer to that question is kept for the backend's life (the
  memory modes and options it lists are still reused for 30 s, as above); a question that fails
  (timeout, connection error, an HTTP 5xx other than 501) is asked again after 30 s, and until then the remote's kind
  is unknown. A
  remote that serves them receives `extra` nested, as described under New, and the model as named; any other
  remote, or one whose kind is unknown, receives `extra` at the top level and the model without the prefix. A model
  named without such a prefix is sent as given to every remote.
- OpenAI-compatible backend and AbstractCore plugin: a base URL is the OpenAI API when its host is `api.openai.com`
  or one of its subdomains (such as `eu.api.openai.com`); a URL that has `api.openai.com` only in its path or
  query, or a look-alike host, is an OpenAI-compatible remote.
- AbstractCore plugin: a backend given through `vision_backend_instance` or `vision_backend_factory` (called once,
  when the capability is built) that lacks `has_loaded_model`, `preload`, `unload`, `supported_memory_modes`,
  `list_provider_models` or `list_provider_adapters` is refused when the capability is built, with a `TypeError`
  that names the missing methods and says to inherit `abstractvision.backends.base_backend.VisionBackend`, which
  provides them. Unloading the configured backend by provider or `load_id` calls its `unload()` whether or not it
  is listed. A `vision_backend_factory` that raises fails the capability's construction with that error (0.3.36
  ignored it and built the configured default backend), and an injection added to the owner's config after the
  capability is built is not read.
- Diffusers refuses a model that is not in the cache and that generation may not download (downloads off, or a
  repository marked `explicit_download`) before loading anything; a cached model's own load error is raised as it is.
- The cache lookups of model resolution, the Diffusers refusal, the `cache` column and `show-model` also read the
  cache huggingface_hub reads, by its own precedence (`HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME`,
  `XDG_CACHE_HOME`), so a model that only `HUGGINGFACE_HUB_CACHE` or `XDG_CACHE_HOME` locates is found; downloads
  write where they did.
- `HuggingFaceAccessError` says that the repository is gated and names the licence when the catalog marks it gated,
  otherwise what the HTTP status tells (401, 403, 404), and keeps the first line of the upstream error.
- Diffusers recognises a CUDA out-of-memory by its type (`torch.cuda.OutOfMemoryError`), no longer by the words
  "CUDA out of memory" in a message, and its `GpuOutOfMemoryError` no longer says how much more memory the run
  needed (that figure was read from torch's message).

### Fixed

- MLX-Gen reads the `base_model` of a model or adapter card that starts with a byte-order mark; such a card was read
  as having no front matter.
- Downloads of the Wan 2.2 A14B MLX-Gen packages fetch the low-noise expert (`transformer_2/`), and MLX-Gen package
  downloads fetch the chat-template files (`*.jinja`) the Qwen, ERNIE and Bonsai tokenizers load; both were left out.
  Running `download` again on a package downloaded by 0.3.36 fetches them.
- MLX-Gen releases the loaded model, and clears the MLX cache, before it loads another one (another model or
  adapter set): two instances are never in memory together.
- A failed MLX-Gen `preload()` drops the model it loaded before raising, and the AbstractCore plugin does not list
  it as loaded. A preload refused for its values, or one that fails after reusing the model already held, leaves
  that model held. A failed preload that had to replace the held model (another `quantize`, mode or adapter set)
  leaves nothing loaded.
- AbstractCore plugin: `control_image` and `control_strength` reach `t2i` (they were dropped), where every built-in
  backend refuses `control_image`, and MLX-Gen also refuses `control_strength` given alone (the other built-in
  backends do not use it).

### Removed

- `abstractvision.cli.looks_like_hf_repo_id`, a name the CLI module imported: use
  `abstractvision.model_downloads.looks_like_hf_repo_id`, where it is defined.
- `abstractvision.cli.download_hf_repo_snapshot`, a name the CLI module imported: use
  `abstractvision.model_downloads.download_hf_repo_snapshot`, where it is defined (it now plans and checks the disk
  before fetching).
- `abstractvision.backends.mflux.WAN_VIDEO_MLX_CACHE_LIMIT_CEILING_BYTES` and `WAN_VIDEO_MLX_CACHE_LIMIT_FLOOR_BYTES`:
  AbstractVision no longer sets an MLX cache limit for Wan `default` runs; MLX-Gen applies its own (RAM/8, 1 to
  8 GiB), and `low` / `lowest` runs hold 1 GB (see `memory_mode`).

## 0.3.36 - 2026-11-11

- **The one-shot commands run locally by default**: `abstractvision t2i`, `i2i` and the other generation commands
  pick this machine's engine and FLUX.2 [klein] 4B in 8 bits when no `--provider`, `--model`, `ABSTRACTVISION_PROVIDER`
  or base URL is given: MLX-Gen with `AbstractFramework/flux.2-klein-4b-8bit` on Apple silicon, stable-diffusion.cpp
  with the Q8_0 GGUF of `black-forest-labs/FLUX.2-klein-4B` on an NVIDIA card (the artifact `abstractvision download`
  already picks there), Diffusers with the same model on a processor-only host. The command prints its choice and the
  download command for the model. A `--base-url` (or `$OPENAI_BASE_URL`) selects the OpenAI-compatible HTTP backend as
  before; `--provider` keeps winning. Until now the commands defaulted to the remote backend on every platform and
  stopped with `Missing --base-url`. Tests cover the three profiles, the base URL and the explicit provider.
- Docs: a platform check of 0.3.35 (`docs/benchmark.md`) with the measured generation times per model on Linux
  without a GPU, Linux with an NVIDIA card and Apple silicon, linked from the README and the docs index.

## 0.3.35 - 2026-11-10

Linux release check: this version was run end to end on an Ubuntu 24.04 host with an NVIDIA Quadro RTX 5000
(16 GB, Turing) and on a processor-only Ubuntu 26.04 host with Python 3.14. On the card, `abstractvision[gpu]`
installs with no compiler, the Diffusers backend picks `cuda` and float16 (the fast dtype on Turing, which has no
native bfloat16), FLUX.2 [klein] 4B generates 1024² in 18-20 s with 8.6 GB of GPU memory and about 24 GB of host RAM
for model CPU offload, the model stays resident between calls and `unload` returns the memory; the SDXL family, the
turbo models, SD 1.5, Kolors, Lumina-2, PixArt-Sigma, the inpaint and instruct-pix2pix pipelines and the GGUF path
(FLUX.2 klein, Z-Image, Qwen-Image-2512, Qwen-Image-Edit-2511 through `sd-cli`) all ran at their catalog defaults.
On the processor the backend picks `cpu` and float32 and generates without a CUDA warning.

- One-line standalone install: `curl -LsSf https://abstractframework.ai/vision/install.sh | sh` (macOS/Linux) and
  `powershell -ExecutionPolicy ByPass -c "irm https://abstractframework.ai/vision/install.ps1 | iex"` (Windows),
  `scripts/install.sh` / `scripts/install.ps1`: no admin rights, uv and Python 3.12 installed when missing,
  `abstractvision[<profile>]==<version>` as an isolated user tool; the profile is detected like the framework
  installer's (Apple Silicon + macOS 14 -> `apple`, NVIDIA or ROCm -> `gpu`, else `light`); `--print`, `--profile`,
  `--pin`, `--from`; a re-run upgrades in place.
- Diffusers on the processor loads every component in float32 (the CPU default dtype was unset, so Transformers 5
  loaded float16 text encoders next to a float32 UNet and every generation failed with a dtype mismatch).
- `local_model_profile()` reports `cpu` when `CUDA_VISIBLE_DEVICES` is empty or `-1` (it said `cuda`).
- A CUDA out-of-memory during a Diffusers generation is raised as `GpuOutOfMemoryError`, naming the model's size,
  its placement, what was free and what to change, instead of torch's traceback.
- 8-bit GGUF presets download and run their `Q8_0` file only (they fetched every quantization of the repository:
  258 GB for Qwen-Image-2512's 22 GB file, 92 GB for Z-Image, and then ran the BF16 file).
- `sd-cli` auto-install picks the newest stable-diffusion.cpp release whose Linux build the host's glibc can run
  and skips ROCm builds (since master-1044 the Linux builds need glibc 2.43, so every GGUF call failed on Ubuntu
  24.04); a host that already installed a newer build deletes `~/.abstractvision/bin` once.
- Z-Image GGUF presets bring their text encoder (Qwen3-4B Q4_K_M) and VAE, so `sd-cli` can build its context.
- The GGUF path runs at the catalog's steps and guidance (it ran `sd-cli`'s 20 steps and cfg 7 for every model);
  guidance below 1 is sent as cfg 1 for turbo and distilled models.
- Qwen-Image-Edit-2511 on `sd-cli` passes `--model-args qwen_image_zero_cond_t=true` (the flag it used did not exist).
- On a CUDA host, `abstractvision download <model>` no longer picks the FP8 side artifacts (for example
  `black-forest-labs/FLUX.2-klein-4b-fp8`): no AbstractVision backend loads them. The CUDA auto order is gguf,
  diffusers, hf-snapshot; FP8 is left out rather than moved last because downloads are 8-bit by default and FP8 is
  the only 8-bit artifact of flux1-dev and flux2-klein-9b-kv. `--target fp8` still lists and downloads them.
- A Diffusers download fetches only what the backend loads on this host: each component listed in
  `model_index.json` with one safetensors set (fp16 on CUDA and Apple silicon, the default set on the processor),
  without root single-file checkpoints or `.bin` / `.onnx` / `.msgpack` copies. Measured on the Quadro RTX 5000:
  stabilityai/sdxl-turbo went from 41.6 GB in 162 s to 6.9 GB in 38 s and generated the same image; FLUX.2-klein-4B
  goes from 23.7 GB to 16.0 GB. `abstractvision download --all-files` restores the full download. AbstractCore's own
  Diffusers route downloads go through its model materializer and are not trimmed by this.
- `sd-cli` guidance follows the model family: FLUX dev models get `--guidance <catalog value>` with `--cfg-scale 1`
  (their guidance is distilled, not classifier-free), Qwen-Image-Edit defaults to cfg 2.5 as stable-diffusion.cpp
  documents (checked on one edit with the same seed: 2.5 gives the asked-for subject, 1.0 a hybrid). The sd-cli
  installer requires the host architecture in the build it picks, refuses Ubuntu builds newer than its glibc table and
  says when no build fits; GGUF preset downloads take root files only; a valueless `--model-args` gets its value in
  place; on a GPU out-of-memory the failed step's memory is released before the free figure is reported.

## 0.3.34 - 2026-10-18

- Endpoint profiles: a request whose provider is `endpoint:<id>` (an OpenAI-compatible connection kept as an
  AbstractCore provider profile or by a host such as AbstractGateway) now runs on that endpoint, with the
  profile's base URL and API key, through the OpenAI-compatible backend. Before, the provider fell through
  to the configured default backend, so an image route set to an endpoint serving `gpt-image-2` could not
  reach it. The profile is resolved by the owning Core provider's host resolver
  (`resolve_provider_endpoint_profile`), then the ambient resolver (`abstractcore.providers.endpoint_context`),
  then AbstractCore's `provider_profiles`; an unknown or disabled profile, or one of another family than
  `openai` / `openai-compatible`, fails with an error that never contains the key. An `openai-compatible`
  endpoint without a key never borrows `OPENAI_API_KEY`. Tests: `tests/test_endpoint_profile_routing.py`.

## 0.3.33 - 2026-09-29

- Diffusers on CUDA: `cpu_offload="auto"` falls back to Diffusers' sequential CPU offload when the
  GPU's free memory (minus the working reserve) cannot hold the pipeline's largest component,
  which model CPU offload moves to the GPU whole. This covers a GPU shared with another loaded
  model: on a 16 GB Quadro RTX 5000 with 6.5 GiB held by a text model (about 7.8 GiB free),
  FLUX.2 [klein] 4B in float16 (largest component about 7.5 GiB) leaves model CPU offload no working
  room. Sequential CPU offload uses much less GPU memory (about 1.4 GiB peak there) and each step is
  slower: measured at 768x768 and 4 steps, 10.9-15.6 s against 13.4-18.5 s with model CPU offload,
  but about 2.3 s per step against 1.5 s, so it is slower for models that run many steps. The
  backend logs a warning that says why it was chosen, and the result metadata reports
  `cpu_offload: "sequential"`. The whole-pipeline and model CPU offload placements are unchanged
  when they fit.
- `HuggingFaceDiffusersBackendConfig(cpu_offload=...)` accepts `"sequential"` to always use
  sequential CPU offload; `"model"` and `"none"` keep their meaning. A pipeline that does not
  support sequential offload logs a warning and uses model CPU offload instead.
- Diffusers: unloading a model now releases its memory. The pipeline (including its CPU offload
  hooks) is freed when the unload returns, and on Linux the freed memory is returned to the
  operating system. Measured on CUDA (FLUX.2 [klein] 4B with model CPU offload): 0.86 GB still
  held after the unload, down from 16.3 GB. This matters when a host such as AbstractGateway
  unloads an image model to make room for the next one.
- AbstractCore plugin: `load_resident_model` reports `loaded_new` — `true` when the call loaded the
  model, `false` when it was already in memory — so AbstractCore and AbstractRuntime report a fresh
  image-model load as `loaded` instead of `already_loaded`.

## 0.3.32 - 2026-09-29

- Install profiles: the `gpu` and `all-gpu` extras no longer install MLX-Gen. They are the
  NVIDIA/Linux profiles: `gpu` installs Diffusers/torch and `all-gpu` adds stable-diffusion.cpp.
  On Linux, MLX-Gen pulled `mlx[cuda13]` (about 2.1 GB of CUDA 13 wheels next to torch's own CUDA
  runtime) that AbstractCore never routes on CUDA hosts, and it required glibc 2.35 or newer. The
  `gpu` and `all-gpu` installs are now that much smaller and install on Linux with an older glibc.
  MLX-Gen still comes with `apple`, `all-apple` and `all`; to try it on Linux, install the explicit
  `mlx-gen` extra (it installs there but is not validated; see backlog 030).
- Install profiles: `all-gpu` no longer installs stable-diffusion.cpp on Windows
  (`stable-diffusion-cpp-python; sys_platform != 'win32'`). PyPI ships it as a source build only,
  and that build needs MSVC (an elevated install), so `abstractcore[gpu]` could not install on
  Windows with wheels. Diffusers on PyTorch's CUDA build covers image generation there (not yet validated on Windows
  hardware); Linux and
  macOS keep stable-diffusion.cpp in `all-gpu` (framework backlog 0988).
- Diffusers on CUDA: a pipeline whose weights do not fit the GPU's free memory (minus a working
  reserve of max(1.5 GiB, 10% of the card)) now loads with Diffusers' model CPU offload (one
  component on the GPU at a time) instead of being moved whole and failing with CUDA out of memory.
  Measured on a 16 GB Quadro RTX 5000 (framework backlog 0989): FLUX.2 [klein] 4B in float16
  (14.9 GiB of weights) went from OOM to 768x768 in about 17 s at 4 steps, 7.8-8.3 GiB peak.
  `HuggingFaceDiffusersBackendConfig(cpu_offload="auto" | "model" | "none")`; the result metadata
  reports the execution device (`cuda:0`, not the CPU the idle weights wait on) and
  `cpu_offload: "model"`.

## 0.3.31 - 2026-09-29

- MLX-Gen Wan video: every Wan model has its own default generation size, kept in one table
  (`WAN_VIDEO_DEFAULT_CANVASES` in `backends/mflux.py`) that the backend, provider discovery and
  the packaged capability registry follow. TI2V-5B now defaults to 832x480, 121 frames at 24 fps,
  50 steps (was 1280x704), with `flow_shift=3.0` at that size; T2V-A14B and I2V-A14B default to
  832x480, 81 frames at 16 fps, 40 steps (was 1280x720). Pass `width`, `height`, `num_frames`,
  `fps` and `steps` (CLI `--width/--height/--frames/--fps/--steps`) to generate at any other size.
- MLX-Gen Wan video decodes in overlapping spatial tiles by default, so the VAE decode no longer
  sets the run's memory peak. Measured on Apple silicon (mlx-gen 0.38): TI2V-5B at 832x480 peaks at
  16.3 GiB (23.4 GiB untiled) and at 1280x704 at 25.4 GiB (51.2 GiB untiled); T2V-A14B at 832x480
  peaks at 38.3 GiB (47.4 GiB untiled). Pass `extra={"vae_tiling": False}` (CLI `--no-vae-tiling`)
  to decode whole frames at once. Generated video metadata records `vae_tiling`.
- MLX-Gen Wan video caps MLX's allocator cache while a video generates and decodes, and releases the
  TI2V-5B denoiser before the decode (the next TI2V request loads it again, which takes about a minute). This keeps the
  process footprint close to the MLX peak instead of holding every freed decode buffer.

## 0.3.30 - 2026-09-27

- MLX-Gen: unloading a model now returns MLX's cached Metal memory to the operating system. Previously the released weights could stay reserved in the MLX allocator cache, so a process with no loaded model could still hold most of the model's memory. No API change.

## 0.3.29 - 2026-09-23

- Runtime dependency: raise the optional MLX-Gen runtime floor to `mlx-gen>=0.38.0,<0.39.0` in every extra that installs it (`mlx-gen`, `mflux`, `apple`, `gpu`, `all`, `all-apple`, `all-gpu`). The MLX-Gen 0.38 line requires Python 3.10+, `transformers>=5`, and `Pillow>=12.1.1`; the AbstractVision backend call surface is unchanged.

## 0.3.28 - 2026-06-16

- MLX-Gen capabilities: align advertised `supports_mask` with the backend family gate so FLUX.2 models do not report mask support from inpaint-only route metadata when masked edits are implemented only for Qwen Image Edit and FIBO Edit families.
- AbstractCore plugin: probe provider catalogs concurrently so `list_provider_models` waits for the slowest backend instead of the sum of serial scans.

## 0.3.27 - 2026-06-16

- Runtime dependency: raise the optional MLX-Gen runtime floor to `mlx-gen>=0.18.19,<0.19.0`.
- MLX-Gen Qwen routes: expose validated base-Qwen structured control through typed `control_image` / `control_strength` and validated Qwen 2511 masked edit through `mask`, with explicit fail-closed backend gating instead of older family-only assumptions.
- Adapter capability registry: add packaged `vision_adapter_capabilities.json` plus `VisionAdapterCapabilitiesRegistry` so package-owned adapter defaults, compatibilities, docs links, and quantization guidance are data-driven instead of hard-coded in Python branches.
- CLI, provider catalog, and docs: surface structured-control parameters, adapter recommendations, registry override paths, and the updated MLX-Gen Qwen route guidance consistently across the CLI, REPL, AbstractCore integration, troubleshooting docs, and AI-readable indexes.

## 0.3.26 - 2026-06-13

- AbstractCore plugin surface: expose `list_provider_adapters(...)` plus first-class batch methods for `t2i`, `i2i`, `t2v`, and `i2v`, keeping stacked LoRA adapters, explicit seed planning, and route-specific video controls on the same typed boundary used by direct `VisionManager` callers.
- AbstractCore tool helpers: bring `make_vision_tools(...)` up to the same contract with adapter discovery, stacked LoRA adapters, batch helpers, and multi-reference image-edit inputs.
- Core-hosted proof: expand regression coverage so the plugin contract is explicitly validated for installed-adapter discovery, image batch seed expansion, and video batch forwarding.
- Packaging docs: clarify that `abstractvision[local]` covers the Diffusers + `sdcpp` runtime stack, while MLX-Gen stays explicit through `abstractvision[mlx-gen]` or the aggregate `all` / `all-apple` profiles.

## 0.3.25 - 2026-06-13

- Provider discovery: fix SeedVR2 local prepared-folder discovery for bare local folder names such as `seedvr2-7b-8bit` and `seedvr2-7b-4bit`, so clean environments surface both exact local upscaler variants without relying on a populated Hugging Face cache.

## 0.3.24 - 2026-06-13

- Provider discovery: fix SeedVR2 local prepared-folder discovery so `AbstractFramework/seedvr2-7b-8bit` and `AbstractFramework/seedvr2-7b-4bit` both surface correctly from a configured local model directory even when no Hugging Face cache is available, while keeping the canonical base alias aligned with the preferred local variant.

## 0.3.23 - 2026-06-13

- Runtime dependency: raise the MLX-Gen optional runtime floor to `mlx-gen>=0.18.18,<0.19.0` and expose the same upstream install surface that now includes Linux/CUDA markers in addition to Apple Silicon.
- Shared LoRA contract: add `LoRAAdapterSpec` plus typed `lora_adapters` request fields across `ImageGenerationRequest`, `ImageEditRequest`, `VideoGenerationRequest`, and `ImageToVideoRequest`.
- MLX-Gen integration: route typed LoRA adapters through local `t2i`, `i2i`, `t2v`, and `i2v`, keep backend-default adapters as a separate config path, and preserve adapter provenance in generated metadata.
- CLI/Core integration: add first-class repeated `--lora*` flags for one-shot MLX-Gen image/video commands and forward `lora_adapters` plus typed `guidance_2` through the AbstractCore plugin surface.
- Provider discovery: surface MLX-Gen route-level LoRA truth (`supports_lora`, `lora_status`, `lora_target_roles`, `lora_validation_profile`) through catalog/provider-model discovery instead of duplicating compatibility truth in the packaged registry.
- Batch generation: add and document first-class batch orchestration through `VisionManager.generate_image_batch(...)`, `edit_image_batch(...)`, `generate_video_batch(...)`, `image_to_video_batch(...)`, plus CLI `--count` / `--seeds`.
- Adapter inventory hygiene: add route-aware installed-adapter discovery through `abstractvision adapters ...` / `VisionManager.list_provider_adapters(...)` and filter full-model component snapshots out of that inventory so only optional overlays are surfaced as adapters.
- Wan video validation: reject undersized TI2V-5B requests below `832x480` / `480x832` in the MLX-Gen backend boundary and publish only validated `832x480` TI2V and `480x240` task-specific Wan A14B proof assets.
- Docs/tests: add a bundled MLX-Gen proof page with current batch image/video runs, stacked-adapter image-edit evidence, validated TI2V-5B and Wan A14B visual proofs, progress logs, provider-adapter discovery evidence, and expanded regression coverage for exact Qwen edit routing, CLI propagation, plugin propagation, Diffusers compatibility bridging, adapter inventory filtering, and packaging markers.

## 0.3.22 - 2026-06-07

- Runtime dependency: raise the MLX-Gen optional runtime floor to `mlx-gen>=0.18.14,<0.19.0` so Apple Silicon installs pick up SeedVR2 large-output VAE decode stability, official base-model support, canonical q8/q4 packages, and the latest image-edit fixes.
- Image upscaling: add first-class `ImageUpscaleRequest`, `VisionManager.upscale_image(...)`, backend `upscale_image_with_progress(...)`, and canonical `image_upscale` capability metadata.
- MLX-Gen SeedVR2: surface official `ByteDance-Seed/SeedVR2-3B` and `ByteDance-Seed/SeedVR2-7B` base models plus canonical `AbstractFramework/seedvr2-{3b,7b}-{8bit,4bit}` packages. Short selectors such as `seedvr2-3b` resolve to the recommended 8-bit package, while exact q4/q8 model ids remain exact.
- CLI/Core integration: add one-shot `abstractvision upscale` with default `mlx-gen/AbstractFramework/seedvr2-3b-8bit`, AbstractCore plugin tool support, and residency/catalog support for upscaler-only models.
- Progress events: relay SeedVR2 denoise-step progress through the same normalized progress event surface used by image/video generation, with `task="image_upscale"`.

## 0.3.21 - 2026-06-04

- Wan A14B guidance controls: promote MLX-Gen `guidance_2` to a first-class `VideoGenerationRequest` / `ImageToVideoRequest` field, CLI `--guidance-2` flag, REPL flag, playground job parameter, and AbstractCore tool parameter.
- Model registry: declare `guidance_2` defaults for Wan 2.2 A14B text-to-video and image-to-video tasks so discovery, tool defaults, and provider catalogs agree.
- OpenAI-compatible video: forward typed `guidance_2` to custom text-to-video and image-to-video endpoints that support the Wan-style two-stage guidance contract.

## 0.3.20 - 2026-06-04

- Runtime dependency: raise the MLX-Gen optional runtime floor to `mlx-gen>=0.18.10,<0.19.0` so Apple Silicon installs pick up the current capability planner, shared image/video progress events, and multi-reference edit routing.
- MLX-Gen progress: surface image generation and image edit progress through backend `*_with_progress(...)`, Python/Core `on_progress(event)` callbacks, and opt-in one-shot CLI `--progress` flags while preserving video progress availability for shell, Python, and AbstractCore callers.
- MLX-Gen progress semantics: align normalized video progress, the CLI display, and backend progress-bar callbacks with MLX-Gen denoise-step progress while retaining video frame counters as contextual event fields.
- MLX-Gen multi-reference edits: allow FLUX.2 and Qwen Image Edit requests to pass additional reference images through `extra.reference_images` / CLI `--reference-image`, and mark the supported model task metadata accordingly.
- Docs/tests: document image/edit/video progress across CLI, Python, and AbstractCore plugin calls; add propagation coverage for manager, CLI, plugin, Flux.2 multi-reference edit paths, and MLX-Gen denoise-step video progress normalization.
- Examples: add a bundled MLX-Gen local example gallery with copy-paste image/edit/video commands, generated PNG/MP4 outputs, contact sheet, manifest, and AbstractCore plugin progress-event capture.

## 0.3.19 - 2026-06-03

- Docs: clarify MLX-Gen `0.18.8` setup, Wan 2.2 A14B text-to-video/image-to-video usage, and generated-media configuration references.
- AI-readable docs: refresh `llms.txt` and `llms-full.txt` for the current AbstractVision Apple profile and video model surface.

## 0.3.18 - 2026-05-31

- Runtime dependency: raise the MLX-Gen optional runtime floor to `mlx-gen>=0.18.8,<0.19.0` so Apple Silicon installs pick up Wan 2.2 A14B text-to-video and image-to-video support.
- Wan A14B catalog: expose `Wan-AI/Wan2.2-T2V-A14B-Diffusers` and `Wan-AI/Wan2.2-I2V-A14B-Diffusers` through the canonical `mlx-gen` provider with task-specific defaults and capability metadata.
- Core propagation: prepare AbstractVision `0.3.18` so AbstractCore can depend on the corrected MLX-Gen runtime contract.

## 0.3.17 - 2026-05-29

- Docs: update the docs-site favicon and add an embedded text-to-video example carousel (5s MP4) to `docs/getting-started.md`.
- Runtime dependency: raise the MLX-Gen optional runtime floor to `mlx-gen>=0.18.7,<0.19.0` so Apple Silicon installs pick up the latest Wan video/model fixes.
- Core propagation: prepare the next AbstractVision package version (`0.3.17`) so AbstractCore can depend on the corrected MLX-Gen runtime contract.

## 0.3.16 - 2026-05-26

- MLX-Gen video progress: surface Wan `text_to_video` / `image_to_video` progress events through one-shot `abstractvision t2v` / `i2v`, interactive `/t2v` / `/i2v`, and Python/Core `on_progress(event)` callbacks.
- Backend API: add normalized `VideoProgressEvent` objects with phase, frame, total frame, step, total step, and progress fields while preserving backend `*_with_progress(...)` two-argument callbacks for generic progress bars.
- Capability filtering: keep Diffusers task support aligned with backend-specific registry requirements so MLX-Gen-only edit paths do not leak into Diffusers capability reports.
- AbstractCore integration: forward `on_progress` through plugin `t2v` / `i2v` requests so gateway/runtime callers can observe local MLX-Gen video runs without hardcoded backend plumbing.
- Docs/tests: document video progress usage across shell, REPL, Python, and AbstractCore examples; expand regression coverage for MLX-Gen progress bridging and quiet CLI flags.

## 0.3.15 - 2026-05-26

- CLI compatibility: canonicalize legacy `mflux/<model>` model prefixes to the `mlx-gen` runtime path for one-shot commands, matching `--provider mflux` and canonical `--provider mlx-gen` behavior.
- Tests: add one-shot `t2i`/`i2i` coverage proving the legacy `mflux` alias routes through the MLX-Gen backend without changing catalog/provider identity.

## 0.3.14 - 2026-05-26

- MLX-Gen migration: switch the Apple Silicon image runtime dependency to `mlx-gen`, expose `mlx-gen` as the canonical provider/engine, and keep `mflux` names as compatibility aliases.
- Catalog/model coverage: prioritize the AbstractFramework Hugging Face MLX-Gen q4/q8 prepared model collection, default q4 for local memory efficiency, and keep q8 variants discoverable for quality-focused runs.
- Local image support: expand the MLX-Gen backend/catalog surface for FLUX.2 klein/base, Qwen Image, Qwen Image Edit, Z-Image, Z-Image Turbo, and ERNIE Image Turbo q4/q8, including Qwen Image Edit `image_to_image` routing.
- Runtime dependency: require `mlx-gen>=0.18.5` for ERNIE Image Turbo q4/q8 prepared-folder support.
- Core/Gateway propagation: surface canonical `mlx-gen` provider/model metadata through AbstractCore and Gateway catalog paths while preserving legacy routed ids.
- Docs/tests: refresh CLI, backend, AbstractCore integration, and agent docs for MLX-Gen naming and q4/q8 guidance.

## 0.3.13 - 2026-05-23

- MFLUX edits: enable `image_to_image` for the curated FLUX.2 klein presets (`flux2-klein-4b`, `flux2-klein-9b`) so they surface in the playground edit UI and can be called through `VisionManager.edit_image(...)` (mask edits are not supported yet).
- Request normalization: map edit `strength` into the underlying MFLUX `image_strength` parameter and infer width/height from the input image when omitted so the UI/CLI behave consistently across backends.
- macOS GGUF policy: re-enable stable-diffusion.cpp GGUF presets on Apple Silicon by default and add an explicit opt-out (`ABSTRACTVISION_DISABLE_GGUF_ON_MACOS=1`) to hide GGUF catalog entries and reject GGUF execution for operators who want that behavior.
- Docs/tests: refresh README + core docs for the new MFLUX edit surface and the GGUF opt-out, and expand regression coverage for catalog surfacing and edit parameter plumbing.

## 0.3.12 - 2026-05-22

- Release packaging: publish the current clean tracked repository state as a fresh patch release.

## 0.3.11 - 2026-05-22

- Hugging Face cache integrity: treat repo-level `.incomplete` blobs as an incomplete snapshot, so interrupted Diffusers downloads no longer masquerade as a usable cached model.
- Qwen Image Edit downloads: `abstractvision download qwen-image-edit-2511 --provider diffusers` now resumes the real 16-bit snapshot instead of incorrectly short-circuiting on a partial cache.

## 0.3.10 - 2026-05-22

- Catalog/handles: make the curated Qwen Image Edit aliases consistent across the registry, downloads, and CLI catalog surfaces (`qwen-image-edit`, `qwen-image-edit-2509`, `qwen-image-edit-2511`), and re-verify all packaged Hugging Face repo ids in `vision_model_capabilities.json`.
- Runtime truth: quarantine unreliable local runtime task paths until they are re-validated. Local Diffusers `GLM-Image` is now hidden/disabled, local MFLUX is surfaced for `text_to_image` only, and local Diffusers `text_to_video` is marked experimental and disabled from the normal local surfaces.
- Playground/catalog UX: keep the per-task playground tabs with per-tab load/unload controls, but filter task selectors through backend-supported runtime truth so disabled local tasks do not surface. Update the `download` command messaging so curated 16-bit Diffusers snapshots are reported honestly instead of looking like a failed 8-bit fallback.
- Docs/backlog/tests: add the planned quarantine follow-up item, refresh the core docs around current local support policy, expand regression coverage for the runtime blacklists and catalog filtering, and regenerate `llms-full.txt`.

## 0.3.9 - 2026-05-21

- Local Diffusers video: implement the first in-process `text_to_video` path through `zai-org/CogVideoX-2b` / `THUDM/CogVideoX-2b`, including registry-backed request normalization, preload warmup, unload support, MPS-safe explicit FP16 device moves, and MP4 artifact outputs packaged through `ffmpeg`.
- Interactive surfaces: add one-shot `abstractvision t2v`, REPL `/t2v`, and playground `Text → Video` generation with backend/task-aware model discovery, progress reporting, and catalog surfacing for local video-capable models.
- AbstractCore integration: route plugin `t2v` / `i2v` calls through the same explicit backend binding path used by image generation, extend residency task aliases for video, and track request-warm local `text_to_video` models in the loaded-model inventory.
- Catalog/tests/docs: add curated capability/download metadata for `zai-org/CogVideoX-2b`, keep local image-edit surfaces intact, expand backend/CLI/playground/plugin regression coverage, refresh README/reference/FAQ/getting-started docs, and regenerate `llms-full.txt`.

## 0.3.8 - 2026-05-20

- Model registry/catalog: expand the curated `vision_model_capabilities.json` inventory with more current official or reputable vision model families and cross-platform download variants, tighten provider provenance, remove adapter-only repos from the standalone model catalog, and expose richer model notes / `task_specs` through the CLI catalog surfaces.
- Apple/MLX semantics: treat `mlx` as a download target rather than a runnable backend, fail fast on generic `mlx` provider/model routing across CLI, playground, and AbstractCore, and limit shipped MFLUX presets/docs to the families the backend actually supports today.
- GGUF / local-runtime routing: improve stable-diffusion.cpp bundle resolution for curated multi-artifact models and legacy Qwen GGUF repo ids, resolve curated `sdcpp` model keys consistently in both one-shot CLI and REPL paths, and make AbstractCore request-scoped `sdcpp` routing explicit while canonicalizing MFLUX aliases for backend reuse.
- Docs/policy/tests: add the ADR set and planned backlog items for curated catalog ownership, adapter-aware model graphs, and the long-term native MLX direction; refresh README/reference/FAQ/agent docs; regenerate `llms-full.txt`; and expand regression coverage for catalog aliases, GLM defaults, MFLUX routing, and local backend bundle selection.

## 0.3.7 - 2026-05-19

- AbstractCore plugin: add explicit local model residency control for in-process `diffusers`, `mflux`, and `sdcpp` backends through `load_resident_model(...)`, `list_loaded_models(...)`, `list_resident_models(...)`, and `unload_resident_model(...)`, with `load_model(...)` / `unload_model(...)` compatibility aliases for Core route adapters.
- Residency semantics: preserve explicit resident models across model switches while keeping the previous unload-on-switch behavior for non-resident request-warm backends, and report stable process-local `load_id` / `backend_kind` / `resident` metadata instead of relying on backend private fields.
- Safety/robustness: reject OpenAI/OpenAI-compatible HTTP backends for residency control even on localhost, normalize provider/task filters consistently, reject ambiguous unload requests, and keep unload behavior deterministic for injected local backends.
- Tests/docs: expand AbstractCore plugin coverage for local preload/list/unload flows, request-warm loaded-state reporting, task-aware loaded-model filters, injected backend residency, and switch-survival behavior; document the new local residency control surface and regenerate `llms-full.txt`.

## 0.3.6 - 2026-05-18

- Model registry/downloads: expand `vision_model_capabilities.json` and curated preset coverage for current Hugging Face text-to-image and image-to-image models, including FLUX.1/2, Qwen Image, ERNIE Image, GLM Image, Z-Image, SDXL, SD Turbo, and SD3.5. The registry now remains the packaged source of truth for both downloadable artifacts and task capability metadata.
- Cache/catalog: keep curated downloads in the Hugging Face cache by default, migrate older `~/models/<preset>` trees into that cache on first use, and align Diffusers/MFLUX cache discovery across CLI, REPL, playground, and AbstractCore so cached local models surface consistently.
- API-level normalization: move model-specific parameter normalization into shared backend hooks used by `VisionManager` and the playground server. Model constraints such as MFLUX distilled `guidance_scale=1.0`, unsupported negative prompts, and GLM `steps/guidance_scale/32-multiple dimensions` now apply consistently through the CLI/REPL, playground, and AbstractCore integration.
- Playground/API: expose structured per-model `task_specs` from `/v1/vision/models`, only enable the edit surface for models that advertise `image_to_image`, switch models by auto-loading the new selection, and truncate `b64_json` in logs instead of dumping the full base64 payload.
- AbstractCore/provider catalogs: keep remote provider/model listing robust when internet or API credentials are unavailable, and keep local provider/model catalogs aligned with the same cache-backed discovery used by the interactive surfaces.
- CLI/docs: keep `abstractvision cli` as the canonical interactive command (with `repl` as a legacy alias), expand model catalog/docs coverage, refresh backend/configuration/AbstractCore/playground references, and regenerate `llms.txt` / `llms-full.txt` for release.

## 0.3.5 - 2026-05-13

- AbstractCore plugin: accept runtime output metadata such as `provider` and `size` without rejecting image generation requests, and forward per-call image model selectors to OpenAI-compatible backends.


## 0.3.4 - 2026-05-09

- Packaging: constrain the optional `stable-diffusion-cpp-python` binding to
  `<0.4.6` for `sdcpp`, `local`, `apple`, `all`, `all-apple`, and `all-gpu`
  extras. Version `0.4.6` currently ships an sdist that fails native Linux
  builds because required vendored libwebm CMake files are missing.

## 0.3.3 - 2026-05-08

- Packaging: add shared native install profiles. `abstractvision[apple]` and
  `abstractvision[all-apple]` install the full local macOS vision stack;
  `abstractvision[gpu]` installs the Diffusers/Torch GPU stack; and
  `abstractvision[all-gpu]` installs Diffusers plus stable-diffusion.cpp
  bindings for full local GPU-capable deployments.
- Docs/tests: document and verify the Apple/GPU profile aliases while keeping
  the base package lightweight and remote/OpenAI-compatible by default.

## 0.3.2 - 2026-05-08

- AbstractCore plugin: switch the default backend id to `abstractvision:openai` with official OpenAI defaults, while keeping `abstractvision:openai-compatible` registered as a legacy-compatible backend id.
- OpenAI configuration: use standard `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_IMAGE_MODEL_ID`, and `OPENAI_IMAGE_MODEL` aliases for the plugin default path; keep compatible endpoints explicit with `ABSTRACTVISION_BACKEND=openai-compatible` and `OPENAI_BASE_URL`.
- Provider catalogs: add OpenAI/OpenAI-compatible `/models` discovery through `VisionManager.list_provider_models(...)`, `abstractvision provider-models`, REPL `/provider-models`, and `llm.vision.list_provider_models(...)` in the AbstractCore plugin.
- API/contracts: add `ProviderModelInfo`, provider catalog backend hooks, task filtering for OpenAI image model catalogs, and JSON-safe bounded provider metadata at the AbstractCore plugin boundary.
- Docs/agent context: refresh AbstractCore integration, backend/configuration/API docs, playground production-boundary notes, completed backlog reports, `llms.txt`, and regenerated `llms-full.txt`.
- Tests: add coverage for OpenAI defaults, legacy compatible aliases, provider catalog parsing/filtering, plugin catalog exposure, CLI/REPL catalog commands, manager delegation, and a credential-gated live OpenAI catalog check.

## 0.3.1 - 2026-05-07

- CI/release: move GitHub Actions checkout/setup-python/artifact/release actions to Node 24-compatible major versions, removing the Node 20 deprecation warnings from release runs.
- Tests: strengthen packaging metadata coverage so Diffusers aliases and `local`/`all` runtime bundles cannot drift from their intended dependency sets.
- Docs: separate contributor-only extras from runtime install profiles and explicitly mark `dev` as unsuitable for application/runtime dependency declarations.

## 0.3.0 - 2026-05-07

- Packaging: make the base install lightweight. `pip install abstractvision` no longer installs Torch, Diffusers, Transformers, Pillow, or local inference runtimes by default.
- Extras: add canonical runtime profiles `abstractvision[openai]`, `abstractvision[openai-compatible]`, `abstractvision[diffusers]`, `abstractvision[sdcpp]`, `abstractvision[local]`, and `abstractvision[all]`; keep `huggingface`/`huggingface-dev` compatibility aliases.
- Runtime defaults: keep AbstractCore plugin and one-shot CLI remote-first, and stop the REPL/playground from silently selecting Diffusers when no backend is configured.
- Errors/tests/CI: improve local-backend missing-extra hints, add stronger packaging/import-light/OpenAI-compatible coverage, and split CI into lightweight base and local Diffusers paths.
- Docs: update install guidance, backend references, and internal backlog policy for explicit local runtime extras.

## 0.2.6 - 2026-05-06

- Docs: refresh install extras, AbstractCore integration, playground ownership, OpenAI-compatible request-shape notes, and backend/config references so the public docs match the current code.
- Agent docs: update `llms.txt`, include playground endpoint docs in the generated `llms-full.txt` bundle, and regenerate the AI-ready documentation from current sources.
- Contributing: clarify that `abstractvision[abstractcore]` is a compatibility marker and that AbstractCore is supplied by the host application.

## 0.2.5 - 2026-05-06

- Packaging: keep the default Diffusers backend installable while moving `stable-diffusion-cpp-python` out of the base dependency set and into the explicit `sdcpp`/`local` extras. This keeps AbstractCore plugin installs from failing on platforms where stable-diffusion.cpp bindings need a local native build.
- AbstractCore plugin: restore the OpenAI-compatible HTTP backend as the default while keeping local `diffusers` and `sdcpp` backends explicit through config/env.
- OpenAI-compatible backend: shape requests correctly for real OpenAI GPT image models while preserving local OpenAI-compatible extensions for unknown model ids.
- Playground: capture the active backend at job submission time so background jobs do not accidentally run on a newly selected model.
- Docs/tests: clarify the default install shape, document when to install `abstractvision[sdcpp]`, and add metadata/OpenAI/playground coverage so these paths do not regress.

## 0.2.4 - 2026-05-06

- Playground: add a self-contained `abstractvision playground` command that serves both the web UI and `/v1/vision/*` API locally, so playground testing no longer depends on an AbstractCore server.
- Playground: package the HTML asset in the wheel, default the UI to the serving origin, and avoid stale persisted API URLs that could keep calling an older AbstractCore endpoint.
- Playground model loading: accept raw Hugging Face ids such as `runwayml/stable-diffusion-v1-5` directly, while still accepting explicit backend prefixes like `diffusers/...`, `sdcpp/...`, and `openai-compatible/...`.
- Packaging/CI: keep AbstractCore out of AbstractVision dependency metadata and test workflows; AbstractCore remains an optional host integration loaded lazily when present.
- Docs/tests: refresh playground docs around the self-contained local server and add coverage for cached model listing, raw model loading, playground jobs, and tool integration without installing AbstractCore.

## 0.2.3 - 2026-05-06

- AbstractCore plugin: support local Diffusers and stable-diffusion.cpp backends through `llm.vision`, not only OpenAI-compatible HTTP. The default plugin path now matches the REPL default: local Diffusers with `runwayml/stable-diffusion-v1-5`, cache-only unless `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1` is set.
- AbstractCore plugin: keep OpenAI-compatible usage available with `ABSTRACTVISION_BACKEND=openai` plus `OPENAI_BASE_URL`, and preserve artifact-store behavior for generated media outputs.

## 0.2.2 - 2026-05-06

- Release automation: add GitHub Actions CI/release workflows, issue templates, pre-commit config, MkDocs config, PyPI trusted publishing, GitHub Releases, and release-time docs deployment to `gh-pages`.
- Packaging: support Python 3.9-3.13, modernize license metadata, include package data explicitly, and add test/docs/dev extras.
- Defaults: make the REPL default to local Diffusers with `runwayml/stable-diffusion-v1-5`, `ABSTRACTVISION_DIFFUSERS_DEVICE=auto`, and cache-only/offline runtime downloads disabled by default.
- Diffusers backend: keep `allow_download=False` as the default, force Hugging Face offline/cache-only env during loads/calls, disable implicit HF token use offline, and load from cached snapshot paths when present.
- Diffusers backend: ignore unknown REPL `extra` flags that the pipeline `__call__` does not accept, avoiding `unexpected keyword argument` crashes.
- Diffusers backend: add better fp16 variant fallback behavior, MPS dtype/invalid-output retry handling, LoRA/Rapid-AIO offline handling, and clearer missing-local-model errors.
- Capability registry: add `black-forest-labs/FLUX.2-klein-9B` and normalize FLUX license id to `flux-non-commercial-license`.
- Docs: refresh quickstarts around Stable Diffusion 1.5 first, add clearer macOS Metal/NVIDIA CUDA/CPU guidance, expand stable-diffusion.cpp notes, and point users to cache-only local workflows.
- Tooling: add `scripts/download_model_sets.py` for explicit heavyweight model downloads (Stable Diffusion 1.5, FLUX 2 GGUF/Diffusers, and Qwen Image snapshots).
- Cleanup: remove the misspelled duplicate `ACKNOWLEDMENTS.md`.

## 0.2.1

- Documentation refresh for public release:
  - add `docs/api.md` and strengthen cross-linking between README and docs
  - add `CONTRIBUTING.md`, `SECURITY.md`, and `ACKNOWLEDGMENTS.md`
  - add `llms.txt` and generated `llms-full.txt` for agent-oriented context
  - clarify playground/server endpoint expectations (`/v1/vision/*`)

## 0.2.0

- Add stable-diffusion.cpp (`sd-cli`) backend for local GGUF diffusion models.
- REPL: forward unknown `--flags` as backend `extra` parameters.
- Add a tiny web playground (`playground/vision_playground.html`) for testing via AbstractCore Server vision endpoints (`/v1/vision/*`).

## 0.1.0

- Initial MVP: capability registry + schema validation.
- Artifact-first outputs via `LocalAssetStore` and runtime adapter.
- OpenAI-compatible HTTP backend for image generation/editing (optional video endpoints via config).
- Local Diffusers backend for images (opt-in deps).
- AbstractCore tool integration (`make_vision_tools`) with artifact refs.
- CLI/REPL for interactive manual testing.
