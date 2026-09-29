"""Spatially tiled Wan VAE decode for MLX-Gen video routes.

Why this exists: the Wan VAE decode is the peak of a Wan video run. MLX lowers
each 3D convolution to a matrix product over an unfolded copy of its input, so
at full resolution a single convolution holds a buffer many times the size of
the activation. Measured on an M5 Max (mlx-gen 0.38.0, TI2V-5B at
1280x704x121) the untiled decode alone peaked at about 49 GiB above the
weights.

Decoding the latent in overlapping spatial tiles bounds that buffer by the
tile size instead of the frame size. Each tile runs the full causal decode
over all frames with its own feature cache; neighbouring tiles overlap and are
cross-faded linearly, the scheme Diffusers' ``AutoencoderKLWan.tiled_decode``
uses.

- Wan2.1 VAE (patch size 1, used by the A14B routes): mlx-gen ships this tiled
  decode itself; :func:`install_tiled_decode` only switches it on.
- Wan2.2 VAE (patch size 2, used by TI2V-5B): mlx-gen 0.38 refuses to tile it,
  because its decoder output lives at half the final resolution until it is
  unpatchified. This module runs the same algorithm with every sample-space
  tile, stride and blend width divided by the patch size, as Diffusers does.

The helpers use the mlx-gen 0.38 VAE surface (``post_quant_conv``, ``decoder``,
``unpatchify``, the feature-cache helpers, ``latents_mean``/``latents_std``).
AbstractVision pins ``mlx-gen>=0.38.0,<0.39.0``; a change to that surface fails
loudly here rather than silently decoding untiled.
"""

from __future__ import annotations

import gc
from typing import Any, Callable, Iterator, List, Optional

# Sample-space tile and stride, in output pixels, for the Wan2.2 (TI2V-5B)
# VAE. Measured on its decode (mlx-gen 0.38.0, 832x480x121): 512/384 cut the
# decode's own peak from 12.5 to 7.4 GiB above the weights for 1.4x the decode
# time; 256/192 (the Diffusers/mlx-gen Wan2.1 policy) reached 4.7 GiB but took
# 2.4x as long. With 512/384 the decode stays below the denoising peak at
# both 832x480 and 1280x704, which is what bounds the run.
WAN_VAE_TILE_SAMPLE_PX = 512
WAN_VAE_TILE_STRIDE_PX = 384

def _mx():
    import mlx.core as mx  # type: ignore

    return mx


def _blend_rows(above: Any, current: Any, extent: int) -> Any:
    mx = _mx()
    extent = min(above.shape[-2], current.shape[-2], extent)
    if extent <= 0:
        return current
    weight = mx.arange(extent, dtype=mx.float32).reshape(1, 1, 1, extent, 1) / extent
    blended = above[:, :, :, -extent:, :] * (1 - weight) + current[:, :, :, :extent, :] * weight
    return mx.concatenate([blended.astype(current.dtype), current[:, :, :, extent:, :]], axis=-2)


def _blend_columns(left: Any, current: Any, extent: int) -> Any:
    mx = _mx()
    extent = min(left.shape[-1], current.shape[-1], extent)
    if extent <= 0:
        return current
    weight = mx.arange(extent, dtype=mx.float32).reshape(1, 1, 1, 1, extent) / extent
    blended = left[:, :, :, :, -extent:] * (1 - weight) + current[:, :, :, :, :extent] * weight
    return mx.concatenate([blended.astype(current.dtype), current[:, :, :, :, extent:]], axis=-1)


def _tile_starts(length: int, tile: int, stride: int) -> List[int]:
    """Tile origins along one axis, stopping once a tile reaches the end.

    Diffusers and mlx-gen start a tile at every stride multiple, so trailing
    tiles lie wholly inside the one before them and are decoded for nothing.
    Here the last tile keeps its full extent instead of being cropped to the
    stride, which covers the same pixels with fewer tiles.
    """
    starts = [0]
    while starts[-1] + tile < length:
        starts.append(starts[-1] + stride)
    return starts


def _release() -> None:
    mx = _mx()
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def iter_tiled_decode_slices(
    vae: Any,
    latents: Any,
    *,
    tile_px: int = WAN_VAE_TILE_SAMPLE_PX,
    stride_px: int = WAN_VAE_TILE_STRIDE_PX,
) -> Iterator[Any]:
    """Decode denormalized Wan latents ``[B, C, F, H, W]`` in spatial tiles.

    Yields decoded frame groups clipped to [-1, 1], one group per latent frame,
    exactly like mlx-gen's streamed ``iter_decode_slices`` (1 frame for the
    first latent frame, ``temporal_scale`` frames for each following one).
    Latents no larger than one tile decode untiled.
    """
    mx = _mx()
    if latents.ndim == 4:
        latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])
    patch = int(vae.patch_size)
    scale = int(vae.spatial_scale)
    tile_latent = tile_px // scale
    stride_latent = stride_px // scale
    latent_h, latent_w = latents.shape[-2:]
    if latent_h <= tile_latent and latent_w <= tile_latent:
        yield from vae.iter_decode_slices(latents)
        return

    # Decoder output coordinates (before unpatchify).
    stride_out = stride_px // patch
    blend_out = (tile_px - stride_px) // patch
    out_h = latent_h * scale // patch
    out_w = latent_w * scale // patch

    frames_per_latent: Optional[List[int]] = None
    rows: List[List[Any]] = []
    row_starts = _tile_starts(latent_h, tile_latent, stride_latent)
    column_starts = _tile_starts(latent_w, tile_latent, stride_latent)
    for top in row_starts:
        row: List[Any] = []
        for left in column_starts:
            feat_cache = vae._new_feature_cache()
            pieces = []
            counts = []
            for frame_index in range(latents.shape[2]):
                tile = latents[
                    :, :, frame_index : frame_index + 1, top : top + tile_latent, left : left + tile_latent
                ]
                tile = vae.post_quant_conv(tile)
                decoded = vae.decoder(tile, feat_cache=feat_cache, feat_idx=[0], first_chunk=frame_index == 0)
                if frame_index == 0 and decoded.shape[2] > 1:
                    decoded = decoded[:, :, 1:, :, :]
                vae._materialize_feature_cache(decoded, feat_cache)
                pieces.append(decoded)
                counts.append(int(decoded.shape[2]))
            if frames_per_latent is None:
                frames_per_latent = counts
            decoded_tile = mx.contiguous(mx.concatenate(pieces, axis=2))
            mx.eval(decoded_tile)
            row.append(decoded_tile)
            del pieces, feat_cache, decoded, tile
            _release()
        rows.append(row)

    result_rows = []
    for row_index, row in enumerate(rows):
        strips = []
        for column_index, tile in enumerate(row):
            if row_index > 0:
                tile = _blend_rows(rows[row_index - 1][column_index], tile, blend_out)
            if column_index > 0:
                tile = _blend_columns(row[column_index - 1], tile, blend_out)
            row[column_index] = tile
            # Every tile contributes its first stride; the last tile of a row
            # or column contributes everything up to the frame edge.
            keep_h = None if row_index == len(rows) - 1 else stride_out
            keep_w = None if column_index == len(row) - 1 else stride_out
            strips.append(tile[:, :, :, :keep_h, :keep_w])
        result_rows.append(mx.concatenate(strips, axis=-1))
    decoded = mx.concatenate(result_rows, axis=-2)[:, :, :, :out_h, :out_w]
    decoded = vae.unpatchify(decoded, patch_size=patch)
    decoded = mx.contiguous(mx.clip(decoded, -1.0, 1.0))
    mx.eval(decoded)
    del rows, result_rows
    _release()

    start = 0
    for count in frames_per_latent or []:
        group = mx.contiguous(decoded[:, :, start : start + count])
        mx.eval(group)
        yield group
        start += count


def install_tiled_decode(
    vae: Any,
    *,
    tile_px: int = WAN_VAE_TILE_SAMPLE_PX,
    stride_px: int = WAN_VAE_TILE_STRIDE_PX,
) -> Callable[[], None]:
    """Route this VAE instance's streamed decode through the tiled decode.

    mlx-gen's Wan pipeline decodes through
    ``vae.iter_decode_normalized_latent_slices``; this shadows that method on
    the instance (never the class) and returns a callable that restores it.
    """
    original = vae.iter_decode_normalized_latent_slices
    shadowed = "iter_decode_normalized_latent_slices" in vars(vae)

    def tiled(latents: Any, *, clear_cache_each_slice: bool = False, tile_spatial: bool = False) -> Iterator[Any]:
        if int(vae.patch_size) == 1:
            # mlx-gen tiles the Wan2.1 VAE itself.
            yield from original(latents, clear_cache_each_slice=clear_cache_each_slice, tile_spatial=True)
            return
        mx = _mx()
        z_dim = int(vae.z_dim)
        mean = mx.array(vae.latents_mean).reshape(1, z_dim, 1, 1, 1)
        std = mx.array(vae.latents_std).reshape(1, z_dim, 1, 1, 1)
        yield from iter_tiled_decode_slices(vae, latents * std + mean, tile_px=tile_px, stride_px=stride_px)

    vae.iter_decode_normalized_latent_slices = tiled

    def restore() -> None:
        if vars(vae).get("iter_decode_normalized_latent_slices") is not tiled:
            return
        if shadowed:
            vae.iter_decode_normalized_latent_slices = original
        else:
            del vae.iter_decode_normalized_latent_slices

    return restore
