"""Tiled Wan VAE decode: tiling, blending and frame streaming against an untiled decode.

Uses a tiny stand-in VAE (no model download): its "decoder" upsamples each latent
pixel with a smooth per-position response, so a tiled decode must reproduce the
untiled one wherever tiles overlap. Requires MLX (Apple Silicon); skipped elsewhere.
"""

from __future__ import annotations

import unittest

try:
    import mlx.core as mx  # type: ignore
except Exception:  # pragma: no cover - non-Apple hosts
    mx = None


class _TinyWanVae:
    """Mimics the mlx-gen 0.38 Wan VAE decode surface with a spatially pointwise decoder."""

    spatial_scale = 16
    temporal_scale = 4
    z_dim = 2

    def __init__(self, patch_size: int):
        self.patch_size = patch_size
        self.latents_mean = [0.5, -0.25]
        self.latents_std = [2.0, 0.5]
        self.decoder_calls = 0

    def post_quant_conv(self, x):
        return x

    @staticmethod
    def _new_feature_cache():
        return [None] * 4

    @staticmethod
    def _materialize_feature_cache(output, feat_cache):
        mx.eval(output)

    @staticmethod
    def unpatchify(x, patch_size):
        if patch_size == 1:
            return x
        b, cp, f, h, w = x.shape
        c = cp // (patch_size * patch_size)
        x = mx.reshape(x, (b, c, patch_size, patch_size, f, h, w))
        x = mx.transpose(x, (0, 1, 4, 5, 3, 6, 2))
        return mx.reshape(x, (b, c, f, h * patch_size, w * patch_size))

    def decoder(self, x, *, feat_cache, feat_idx, first_chunk):
        self.decoder_calls += 1
        up = self.spatial_scale // self.patch_size
        frames = 1 if first_chunk else self.temporal_scale
        x = mx.tanh(x * 0.3)
        x = mx.repeat(mx.repeat(x, up, axis=3), up, axis=4)
        channels = 3 * self.patch_size * self.patch_size
        x = mx.concatenate([x[:, i % 2 : i % 2 + 1] * (i + 1) / channels for i in range(channels)], axis=1)
        return mx.concatenate([x * (1.0 - 0.1 * t) for t in range(frames)], axis=2)

    def iter_decode_slices(self, latents):
        for i in range(latents.shape[2]):
            out = self.decoder(latents[:, :, i : i + 1], feat_cache=None, feat_idx=[0], first_chunk=i == 0)
            yield mx.clip(self.unpatchify(out, self.patch_size), -1.0, 1.0)

    def iter_decode_normalized_latent_slices(self, latents, *, clear_cache_each_slice=False, tile_spatial=False):
        self.native_tile_spatial = tile_spatial
        yield from self.iter_decode_slices(latents)


@unittest.skipIf(mx is None, "MLX is not available")
class TestWanVaeTiling(unittest.TestCase):
    def _latents(self, h, w, frames=3):
        import numpy as np

        rng = np.random.default_rng(0)
        return mx.array(rng.normal(size=(1, 2, frames, h, w)).astype("float32"))

    def test_patch_size_2_tiled_decode_matches_untiled_and_streams_frame_groups(self):
        from abstractvision.backends.wan_vae_tiling import iter_tiled_decode_slices

        vae = _TinyWanVae(patch_size=2)
        latents = self._latents(30, 52)  # 480x832 output in 256px tiles, stride 192
        reference = list(vae.iter_decode_slices(latents))
        vae.decoder_calls = 0
        tiled = list(iter_tiled_decode_slices(vae, latents, tile_px=256, stride_px=192))

        self.assertEqual([g.shape[2] for g in tiled], [1, 4, 4])
        self.assertEqual([g.shape for g in tiled], [g.shape for g in reference])
        self.assertEqual(tiled[0].shape[-2:], (480, 832))
        for got, want in zip(tiled, reference):
            self.assertLess(float(mx.max(mx.abs(got - want))), 1e-5)
        # It really tiled: rows start at 0/12/24 and columns at 0/12/24/36 (latent px);
        # a tile at 48 would lie inside the one at 36. One decoder call per tile per latent frame.
        self.assertEqual(vae.decoder_calls, 3 * 4 * 3)

    def test_tile_starts_stop_once_a_tile_reaches_the_edge(self):
        from abstractvision.backends.wan_vae_tiling import _tile_starts

        self.assertEqual(_tile_starts(52, 16, 12), [0, 12, 24, 36])
        self.assertEqual(_tile_starts(30, 32, 24), [0])
        self.assertEqual(_tile_starts(80, 32, 24), [0, 24, 48])
        self.assertEqual(_tile_starts(44, 32, 24), [0, 24])
        for length in range(1, 120):
            starts = _tile_starts(length, 32, 24)
            self.assertGreaterEqual(starts[-1] + 32, length)
            if len(starts) > 1:
                # the last tile adds pixels the previous one did not cover
                self.assertLess(starts[-2] + 32, length)

    def test_default_tiles_cover_a_480p_frame_in_two_tiles(self):
        from abstractvision.backends.wan_vae_tiling import iter_tiled_decode_slices

        vae = _TinyWanVae(patch_size=2)
        latents = self._latents(30, 52, frames=2)
        reference = list(vae.iter_decode_slices(latents))
        vae.decoder_calls = 0
        tiled = list(iter_tiled_decode_slices(vae, latents))
        self.assertEqual(vae.decoder_calls, 2 * 2)
        for got, want in zip(tiled, reference):
            self.assertLess(float(mx.max(mx.abs(got - want))), 1e-5)

    def test_small_latent_decodes_untiled(self):
        from abstractvision.backends.wan_vae_tiling import iter_tiled_decode_slices

        vae = _TinyWanVae(patch_size=2)
        latents = self._latents(16, 16, frames=2)
        groups = list(iter_tiled_decode_slices(vae, latents))
        self.assertEqual(vae.decoder_calls, 2)
        self.assertEqual([g.shape[2] for g in groups], [1, 4])

    def test_install_denormalizes_for_patch_size_2_and_restores(self):
        from abstractvision.backends.wan_vae_tiling import install_tiled_decode

        vae = _TinyWanVae(patch_size=2)
        latents = self._latents(30, 40, frames=2)
        mean = mx.array(vae.latents_mean).reshape(1, 2, 1, 1, 1)
        std = mx.array(vae.latents_std).reshape(1, 2, 1, 1, 1)
        want = list(vae.iter_decode_slices(latents * std + mean))

        restore = install_tiled_decode(vae)
        got = list(vae.iter_decode_normalized_latent_slices(latents, clear_cache_each_slice=False))
        restore()

        self.assertNotIn("iter_decode_normalized_latent_slices", vars(vae))
        for g, w in zip(got, want):
            self.assertLess(float(mx.max(mx.abs(g - w))), 1e-5)

    def test_install_uses_mlx_gen_native_tiling_for_patch_size_1(self):
        from abstractvision.backends.wan_vae_tiling import install_tiled_decode

        vae = _TinyWanVae(patch_size=1)
        restore = install_tiled_decode(vae)
        list(vae.iter_decode_normalized_latent_slices(self._latents(40, 40, frames=1)))
        restore()
        self.assertIs(vae.native_tile_spatial, True)


if __name__ == "__main__":
    unittest.main()
