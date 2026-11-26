"""A source repository id whose own snapshot is cached is that model, even where it maps to several prepared
packages; a source id that is not cached and maps to several packages stays ambiguous.
"""

from __future__ import annotations

import unittest

from _mlx_gen_route_fixtures import EVENTS, FLUX2, QWEN_2512, RouteCase, snapshot
from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401

KLEIN_4B_SOURCE = "black-forest-labs/FLUX.2-klein-4B"
KLEIN_9B_SOURCE = "black-forest-labs/FLUX.2-klein-9B"
KLEIN_BASE_9B_SOURCE = "black-forest-labs/FLUX.2-klein-base-9B"
QWEN_2512_SOURCE = "Qwen/Qwen-Image-2512"
# The twelve source ids that map to several prepared packages.
SOURCES = (
    KLEIN_4B_SOURCE, KLEIN_9B_SOURCE, "black-forest-labs/FLUX.2-klein-base-4B", KLEIN_BASE_9B_SOURCE,
    "Tongyi-MAI/Z-Image", "Tongyi-MAI/Z-Image-Turbo", "Qwen/Qwen-Image", QWEN_2512_SOURCE, "Qwen/Qwen-Image-Edit",
    "Qwen/Qwen-Image-Edit-2509", "Qwen/Qwen-Image-Edit-2511", "baidu/ERNIE-Image-Turbo",
)


def engine_source_widths():
    from abstractvision.backends.mflux import _mlx_gen_quantize_choices
    from abstractvision.errors import OptionalDependencyMissingError

    try:
        return _mlx_gen_quantize_choices()
    except OptionalDependencyMissingError:
        return None  # MLX-Gen is not installed: no load-time widths


class TestSourceRepositoryIds(RouteCase):
    def cache_source(self, repo_id):
        path = snapshot(self.snapshots[FLUX2].parents[2], repo_id)
        (path / "model_index.json").write_text("{}", encoding="utf-8")
        self.snapshots[repo_id] = path
        return path

    def assert_ambiguous(self, repo_id):
        from abstractvision.errors import OptionalDependencyMissingError

        with self.assertRaisesRegex(OptionalDependencyMissingError, f"selector {repo_id!r} is ambiguous"):
            self.backend(repo_id)._resolve()

    def generate(self, model):
        from abstractvision.types import ImageGenerationRequest

        backend = self.backend(model)
        backend.generate_image(ImageGenerationRequest(prompt="a red kite", width=64, height=64, steps=2, seed=1))
        (init,) = [event for event in EVENTS if event[0] == "init"]
        return backend, init

    def test_a_cached_source_without_packages_runs_as_released(self):
        source = self.cache_source(KLEIN_4B_SOURCE)
        backend, init = self.generate(KLEIN_4B_SOURCE)
        self.assertEqual(init[1], "Flux2Klein")
        self.assertEqual((init[2]["model_path"], init[2].get("quantize")), (str(source), None))
        self.assertEqual(backend._resolve().base_model, "flux2-klein-4b")
        self.assertFalse(backend._resolve().prepared)

    def test_a_source_not_cached_that_maps_to_several_packages_stays_ambiguous(self):
        from abstractvision.errors import OptionalDependencyMissingError

        with self.assertRaisesRegex(
            OptionalDependencyMissingError,
            r"MLX-Gen model selector 'black-forest-labs/FLUX\.2-klein-base-9B' is ambiguous because it maps to "
            r"multiple prepared AbstractFramework q4/q8 repos\. Use an exact model id: "
            r"AbstractFramework/flux\.2-klein-base-9b-4bit, AbstractFramework/flux\.2-klein-base-9b-8bit\.",
        ):
            self.generate(KLEIN_BASE_9B_SOURCE)
        self.assertEqual(self.inits(), [])

    def test_a_cached_source_means_the_source_even_where_a_package_is_cached(self):
        source = self.cache_source(KLEIN_9B_SOURCE)
        _backend, init = self.generate(KLEIN_9B_SOURCE)
        self.assertEqual((init[2]["model_path"], init[2].get("quantize")), (str(source), None))

    def test_published_answers_and_preload_describe_the_source(self):
        source = self.cache_source(KLEIN_4B_SOURCE)
        backend = self.backend(KLEIN_4B_SOURCE)
        self.assertIn("text_to_image", backend.get_capabilities().supported_tasks)
        backend.preload(warm_up=False)
        (init,) = [event for event in EVENTS if event[0] == "init"]
        self.assertEqual((init[1], init[2]["model_path"], init[2].get("quantize")), ("Flux2Klein", str(source), None))

    def test_a_prepared_package_id_is_unaffected(self):
        self.cache_source(KLEIN_9B_SOURCE)
        self.cache_source(QWEN_2512_SOURCE)
        _backend, init = self.generate(FLUX2)
        self.assertEqual(init[2]["model_path"], str(self.snapshots[FLUX2]))
        package = self.backend(QWEN_2512)
        self.assertTrue(package._resolve().prepared)
        self.assertEqual((package._resolve().model_path, package.quantize_bits()), (str(self.snapshots[QWEN_2512]), ()))

    def test_each_cached_source_is_its_own_snapshot_not_a_prepared_package(self):
        widths = engine_source_widths()
        for repo_id in SOURCES:
            with self.subTest(repo_id=repo_id):
                source = self.cache_source(repo_id)
                backend = self.backend(repo_id)
                resolution = backend._resolve()
                self.assertEqual((resolution.model_path, resolution.prepared), (str(source), False))
                if widths is None:
                    continue
                self.assertEqual(backend.quantize_bits(), widths)
                self.assertEqual(backend._checked_load_quantize(8), 8)

    def test_the_qwen_image_2512_source_runs_quantized_as_an_id_and_as_a_folder(self):
        from abstractvision.types import ImageGenerationRequest

        widths = engine_source_widths()
        if widths is None:
            self.skipTest("MLX-Gen is not installed: no load-time widths to compare")
        source = self.cache_source(QWEN_2512_SOURCE)
        for model in (QWEN_2512_SOURCE, str(source)):
            with self.subTest(model=model):
                EVENTS.clear()
                backend = self.backend(model)
                self.assertEqual(backend.get_capabilities().quantize_bits, widths)
                backend.generate_image(
                    ImageGenerationRequest(
                        prompt="a red kite", width=64, height=64, steps=2, seed=1, extra={"quantize": 8}
                    )
                )
                (init,) = [event for event in EVENTS if event[0] == "init"]
                self.assertEqual(
                    (init[1], init[2]["model_path"], init[2].get("quantize")), ("QwenImage", str(source), 8)
                )

    def test_the_qwen_image_2512_source_reaches_the_backend_through_the_plugin(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _Owner:
            config = {}

        source = self.cache_source(QWEN_2512_SOURCE)
        cap = _AbstractVisionCapability(_Owner())
        out = cap.t2i("a red kite", provider="mlx-gen", model=QWEN_2512_SOURCE, width=64, height=64, steps=2, seed=1)
        self.assertTrue(out.startswith(b"\x89PNG"))
        (init,) = [event for event in EVENTS if event[0] == "init"]
        self.assertEqual((init[1], init[2]["model_path"]), ("QwenImage", str(source)))

    def test_the_plugin_passes_a_cached_source_id_on_and_keeps_its_answer_for_one_not_cached(self):
        from abstractvision.integrations.abstractcore_plugin import _canonical_mflux_model_id

        uncached = {repo_id: _canonical_mflux_model_id(repo_id) for repo_id in SOURCES}
        # Not cached: 0.3.37's answers, Qwen/Qwen-Image-2512 rewritten to its catalog key.
        self.assertEqual(uncached, {**{k: k for k in SOURCES}, QWEN_2512_SOURCE: "qwen-image"})
        for repo_id in SOURCES:
            self.cache_source(repo_id)
        self.assertEqual({repo_id: _canonical_mflux_model_id(repo_id) for repo_id in SOURCES}, {k: k for k in SOURCES})

    def test_a_snapshot_that_cannot_load_as_the_model_stays_ambiguous(self):
        transformer = self.cache_source(KLEIN_9B_SOURCE) / "transformer"
        (transformer / "0.safetensors").unlink()
        (transformer.parent / "vae").mkdir()
        (transformer.parent / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
        self.assert_ambiguous(KLEIN_9B_SOURCE)  # only vae/ weights
        (transformer.parent / "flux-2-klein-9b.safetensors").write_bytes(b"x")
        self.assert_ambiguous(KLEIN_9B_SOURCE)  # and a root single-file checkpoint
        (transformer / "0.safetensors").symlink_to(transformer.parents[2] / "blobs" / "missing")
        self.assert_ambiguous(KLEIN_9B_SOURCE)  # and a dangling transformer link
        (transformer / "0.safetensors").unlink()
        (transformer / "0.safetensors").write_bytes(b"x")
        (transformer.parent / "model_index.json").unlink()
        self.assert_ambiguous(KLEIN_9B_SOURCE)  # transformer weights without model_index.json


if __name__ == "__main__":
    unittest.main()
