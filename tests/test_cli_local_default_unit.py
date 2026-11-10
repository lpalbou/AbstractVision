"""The one-shot commands default to this machine's local engine and FLUX.2 [klein] 4B in 8 bits (0.3.36).

With no --provider, --model, ABSTRACTVISION_PROVIDER/BACKEND or base URL, `abstractvision t2i "..."` used to pick
the remote OpenAI-compatible backend on every platform and stop with "Missing --base-url". Now: MLX-Gen with
AbstractFramework/flux.2-klein-4b-8bit on Apple silicon, stable-diffusion.cpp (the Q8_0 GGUF) with
black-forest-labs/FLUX.2-klein-4B on an NVIDIA card, Diffusers with the same model on the processor. A configured
base URL or an explicit provider keeps winning. Backends are only constructed here, never loaded.
"""

from __future__ import annotations

import argparse
import os
import unittest
from unittest.mock import patch



def _cli():
    # Imported inside each test, like tests/test_cli_smoke.py: test_abstractcore_plugin pops and reloads the backend
    # modules, so a module-level import here would hold stale backend classes for the whole run.
    from abstractvision import cli

    return cli


def _args(**overrides) -> argparse.Namespace:
    base = dict(provider=None, backend=None, model=None, model_id=None, mflux_model=None, base_url=None, api_key=None,
                store_dir=None, images_generations_path="/v1/images/generations", images_edits_path="/v1/images/edits",
                text_to_video_path=None, image_to_video_path=None, image_to_video_mode="auto", timeout_s=None,
                sdcpp_bin=None, sdcpp_model=None, sdcpp_diffusion_model=None, sdcpp_vae=None, sdcpp_llm=None,
                sdcpp_llm_vision=None, sdcpp_extra_args=None, diffusers_device=None, diffusers_torch_dtype=None,
                diffusers_allow_download=False, diffusers_auto_retry_fp32=True, mflux_base_model=None,
                mflux_model_dir=None, mflux_allow_download=False)
    base.update(overrides)
    return argparse.Namespace(**base)


_CLEAR_ENV = {k: "" for k in ("ABSTRACTVISION_PROVIDER", "ABSTRACTVISION_BACKEND", "ABSTRACTVISION_MODEL",
                              "ABSTRACTVISION_MODEL_ID", "ABSTRACTVISION_MFLUX_MODEL", "OPENAI_BASE_URL")}


class LocalDefaultTests(unittest.TestCase):
    def setUp(self):
        global cli
        cli = _cli()

    def test_default_provider_follows_the_local_profile(self):
        self.assertEqual(cli.default_local_provider("apple-silicon"), "mlx-gen")
        self.assertEqual(cli.default_local_provider("cuda"), "sdcpp")
        self.assertEqual(cli.default_local_provider("cpu"), "diffusers")
        self.assertEqual(cli.default_local_model_id("mlx-gen"), "AbstractFramework/flux.2-klein-4b-8bit")
        self.assertEqual(cli.default_local_model_id("sdcpp"), "black-forest-labs/FLUX.2-klein-4B")

    def test_apple_silicon_defaults_to_mlx_gen_klein_4b_8bit(self):
        with patch.dict(os.environ, _CLEAR_ENV), patch.object(cli, "local_model_profile", return_value="apple-silicon"):
            vm = cli._build_manager_from_args(_args())
        self.assertIsInstance(vm.backend, cli.MFluxVisionBackend)
        self.assertEqual(vm.backend._cfg.model, "AbstractFramework/flux.2-klein-4b-8bit")

    def test_nvidia_defaults_to_sdcpp_klein_4b(self):
        fake = cli.resolve_sdcpp_model_selection.__class__  # noqa: F841 - keep the real function importable
        with patch.dict(os.environ, _CLEAR_ENV), patch.object(cli, "local_model_profile", return_value="cuda"), \
                patch.object(cli, "resolve_sdcpp_model_selection", side_effect=ValueError("not cached here")):
            vm = cli._build_manager_from_args(_args())
        self.assertIsInstance(vm.backend, cli.StableDiffusionCppVisionBackend)
        self.assertEqual(vm.backend._cfg.model, "black-forest-labs/FLUX.2-klein-4B")

    def test_processor_defaults_to_diffusers_klein_4b(self):
        with patch.dict(os.environ, _CLEAR_ENV), patch.object(cli, "local_model_profile", return_value="cpu"), \
                patch.object(cli, "_resolve_cached_diffusers_model_id", side_effect=lambda m: m):
            vm = cli._build_manager_from_args(_args())
        self.assertIsInstance(vm.backend, cli.HuggingFaceDiffusersVisionBackend)
        self.assertEqual(vm.backend._cfg.model_id, "black-forest-labs/FLUX.2-klein-4B")

    def test_a_configured_base_url_keeps_the_remote_backend(self):
        with patch.dict(os.environ, _CLEAR_ENV), patch.object(cli, "local_model_profile", return_value="cuda"):
            vm = cli._build_manager_from_args(_args(base_url="http://localhost:1234/v1"))
        self.assertIsInstance(vm.backend, cli.OpenAICompatibleVisionBackend)
        with patch.dict(os.environ, {**_CLEAR_ENV, "OPENAI_BASE_URL": "http://localhost:1234/v1"}), \
                patch.object(cli, "local_model_profile", return_value="cuda"):
            vm = cli._build_manager_from_args(_args(base_url="http://localhost:1234/v1"))
        self.assertIsInstance(vm.backend, cli.OpenAICompatibleVisionBackend)

    def test_an_explicit_provider_or_env_wins_over_the_local_default(self):
        with patch.dict(os.environ, _CLEAR_ENV), patch.object(cli, "local_model_profile", return_value="cuda"), \
                patch.object(cli, "_resolve_cached_diffusers_model_id", side_effect=lambda m: m):
            vm = cli._build_manager_from_args(_args(provider="diffusers", model="qwen-image"))
        self.assertIsInstance(vm.backend, cli.HuggingFaceDiffusersVisionBackend)
        with patch.dict(os.environ, {**_CLEAR_ENV, "ABSTRACTVISION_PROVIDER": "openai"}), \
                patch.object(cli, "local_model_profile", return_value="cuda"):
            with self.assertRaises(SystemExit):  # the remote provider still needs its base URL
                cli._build_manager_from_args(_args())


if __name__ == "__main__":
    unittest.main()
