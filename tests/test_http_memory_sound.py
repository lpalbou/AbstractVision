import base64
import json
import sys
import threading
import time
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from abstractvision.errors import CapabilityNotSupportedError, GenerationCancelledError  # noqa: E402
from abstractvision.types import GeneratedAsset, GeneratedAudioTrack  # noqa: E402

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"stand-in video"
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 16
PNG = b"\x89PNG\r\n\x1a\n" + b"stand-in image"
AUDIO_FIELDS = ["soundscape", "music", "generate_audio", "audio_shift"]
# What a plain OpenAI image endpoint may receive from the backend.
OPENAI_IMAGE_FIELDS = {"prompt", "n", "model", "response_format", "size", "width", "height", "seed", "steps", "guidance_scale"}


def _b64(data):
    return base64.b64encode(data).decode("ascii")


def _serve(handler_cls, test):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    test.addCleanup(server.server_close)
    test.addCleanup(server.shutdown)
    return f"http://127.0.0.1:{server.server_address[1]}"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return None

    def _reply(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        content_type = self.headers.get("Content-Type") or ""
        if content_type.startswith("multipart/"):
            from abstractvision.playground_server import _parse_multipart

            fields, _files = _parse_multipart(content_type, raw)
            return fields
        return json.loads(raw.decode("utf-8")) if raw else {}


class _ContractServer(_Handler):
    """A stand-in for AbstractCore's server: the memory and sound fields, discovery, jobs, cancel."""

    modes = ["default", "low", "lowest"]
    posts = []
    gets = []
    jobs = {}
    # Set by tests: an HTTP status for GET /vision/models, a status for job polls, a server busy
    # with another generation, and jobs that succeed at their first poll.
    discovery_status = None
    poll_status = None
    busy = False
    jobs_succeed = False

    def do_GET(self):
        path = urlparse(self.path).path
        self.gets.append(path)
        if path == "/v1/vision/models" and self.discovery_status is not None:
            self._reply(self.discovery_status, {"error": {"message": "stand-in status"}})
        elif path == "/v1/vision/models":
            entry = {"id": "stand-in/h3", "memory_modes": list(self.modes), "supports_audio": True, "audio_fields": AUDIO_FIELDS}
            self._reply(200, {"models": [entry, {"id": "stand-in/image", "memory_modes": ["default"], "supports_audio": False, "audio_fields": []}], "active": None})
        elif path == "/v1/models":
            self._reply(200, {"data": [{"id": "stand-in/h3"}, {"id": "other"}]})
        elif path.startswith("/v1/vision/jobs/") and self.poll_status is not None:
            self._reply(self.poll_status, {"error": {"message": "stand-in poll failure"}})
        elif path.startswith("/v1/vision/jobs/"):
            job = self.jobs.get(path.rsplit("/", 1)[-1])
            if job and self.jobs_succeed and job["state"] == "running":
                job["state"] = "succeeded"
                if "/videos/" in job["route"]:
                    job["result"] = self._result(job["body"])
                else:
                    # As AbstractCore's image job routes: the mode on the job, not in its result.
                    job["memory_mode_used"] = job["body"].get("memory_mode") or "default"
                    job["result"] = {"created": 1, "data": [{"b64_json": _b64(PNG)}]}
            self._reply(200, {k: v for k, v in job.items() if k not in {"route", "body"}} if job else {"state": "failed", "error": "unknown job"})
        else:
            self._reply(404, {"error": {"message": "not found", "type": "http_error"}})

    def _result(self, body):
        mode = body.get("memory_mode") or "default"
        audio = {"present": True, "sample_rate": 48000, "channels": 2, "codec": "pcm_s16le", "muxed": False,
                 "duration_seconds": 5.2, "wav": {"b64_json": _b64(WAV), "mime_type": "audio/wav"}}
        return {"created": 1, "memory_mode_used": mode, "audio": audio,
                "data": [{"b64_json": _b64(MP4), "mime_type": "video/mp4", "audio": audio}]}

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._body()
        self.posts.append((path, body))
        if path.endswith("/cancel"):
            self.jobs[path.split("/")[-2]]["state"] = "cancelled"
            self._reply(200, {"ok": True})
            return
        mode = body.get("memory_mode")
        if mode and mode not in self.modes:
            self._reply(422, {"error": {
                "message": f"memory_mode={mode!r} is not supported for stand-in/h3 (supported: {', '.join(self.modes)}).",
                "type": "invalid_request_error", "code": "capability_not_supported",
                "param": "memory_mode", "supported": list(self.modes)}})
            return
        if path.startswith("/v1/vision/jobs/") and self.busy:
            self._reply(409, {"detail": "Another generation is already running; wait for it to finish."})
        elif path.startswith("/v1/vision/jobs/"):
            job_id = uuid.uuid4().hex
            # The job runs until it is cancelled, unless jobs_succeed.
            self.jobs[job_id] = {"job_id": job_id, "state": "running", "route": path, "body": body}
            self._reply(200, {"job_id": job_id})
        elif path.startswith("/v1/videos/"):
            self._reply(200, self._result(body))
        else:
            self._reply(200, {"data": [{"b64_json": _b64(PNG)}]})


class _PlainOpenAIServer(_Handler):
    seen = []

    def do_GET(self):
        self.seen.append(("GET", self.path, None))
        if self.path == "/v1/models":
            self._reply(200, {"data": [{"id": "image-model"}]})
            return
        self._reply(404, {"error": {"message": "not found"}})

    def do_POST(self):
        self.seen.append(("POST", self.path, self._body()))
        self._reply(200, {"data": [{"b64_json": _b64(PNG)}]})


def _client(base_url, **config):
    from abstractvision.backends.openai_compatible import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend

    return OpenAICompatibleVisionBackend(config=OpenAICompatibleBackendConfig(base_url=f"{base_url}/v1", **config))


class TestOpenAICompatibleAgainstContractServer(unittest.TestCase):
    def setUp(self):
        _ContractServer.modes = ["default", "low", "lowest"]
        _ContractServer.posts = []
        _ContractServer.gets = []
        _ContractServer.jobs = {}
        _ContractServer.discovery_status = None
        _ContractServer.poll_status = None
        _ContractServer.busy = False
        _ContractServer.jobs_succeed = False
        self.root = _serve(_ContractServer, self)
        self.client = _client(
            self.root, model_id="stand-in/h3", text_to_video_path="/videos/generations", image_to_video_path="/videos/edits"
        )

    def test_video_fields_are_sent_typed_and_the_soundtrack_comes_back(self):
        from abstractvision import VisionManager

        asset = VisionManager(backend=self.client).generate_video(
            "a fox in the snow", memory_mode="lowest", soundscape="wind over snow", music="sparse piano",
            generate_audio=True, audio_shift=2.5,
        )
        path, body = _ContractServer.posts[-1]
        self.assertEqual(path, "/v1/videos/generations")
        self.assertEqual(body["memory_mode"], "lowest")
        self.assertEqual((body["soundscape"], body["music"]), ("wind over snow", "sparse piano"))
        self.assertIs(body["generate_audio"], True)
        self.assertEqual(body["audio_shift"], 2.5)
        self.assertEqual(body["model"], "stand-in/h3")
        self.assertEqual(asset.data, MP4)
        self.assertIsInstance(asset.audio, GeneratedAudioTrack)
        self.assertEqual((asset.audio.sample_rate, asset.audio.channels, asset.audio.muxed), (48000, 2, False))
        self.assertEqual(asset.audio.data, WAV)
        self.assertEqual(asset.audio.mime_type, "audio/wav")
        self.assertEqual(asset.metadata["memory_mode_used"], "lowest")
        # The same summary as a local backend gives: the codec is the remote's (the WAV's PCM format).
        self.assertEqual(
            asset.audio_summary(),
            {"present": True, "sample_rate": 48000, "channels": 2, "codec": "pcm_s16le", "muxed": False,
             "duration_seconds": 5.2, "wav": WAV},
        )

    def test_multipart_image_to_video_sends_form_fields(self):
        from abstractvision.types import ImageToVideoRequest

        self.client.image_to_video(ImageToVideoRequest(image=PNG, prompt="fox", memory_mode="low", generate_audio=False, quantize=8))
        _path, form = _ContractServer.posts[-1]
        self.assertEqual((form["memory_mode"], form["generate_audio"], form["quantize"]), ("low", "false", "8"))

    def test_discovery_lists_modes_per_model(self):
        models = {m.id: m for m in self.client.list_provider_models()}
        self.assertEqual(models["stand-in/h3"].raw["memory_modes"], ["default", "low", "lowest"])
        self.assertTrue(models["stand-in/h3"].raw["supports_audio"])
        self.assertNotIn("memory_modes", models["other"].raw)
        caps = self.client.get_capabilities()
        self.assertEqual(tuple(caps.memory_modes), ("default", "low", "lowest"))
        self.assertTrue(caps.generates_audio)
        self.assertEqual(self.client.supported_request_options("text_to_video"), frozenset({"quantize", *AUDIO_FIELDS}))
        self.assertEqual(self.client.supported_request_options("text_to_image"), frozenset({"quantize"}))
        image_client = _client(self.root, model_id="stand-in/image")
        image_client.list_provider_models()
        self.assertEqual(tuple(image_client.get_capabilities().memory_modes), ("default",))

    def test_plain_requests_make_no_discovery_request_and_one_that_sets_a_field_makes_one(self):
        from abstractvision import VisionManager

        vm = VisionManager(backend=self.client)
        for _ in range(3):
            vm.generate_image("a red cube", width=64, height=64)
        self.client.get_capabilities()
        self.assertEqual(_ContractServer.gets, [])
        self.assertEqual([path for path, _body in _ContractServer.posts], ["/v1/images/generations"] * 3)
        for _ in range(2):
            vm.generate_image("a red cube", memory_mode="low")
        self.assertEqual(_ContractServer.gets, ["/v1/vision/models"])  # then reused
        self.assertEqual(_ContractServer.posts[-1][1]["memory_mode"], "low")

    def test_a_failed_discovery_is_not_taken_for_a_plain_server(self):
        from abstractvision.types import ImageGenerationRequest

        _ContractServer.discovery_status = 503
        with self.assertRaisesRegex(CapabilityNotSupportedError, r"memory_mode='low'.*GET /vision/models failed.*503"):
            self.client.generate_image(ImageGenerationRequest(prompt="fox", memory_mode="low"))
        self.assertEqual(_ContractServer.posts, [])
        _ContractServer.discovery_status = None  # the server answers now: asked again, not cached
        self.client.generate_image(ImageGenerationRequest(prompt="fox", memory_mode="low"))
        self.assertEqual(_ContractServer.posts[-1][1]["memory_mode"], "low")
        self.assertEqual(_ContractServer.gets, ["/v1/vision/models"] * 2)

    def test_a_definite_answer_without_modes_is_cached_as_a_plain_server(self):
        from abstractvision.types import ImageGenerationRequest

        _ContractServer.discovery_status = 404
        for _ in range(2):
            with self.assertRaisesRegex(CapabilityNotSupportedError, r"supported: default\)"):
                self.client.generate_image(ImageGenerationRequest(prompt="fox", memory_mode="low"))
        self.assertEqual(_ContractServer.gets, ["/v1/vision/models"])

    def test_job_results_carry_the_mode_used_from_the_result_or_the_job(self):
        from abstractvision import VisionManager

        _ContractServer.jobs_succeed = True
        vm = VisionManager(backend=self.client)
        image = vm.generate_image("fox", memory_mode="low", cancel_event=threading.Event())
        self.assertEqual(_ContractServer.posts[-1][0], "/v1/vision/jobs/images/generations")
        self.assertEqual(image.metadata["memory_mode_used"], "low")
        video = vm.generate_video("fox", memory_mode="lowest", cancel_event=threading.Event())
        self.assertEqual(video.metadata["memory_mode_used"], "lowest")
        self.assertEqual(video.audio.codec, "pcm_s16le")

    def test_a_server_busy_with_another_generation_is_named(self):
        from abstractvision.types import ImageGenerationRequest

        _ContractServer.busy = True
        with self.assertRaisesRegex(RuntimeError, "Another generation is running on the server"):
            self.client.generate_image(ImageGenerationRequest(prompt="fox", cancel_event=threading.Event()))

    def test_unsupported_values_are_capability_errors(self):
        from abstractvision.types import ImageGenerationRequest, VideoGenerationRequest

        self.client.list_provider_models()
        # The remote stops offering "lowest" after the client read its list: its 422 comes back
        # as CapabilityNotSupportedError with the remote's message.
        _ContractServer.modes = ["default", "low"]
        with self.assertRaisesRegex(CapabilityNotSupportedError, r"memory_mode='lowest' is not supported for stand-in/h3"):
            self.client.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="lowest"))
        count = len(_ContractServer.posts)
        image_client = _client(self.root, model_id="stand-in/image")
        with self.assertRaises(CapabilityNotSupportedError):
            image_client.generate_image(ImageGenerationRequest(prompt="fox", memory_mode="low"))
        with self.assertRaises(ValueError):
            self.client.generate_image(ImageGenerationRequest(prompt="fox", memory_mode="tiny"))
        self.assertEqual(len(_ContractServer.posts), count)

    def test_cancel_event_cancels_the_remote_job(self):
        from abstractvision.types import VideoGenerationRequest

        cancel = threading.Event()
        outcome = {}

        def run():
            try:
                self.client.generate_video(VideoGenerationRequest(prompt="fox", cancel_event=cancel))
            except Exception as e:  # noqa: BLE001 - the test inspects the type
                outcome["error"] = e

        worker = threading.Thread(target=run)
        worker.start()
        for _ in range(200):
            if _ContractServer.jobs:
                break
            time.sleep(0.01)
        self.assertEqual(_ContractServer.posts[0][0], "/v1/vision/jobs/videos/generations")
        cancel.set()
        worker.join(5)
        self.assertIsInstance(outcome.get("error"), GenerationCancelledError)
        job_id = next(iter(_ContractServer.jobs))
        self.assertEqual(_ContractServer.posts[-1][0], f"/v1/vision/jobs/{job_id}/cancel")
        self.assertEqual(_ContractServer.jobs[job_id]["state"], "cancelled")

    def test_any_cancel_event_with_is_set_cancels_the_remote_job(self):
        from types import SimpleNamespace

        from abstractvision.types import VideoGenerationRequest

        checks = iter([False, False, False])
        flag = SimpleNamespace(is_set=lambda: next(checks, True))
        with self.assertRaises(GenerationCancelledError):
            self.client.generate_video(VideoGenerationRequest(prompt="fox", cancel_event=flag))
        job_id = next(iter(_ContractServer.jobs))
        self.assertEqual(_ContractServer.posts[-1][0], f"/v1/vision/jobs/{job_id}/cancel")

    def test_polling_that_ends_abnormally_cancels_the_remote_job_once(self):
        from types import SimpleNamespace

        from abstractvision.types import VideoGenerationRequest

        def interrupted():  # Ctrl-C once the job has been polled
            if any(path.startswith("/v1/vision/jobs/") for path in _ContractServer.gets):
                raise KeyboardInterrupt
            return False

        for cancel_event, status, error in (
            (threading.Event(), 503, RuntimeError),  # a poll the server fails
            (SimpleNamespace(is_set=interrupted), None, KeyboardInterrupt),
        ):
            _ContractServer.jobs = {}
            _ContractServer.posts = []
            _ContractServer.gets = []
            _ContractServer.poll_status = status
            with self.assertRaises(error):
                self.client.generate_video(VideoGenerationRequest(prompt="fox", cancel_event=cancel_event))
            job_id = next(iter(_ContractServer.jobs))
            cancels = [path for path, _body in _ContractServer.posts if path.endswith("/cancel")]
            self.assertEqual(cancels, [f"/v1/vision/jobs/{job_id}/cancel"], error)


class TestOpenAICompatibleAgainstPlainEndpoint(unittest.TestCase):
    def test_a_remote_without_discovery_gets_no_unknown_field(self):
        from abstractvision.types import ImageGenerationRequest, VideoGenerationRequest

        _PlainOpenAIServer.seen = []
        client = _client(_serve(_PlainOpenAIServer, self), model_id="image-model", text_to_video_path="/videos/generations")
        self.assertEqual(tuple(client.get_capabilities().memory_modes), ("default",))
        for request in (
            VideoGenerationRequest(prompt="fox", memory_mode="low"),
            VideoGenerationRequest(prompt="fox", soundscape="wind"),
            ImageGenerationRequest(prompt="fox", quantize=8),
            ImageGenerationRequest(prompt="fox", extra={"soundscape": "wind"}),
        ):
            with self.assertRaises(CapabilityNotSupportedError):
                getattr(client, "generate_video" if isinstance(request, VideoGenerationRequest) else "generate_image")(request)
        client.generate_image(ImageGenerationRequest(prompt="fox", memory_mode="default", width=64, height=64))
        posts = [payload for method, _path, payload in _PlainOpenAIServer.seen if method == "POST"]
        self.assertEqual(len(posts), 1)
        self.assertLessEqual(set(posts[0]), OPENAI_IMAGE_FIELDS)
        models = client.list_provider_models()
        self.assertEqual([m.id for m in models], ["image-model"])
        self.assertNotIn("memory_modes", models[0].raw)


class TestPlaygroundPassesTheFieldsThrough(unittest.TestCase):
    def _state(self, backend):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))
        state._active_backend = backend
        state._active_backend_kind = "mlx-gen"
        state._active_model_id = "mlx-gen/stand-in/h3"
        return state

    def _wait(self, state, job):
        for _ in range(200):
            snap = state.get_job(job["job_id"])
            if snap["state"] in {"succeeded", "failed"}:
                return snap
            time.sleep(0.01)
        return snap

    def test_video_job_builds_typed_fields_from_top_level_and_extra(self):
        seen = {}

        class Backend:
            def generate_video_with_progress(self, request, progress_callback=None):
                seen["request"] = request
                return GeneratedAsset(media_type="video", data=MP4, mime_type="video/mp4")

        state = self._state(Backend())
        job = state.start_video_generation_job(
            {"prompt": "fox", "memory_mode": "low", "quantize": "8", "generate_audio": "false",
             "audio_shift": "1.5", "extra": {"soundscape": "wind", "music": "piano", "vae_tiling": False}}
        )
        self.assertEqual(self._wait(state, job)["state"], "succeeded")
        request = seen["request"]
        self.assertEqual((request.memory_mode, request.quantize, request.generate_audio), ("low", 8, False))
        self.assertEqual((request.audio_shift, request.soundscape, request.music), (1.5, "wind", "piano"))
        self.assertEqual(request.extra, {"vae_tiling": False})

    def test_backend_refusal_and_conflicts_surface(self):
        from abstractvision.backends.base_backend import VisionBackend

        class Backend(VisionBackend):
            def generate_image(self, request):
                self.resolve_memory_request(request, task="text_to_image")
                return GeneratedAsset(media_type="image", data=PNG, mime_type="image/png")

            edit_image = generate_video = image_to_video = generate_angles = generate_image

        state = self._state(Backend())
        snap = self._wait(state, state.start_image_generation_job({"prompt": "fox", "memory_mode": "lowest"}))
        self.assertEqual(snap["state"], "failed")
        self.assertIn("memory_mode='lowest'", snap["error"])
        with self.assertRaises(CapabilityNotSupportedError):
            state.start_image_generation_job({"prompt": "fox", "soundscape": "wind"})
        with self.assertRaisesRegex(ValueError, "memory_mode"):
            state.start_video_generation_job({"prompt": "fox", "memory_mode": "low", "extra": {"memory_mode": "lowest"}})

    def test_model_load_refuses_memory_fields_instead_of_dropping_them(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState, _make_handler

        server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(PlaygroundState(PlaygroundServerConfig(default_model_id=""))))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        body = json.dumps({"model_id": "mlx-gen/stand-in/h3", "memory_mode": "low"}).encode("utf-8")
        req = Request(f"http://127.0.0.1:{server.server_address[1]}/v1/vision/model/load", data=body, method="POST")
        with self.assertRaises(HTTPError) as ctx:
            urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 400)
        self.assertIn("memory_mode", json.loads(ctx.exception.read().decode("utf-8"))["error"]["message"])


if __name__ == "__main__":
    unittest.main()
