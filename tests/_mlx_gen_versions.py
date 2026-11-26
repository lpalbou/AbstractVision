# The `_require_runtime_versions` stub the MLX-Gen stand-in test modules share (`from _mlx_gen_versions import ...`).
from unittest.mock import patch

# The tests drive stand-in engines: the mlx-gen and MLX installed beside them are not the engine under test, so their
# versions are not checked there (test_mflux_backend.py tests the check itself through `real_runtime_versions()`).
_STUB = patch("abstractvision.backends.mflux._require_runtime_versions")
_REAL = {}


def setUpModule():
    from abstractvision.backends import mflux

    _REAL["check"] = mflux._require_runtime_versions
    _STUB.start()


def tearDownModule():
    _STUB.stop()


def real_runtime_versions():
    return _REAL["check"]
