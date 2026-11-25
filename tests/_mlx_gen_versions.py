# Replaces the copy of the `_require_runtime_versions` stub each MLX-Gen stand-in test module carried: one shared stub.
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
