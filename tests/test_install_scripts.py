"""The standalone one-line installers (scripts/install.sh, scripts/install.ps1).

They are served at https://abstractframework.ai/vision/install.sh and .../install.ps1, so these
checks keep them in step with the package: the default pin is this release
(src/abstractvision/__init__.py), they install only the three install settings (light = no extra,
`apple`, `gpu`), the commands they guard are exactly `[project.scripts]`, the prebuilt-wheel
handling targets a real requirement, and the documented one-liners are the exact URLs.

unittest only, no tomllib: CI runs `python -m unittest discover` on Python 3.9 to 3.13 with no
test dependency installed, so pyproject.toml is read with the same block parsing as
tests/test_packaging_metadata.py.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SH = ROOT / "scripts" / "install.sh"
PS1 = ROOT / "scripts" / "install.ps1"
README = ROOT / "README.md"
INSTALL_DOC = ROOT / "docs" / "getting-started.md"  # the docs' install page (docs/README.md)

SH_LINE = "curl -LsSf https://abstractframework.ai/vision/install.sh | sh"
PS1_LINE = 'powershell -ExecutionPolicy ByPass -c "irm https://abstractframework.ai/vision/install.ps1 | iex"'
PIP_LINES = ("pip install abstractvision", 'pip install "abstractvision[apple]"', 'pip install "abstractvision[gpu]"')


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _version() -> str:
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', _read(ROOT / "src" / "abstractvision" / "__init__.py"), re.MULTILINE)
    assert m, "src/abstractvision/__init__.py has no __version__"
    return m.group(1)


def _one(pattern: str, text: str, what: str) -> str:
    m = re.search(pattern, text, re.MULTILINE)
    assert m, f"{what} not found"
    return m.group(1)


def _section(text: str, header: str) -> str:
    start = text.index(f"\n[{header}]\n") + len(header) + 3
    end = text.find("\n[", start)
    return text[start:] if end == -1 else text[start:end]


def _extras() -> dict:
    """[project.optional-dependencies]: extra name -> the package names it requires."""
    out = {}
    body = _section(_read(ROOT / "pyproject.toml"), "project.optional-dependencies")
    for m in re.finditer(r"^([A-Za-z0-9_-]+) = \[(.*?)\]$", body, re.MULTILINE | re.DOTALL):
        reqs = re.findall(r'^\s*"([^"]+)"', m.group(2), re.MULTILINE)
        out[m.group(1)] = {re.split(r"[\s\[<>=!~;]", r, maxsplit=1)[0].lower() for r in reqs}
    return out


def _project_scripts() -> set:
    body = _section(_read(ROOT / "pyproject.toml"), "project.scripts")
    return set(re.findall(r"^([A-Za-z0-9_.-]+)\s*=", body, re.MULTILINE))


class InstallScriptsTest(unittest.TestCase):
    def test_both_scripts_exist(self):
        self.assertTrue(SH.is_file())
        self.assertTrue(PS1.is_file())

    @unittest.skipIf(shutil.which("sh") is None, "no POSIX sh")
    def test_install_sh_parses_and_prints_help(self):
        subprocess.run(["sh", "-n", str(SH)], check=True)
        out = subprocess.run(["sh", str(SH), "--help"], check=True, capture_output=True, text=True).stdout
        for opt in ("--profile auto|light|apple|gpu", "--pin VERSION|latest", "--from PATH", "--print"):
            self.assertIn(opt, out)

    @unittest.skipIf(shutil.which("sh") is None, "no POSIX sh")
    def test_install_sh_cut_mid_download_runs_nothing(self):
        """The body is one function called by `{ main "$@"; }` on the last line: a truncated `curl | sh`
        stream executes nothing, also when it ends inside that last line (a bare `main "$@"` cut
        after `main` would run the install without its arguments, since sh runs an unterminated
        last line)."""
        text = _read(SH)
        self.assertEqual(text.rstrip().splitlines()[-1], '{ main "$@"; }')
        data = text.encode("utf-8")
        last = len(data.rstrip()) - len(b'{ main "$@"; }')
        cuts = [int(len(data) * frac) for frac in (0.2, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99)]
        cuts += list(range(last - 2, len(data.rstrip())))  # every byte up to the closing brace
        for cut in cuts:
            out = subprocess.run(["sh", "-s", "--", "--print"], input=data[:cut], capture_output=True).stdout
            self.assertNotIn(b"AbstractVision installer", out, f"a stream cut at byte {cut} of {len(data)} ran the installer")
        out = subprocess.run(["sh", "-s", "--", "--print", "--profile", "light"], input=data, capture_output=True).stdout
        self.assertIn(b"AbstractVision installer", out)  # the whole stream does run

    def test_option_parity_between_scripts(self):
        sh, ps1 = _read(SH), _read(PS1)
        for opt in ("--profile", "--pin", "--from", "--no-modify-path", "--print", "--dry-run", "--verbose"):
            self.assertIn(opt, sh, f"install.sh lacks {opt}")
        self.assertIn("[Alias('Profile')]", ps1)
        self.assertIn("[Alias('DryRun')]", ps1)
        for param in ("$Pin", "$From", "$NoModifyPath", "$Print"):
            self.assertIn(f"]{param}", ps1, f"install.ps1 lacks {param}")
        self.assertIn("[CmdletBinding()]", ps1)
        self.assertIn("$VerbosePreference -eq 'Continue'", ps1)  # -Verbose
        for env in ("AVIS_PROFILE", "AVIS_PIN", "AVIS_FROM"):
            self.assertIn(env, sh)
            self.assertIn(f"$env:{env}", ps1)

    def test_review_fixes_carried_from_abstractvoice_and_abstractcore(self):
        sh, ps1 = _read(SH), _read(PS1)
        # nvidia-smi installed but not answering keeps its reason, in both scripts.
        for text in (sh, ps1):
            self.assertIn("nvidia-smi is installed but does not answer (driver not loaded?)", text)
            self.assertIn("A standalone AbstractVision beside the framework", text)
            self.assertIn("already includes AbstractVision", text)
        self.assertIn("${NV_ERR:-", sh)
        self.assertIn("$gpu.Error", ps1)
        # uv's installer is a downloaded file run as its own step, never `curl | sh` inside sh -c.
        self.assertIn('env UV_NO_MODIFY_PATH=1 sh "$WORK/uv-install.sh"', sh)
        self.assertNotIn("sh -c", sh)
        # The PATH check accepts $HOME/..., ${HOME}/..., ~/... (and so uv's . "$HOME/.local/bin/env").
        self.assertIn('-e "\\$HOME$_rel" -e "\\${HOME}$_rel" -e "~$_rel"', sh)
        # Windows: the caller's ErrorActionPreference is restored; `exit` only when run as a file.
        self.assertIn("if ($script:RanAsFile) { exit 1 }", ps1)

    def test_install_ps1_passes_no_inline_code_with_double_quotes(self):
        """Windows PowerShell 5.1 strips double quotes inside native arguments (pwsh 7 does not), so a
        `python -c "..."` (or `powershell -Command "..."`) whose code holds a `"` runs altered there.
        Every -c / -Command argument install.ps1 builds is checked (a literal, or the variable it
        names); the Python check runs from a .py file instead (Invoke-Smoke)."""
        ps1 = _read(PS1)
        args = re.findall(r"'-(?:c|Command)',\s*(\$\w+|'[^']*'|\"[^\"]*\")", ps1)
        self.assertGreaterEqual(len(args), 2, f"expected the -c/-Command arguments install.ps1 builds, found {args}")
        for arg in args:
            if arg.startswith("$"):
                values = re.findall(rf"^\s*\{arg}\s*=\s*(.+)$", ps1, re.MULTILINE)
                self.assertTrue(values, f"{arg} (a -c/-Command argument) is never assigned")
                for value in values:
                    body = value.strip()
                    if body.startswith("'"):
                        inner = body[1 : body.rindex("'")].replace("''", "'")
                    elif body.startswith('"'):
                        inner = body[1 : body.rindex('"')].replace('`"', '"')
                    else:
                        self.fail(f"{arg} = {body[:60]}: not a plain string, cannot be checked")
                    self.assertNotIn('"', inner, f"{arg} passes inline code with a double quote: {inner[:80]}")
            else:
                self.assertNotIn('"', arg[1:-1], f"inline -c argument with a double quote: {arg[:80]}")
        # No Python is run inline: python.exe is only ever given a script path.
        self.assertIsNone(re.search(r"&\s*\$\w*[Pp]y\w*\s+['\"]?-c", ps1))
        self.assertIn("& $Python $file", ps1)
        self.assertIn("$script:WorkDir", ps1)

    @unittest.skipIf(shutil.which("pwsh") is None, "no PowerShell")
    def test_install_ps1_parses(self):
        cmd = (
            "$e = $null; [System.Management.Automation.Language.Parser]::ParseFile("
            f"'{PS1}', [ref]$null, [ref]$e) | Out-Null; $e.Count"
        )
        out = subprocess.run(["pwsh", "-NoProfile", "-Command", cmd], check=True, capture_output=True, text=True).stdout
        self.assertEqual(out.strip(), "0")

    def test_default_pin_is_this_release(self):
        self.assertEqual(_one(r'^AVIS_PIN_DEFAULT="([^"]+)"', _read(SH), "AVIS_PIN_DEFAULT in install.sh"), _version())
        self.assertEqual(_one(r"^\$AvisPinDefault = '([^']+)'", _read(PS1), "$AvisPinDefault in install.ps1"), _version())

    def test_profiles_install_only_the_three_settings(self):
        sh, ps1 = _read(SH), _read(PS1)
        extras = _extras()
        # The profile -> extra mapping each script installs; light is the base package, no extra.
        self.assertEqual(dict(re.findall(r"(apple|gpu)\) EXTRAS=(\w+)", sh)), {"apple": "apple", "gpu": "gpu"})
        self.assertIn('*) EXTRAS="" ;;', sh)
        self.assertIn("$extras = if ($profileName -eq 'gpu') { 'gpu' } else { '' }", ps1)
        # Every abstractvision[...] the scripts or their headers name is `apple` or `gpu`, and exists.
        for text, name, expected in ((sh, "install.sh", {"apple", "gpu"}), (ps1, "install.ps1", {"gpu"})):
            named = set()
            for group in re.findall(r"abstractvision\[([a-z0-9,-]+)\]", text):
                named.update(group.split(","))
            self.assertEqual(named, expected, f"{name} names abstractvision[...] extras {sorted(named)}")
            self.assertLessEqual(named, set(extras))
        # Both scripts offer the same profiles (apple is refused on Windows, with a sentence).
        for p in ("auto", "light", "apple", "gpu"):
            self.assertIn(f"'{p}'", ps1, f"install.ps1 does not handle profile {p}")
            self.assertRegex(sh, rf"(?m)^\s*{p}[|)]", f"install.sh does not handle profile {p}")

    def test_commands_are_exactly_the_project_scripts(self):
        scripts = _project_scripts()
        self.assertTrue(scripts)
        sh_names = _one(r'^AVIS_COMMANDS="([^"]+)"', _read(SH), "AVIS_COMMANDS in install.sh").split()
        ps1_names = re.findall(r"'([^']+)'", _one(r"^\$AvisCommands = @\((.+)\)$", _read(PS1), "$AvisCommands in install.ps1"))
        self.assertEqual(len(sh_names), len(set(sh_names)))
        self.assertEqual(set(sh_names), scripts)
        self.assertEqual(set(ps1_names), scripts)

    def test_prebuilt_wheel_handling_targets_real_requirements(self):
        """apple carries stable-diffusion-cpp-python, which PyPI publishes as a source build only:
        the overrides file and --no-build-package drop it. gpu must not carry it (neither script
        overrides anything there) and light has no dependency at all."""
        sh, ps1 = _read(SH), _read(PS1)
        extras = _extras()
        no_wheel = _one(r'^AVIS_NO_WHEEL="([^"]+)"', sh, "AVIS_NO_WHEEL").split()
        self.assertEqual(no_wheel, ["stable-diffusion-cpp-python"])
        for name in no_wheel:
            self.assertIn(name, extras["apple"], "an override for a package the apple setting does not carry")
            self.assertNotIn(name, extras["gpu"], "the gpu setting gained a package with no wheel: drop it there too")
        self.assertIn("--overrides uv-overrides.txt", sh)
        self.assertIn('set -- "$@" --no-build-package "$_p"', sh)
        self.assertIn('if [ "$EXTRAS" = apple ]; then', sh)
        self.assertNotIn("--overrides", ps1)  # Windows installs no apple tree
        self.assertEqual(_one(r"^dependencies = (\[.*\])$", _read(ROOT / "pyproject.toml"), "dependencies"), "[]")
        # The summary says what is left out and that sd-cli installs itself at the first GGUF run.
        self.assertIn("left out:  stable-diffusion-cpp-python", sh)
        for text in (sh, ps1):
            self.assertIn("ABSTRACTVISION_SDCPP_AUTO_INSTALL=0", text)
        self.assertIn("ABSTRACTVISION_SDCPP_AUTO_INSTALL", _read(ROOT / "src" / "abstractvision" / "backends" / "stable_diffusion_cpp.py"))

    def test_device_check_imports_torch_and_torchvision(self):
        sh, ps1 = _read(SH), _read(PS1)
        for text in (sh, ps1):
            self.assertIn("import torchvision", text)  # a torch / torchvision pair from two indexes fails here
            self.assertIn("torch.cuda.is_available()", text)
        self.assertIn('"$TOOL_PY" "$WORK/torch-check.py"', sh)
        self.assertIn("Invoke-Smoke $toolPy $AvisTorchSmoke", ps1)

    def test_scripts_and_docs_name_the_exact_one_liners(self):
        readme, install = _read(README), _read(INSTALL_DOC)
        for doc, name in ((readme, "README.md"), (install, "docs/getting-started.md")):
            for line in (SH_LINE, PS1_LINE) + PIP_LINES:
                self.assertIn(line, doc, f"{name} lacks: {line}")
        self.assertIn("## One-line install", install)
        self.assertIn("getting-started.md", _read(ROOT / "docs" / "README.md"))
        self.assertIn(SH_LINE, _read(SH))
        self.assertIn(PS1_LINE, _read(PS1))
        for text in (_read(SH), _read(PS1), readme):
            self.assertIn("docs/getting-started.md#one-line-install", text)

    def test_linux_without_a_gpu_is_documented_as_measured(self):
        sh, install = _read(SH), _read(INSTALL_DOC)
        # light has no local generation; --profile gpu runs Diffusers on the processor.
        self.assertIn("### Linux without a GPU", install)
        self.assertIn("`--profile gpu` is the way to generate locally on the processor", install)
        # The smaller footprint: torch and torchvision both from PyTorch's CPU index, the same line
        # in the summary and the docs.
        slim = 'uv tool install --python 3.12 --torch-backend cpu --reinstall "abstractvision[gpu]==<version>"'
        self.assertIn(slim, install)
        self.assertIn("uv tool install --python $AVIS_PYTHON --torch-backend cpu --reinstall", sh)
        self.assertIn("Both must come from the same index", install)


if __name__ == "__main__":
    unittest.main()
