<#
.SYNOPSIS
    AbstractVision standalone installer for Windows 10/11 (PowerShell 5.1 and 7).

.DESCRIPTION
    One line, no admin rights, no system Python needed:

        powershell -ExecutionPolicy ByPass -c "irm https://abstractframework.ai/vision/install.ps1 | iex"

    With options (a script block keeps the parameters):

        & ([scriptblock]::Create((irm https://abstractframework.ai/vision/install.ps1))) -Profile gpu

    The Windows twin of scripts/install.sh (its header has the reasoning): AbstractVision alone,
    the abstractvision command, no gateway, no services. Each step prints its command; -Print
    shows them and changes nothing. Steps: preflight; uv into %USERPROFILE%\.local\bin when
    missing; Python 3.12 through uv; uv tool install --python 3.12 "abstractvision[<profile>]==<version>"
    (prebuilt wheels only, no MSVC Build Tools); the PyTorch device check (gpu); `uv tool
    update-shell` for the user PATH.

    Profiles (-Profile auto|light|gpu; auto takes the first that fits):
      gpu    nvidia-smi answers: abstractvision[gpu] (Diffusers) with PyTorch's CUDA build for the
             driver (580+ and compute capability 7.5+: --torch-backend cu130; 525+: cu126; else
             PyPI's CPU build). --torch-backend takes torch and torchvision from the same index
             (a torch from one index and torchvision from another do not load together). Asked
             for without a card: the same stack on the processor (PyPI's PyTorch is the CPU
             build on Windows). Windows on ARM: refused (PyTorch publishes no ARM64 wheels).
      light  abstractvision with no extra (no dependency at all): the CLI, the catalog and the
             OpenAI-compatible backend; no local generation, which needs the gpu profile
      apple  Apple Silicon Macs only: refused here
    Every package of the gpu tree has a Windows x64 wheel for Python 3.12; stable-diffusion-cpp-python
    is not in abstractvision[gpu], so no override is needed (install.sh drops it from apple). GGUF
    models run through stable-diffusion.cpp's sd-cli, which AbstractVision downloads itself into
    %USERPROFILE%\.abstractvision\bin at the first GGUF generation; this installer does not.

    Upgrade: run the same line again ("AbstractVision <old> found: upgrading to <new>", or "already
    up to date"); the profile is kept unless -Profile says otherwise. Remove: uv tool uninstall abstractvision

    Options (environment twins AVIS_PROFILE, AVIS_PIN, AVIS_FROM): -Profile auto|light|gpu;
    -Pin VERSION|latest (default: the release this script ships with; latest = the newest on PyPI);
    -From PATH (a checkout, sdist or wheel, for testing); -NoModifyPath; -Print (alias -DryRun);
    -Verbose (every command's full output). The body runs only from Main, called on the last lines.

.EXAMPLE
    .\install.ps1 -Print
#>
[CmdletBinding()]
param(
    # No [ValidateSet]: `irm | iex` evaluates this block as statements; Main validates the value.
    [Alias('Profile')]
    [string]$InstallProfile = '',
    [string]$Pin = '',
    [string]$From = '',
    [switch]$NoModifyPath,
    [Alias('DryRun')]
    [switch]$Print
)

# The release this script ships with = src/abstractvision/__init__.py (tests/test_install_scripts.py).
$AvisPinDefault = '0.3.36'
$AvisPython = '3.12'
$AvisScriptUrl = 'https://abstractframework.ai/vision/install.ps1'
$AvisDocs = 'https://github.com/lpalbou/AbstractVision/blob/main/docs/getting-started.md#one-line-install'
# Every name of [project.scripts] in pyproject.toml (the test compares them): uv refuses the
# whole install when one of them already exists.
$AvisCommands = @('abstractvision')
# The device check, in a fresh process of the tool's environment: torch and torchvision import
# (a mismatched pair fails here), CUDA. One JSON line "AVISMOKE {...}".
$AvisTorchSmoke = @'
import json
r = {"version": "", "cuda_build": "", "cuda": False, "gpu": "", "torchvision": "", "error": ""}
try:
    import torch
    r["version"] = torch.__version__
    r["cuda_build"] = str(torch.version.cuda or "")
    r["cuda"] = bool(torch.cuda.is_available())
    if r["cuda"]:
        r["gpu"] = torch.cuda.get_device_name(0)
    import torchvision
    r["torchvision"] = torchvision.__version__
except BaseException as e:
    r["error"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0] if str(e) else "")
print("AVISMOKE " + json.dumps(r))
'@

$script:RanAsFile = [bool]$PSCommandPath; $script:DryRun = [bool]$Print; $script:LogFile = ''; $script:StepNo = 0
$script:ShowAll = ($VerbosePreference -eq 'Continue')   # -Verbose (CmdletBinding): show every command's output

function Write-Step([string]$Text) { $script:StepNo++; Write-Host ''; Write-Host "[$($script:StepNo)] $Text" -ForegroundColor White }
function Write-Ok([string]$Text) { Write-Host '  + ' -ForegroundColor Green -NoNewline; Write-Host $Text }
function Write-Info([string]$Text) { Write-Host '  . ' -ForegroundColor Cyan -NoNewline; Write-Host $Text }
function Write-Warn2([string]$Text) { Write-Host '  ! ' -ForegroundColor Yellow -NoNewline; Write-Host $Text }
function Stop-Install([string]$Text) { throw "AVISBOOT: $Text" }
function Test-Command([string]$Name) { return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }
function Format-Arg([string]$Value) { if ($Value -match '^[A-Za-z0-9_./:=@,+%\\-]+$') { return $Value }; return "'" + ($Value -replace "'", "''") + "'" }
function Format-Cmd([string[]]$Argv) { return (($Argv | ForEach-Object { Format-Arg $_ }) -join ' ') }
function Get-Http([string]$Url) { try { return [string](Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 20 -ErrorAction Stop -Verbose:$false).Content } catch { return $null } }

# Run a native command: print it, log its output (and show it with -Live), return $true on exit 0.
# A failure stops the install unless -Soft, which warns and returns $false.
function Invoke-Native {
    param([string]$Description, [string[]]$Argv, [switch]$Soft, [switch]$Live, [string]$Shown = '')
    if (-not $Shown) { $Shown = Format-Cmd $Argv }
    Write-Host "  `$ $Shown" -ForegroundColor DarkGray
    if ($script:DryRun) { return $true }
    $exe = $Argv[0]; $rest = @(); if ($Argv.Count -gt 1) { $rest = $Argv[1..($Argv.Count - 1)] }
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'   # PS 5.1 turns native stderr into errors
    try {
        Add-Content -Path $script:LogFile -Value "`r`n`$ $Shown" -Encoding UTF8
        & $exe @rest 2>&1 | ForEach-Object {
            $line = "$_"; Add-Content -Path $script:LogFile -Value $line -Encoding UTF8
            if ($Live -or $script:ShowAll) { Write-Host "    $line" -ForegroundColor DarkGray }
        }
        $code = $LASTEXITCODE
    } catch { Add-Content -Path $script:LogFile -Value "$_" -Encoding UTF8; $code = 1 } finally { $ErrorActionPreference = $old }
    if ($code -eq 0) { return $true }
    if ($Soft) { Write-Warn2 "$Description did not succeed (exit $code; log: $($script:LogFile))"; return $false }
    Get-Content -Path $script:LogFile -Tail 15 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Stop-Install "$Description failed (exit $code). Log: $($script:LogFile)`nWhat to do: run the installer again (it repairs a half-finished install); if it stops at the same step, report it with that log ($AvisDocs)."
}

# The framework installer's Get-NvidiaInfo: Ok, Driver, DriverMajor, Cc (the lowest compute
# capability, $null when the driver cannot report it), Name, Error (why an installed nvidia-smi
# gives no GPU; empty when there is no nvidia-smi at all).
function Get-NvidiaInfo {
    $info = @{ Ok = $false; Driver = ''; DriverMajor = 0; Cc = $null; Name = ''; Error = '' }
    $noAnswer = 'nvidia-smi is installed but does not answer (driver not loaded?)'
    if (-not (Test-Command 'nvidia-smi')) { return $info }
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try {
        $withCc = $true
        $lines = @(& nvidia-smi '--query-gpu=driver_version,compute_cap,name' '--format=csv,noheader' 2>$null | ForEach-Object { "$_" })
        if ($LASTEXITCODE -ne 0) {   # drivers too old to report compute_cap: the driver alone
            $withCc = $false; $lines = @(& nvidia-smi '--query-gpu=driver_version,name' '--format=csv,noheader' 2>$null | ForEach-Object { "$_" }) }
        if ($LASTEXITCODE -ne 0) { $info.Error = $noAnswer; return $info }
    } catch { $info.Error = $noAnswer; return $info } finally { $ErrorActionPreference = $old }
    $inv = [Globalization.CultureInfo]::InvariantCulture; $nameAt = if ($withCc) { 2 } else { 1 }
    foreach ($l in ($lines | Where-Object { $_.Trim() })) {
        $parts = $l.Split(',', $nameAt + 1); $major = 0; $cc = 0.0
        if (-not [int]::TryParse(($parts[0].Trim().Split('.')[0]), [ref]$major)) { continue }
        if (-not $info.Driver) { $info.Driver = $parts[0].Trim(); $info.DriverMajor = $major }
        if ($withCc -and $parts.Count -ge 2 -and [double]::TryParse($parts[1].Trim(), [Globalization.NumberStyles]::Float, $inv, [ref]$cc)) {
            if ($null -eq $info.Cc -or $cc -lt $info.Cc) { $info.Cc = $cc }
        }
        if (-not $info.Name -and $parts.Count -gt $nameAt) { $info.Name = $parts[$nameAt].Trim() }
    }
    $info.Ok = [bool]$info.Driver; if (-not $info.Ok) { $info.Error = 'nvidia-smi gave no driver version' }; return $info
}

function Find-Uv([string]$HomeDir) {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($d in @($env:UV_INSTALL_DIR, $env:XDG_BIN_HOME, [IO.Path]::Combine($HomeDir, '.local', 'bin'), [IO.Path]::Combine($HomeDir, '.cargo', 'bin'))) {
        if ($d -and (Test-Path -LiteralPath (Join-Path $d 'uv.exe'))) { return (Join-Path $d 'uv.exe') }
    }
    return ''
}
function Get-ToolList([string]$Uv) {  # `uv tool list` lines (none without uv)
    if (-not $Uv) { return @() }
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { return @(& $Uv tool list 2>$null | ForEach-Object { "$_" }) } catch { return @() } finally { $ErrorActionPreference = $old }
}
function Get-ToolVersion([string]$Uv) { foreach ($l in (Get-ToolList $Uv)) { if ($l -match '^abstractvision v(\S+)') { return $Matches[1] } }; return '' }
# Run one smoke script with the tool environment's Python: the parsed JSON, or $null. Always from a
# .py file in the installer's folder (kept, so the printed command can be run again), run by path,
# never `python -c`: Windows PowerShell 5.1 strips double quotes inside native arguments (pwsh 7
# does not), so inline code holding a `"` would run altered and misreport there.
# tests/test_install_scripts.py rejects any -c argument with a `"`.
function Invoke-Smoke([string]$Python, [string]$Code, [string]$File) {
    if (-not $Python -or -not (Test-Path -LiteralPath $Python)) { return $null }
    [System.IO.File]::WriteAllText($File, $Code, (New-Object System.Text.UTF8Encoding($false)))
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $out = @(& $Python $File 2>&1 | ForEach-Object { "$_" }) } catch { $out = @("$_") } finally { $ErrorActionPreference = $old }
    if ($script:LogFile) { Add-Content -Path $script:LogFile -Value (@("`r`n`$ $Python <smoke check>") + $out) -Encoding UTF8 }
    $line = $out | Where-Object { $_ -like 'AVISMOKE *' } | Select-Object -Last 1
    if (-not $line) { return $null }
    try { return ($line.Substring(9) | ConvertFrom-Json) } catch { return $null }
}

function Main {
    $onWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or ($IsWindows -eq $true)
    if (-not $onWindows -and -not $script:DryRun) {
        Stop-Install 'this installer is for Windows; on macOS and Linux run: curl -LsSf https://abstractframework.ai/vision/install.sh | sh'
    }
    $homeDir = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
    $profileName = $InstallProfile
    if (-not $profileName) { $profileName = if ($env:AVIS_PROFILE) { $env:AVIS_PROFILE } else { 'auto' } }
    $profileName = $profileName.ToLowerInvariant()   # -Profile GPU works, as in install.sh
    if (-not $Pin -and $env:AVIS_PIN) { $Pin = $env:AVIS_PIN }
    if (-not $From -and $env:AVIS_FROM) { $From = $env:AVIS_FROM }
    $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }

    Write-Host "AbstractVision installer  $(if ($script:DryRun) { '(-Print: preflight only, nothing is installed)' } else { $AvisScriptUrl })" -ForegroundColor White
    Write-Step 'Preflight'
    Write-Ok "system: Windows $([Environment]::OSVersion.Version) $arch"

    $uv = Find-Uv $homeDir; $toolBin = ''; $toolDir = ''
    if ($uv) { $toolBin = (& $uv tool dir --bin 2>$null | Select-Object -First 1); $toolDir = (& $uv tool dir 2>$null | Select-Object -First 1) }
    if (-not $toolBin) { $toolBin = if ($env:UV_TOOL_BIN_DIR) { $env:UV_TOOL_BIN_DIR } else { [IO.Path]::Combine($homeDir, '.local', 'bin') } }
    if (-not $toolDir) { $toolDir = if ($env:UV_TOOL_DIR) { $env:UV_TOOL_DIR } else { [IO.Path]::Combine($(if ($env:APPDATA) { $env:APPDATA } else { $homeDir }), 'uv', 'tools') } }
    $before = Get-ToolVersion $uv; $prevProfile = ''; $otherExtras = ''   # the existing install: its version, and its profile from the uv receipt
    $receipt = [IO.Path]::Combine($toolDir, 'abstractvision', 'uv-receipt.toml')
    if ($before -and (Test-Path -LiteralPath $receipt)) {
        $rc = Get-Content -LiteralPath $receipt -Raw
        if ($rc -match 'name = "abstractvision", extras = \[[^\]]*"(gpu|all-gpu)"') { $prevProfile = 'gpu' }
        elseif ($rc -match 'name = "abstractvision", extras = \[([^\]]*)\]') { $otherExtras = ($Matches[1] -replace '[" ]', '') }   # e.g. diffusers
        elseif ($rc -match 'name = "abstractvision"') { $prevProfile = 'light' } }
    if ($otherExtras) {
        if ($profileName -eq 'auto') { Write-Warn2 "the previous install used the extras [$otherExtras], which are not a profile of this installer: re-detecting the profile (-Profile picks one)" }
        else { Write-Warn2 "the previous install used the extras [$otherExtras]: replacing them with the $profileName profile" } }

    $gpu = Get-NvidiaInfo; $local = 'no local generation: that needs the gpu profile'
    switch ($profileName) {
        'auto' {
            if ($prevProfile) { $profileName = $prevProfile; $why = 'kept from the previous install' }
            elseif ($arch -eq 'ARM64') { $profileName = 'light'; $why = "Windows on ARM: PyTorch publishes no ARM64 wheels, so $local" }
            elseif ($gpu.Ok) { $profileName = 'gpu'; $why = "nvidia-smi found $(if ($gpu.Name) { $gpu.Name } else { 'a GPU' })" }
            elseif ($gpu.Error) { $profileName = 'light'; $why = "$($gpu.Error), so cloud providers only; once the driver works, run the installer again" }
            else { $profileName = 'light'; $why = 'no GPU found, so cloud providers only; -Profile gpu generates locally on the processor' }
            Write-Ok "profile: $profileName ($why; override with -Profile)"
        }
        'light' { Write-Ok "profile: light (the CLI and the OpenAI-compatible backend; $local)" }
        'gpu' {
            if ($arch -eq 'ARM64') { Stop-Install 'the gpu profile has no Windows on ARM build (PyTorch publishes no ARM64 wheels); use -Profile light' }
            if (-not $gpu.Ok) { Write-Warn2 "gpu profile requested, but $(if ($gpu.Error) { $gpu.Error } else { 'there is no nvidia-smi' }): Diffusers will run on the processor" }
            Write-Ok 'profile: gpu (Diffusers on PyTorch: CUDA when a card works, else the processor)'
        }
        'apple' { Stop-Install 'the apple profile is for Apple Silicon Macs; use -Profile light or gpu' }
        default { Stop-Install "unknown profile '$profileName' (expected auto, light or gpu)" }
    }
    $extras = if ($profileName -eq 'gpu') { 'gpu' } else { '' }

    # PyTorch's build (gpu profile; the framework installer's Select-GpuStack, without llama.cpp).
    $torchArgs = @()
    if ($profileName -eq 'gpu') {
        $torch = ''
        if ($gpu.Ok -and $gpu.DriverMajor -ge 580 -and $null -ne $gpu.Cc -and $gpu.Cc -ge 7.5) { $torch = 'cu130' }
        elseif ($gpu.Ok -and $gpu.DriverMajor -ge 525) { $torch = 'cu126' }
        if ($torch) {
            $help = if ($uv) { (& $uv tool install --help 2>$null) -join "`n" } else { '--torch-backend' }
            if ($help -match '--torch-backend') { $torchArgs = @('--torch-backend', $torch); Write-Ok "PyTorch: CUDA build $torch (driver $($gpu.Driver)), torch and torchvision from the same index" }
            else { Write-Warn2 "this uv has no --torch-backend: PyTorch's CPU build is installed; update uv (uv self update) and run the installer again" }
        } else { Write-Info "PyTorch: PyPI's CPU build ($(if ($gpu.Ok) { "driver $($gpu.Driver) is older than 525" } else { 'no NVIDIA GPU' }))" }
    }

    if ($From) {
        if (-not (Test-Path -LiteralPath $From)) { Stop-Install "-From: no such file or folder: $From" }
        $From = (Resolve-Path -LiteralPath $From).Path; $target = "(the build from $From)"; $pinWhy = '-From'
    } elseif ($Pin -eq 'latest') {
        $latest = ''; $j = Get-Http 'https://pypi.org/pypi/abstractvision/json'
        if ($j -and $j -match '"version":\s*"([^"]+)"') { $latest = $Matches[1] }
        $target = if ($latest) { $latest } else { 'the newest release on PyPI' }; $pinWhy = '-Pin latest'
    } else {
        if ($Pin) { $pinWhy = '-Pin' } else { $Pin = $AvisPinDefault; $pinWhy = 'the release this installer ships with' }; $target = $Pin
    }
    function Get-Spec([string]$Extras) {  # the requirement given to uv
        $x = if ($Extras) { "[$Extras]" } else { '' }
        if ($From) { return "abstractvision$x @ " + [Uri]::new($From, [UriKind]::Absolute).AbsoluteUri }
        if ($Pin -eq 'latest') { return "abstractvision$x" }; return "abstractvision$x==$Pin"
    }
    Write-Ok "package: $(Get-Spec $extras)  (version: $pinWhy)"
    if (-not $before) { Write-Ok "No AbstractVision install found: installing AbstractVision $target" }
    elseif ($From) { Write-Ok "AbstractVision $before found: replacing it with the build from $From (-From)" }
    elseif ($before -eq $target) { Write-Ok "AbstractVision $before found: already up to date (checked, repaired if needed)" }
    elseif ($Pin -ne 'latest' -and $(try { [version]($before -replace '[^0-9.].*$', '') -gt [version]($Pin -replace '[^0-9.].*$', '') } catch { $false })) {
        Write-Ok "AbstractVision $before found: going back to AbstractVision $target (-Pin)"
    } else { Write-Ok "AbstractVision $before found: upgrading to AbstractVision $target" }

    # A command of the same name from another program (often AbstractFramework's gateway tool):
    # uv would refuse the install, so stop and say why.
    if (-not $before) {
        foreach ($n in $AvisCommands) {
            $exe = Join-Path $toolBin "$n.exe"
            if (-not (Test-Path -LiteralPath $exe)) { continue }
            $owner = ''; $t = ''
            foreach ($l in (Get-ToolList $uv)) { if ($l -match '^(\S+) v') { $t = $Matches[1] } elseif ($l -eq "- $n") { $owner = $t; break } }
            $msg = "$exe already exists ($(if ($owner) { "from the uv tool '$owner'" } else { "another program's" }))"
            if ($owner -eq 'abstractgateway') { $msg += ': AbstractFramework is installed here and already includes AbstractVision; upgrade it with: powershell -ExecutionPolicy ByPass -c "irm https://abstractframework.ai/install.ps1 | iex". A standalone AbstractVision beside the framework''s gateway tool is not supported (there is no -Force: both would own the same commands)' }
            if ($script:DryRun) { Write-Warn2 "$msg (the real run stops here; nothing is changed)"; break }
            Stop-Install "$msg`nNothing was changed."
        }
    }

    $needMb = if ($profileName -eq 'gpu') { 7000 } else { 500 }
    try {
        $root = [System.IO.Path]::GetPathRoot($homeDir)
        $freeMb = [int]((New-Object System.IO.DriveInfo($root)).AvailableFreeSpace / 1MB)
        if ($freeMb -lt 300) { Stop-Install "only $freeMb MB free on $root (about $needMb MB needed)" }
        elseif ($freeMb -lt $needMb) { Write-Warn2 "only $freeMb MB free on $root; the $profileName profile needs about $needMb MB (models need more)" }
        else { Write-Ok "disk: $freeMb MB free (about $needMb MB needed, models extra)" }
    } catch { if ("$_" -like 'AVISBOOT:*') { throw } }
    if ($null -ne (Get-Http 'https://pypi.org/simple/pip/')) { Write-Ok 'internet: pypi.org reachable' }
    elseif ($script:DryRun) { Write-Warn2 'pypi.org is not reachable (a real run stops here)' }
    else { Stop-Install 'this computer cannot reach pypi.org, where AbstractVision is downloaded (check the connection or the proxy). Nothing was changed.' }
    if (-not $From -and $Pin -ne 'latest' -and $null -eq (Get-Http "https://pypi.org/pypi/abstractvision/$Pin/json")) {
        $m = "abstractvision $Pin is not on PyPI (yet): use -Pin latest for the newest published version"
        if ($script:DryRun) { Write-Warn2 $m } else { Stop-Install "$m. Nothing was changed." }
    }
    $work = [IO.Path]::Combine($(if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { $homeDir }), 'abstractvision', 'installer')
    if (-not $script:DryRun) { New-Item -ItemType Directory -Force -Path $work | Out-Null
        $script:LogFile = Join-Path $work 'install.log'; Set-Content -Path $script:LogFile -Value '' -Encoding UTF8 }

    # --- 2. uv and Python ------------------------------------------------------------------------
    Write-Step 'uv (Python toolchain manager)'
    if ($uv) { Write-Ok "uv found: $uv" }
    else {
        Write-Info 'installing uv from astral.sh into %USERPROFILE%\.local\bin (no admin)'
        # A child PowerShell with a per-process ByPass policy, so the vendor script's `exit` cannot end this one.
        $command = "`$env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex"
        Invoke-Native -Description 'install uv' -Argv @((Get-Process -Id $PID).Path, '-NoProfile', '-ExecutionPolicy', 'ByPass', '-Command', $command) `
            -Shown "powershell -ExecutionPolicy ByPass -c `"$command`"" | Out-Null
        if ($script:DryRun) { $uv = 'uv' }
        else { $uv = Find-Uv $homeDir; if (-not $uv) { Stop-Install 'uv was installed but cannot be found in %USERPROFILE%\.local\bin' }
            Write-Ok "uv installed: $uv"
            $toolBin = (& $uv tool dir --bin 2>$null | Select-Object -First 1); $toolDir = (& $uv tool dir 2>$null | Select-Object -First 1) }
    }
    Write-Step "Python $AvisPython (managed by uv, isolated from any system Python)"
    Invoke-Native -Description "install Python $AvisPython" -Argv @($uv, 'python', 'install', $AvisPython) | Out-Null

    # --- 3. AbstractVision -----------------------------------------------------------------------
    Write-Step 'AbstractVision'
    function Install-Tool([string]$Extras, [string[]]$More = @()) {
        $argv = @($uv, 'tool', 'install', '--python', $AvisPython) + $More
        if ($Extras -eq 'gpu') { $argv += $torchArgs }
        if ($From) { $argv += @('--reinstall-package', 'abstractvision') }      # same version number, new code
        elseif ($Pin -eq 'latest') { $argv += '--upgrade' }                     # else uv keeps a satisfying install
        $argv += (Get-Spec $Extras)
        return (Invoke-Native -Description "install $(Get-Spec $Extras)" -Argv $argv -Live)
    }
    Install-Tool $extras | Out-Null
    $avisExe = Join-Path $toolBin 'abstractvision.exe'
    $after = $before
    if (-not $script:DryRun) {
        # Repair: uv can report the tool installed while its command cannot start; reinstall in place.
        & $avisExe --help *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Warn2 'the installed abstractvision does not start (an earlier install was interrupted or damaged): reinstalling it'
            Install-Tool $extras -More @('--reinstall') | Out-Null
            & $avisExe --help *> $null
            if ($LASTEXITCODE -ne 0) { Stop-Install "abstractvision still does not start after reinstalling it (log: $($script:LogFile))" }
        }
        $after = Get-ToolVersion $uv
        if (-not $before) { Write-Ok "installed abstractvision $after" }
        elseif ($before -ne $after) { Write-Ok "abstractvision $before -> $after" }
        elseif ($From) { Write-Ok "abstractvision $after reinstalled from $From" }
        else { Write-Ok "abstractvision $after already installed" }
    }

    # --- 4. PyTorch: the device Diffusers runs on (gpu) -----------------------------------------
    $torchResult = ''
    if ($extras) {
        Write-Step 'PyTorch (Diffusers): the device it runs on'
        $toolPy = [IO.Path]::Combine($toolDir, 'abstractvision', 'Scripts', 'python.exe')
        $checkPy = Join-Path $work 'torch-check.py'
        Write-Host "  `$ $(Format-Cmd @($toolPy, $checkPy))   # imports torch and torchvision, reports CUDA" -ForegroundColor DarkGray
        if ($script:DryRun) { $torchResult = 'planned: the check above (torch and torchvision import; CUDA)' }
        else {
            $r = Invoke-Smoke $toolPy $AvisTorchSmoke $checkPy
            if (-not $r -or $r.error -or -not $r.version) {
                $torchResult = "does not import ($(if ($r -and $r.error) { $r.error } else { 'no answer' }); log: $($script:LogFile)): Diffusers cannot run"; Write-Warn2 "PyTorch $torchResult"
            } elseif ($r.cuda) {
                $torchResult = "$($r.version) (CUDA $($r.cuda_build)), torchvision $($r.torchvision): on the GPU ($($r.gpu))"; Write-Ok "PyTorch $torchResult"
            } else {
                $why = if (-not $gpu.Ok) { $(if ($gpu.Error) { $gpu.Error } else { 'no NVIDIA GPU' }) } elseif (-not $r.cuda_build) { "PyPI's CPU build" } else { 'it does not see the GPU' }
                $torchResult = "$($r.version), torchvision $($r.torchvision): on the processor ($why)"; Write-Info "PyTorch $torchResult"
            }
        }
    }

    # --- 5. PATH and summary -----------------------------------------------------------------------
    Write-Step 'PATH'
    if (($env:PATH -split [IO.Path]::PathSeparator) -contains $toolBin) { Write-Ok "$toolBin is on PATH" }
    elseif ($NoModifyPath) { Write-Warn2 "$toolBin is not on PATH (-NoModifyPath); add it to your user PATH or use $avisExe" }
    elseif (Invoke-Native -Description "add $toolBin to your user PATH" -Argv @($uv, 'tool', 'update-shell') -Soft) {
        Write-Info "open a new terminal for the 'abstractvision' command to be on PATH"
    } else { Write-Info "add $toolBin to your user PATH yourself (Settings > System > About > Advanced system settings > Environment Variables)" }

    Write-Host ''
    Write-Host $(if ($script:DryRun) { 'Plan (-Print: nothing was changed)' } else { 'AbstractVision is installed' }) -ForegroundColor White
    $ver = if ($script:DryRun -or -not $after) { $target } else { $after }
    Write-Host "  version:   $ver$(if (-not $script:DryRun -and $before -and $before -ne $after) { " (was $before)" })"
    Write-Host "  profile:   $profileName ($(Get-Spec $extras))"
    if ($extras) { Write-Host '  engines:   Diffusers (PyTorch), OpenAI-compatible endpoints' }
    else {
        Write-Host '  engines:   OpenAI-compatible endpoints only (OpenAI, any /v1/images server); local generation needs'
        Write-Host '             the gpu profile (Diffusers; without a GPU on the processor): run the installer again with -Profile gpu'
    }
    if ($torchResult) { Write-Host "  PyTorch:   $torchResult" }
    if ($extras) { Write-Host "  GGUF:      stable-diffusion.cpp's sd-cli is not installed here: AbstractVision downloads it into %USERPROFILE%\.abstractvision\bin at the first GGUF generation (ABSTRACTVISION_SDCPP_AUTO_INSTALL=0 turns that off)" }
    Write-Host "  commands:  $avisExe"
    if (-not $script:DryRun) { Write-Host "  log:       $($script:LogFile)" }
    Write-Host ''
    Write-Host 'Try it:'
    Write-Host '  abstractvision --help'
    Write-Host '  abstractvision catalog            # the models this machine can download and run'
    Write-Host "Docs: $AvisDocs"
}

# The caller's $ErrorActionPreference is not ours (`irm | iex` shares the caller's scope; under
# 'Stop', PowerShell 5.1 turns any native stderr line into a terminating error), so Main runs under
# 'Continue', as with -File, and the caller's value comes back after.
$AvisCallerErrorAction = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    Main
} catch {
    Write-Host ''
    Write-Host "ERROR: $("$_" -replace '^AVISBOOT: ', '')" -ForegroundColor Red
    if ("$_" -notlike 'AVISBOOT:*') { Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray }
    # `exit` would close the window of someone who pasted `irm | iex`; only exit when run as a file.
    if ($script:RanAsFile) { exit 1 }
} finally {
    $ErrorActionPreference = $AvisCallerErrorAction
}
