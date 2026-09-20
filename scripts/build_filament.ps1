# Build the pinned (and patched) Filament checkout with MSVC + Ninja, Vulkan backend only, samples skipped, and
# install it to third_party/filament-install (git-ignored): include/, lib/x86_64/{filament,gltfio,...}.lib, bin/
# (matc, resgen, cmgen). The demo links against that install. Run scripts\fetch_tools.ps1 and
# scripts\fetch_filament.ps1 first. Incremental: re-running only rebuilds what changed. The build tree is
# %LOCALAPPDATA%\dlss5-vulkan\filament-<config> (or DLSS5_FILAMENT_BUILD_DIR): MSVC cannot open object files
# whose path exceeds MAX_PATH, and Filament's hashed object paths under a long checkout path do.
#   powershell -File scripts\build_filament.ps1 [-Debug]
param([switch]$Debug)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "third_party\filament"
$config = if ($Debug) { "Debug" } else { "Release" }
$out = if ($env:DLSS5_FILAMENT_BUILD_DIR) { $env:DLSS5_FILAMENT_BUILD_DIR } else { Join-Path $env:LOCALAPPDATA ("dlss5-vulkan\filament-" + $config.ToLower()) }
$install = Join-Path $root "third_party\filament-install"
$cmake = Join-Path $root "tools\cmake\bin\cmake.exe"
$ninja = Join-Path $root "tools\ninja"
$vcvars = & (Join-Path $PSScriptRoot "find_vcvars.ps1")
New-Item -ItemType Directory -Force $out | Out-Null
# (PATH is extended here, not inside the cmd line: %PATH% there would expand before vcvars runs)
$env:PATH = "$ninja;" + $env:PATH
$configure = "`"$vcvars`" >nul && `"$cmake`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$ninja\ninja.exe`" -DCMAKE_BUILD_TYPE=$config " +
  "-DCMAKE_INSTALL_PREFIX=`"$install`" -DFILAMENT_SUPPORTS_VULKAN=ON -DFILAMENT_SUPPORTS_OPENGL=OFF " +
  "-DFILAMENT_SKIP_SAMPLES=ON -DFILAMENT_ENABLE_MATDBG=OFF -DFILAMENT_ENABLE_FGVIEWER=OFF `"$src`""
Push-Location $out
cmd /c $configure
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "cmake configure failed" }
$t0 = Get-Date
# (parallel jobs: MSVC takes up to a GB per job on Filament's bigger units; DLSS5_BUILD_JOBS overrides)
$jobs = if ($env:DLSS5_BUILD_JOBS) { $env:DLSS5_BUILD_JOBS } else { 8 }
cmd /c "`"$vcvars`" >nul && `"$cmake`" --build . --target install --parallel $jobs"
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { throw "filament build failed" }
Write-Host ("Filament installed to $install in " + [int]((Get-Date) - $t0).TotalMinutes + " min")
