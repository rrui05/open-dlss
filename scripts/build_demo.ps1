# Build the DLSS 5 NR demo: the demo shaders (glslang -> build\data\shaders), the demo materials (Filament's matc
# -> build\data\materials), then demo\CMakeLists.txt with CMake + Ninja + MSVC into build\demo-build, the executable
# landing in build\demo\dlss5-demo.exe. Needs scripts\fetch_tools.ps1, scripts\fetch_filament.ps1 and
# scripts\build_filament.ps1 (third_party\filament-install) first, and scripts\build.ps1 for the NR kernels.
# Run the result from anywhere: it addresses build\data, build\shaders and build\scenes relative to itself.
#   powershell -File scripts\build_demo.ps1 [-Debug]
param([switch]$Debug)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$install = Join-Path $root "third_party\filament-install"
$glslang = Join-Path $root "tools\glslang\bin\glslang.exe"
$matc = Join-Path $install "bin\matc.exe"
$cmake = Join-Path $root "tools\cmake\bin\cmake.exe"
$ninja = Join-Path $root "tools\ninja"
if (-not (Test-Path $matc)) { throw "$matc not found: run scripts\fetch_filament.ps1 and scripts\build_filament.ps1 first" }
$vcvars = & (Join-Path $PSScriptRoot "find_vcvars.ps1")
$data = Join-Path $root "build\data"

# demo shaders (the NR preprocess / composite, the motion vector unpack)
New-Item -ItemType Directory -Force (Join-Path $data "shaders") | Out-Null
Get-ChildItem (Join-Path $root "demo\shaders") -Include *.comp -Recurse | ForEach-Object {
  $target = Join-Path $data ("shaders\" + $_.Name + ".spv")
  & $glslang -V --target-env vulkan1.3 -I"$root\demo\shaders" $_.FullName -o $target
  if ($LASTEXITCODE -ne 0) { throw "shader compilation failed: $($_.Name)" }
}
# demo materials (Filament packages for the Vulkan backend)
New-Item -ItemType Directory -Force (Join-Path $data "materials") | Out-Null
Get-ChildItem (Join-Path $root "demo\materials\*.mat") | ForEach-Object {
  $target = Join-Path $data ("materials\" + $_.BaseName + ".filamat")
  & $matc -a vulkan -p desktop -o $target $_.FullName
  if ($LASTEXITCODE -ne 0) { throw "material compilation failed: $($_.Name)" }
}
Write-Host "demo shaders and materials -> $data"

# the executable
$config = if ($Debug) { "Debug" } else { "Release" }
$out = Join-Path $root "build\demo-build"
New-Item -ItemType Directory -Force $out | Out-Null
New-Item -ItemType Directory -Force (Join-Path $root "build\demo") | Out-Null
$env:PATH = "$ninja;" + $env:PATH
$configure = "`"$vcvars`" >nul && `"$cmake`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$ninja\ninja.exe`" -DCMAKE_BUILD_TYPE=$config `"$root\demo`""
Push-Location $out
cmd /c $configure
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "cmake configure failed" }
$jobs = if ($env:DLSS5_BUILD_JOBS) { $env:DLSS5_BUILD_JOBS } else { 8 }
cmd /c "`"$vcvars`" >nul && `"$cmake`" --build . --parallel $jobs"
$code = $LASTEXITCODE
Pop-Location
if ($code -ne 0) { throw "demo build failed" }
Write-Host "built $root\build\demo\dlss5-demo.exe"
