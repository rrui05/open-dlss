[CmdletBinding()]
param(
    [string]$RepositoryRoot = '',
    [string]$ToolchainRoot = '',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
if (-not $RepositoryRoot) { $RepositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..')) }
if (-not $ToolchainRoot) { $ToolchainRoot = Join-Path $RepositoryRoot 'tools' }
$benchmarkRoot = Split-Path -Parent $PSScriptRoot
$generatedRoot = Join-Path $RepositoryRoot 'build/benchmark_native'
& $Python (Join-Path $benchmarkRoot 'prepare_native_timed_main.py') --repository-root $RepositoryRoot --output $generatedRoot
if ($LASTEXITCODE -ne 0) { throw "Timing entry generation failed: $LASTEXITCODE" }
$vcvars = & (Join-Path $RepositoryRoot 'scripts/find_vcvars.ps1')
$sources = @((Get-ChildItem -LiteralPath (Join-Path $RepositoryRoot 'src') -Filter '*.cpp' |
    Where-Object Name -NE 'main.cpp' | ForEach-Object { '"' + $_.FullName + '"' }))
$sources += '"' + (Join-Path $generatedRoot 'main_timed.cpp') + '"'
$volk = '"' + (Join-Path $ToolchainRoot 'volk/volk.c') + '"'
$include = '/I"' + (Join-Path $ToolchainRoot 'Vulkan-Headers/include') + '" /I"' +
    (Join-Path $ToolchainRoot 'volk') + '" /I"' + (Join-Path $RepositoryRoot 'src') + '"'
$objectRoot = Join-Path $generatedRoot 'obj'
New-Item -ItemType Directory -Force -Path $objectRoot | Out-Null
$outputExe = Join-Path $RepositoryRoot 'build/dlss5vk_timed.exe'
$sourceArguments = $sources -join ' '
$compileCommand = "`"$vcvars`" >nul && cl /nologo /std:c++20 /EHsc /W3 /O2 /DNOMINMAX /DWIN32_LEAN_AND_MEAN /D_CRT_SECURE_NO_WARNINGS /DVK_ENABLE_BETA_EXTENSIONS $include /Fo`"$objectRoot\\`" $sourceArguments $volk /Fe:`"$outputExe`" /link /SUBSYSTEM:CONSOLE"
cmd /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "Timed native compilation failed: $LASTEXITCODE" }
Write-Host "built $outputExe"
Write-Host 'Shaders/PTX are reused from the normal scripts/build.ps1 build.'
