[CmdletBinding()]
param(
    [string]$InputPng = '',
    [string]$Model = '',
    [string]$Output = '',
    [string]$Executable = '',
    [string]$Shaders = '',
    [string]$PtxDirectory = '',
    [string]$FeaturesFile = '',
    [string]$Reference = '',
    [ValidateRange(33, 65536)][int]$Width = 896,
    [ValidateRange(33, 65536)][int]$Height = 512,
    [ValidateRange(0, 10000)][int]$Warmups = 5,
    [ValidateRange(1, 100000)][int]$Iterations = 20,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
if (-not $InputPng) { $InputPng = Join-Path $PSScriptRoot 'inputs/street_896x512.png' }
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not $Model) { $Model = Join-Path $repositoryRoot 'models/nr' }
if (-not $Output) { $Output = Join-Path $repositoryRoot 'build/benchmark_native/run' }
if (-not $Executable) { $Executable = Join-Path $repositoryRoot 'build/dlss5vk_timed.exe' }
if (-not $Shaders) { $Shaders = Join-Path $repositoryRoot 'build/shaders' }
if (-not $PtxDirectory) { $PtxDirectory = Join-Path $repositoryRoot 'build/ptx' }
$parameters = @(
    (Join-Path $PSScriptRoot 'native_timed/run_native.py'),
    '--input', $InputPng, '--model', $Model, '--output', $Output,
    '--executable', $Executable, '--shaders', $Shaders, '--ptx-directory', $PtxDirectory,
    '--width', $Width, '--height', $Height, '--warmups', $Warmups, '--iterations', $Iterations
)
if ($FeaturesFile) { $parameters += @('--features-file', $FeaturesFile) }
if ($Reference) { $parameters += @('--reference', $Reference) }
& $Python @parameters
if ($LASTEXITCODE -ne 0) { throw "Native benchmark or collection failed: $LASTEXITCODE" }
