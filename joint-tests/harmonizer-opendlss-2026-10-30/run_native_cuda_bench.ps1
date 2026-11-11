[CmdletBinding()]
param(
    [ValidateRange(0, 10000)][int]$Warmups = 5,
    [ValidateRange(1, 100000)][int]$Iterations = 20
)

$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $experimentRoot)
$nativeExe = Join-Path $workspaceRoot 'open-dlss/build/dlss5vk_timed.exe'
$compatibleModel = Join-Path $experimentRoot 'native_model_compat'
$inputFeatures = Join-Path $experimentRoot 'native_dump/features.f32'
foreach ($requiredFile in @($nativeExe, $inputFeatures, (Join-Path $compatibleModel 'manifest.json'))) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required benchmark artifact is missing: $requiredFile"
    }
}
Write-Host 'Backend: original OpenDLSS Vulkan scheduling with CUDA/PTX kernels.'
Write-Host 'Input: the shared 896x512 image, preprocessed into resident 16-lane features.'
& $nativeExe bench --model $compatibleModel --width 896 --height 512 `
    --features-file $inputFeatures --warmups $Warmups --frames $Iterations
if ($LASTEXITCODE -ne 0) {
    throw "The native benchmark failed with exit code $LASTEXITCODE."
}
