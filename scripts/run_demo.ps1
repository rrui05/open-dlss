# Run the demo: scripts\run_demo.ps1 [-Model <nr model dir>] [-Scene <scene.gltf|.glb> [-Environment <.hdr>]] [extra args]
# Without -Scene the demo starts on the first directory under build\scenes that holds a view.json.
param([string]$Model = "", [string]$Scene = "", [string]$Environment = "",
      [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra)
$root = Split-Path -Parent $PSScriptRoot
$demoArgs = @()
if ($Scene -ne "") { $demoArgs += $Scene; if ($Environment -ne "") { $demoArgs += $Environment } }
if ($Model -ne "") { $demoArgs += @("--model", $Model) }
& (Join-Path $root "build\demo\dlss5-demo.exe") @demoArgs @Extra
