$ErrorActionPreference = 'Stop'
$nativeRoot = 'D:\newDLSS\open-dlss'
$timedMain = Join-Path $PSScriptRoot 'main_timed.cpp'
$vcvars = & (Join-Path $nativeRoot 'scripts\find_vcvars.ps1')
$sources = @((Get-ChildItem -LiteralPath (Join-Path $nativeRoot 'src') -Filter '*.cpp' |
    Where-Object Name -NE 'main.cpp' | ForEach-Object { '"' + $_.FullName + '"' }))
$sources += '"' + $timedMain + '"'
$sourceArguments = $sources -join ' '
$volk = '"' + (Join-Path $nativeRoot 'tools\volk\volk.c') + '"'
$include = '/I"' + (Join-Path $nativeRoot 'tools\Vulkan-Headers\include') + '" /I"' +
    (Join-Path $nativeRoot 'tools\volk') + '" /I"' + (Join-Path $nativeRoot 'src') + '"'
$objectRoot = Join-Path $nativeRoot 'build\obj_timed'
New-Item -ItemType Directory -Force -Path $objectRoot | Out-Null
$outputExe = Join-Path $nativeRoot 'build\dlss5vk_timed.exe'
$compileCommand = "`"$vcvars`" >nul && cl /nologo /std:c++20 /EHsc /W3 /O2 /DNOMINMAX /DWIN32_LEAN_AND_MEAN /D_CRT_SECURE_NO_WARNINGS /DVK_ENABLE_BETA_EXTENSIONS $include /Fo`"$objectRoot\\`" $sourceArguments $volk /Fe:`"$outputExe`" /link /SUBSYSTEM:CONSOLE"
cmd /c $compileCommand
if ($LASTEXITCODE -ne 0) { throw "Timed native compilation failed: $LASTEXITCODE" }
Write-Host "built $outputExe"
