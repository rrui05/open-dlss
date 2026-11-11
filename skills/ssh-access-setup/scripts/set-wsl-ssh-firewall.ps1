[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [Parameter(Mandatory = $true)]
    [string]$ClientAddress,
    [ValidateRange(1, 65535)]
    [int]$Port = 22,
    [switch]$Remove,
    [string]$ResultPath
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$VmCreatorId = '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'
$RuleName = "Codex-WSL-SSH-$ClientAddress-$Port"
$Changes = @()

function Assert-Administrator {
    $principal = New-Object Security.Principal.WindowsPrincipal(
        [Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run this script in an administrator PowerShell window. -WhatIf does not require elevation.'
    }
}

function Get-OptionalRule {
    param([string]$Command, [string]$Store)
    $lookupErrors = @()
    $rules = @(& $Command -PolicyStore $Store -Name $RuleName `
        -ErrorAction SilentlyContinue -ErrorVariable lookupErrors)
    foreach ($lookupError in $lookupErrors) {
        if ($lookupError.CategoryInfo.Category -ne 'ObjectNotFound') {
            throw $lookupError
        }
    }
    if ($rules.Count -gt 1) {
        throw "Multiple rules named $RuleName were found; no matching rule was changed."
    }
    if ($rules.Count -eq 1) { return $rules[0] }
    return $null
}

function Assert-WindowsRule {
    param($Rule)
    if ($null -eq $Rule) { return }
    $address = $Rule | Get-NetFirewallAddressFilter
    $portFilter = $Rule | Get-NetFirewallPortFilter
    if ([string]$Rule.Direction -ne 'Inbound' -or [string]$Rule.Action -ne 'Allow' -or
        [string]$Rule.Profile -ne 'Any' -or
        [string]$portFilter.Protocol -notin @('TCP', '6') -or
        [string]$portFilter.LocalPort -ne [string]$Port -or
        [string]$portFilter.RemotePort -ne 'Any' -or
        [string]$address.LocalAddress -ne 'Any' -or
        [string]$address.RemoteAddress -ne $ClientAddress) {
        throw "Windows rule $RuleName has different settings; it was preserved."
    }
}

function Test-WslRuleScope {
    param($Rule, $VmSettings)
    if ([string]$Rule.VMCreatorId -eq 'Any') {
        # Windows IP/port rules can be inherited by Hyper-V with creator Any.
        return ([string]$Rule.PolicyStoreSourceType -eq 'HostFirewallLocal' -and
                [string]$VmSettings.AllowHostPolicyMerge -eq 'True')
    }
    $actualId = [guid]::Empty
    return ([guid]::TryParse([string]$Rule.VMCreatorId, [ref]$actualId) -and
            $actualId -eq [guid]$VmCreatorId)
}

function Assert-HyperVRule {
    param($Rule, $VmSettings)
    if ($null -eq $Rule) { return }
    if ([string]$Rule.Direction -ne 'Inbound' -or [string]$Rule.Action -ne 'Allow' -or
        [string]$Rule.Protocol -notin @('TCP', '6') -or
        [string]$Rule.LocalPorts -ne [string]$Port -or
        [string]$Rule.RemotePorts -ne 'Any' -or
        [string]$Rule.LocalAddresses -ne 'Any' -or
        [string]$Rule.RemoteAddresses -ne $ClientAddress -or
        -not (Test-WslRuleScope -Rule $Rule -VmSettings $VmSettings)) {
        throw "Hyper-V rule $RuleName has different settings or VM scope; it was preserved."
    }
}

function Write-Report {
    param($Report, [bool]$Persist = $false)
    $json = $Report | ConvertTo-Json -Depth 6
    if ($Persist -and $ResultPath) {
        # Do not silently create directories or write reports during -WhatIf.
        $json | Set-Content -LiteralPath $ResultPath -Encoding UTF8
    }
    Write-Output $json
}

try {
    $parsedAddress = $null
    if ($ClientAddress -notmatch '^\d{1,3}(\.\d{1,3}){3}$' -or
        -not [Net.IPAddress]::TryParse($ClientAddress, [ref]$parsedAddress) -or
        $parsedAddress.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork -or
        $parsedAddress.ToString() -ne $ClientAddress -or
        $parsedAddress.GetAddressBytes()[0] -in @(0, 127) -or
        $parsedAddress.GetAddressBytes()[0] -ge 224) {
        throw 'ClientAddress must be one canonical remote unicast IPv4 address, not a wildcard, loopback, CIDR, multicast or broadcast address.'
    }
    $commands = @(
        'Get-NetFirewallRule', 'Get-NetFirewallAddressFilter', 'Get-NetFirewallPortFilter',
        'New-NetFirewallRule', 'Set-NetFirewallRule', 'Remove-NetFirewallRule',
        'Get-NetFirewallHyperVVMSetting', 'Get-NetFirewallHyperVRule',
        'New-NetFirewallHyperVRule', 'Set-NetFirewallHyperVRule', 'Remove-NetFirewallHyperVRule'
    )
    foreach ($command in $commands) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "Required Windows cmdlet is unavailable: $command. Use a supported Windows 11 WSL host."
        }
    }
    if (-not $WhatIfPreference) { Assert-Administrator }

    $vmSettings = Get-NetFirewallHyperVVMSetting -PolicyStore ActiveStore -Name $VmCreatorId
    $windowsRule = Get-OptionalRule -Command 'Get-NetFirewallRule' -Store 'PersistentStore'
    $activeWindowsRule = Get-OptionalRule -Command 'Get-NetFirewallRule' -Store 'ActiveStore'
    $hyperVRule = Get-OptionalRule -Command 'Get-NetFirewallHyperVRule' -Store 'ActiveStore'
    # Validate both existing rules before any mutation, including removal.
    Assert-WindowsRule -Rule $windowsRule
    Assert-WindowsRule -Rule $activeWindowsRule
    Assert-HyperVRule -Rule $hyperVRule -VmSettings $vmSettings
    if ($activeWindowsRule -and -not $windowsRule) {
        throw "Windows rule $RuleName exists only in external policy; it was preserved."
    }
    $action = if ($Remove) { 'Remove the matching Windows and WSL Hyper-V firewall rules' }
              else { 'Allow this client through the Windows and WSL Hyper-V firewall rules' }
    if (-not $PSCmdlet.ShouldProcess("$RuleName (TCP $Port from $ClientAddress)", $action)) {
        Write-Report -Report ([ordered]@{
            SchemaVersion = 1; Status = 'planned'; Scope = 'firewall-only'
            RuleName = $RuleName; ClientAddress = $ClientAddress; Port = $Port
            Operation = $(if ($Remove) { 'remove' } else { 'configure' })
            RemoteLoginVerified = $false; Changes = @()
        })
        return
    }

    if ($Remove) {
        if ($windowsRule) {
            Remove-NetFirewallRule -PolicyStore PersistentStore -Name $RuleName -Confirm:$false
            $Changes += 'Removed Windows rule'
        }
        if ($hyperVRule -and [string]$hyperVRule.PolicyStoreSourceType -ne 'HostFirewallLocal') {
            Remove-NetFirewallHyperVRule -PolicyStore PersistentStore -Name $RuleName -Confirm:$false
            $Changes += 'Removed explicit Hyper-V rule'
        }
        $remainingWindowsRule = Get-OptionalRule -Command 'Get-NetFirewallRule' -Store 'ActiveStore'
        $remainingHyperVRule = Get-OptionalRule -Command 'Get-NetFirewallHyperVRule' -Store 'ActiveStore'
        if ($remainingWindowsRule -or $remainingHyperVRule) {
            throw 'A matching firewall rule is still active. Inspect policy inheritance; removal was not verified.'
        }
        $status = 'firewall-removed'
    } else {
        if ($windowsRule) {
            if ([string]$windowsRule.Enabled -ne 'True') {
                Set-NetFirewallRule -PolicyStore PersistentStore -Name $RuleName -Enabled True -Confirm:$false
                $Changes += 'Enabled Windows rule'
            }
        } else {
            New-NetFirewallRule -PolicyStore PersistentStore -Name $RuleName `
                -DisplayName "Codex WSL SSH from $ClientAddress TCP $Port" `
                -Direction Inbound -Action Allow -Enabled True -Protocol TCP `
                -LocalPort $Port -RemoteAddress $ClientAddress -Profile Any -Confirm:$false | Out-Null
            $Changes += 'Created Windows rule'
        }
        # A new Windows rule may immediately appear as an inherited Hyper-V rule.
        $hyperVRule = Get-OptionalRule -Command 'Get-NetFirewallHyperVRule' -Store 'ActiveStore'
        Assert-HyperVRule -Rule $hyperVRule -VmSettings $vmSettings
        if ($hyperVRule) {
            if ([string]$hyperVRule.Enabled -ne 'True') {
                if ([string]$hyperVRule.PolicyStoreSourceType -eq 'HostFirewallLocal') {
                    throw 'The inherited Hyper-V rule is disabled although the Windows rule is enabled.'
                }
                Set-NetFirewallHyperVRule -PolicyStore PersistentStore -Name $RuleName `
                    -Enabled True -Confirm:$false
                $Changes += 'Enabled explicit Hyper-V rule'
            }
        } else {
            New-NetFirewallHyperVRule -PolicyStore PersistentStore -Name $RuleName `
                -DisplayName "Codex WSL SSH from $ClientAddress TCP $Port" `
                -Direction Inbound -Action Allow -Enabled True -VMCreatorId $VmCreatorId `
                -Protocol TCP -LocalPorts $Port -RemoteAddresses $ClientAddress -Confirm:$false | Out-Null
            $Changes += 'Created explicit Hyper-V rule'
        }
        $windowsRule = Get-OptionalRule -Command 'Get-NetFirewallRule' -Store 'ActiveStore'
        $hyperVRule = Get-OptionalRule -Command 'Get-NetFirewallHyperVRule' -Store 'ActiveStore'
        Assert-WindowsRule -Rule $windowsRule
        Assert-HyperVRule -Rule $hyperVRule -VmSettings $vmSettings
        if (-not $windowsRule -or -not $hyperVRule -or
            [string]$windowsRule.Enabled -ne 'True' -or [string]$hyperVRule.Enabled -ne 'True') {
            throw 'The expected Windows and Hyper-V firewall rules are not both active.'
        }
        $status = 'firewall-configured'
    }

    Write-Report -Persist $true -Report ([ordered]@{
        SchemaVersion = 1; Status = $status; Scope = 'firewall-only'
        RecordedAt = (Get-Date).ToString('o'); RuleName = $RuleName
        ClientAddress = $ClientAddress; Port = $Port; Changes = @($Changes)
        WslHostPolicyMerge = [string]$vmSettings.AllowHostPolicyMerge
        RemoteLoginVerified = $false
        Note = 'This report verifies only the named firewall rules. Other rules can also allow traffic. Test SSH login from the client separately.'
    })
} catch {
    $failure = [ordered]@{
        SchemaVersion = 1; Status = 'failed'; Scope = 'firewall-only'
        RecordedAt = (Get-Date).ToString('o'); RuleName = $RuleName
        ClientAddress = $ClientAddress; Port = $Port; Changes = @($Changes)
        Error = $_.Exception.Message; RemoteLoginVerified = $false
    }
    # Preserve the original failure even if a requested report cannot be written.
    try { Write-Report -Report $failure -Persist (-not $WhatIfPreference) }
    catch { Write-Output ($failure | ConvertTo-Json -Depth 6) }
    Write-Error $failure.Error -ErrorAction Continue
    exit 1
}
