# 在新电脑部署 WSL Ubuntu SSH

此流程使用 Windows 11 22H2 或更新版本、支持 mirrored 的 WSL2、Ubuntu 和 IPv4。
在新电脑本地执行部署。保持本地控制台可用，直到现有电脑登录成功。

## 1. 获取 skill

在新电脑 Ubuntu 终端执行。Git 稀疏检出避免下载模型和视频：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/rrui05/open-dlss.git open-dlss-skills
git -C open-dlss-skills sparse-checkout set skills/ssh-access-setup
mkdir -p ~/.codex/skills
cp -a open-dlss-skills/skills/ssh-access-setup ~/.codex/skills/
cd ~/.codex/skills/ssh-access-setup
```

已有同名 skill 时先比较，不覆盖本地修改。没有 Git 时，可以从浏览器取得仓库的 `skills/ssh-access-setup`。脚本不依赖 Codex，也可以直接在下载目录运行。

## 2. 检查 Windows 与 WSL

新电脑 Windows PowerShell：

```powershell
Get-ComputerInfo | Select-Object WindowsProductName, OsVersion, OsBuildNumber
wsl --version
wsl -l -v
Get-NetIPConfiguration
Get-Service sshd -ErrorAction SilentlyContinue
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object LocalPort -eq 22
netsh interface portproxy show all
```

没有 WSL/Ubuntu 时，从管理员 PowerShell 执行 `wsl --install -d Ubuntu`。按安装提示重启并创建 Ubuntu 普通用户。已有 WSL 时不重装。不支持 mirrored 时先解决 WSL 版本，或另行采用 NAT；不要将 mirrored 脚本直接当作 NAT 转发脚本。

Windows 原生 sshd、其他发行版或已有转发占用端口时，查清所属用途。可以为新 WSL SSH 选择 `2222`，并在 Linux、防火墙和客户端使用同一端口。不要停止用户未要求停止的服务。

## 3. 合并网络与 systemd 配置

备份新电脑 `%USERPROFILE%\.wslconfig`（若存在），在 `[wsl2]` 节设置：

```ini
[wsl2]
networkingMode=mirrored
firewall=true
```

备份新电脑 Ubuntu `/etc/wsl.conf`（若存在），在 `[boot]` 节设置：

```ini
[boot]
systemd=true
```

保留现有内存、挂载、代理、默认用户设置。不要重复同名节。
在新电脑没有运行工作时，从 Windows PowerShell 执行：

```powershell
wsl --shutdown
wsl -d Ubuntu
```

将 `Ubuntu` 替换为实际发行版名。重新进入新电脑 Ubuntu：

```bash
wslinfo --networking-mode
ps -p 1 -o comm=
id -un
```

结果必须分别为 `mirrored`、`systemd` 和目标普通用户。没有 `wslinfo` 时先解决工具/版本问题，不能只凭配置文件声称模式生效。

## 4. 部署密钥与 sshd

按照 [client-and-verification.md](client-and-verification.md) 在现有电脑准备公钥并确认源地址。
将一个 `.pub` 文件复制到新电脑 Ubuntu，可以通过本地控制台从 Windows 路径复制：

```bash
cp /mnt/c/Users/WINDOWS_USER/Downloads/new_pc_ed25519.pub /tmp/new_pc_ed25519.pub
ssh-keygen -lf /tmp/new_pc_ed25519.pub
```

以下地址为文档示例，不能原样部署。将 `192.0.2.10` 替换为**现有电脑**的实际源 IPv4，将 `192.0.2.20` 替换为新电脑的目标 IPv4。将 `ubuntuuser` 替换为新电脑 Ubuntu 用户。两个地址不能互换。

```bash
cd ~/.codex/skills/ssh-access-setup
bash scripts/setup-linux-sshd.sh \
  --user ubuntuuser --client-ip 192.0.2.10 --target-ip 192.0.2.20 \
  --public-key /tmp/new_pc_ed25519.pub --port 22
sudo bash scripts/setup-linux-sshd.sh \
  --user ubuntuuser --client-ip 192.0.2.10 --target-ip 192.0.2.20 \
  --public-key /tmp/new_pc_ed25519.pub --port 22 --apply
```

预览也检查真实账户和公钥。缺少 `ssh-keygen` 时，在目标 Ubuntu 安装 `openssh-client`。应用时脚本自动安装缺少的 `openssh-server`。

脚本写 `/etc/ssh/sshd_config.d/00-ssh-access-setup.conf`，追加目标账户 `~/.ssh/authorized_keys`。它禁用密码和 root 登录，限制源地址。它将 `ssh.socket` 改为 `ssh.service` 管理，以保证监听跟随 sshd 配置。存在其他 SSH 用途时，先检查这项变化是否符合迁移范围。

应用时必须提供 `--target-ip`，以检查实际目标地址下生效的 `Match` 配置。配置冲突时脚本拒绝继续，并恢复其改动的配置和公钥文件。按输出检查既有 `Include`、`Port`、`ListenAddress`、`AllowUsers`、`Match`，只修改授权范围。软件包安装和目录权限变化不属于文件恢复范围。

在新电脑 Ubuntu 检查：

```bash
sudo /usr/sbin/sshd -t
systemctl is-enabled ssh.service
systemctl is-active ssh.service
sudo ss -ltnp 'sport = :22'
sudo ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
sudo ufw status verbose
```

UFW 为 active 时，添加对应来源和端口的规则，不关闭 UFW：

```bash
sudo ufw allow proto tcp from 192.0.2.10 to any port 22
```

没有 UFW 时检查已有 nftables/iptables 策略，不安装额外防火墙来替代未知策略。

## 5. 放行 Windows 与 Hyper-V

在新电脑管理员 PowerShell 中切换到 skill 的 Windows 可访问路径。可以使用 WSL 文件共享；路径中的发行版和用户必须替换：

```powershell
$SkillDir = '\\wsl.localhost\Ubuntu\home\ubuntuuser\.codex\skills\ssh-access-setup'
Set-Location -LiteralPath $SkillDir
.\scripts\set-wsl-ssh-firewall.ps1 -ClientAddress '192.0.2.10' -Port 22 -WhatIf
.\scripts\set-wsl-ssh-firewall.ps1 -ClientAddress '192.0.2.10' -Port 22
```

执行策略阻止本地审阅过的脚本时，只对本次 PowerShell 进程设置：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

组织策略禁止执行时遵守该策略，不修改机器级策略。
需要本地证据时，传入可写路径：

```powershell
.\scripts\set-wsl-ssh-firewall.ps1 -ClientAddress '192.0.2.10' -Port 22 `
  -ResultPath "$env:TEMP\wsl-ssh-firewall-result.json"
```

脚本检查两层 ActiveStore。Windows 规则继承到 Hyper-V 时，`VMCreatorId=Any` 可以是正确状态；脚本同时检查 `HostFirewallLocal` 和 `AllowHostPolicyMerge`，不重复创建规则。

不将来源改成 `Any`，不设置全局 Hyper-V `DefaultInboundAction Allow`，不关闭 Windows 防火墙。其他已有规则可以另外放行流量，不能凭新增规则宣称整机只接受此客户端。

## 6. 生命周期与撤销

运行 [client-and-verification.md](client-and-verification.md) 的实际连接检查。成功后记录发行版、目标地址、源地址、端口、用户、公钥指纹和主机指纹，私钥不得入记录。

防火墙规则持久保存。`ssh.service` 随 Ubuntu 的 systemd 启动；这不保证 Windows 启动时自动启动 WSL，也不保证 WSL 无人使用时持续在线。首次部署保持新电脑 WSL 终端运行。用户要求无人值守时，另行配置使用该发行版所属 Windows 用户的启动任务/长驻 WSL 进程，实测退出终端和 Windows 重启后的连接。不要用看不到该发行版的 SYSTEM 身份任务，也不要把仅执行一次 `wsl ... true` 当作持续在线保证。睡眠、关机、网络断开时无法连接；电源设置只按用户要求修改。

删除本 skill 创建的单组防火墙规则：

```powershell
.\scripts\set-wsl-ssh-firewall.ps1 -ClientAddress '192.0.2.10' -Port 22 -Remove -WhatIf
.\scripts\set-wsl-ssh-firewall.ps1 -ClientAddress '192.0.2.10' -Port 22 -Remove
```

Linux 脚本每次应用时打印 `/var/backups/ssh-access-setup.XXXXXXXX`。恢复时从**本次应用前**的备份还原配置/公钥；原来不存在配置文件时删除本 skill 的 `00-ssh-access-setup.conf`。先执行 `sudo /usr/sbin/sshd -t`，再重启服务。不要用旧备份覆盖之后新增的公钥。UFW 规则、服务/socket 状态和 WSL 配置按部署前状态单独恢复；删除 Windows 规则不会自动恢复它们。

## 官方资料

- [Microsoft：WSL mirrored 与 Hyper-V 防火墙](https://learn.microsoft.com/en-us/windows/wsl/networking)
- [Microsoft：WSL 的 systemd](https://learn.microsoft.com/en-us/windows/wsl/systemd)
- [Microsoft：`.wslconfig` 与 `wsl.conf`](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
- [Ubuntu：OpenSSH 服务](https://ubuntu.com/server/docs/how-to/security/openssh-server/)
- [OpenSSH：配置项与优先级](https://man.openbsd.org/sshd_config)
