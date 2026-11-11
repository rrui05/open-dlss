---
name: ssh-access-setup
description: 在新的 Windows 电脑上通过 WSL Ubuntu 开放 SSH，让现有电脑直接连接。复用 mirrored 网络、OpenSSH、按客户端地址放行的 Windows/Hyper-V 防火墙和密钥登录方法；适用于部署、迁移和排查此连接，不用于 Windows 原生 sshd。
---

# WSL Ubuntu SSH 连接配置

目标：从现有电脑执行 `ssh new-pc`，直接进入新电脑的 WSL Ubuntu。
部署脚本在新电脑运行；连接检查在现有电脑运行。不要把本机误当成待部署的新电脑。

## 读取与选择

- 默认采用 Windows 11 + WSL2 Ubuntu + mirrored 网络，这是原机器实际使用的方法。
- 部署时读取 [references/wsl-setup.md](references/wsl-setup.md)，包含安装、网络、服务、防火墙、撤销步骤。
- 配置现有电脑并验收时读取 [references/client-and-verification.md](references/client-and-verification.md)。
- 用户询问原机器的方法或迁移差异时读取 [references/original-setup.md](references/original-setup.md)。这是脱敏调查记录，不是新机器的默认参数。
- 目标是原生 Ubuntu/Debian 时，可以只复用 Linux 脚本；Windows 与 Hyper-V 步骤不适用。目标是 Windows 命令行时，本 skill 不适用。

## 部署参数

从目标机器的本地控制台和现有电脑查出以下值。上下文已有的值不要再问。

| 参数 | 含义 | 获取位置 |
| --- | --- | --- |
| 发行版名称 | `wsl -l -v` 的实际名称 | 新电脑 Windows |
| 登录用户 | WSL 内现有非 root 用户 | 新电脑 WSL：`id -un` |
| 目标 IPv4 | 供局域网访问的 Windows IPv4 | 新电脑：`Get-NetIPConfiguration` |
| 客户端 IPv4 | 目标实际看到的连接源地址 | 现有电脑：`ip route get TARGET_IP` 的 `src`，结合目标日志核对 |
| 客户端公钥 | 专用 SSH 密钥对应的 `.pub` 文件 | 现有电脑 |
| 端口 | 默认 `22`，有冲突时明确选择其他端口 | 新电脑服务检查 |

只将客户端公钥复制到新电脑。私钥留在现有电脑。不得发布真实公钥、私钥、密码、`authorized_keys`、`known_hosts`、完整 SSH 配置或运行结果 JSON。

## 执行顺序

1. 确认网络可达。默认面向同一局域网，或已经建立路由的网络。互不连通时先解决路由或既有 VPN；仅打开端口不会解决公网 NAT。不要自动添加路由器公网映射或第三方中继。
2. 在新电脑核对 WSL2、systemd、实际 mirrored 模式和端口占用。只合并相关配置，不覆盖整个 `.wslconfig` 或 `/etc/wsl.conf`。`wsl --shutdown` 会终止全部 WSL 发行版中的工作，在新电脑无运行工作时应用。
3. 在现有电脑选择或生成独立密钥。用本地控制台、可移动介质或已有可信通道传递 `.pub`。首次部署不能依赖尚未开通的 SSH。
4. 在新电脑 WSL 运行 `scripts/setup-linux-sshd.sh` 预览；核对用户、源地址、端口和公钥指纹后，以 `sudo` 加 `--apply` 应用。脚本默认禁用密码认证，与原机器允许密码的配置不同。
5. 在新电脑管理员 PowerShell 运行 `scripts/set-wsl-ssh-firewall.ps1`。先 `-WhatIf`，再应用同一客户端地址与端口。只放行该连接，不关闭整个防火墙。
6. 在现有电脑核对主机指纹，测试全新密钥连接。成功后才配置 `Host new-pc` 别名。
7. 区分临时在线、WSL 重启后在线、Windows 重启后在线。`systemctl enable ssh` 只管理 WSL 内服务，不能证明 Windows 启动后 WSL 自动运行。

若用户只请求整理和上传 skill，不执行任何服务、网络或防火墙变更。将来用户请求在新电脑开放 SSH 时，执行该请求包含的部署动作，无须为常规步骤重复询问。

## 脚本约定

`setup-linux-sshd.sh` 默认只预览，`--apply` 才变更。应用时必须传入 `--target-ip`，以检查真实目标地址下的 `Match` 配置。它安装服务、追加公钥、备份其管理的配置和 `authorized_keys`，用 `sshd -t/-T` 检查配置。配置检查失败时恢复这两个文件；服务启动失败返回非零。它不会修改 Windows 网络或防火墙。配置只允许一个用户从一个客户端 IPv4 或本机回环地址登录。

`set-wsl-ssh-firewall.ps1` 支持 `-ClientAddress`、`-Port`、`-WhatIf`、`-Remove` 和可选 `-ResultPath`。它管理 `Codex-WSL-SSH-<IP>-<PORT>` 同名规则，拒绝覆盖参数不同的同名规则。Hyper-V 正确继承 Windows 规则时不重复创建。输出仅证明这两条规则的配置；其他规则仍须单独检查。

## 验收与汇报

分别报告文件检查、WSL 内服务与监听、两层防火墙生效情况、来自现有电脑的全新密钥登录结果。登录检查必须执行远端 `id -un; hostname`。用户要求无人值守时，另测 Windows 重启、用户登录、WSL 生命周期和睡眠后的连接。

预览、语法检查、`active` 状态或防火墙 JSON 都不能代替远程登录。没有目标机器或没有执行登录时写“未验证”。不得用 `StrictHostKeyChecking=no` 隐藏主机身份问题，不得用 `|| true` 掩盖部署失败。
