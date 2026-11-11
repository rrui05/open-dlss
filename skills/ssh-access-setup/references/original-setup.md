# 原机器的方法与迁移差异

调查时间：2026-11-11。本文件是只读调查记录，不是远程登录验收。
发布记录不包含真实客户端地址、公钥、完整 SSH 客户端配置。

## 原机器的连接路径

`局域网客户端 → Windows 局域网 IPv4:22 → WSL mirrored 网络 → Ubuntu sshd → 普通用户`

- Windows 11，构建号 26300。
- WSL2，WSL 版本 3.0.5。
- Ubuntu 24.04；`wslinfo --networking-mode` 实际返回 `mirrored`。
- `/etc/wsl.conf` 使用 `[boot] systemd=true`。
- `/etc/ssh/sshd_config.d/00-lan-access.conf` 指定 IPv4、`0.0.0.0:22`、非 root 用户和单个客户端地址。
- 原配置为 `PubkeyAuthentication yes`、`PasswordAuthentication yes`、`KbdInteractiveAuthentication no`。
- `systemctl is-enabled ssh` 返回 `enabled`；`systemctl is-active ssh` 返回 `active`。
- `ssh.socket` 为 disabled/inactive。
- Windows 没有原生 sshd 服务；`netsh interface portproxy show all` 没有映射。

## 原有代码位置

路径使用 `%USERPROFILE%` 替代具体用户目录。原文件留在原机器；本 skill 提供脱敏参数化版本。

| 原机器文件 | 用途 | 本 skill 对应文件 |
| --- | --- | --- |
| `%USERPROFILE%\wsl-ssh-access-20261027\enable-wsl-ssh-firewall.ps1` | 创建/核对两层规则 | `scripts/set-wsl-ssh-firewall.ps1` |
| `%USERPROFILE%\wsl-ssh-access-20261027\disable-wsl-ssh-firewall.ps1` | 撤销同组规则 | 同一脚本加 `-Remove` |
| `%USERPROFILE%\wsl-ssh-access-20261027\firewall-result.json` | 本地执行记录，不发布 | 同一脚本可指定 `-ResultPath` |
| `/etc/ssh/sshd_config.d/00-lan-access.conf` | sshd 配置 | `scripts/setup-linux-sshd.sh` 生成独立 drop-in |
| `%USERPROFILE%\.wslconfig` | WSL 全局网络模式 | `references/wsl-setup.md` 合并步骤 |
| `/etc/wsl.conf` | Ubuntu systemd | 同上 |

原实现的关键是 Hyper-V 继承：Windows 规则可同名出现在 Hyper-V ActiveStore，`VMCreatorId` 显示 `Any`。当来源为 `HostFirewallLocal` 且 `AllowHostPolicyMerge=True` 时，这是原机器的正常状态。参数化脚本保留此判断。

## 新机器的差异

- 新机器默认只用公钥。原机器同时允许密码，不能混称为同一种认证设置。
- 用户、目标地址、源地址、公钥、发行版名重新获取，不复制原常量。
- 新配置名为 `00-ssh-access-setup.conf`，不覆盖原机器的 `00-lan-access.conf`。
- 保留其他公钥及其限制。新机器不复制原机器的主机私钥；连接前核对新机器自己的主机指纹。
- 未发现自动启动 WSL 的计划任务或启动目录脚本。`ssh.service enabled` 不能证明 Windows 重启后可无人值守连接。

## 证据边界

调查包括配置读取、实际网络模式、服务状态、监听端口、Windows/Hyper-V 继承检查。原有两份 PowerShell 脚本通过 `Parser::ParseFile` 语法检查。

调查未执行端到端 SSH 登录。读取 drop-in 不等于验证全部有效配置；原机器的 `sudo -n sshd -T` 没有取得权限。新电脑必须完成 `sshd -t/-T` 和来自现有电脑的登录验收。
