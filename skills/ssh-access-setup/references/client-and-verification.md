# 现有电脑的配置与连接验收

以下 Bash 命令在**发起连接的现有电脑**执行。默认客户端使用 Linux/WSL OpenSSH。
示例中 `192.0.2.10` 为客户端，`192.0.2.20` 为新电脑。两者都是文档地址，必须替换。

## 准备独立密钥

先检查是否已有供本次连接使用的密钥。不要打印私钥。
没有专用密钥时：

```bash
install -d -m 700 ~/.ssh
if test -e ~/.ssh/new_pc_ed25519 || test -e ~/.ssh/new_pc_ed25519.pub; then
    printf 'Key path already exists; choose or inspect the existing key.\n'
else
    ssh-keygen -t ed25519 -f ~/.ssh/new_pc_ed25519 -C new-pc-ssh
fi
ssh-keygen -lf ~/.ssh/new_pc_ed25519.pub
```

文件已存在时，不执行生成命令，也不覆盖。命令保留口令交互。使用加密私钥时，先用 `ssh-add ~/.ssh/new_pc_ed25519` 解锁到已有 ssh-agent，再做 BatchMode 检查。未运行 agent 时，先在当前终端运行 `eval "$(ssh-agent -s)"`。

只把 `.pub` 文件交给新电脑。公钥传输可以使用本地文件复制，不依赖首次 SSH 登录。

## 确认源地址与路由

```bash
ip route get 192.0.2.20
```

读取输出中的 `src`。NAT、VPN、多网卡环境下，目标看到的源地址可能不同。结合新电脑 `journalctl -u ssh.service` 和实际网络路径确认，不能盲用 WSL 内部虚拟地址。Linux `AllowUsers` 和 Windows/Hyper-V 规则必须使用目标实际看到的同一源地址。

路由不通时，检查网段、Wi-Fi 客户端隔离、VPN 路由。不要放宽到全网来掩盖源地址错误。DHCP 改变客户端地址时更新三处限制；改变目标地址时更新 HostName。长期使用可在已有路由器上预留地址。

## 核对主机身份并首次登录

从新电脑本地控制台获取主机指纹：

```bash
sudo ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

现有电脑执行一次新连接：

```bash
ssh -F /dev/null -o ProxyCommand=none -o ProxyJump=none \
  -o ControlMaster=no -o ControlPath=none \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=ask \
  -i ~/.ssh/new_pc_ed25519 -p 22 ubuntuuser@192.0.2.20
```

提示的指纹必须与新电脑本地控制台一致。确认一致才接受。`ssh-keyscan` 只能收集密钥，不能单独证明机器身份。已有指纹不一致时查明重装/地址复用原因；核对后只更新相应条目，不清空 `known_hosts`。
进入后检查 `id -un`、`hostname`、`uname -a`，必须指向新电脑的 WSL Ubuntu。

## 验证全新且仅使用密钥的连接

退出首次连接，从现有电脑执行：

```bash
ssh -F /dev/null -o ProxyCommand=none -o ProxyJump=none \
  -o ControlMaster=no -o ControlPath=none \
  -o BatchMode=yes -o PreferredAuthentications=publickey \
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=10 -i ~/.ssh/new_pc_ed25519 -p 22 \
  ubuntuuser@192.0.2.20 'id -un; hostname; uname -a'
```

成功条件：退出码为 `0`，远端用户和主机符合目标。此命令禁用连接复用，防止误用已有会话。加密私钥没有加入 agent 时，BatchMode 无法提示口令；先解锁再检查，不能误判为服务器拒绝公钥。

## 配置简短别名

连接成功后备份 `~/.ssh/config`。添加没有同名冲突的区块，放在通配 `Host *` 前，保留其他主机配置：

```sshconfig
Host new-pc
    HostName 192.0.2.20
    User ubuntuuser
    Port 22
    IdentityFile ~/.ssh/new_pc_ed25519
    IdentitiesOnly yes
    ProxyCommand none
    ProxyJump none
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

```bash
chmod 600 ~/.ssh/config
ssh -G new-pc
ssh -o BatchMode=yes -o ControlMaster=no -o ControlPath=none new-pc 'id -un; hostname'
```

核对 `ssh -G` 的最终 hostname/user/port/identityfile/proxycommand/proxyjump。此前验证直接连接，此处验证别名。不要覆盖现有同名主机。

## 常见失败与证据

| 现象 | 检查 |
| --- | --- |
| timeout | 目标在线状态、IPv4、WSL 实际模式、来源限制、两层防火墙/UFW、客户端隔离 |
| connection refused | 目标端口、`ss -ltnp`、`systemctl status ssh`、`journalctl -u ssh.service -n 50` |
| Permission denied (publickey) | 用户、公钥指纹、文件权限/属主、agent 解锁、`AllowUsers`、`ssh -vvv` |
| 主机指纹变化 | 本地控制台核对、地址复用、系统重装 |
| PowerShell Access denied | 管理员权限、组织策略；不要把无法读取规则当作规则不存在 |
| Windows 重启后不可达 | WSL 是否运行、生命周期、Windows 用户归属、启动任务、睡眠 |

新电脑可检查当前连接的有效配置：

```bash
sudo /usr/sbin/sshd -T -C \
  user=ubuntuuser,host=client,addr=192.0.2.10,laddr=192.0.2.20,lport=22
sudo journalctl -u ssh.service -n 50 --no-pager
```

分享诊断记录前移除真实地址和用户名。完整 `ssh -vvv` 输出包含个人路径和端点，发布前检查。
