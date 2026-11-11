#!/usr/bin/env bash
# Configure a new Ubuntu/Debian SSH endpoint. Run locally on the target machine.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: setup-linux-sshd.sh --user LOGIN --client-ip IPv4 --public-key FILE
                           [--target-ip IPv4] [--port 22] [--dry-run | --apply]

The default is --dry-run: inspect inputs and print the proposed configuration.
--apply requires root and systemd. The target account must already exist.
--apply also requires --target-ip, the actual IPv4 used to reach this endpoint.
FILE must contain exactly one OpenSSH public key, never a private key.
Only LOGIN from IPv4 or 127.0.0.1 is admitted; password login is disabled.
This helper does not change any firewall, WSL networking mode or Windows task.
Run from the target's local console, keeping another administration path open.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
login=''; client_ip=''; target_ip=''; public_key=''; port=22; mode=dry-run; mode_explicit=''
while (($#)); do
    case "$1" in
        --user|--client-ip|--target-ip|--public-key|--port)
            (($# >= 2)) || die "Missing value for $1"
            case "$1" in
                --user) login=$2 ;;
                --client-ip) client_ip=$2 ;;
                --target-ip) target_ip=$2 ;;
                --public-key) public_key=$2 ;;
                --port) port=$2 ;;
            esac
            shift 2 ;;
        --apply|--dry-run)
            next_mode=${1#--}
            [[ -z $mode_explicit || $mode_explicit == "$next_mode" ]] || die 'Choose --apply or --dry-run, not both'
            mode=$next_mode; mode_explicit=$next_mode; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done
[[ -n $login && -n $client_ip && -n $public_key ]] || { usage >&2; exit 1; }
[[ $mode != apply || -n $target_ip ]] || die '--apply requires --target-ip with the actual target IPv4'
[[ $login =~ ^[a-zA-Z_][a-zA-Z0-9_.-]*\$?$ ]] || die 'Invalid login name'
[[ $port =~ ^[0-9]{1,5}$ ]] && ((10#$port > 0 && 10#$port <= 65535)) || die 'Port must be 1..65535'
port=$((10#$port))
validate_remote_ipv4() {
    local address=$1 label=$2 octet
    local -a octets
    [[ $address =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || die "$label address must be an IPv4 literal"
    IFS=. read -r -a octets <<< "$address"
    for octet in "${octets[@]}"; do
        ((10#$octet <= 255)) || die 'Invalid IPv4 octet'
        [[ $octet == 0 || $octet != 0* ]] || die 'IPv4 octets must not have leading zeroes'
    done
    [[ ${octets[0]} != 0 && ${octets[0]} != 127 ]] && ((10#${octets[0]} < 224)) || die "Use the remote $label IPv4, not a loopback, unspecified or multicast address"
}
validate_remote_ipv4 "$client_ip" client
if [[ -n $target_ip ]]; then validate_remote_ipv4 "$target_ip" target; fi

account=$(getent passwd "$login") || die "Account does not exist: $login"
IFS=: read -r account_name _ account_uid account_gid _ account_home account_shell <<< "$account"
[[ $account_name == "$login" && $account_uid != 0 ]] || die 'Use an existing non-root account'
[[ $account_home == /* && -d $account_home && ! -L $account_home ]] || die 'Account home must be an existing directory, not a symbolic link'
[[ $account_shell != */nologin && $account_shell != */false ]] || die 'Account has no login shell'
ssh_dir=$account_home/.ssh
keys_file=$ssh_dir/authorized_keys
[[ ! -L $ssh_dir && ! -L $keys_file ]] || die 'Refusing a symbolic link at .ssh or authorized_keys'
[[ ! -e $ssh_dir || -d $ssh_dir ]] || die '.ssh is not a directory'
[[ ! -e $keys_file || -f $keys_file ]] || die 'authorized_keys is not a regular file'
[[ -f $public_key && -r $public_key ]] || die 'Public key file is not readable'
command -v ssh-keygen >/dev/null || die 'Install openssh-client first: sudo apt-get install openssh-client'
key_line=$(sed 's/\r$//' "$public_key")
[[ $key_line != *$'\n'* && $key_line =~ ^(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(256|384|521)|sk-ssh-ed25519@openssh.com|sk-ecdsa-sha2-nistp256@openssh.com)[[:space:]] ]] || die 'Expected exactly one raw OpenSSH public key line'
ssh-keygen -l -f "$public_key" >/dev/null 2>&1 || die 'Invalid public key'
key_fingerprint=$(ssh-keygen -l -f "$public_key" | awk '{print $2}')

config_file=/etc/ssh/sshd_config.d/00-ssh-access-setup.conf
render_config() {
    cat <<EOF
# Managed by ssh-access-setup. Backup before changing this file.
Port $port
AddressFamily inet
ListenAddress 0.0.0.0
PermitRootLogin no
PermitEmptyPasswords no
AllowUsers $login@$client_ip $login@127.0.0.1
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM yes
EOF
}

printf 'Mode: %s\nTarget account: %s\nTarget IPv4: %s\nConfiguration: %s\n' "$mode" "$login" "${target_ip:-not provided}" "$config_file"
render_config
if [[ $mode == dry-run ]]; then
    printf '\nDry run only: no files, packages, services or firewall rules changed.\n'
    if [[ -z $target_ip ]]; then printf 'Target-address Match conditions are not verified; --target-ip is required for --apply.\n'; fi
    exit 0
fi

((EUID == 0)) || die '--apply requires root; use sudo from the target console'
[[ -d /run/systemd/system ]] || die 'systemd is required; enable it in WSL and restart WSL before retrying'
. /etc/os-release
[[ $ID == ubuntu || $ID == debian || " ${ID_LIKE:-} " == *' debian '* ]] || die 'Only Ubuntu/Debian systems are supported'
command -v apt-get >/dev/null || die 'apt-get was not found'
if ! dpkg-query -W -f='${Status}' openssh-server 2>/dev/null | grep -qx 'install ok installed'; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server
fi
[[ ! -L /etc/ssh/sshd_config && ! -L /etc/ssh/sshd_config.d && ! -L $config_file ]] || die 'Refusing symbolic links in the SSH configuration path'
grep -Eiq '^[[:space:]]*Include[[:space:]]+/etc/ssh/sshd_config\.d/\*\.conf([[:space:]]|$)' /etc/ssh/sshd_config || die 'Main sshd_config must include /etc/ssh/sshd_config.d/*.conf in global scope'

backup_dir=$(mktemp -d /var/backups/ssh-access-setup.XXXXXXXX)
chmod 700 "$backup_dir"
config_existed=false; keys_existed=false
if [[ -e $config_file ]]; then config_existed=true; cp -p -- "$config_file" "$backup_dir/sshd-config.conf"; fi
if [[ -e $keys_file ]]; then keys_existed=true; cp -p -- "$keys_file" "$backup_dir/authorized_keys"; fi
printf 'Configuration backups: %s\n' "$backup_dir"
restore_on_error=true
restore_files() {
    local result=$?
    if [[ $restore_on_error == true && $result != 0 ]]; then
        if [[ $config_existed == true ]]; then cp -p -- "$backup_dir/sshd-config.conf" "$config_file"; else rm -f -- "$config_file"; fi
        if [[ $keys_existed == true ]]; then cp -p -- "$backup_dir/authorized_keys" "$keys_file"; else rm -f -- "$keys_file"; fi
        printf 'Configuration validation failed. Previous configuration and authorized_keys restored.\n' >&2
    fi
}
trap restore_files EXIT

install -d -m 755 /etc/ssh/sshd_config.d
install -d -m 700 -o "$account_uid" -g "$account_gid" "$ssh_dir"
if [[ ! -e $keys_file ]]; then install -m 600 -o "$account_uid" -g "$account_gid" /dev/null "$keys_file"; fi
# Fingerprints ignore comments and retain restrictions on an existing key.
key_present=false
if existing_keys=$(ssh-keygen -l -f "$keys_file" 2>/dev/null); then
    if awk '{print $2}' <<< "$existing_keys" | grep -Fxq -- "$key_fingerprint"; then key_present=true; fi
fi
if [[ $key_present == false ]]; then
    # A preceding newline also preserves an existing last line without a newline.
    printf '\n%s\n' "$key_line" >> "$keys_file"
fi
chmod 600 "$keys_file"
chown "$account_uid:$account_gid" "$keys_file"
render_config > "$config_file"
chmod 644 "$config_file"
chown root:root "$config_file"
install -d -m 755 /run/sshd
/usr/sbin/sshd -t

check_effective() {
    local effective=$1 name expected actual
    while IFS='=' read -r name expected; do
        actual=$(awk -v key="$name" '$1==key { $1=""; sub(/^ /, ""); print }' <<< "$effective")
        [[ $actual == "$expected" ]] || die "Conflicting effective $name: expected '$expected', got '$actual'. Review existing sshd_config/includes/Match blocks."
    done <<EOF
port=$port
addressfamily=inet
listenaddress=0.0.0.0:$port
permitrootlogin=no
permitemptypasswords=no
pubkeyauthentication=yes
authorizedkeysfile=.ssh/authorized_keys
passwordauthentication=no
kbdinteractiveauthentication=no
usepam=yes
EOF
    actual=$(awk '$1=="allowusers" {print $2}' <<< "$effective" | sort)
    expected=$(printf '%s\n%s\n' "$login@$client_ip" "$login@127.0.0.1" | sort)
    [[ $actual == "$expected" ]] || die 'Effective AllowUsers differs; review existing configuration and Match blocks'
}
check_effective "$(/usr/sbin/sshd -T)"
check_effective "$(/usr/sbin/sshd -T -C "user=$login,host=client,addr=$client_ip,laddr=$target_ip,lport=$port")"
# From here the files are valid. A service error must remain visible for diagnosis.
restore_on_error=false
if systemctl list-unit-files ssh.socket --no-legend 2>/dev/null | grep -q '^ssh.socket'; then
    systemctl disable --now ssh.socket
fi
systemctl enable ssh.service
systemctl restart ssh.service
systemctl is-active --quiet ssh.service || die 'ssh.service is not active; inspect journalctl -u ssh.service'
printf '\nSSH service active. Firewall and remote login are not yet verified.\n'
printf 'Configuration checks do not prove login: account policy, DNS-dependent Match rules and key restrictions still require a real client login.\n'
printf 'From the client: ssh -i PRIVATE_KEY -p %s %s@%s\n' "$port" "$login" "$target_ip"
