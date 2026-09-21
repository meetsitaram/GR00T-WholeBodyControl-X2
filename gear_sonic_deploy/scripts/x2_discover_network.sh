#!/usr/bin/env bash
# X2 split-topology network discovery.
#
# Run this once from the laptop WHILE CONNECTED OVER THE WIRE (i.e. the
# PC2 SDK-ethernet IP (PC2_HOST / --pc2-host) is reachable). The script will:
#
#   1. Enumerate every IPv4 address on the laptop's interfaces, tagged
#      ethernet / wifi / virtual.
#   2. SSH into PC2 and do the same on the robot's Jetson Orin NX.
#   3. Cross-match laptop ↔ PC2 IPs by /24 prefix and report each
#      candidate route (e.g. wired SDK ethernet pair, WiFi pair).
#   4. Ping-probe each candidate route to confirm reachability.
#   5. Print copy/paste-ready env blocks for `~/.x2/env.wired` and
#      `~/.x2/env.wifi` so the operator can `source` whichever network
#      they happen to be on for the next x2_pc2_daemons.sh invocation.
#
# Why this script: the laptop typically has 2-3 active interfaces (wire,
# WiFi, docker bridge); PC2 typically has wire (SDK ethernet to PC1) +
# WiFi to the lab AP. Without this, the operator has to ssh in, run
# `hostname -I` / `ip addr`, and eyeball which IPs are on the same
# network -- and the wrong choice silently makes the deploy on PC2
# subscribe to a laptop IP it can't reach, with no error until the
# pose-ref watchdog trips after 0.5 s.
#
# This script does NOT touch any running daemons -- it's read-only.
#
# Usage:
#     ./gear_sonic_deploy/scripts/x2_discover_network.sh
#         [--pc2-host H] [--pc2-user U] [--pc1-host H]
#         [--write-env-dir DIR]    # also write env.wired / env.wifi files
#         [--no-probe]             # skip the ping probe
#
# Exit code: 0 if at least one candidate route is up.

set -u
set -o pipefail

PC2_USER="${PC2_USER:-run}"
# PC2's SDK-ethernet address is printed on the robot's dev-unit label / in the
# AgiBot quick-start; there is no baked-in default here. PC1 is optional.
PC2_HOST="${PC2_HOST:-}"
PC1_HOST="${PC1_HOST:-}"
WRITE_ENV_DIR=""
PROBE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pc2-host) PC2_HOST="$2"; shift 2 ;;
        --pc2-user) PC2_USER="$2"; shift 2 ;;
        --pc1-host) PC1_HOST="$2"; shift 2 ;;
        --write-env-dir) WRITE_ENV_DIR="$2"; shift 2 ;;
        --no-probe) PROBE=0; shift ;;
        -h|--help)
            sed -n '2,33p' "$0"
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${PC2_HOST}" ]]; then
    err "PC2_HOST is not set: pass --pc2-host <PC2_IP> or export PC2_HOST (the SDK-ethernet address of the robot dev unit)."; exit 2
fi

C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_BLUE=$'\e[34m'; C_DIM=$'\e[2m'; C_RESET=$'\e[0m'
section() { printf '\n%s== %s ==%s\n' "${C_BLUE}" "$*" "${C_RESET}"; }
ok()      { printf '  %s[ok]%s   %s\n' "${C_GREEN}"  "${C_RESET}" "$*"; }
warn()    { printf '  %s[warn]%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*"; }
err()     { printf '  %s[err]%s  %s\n' "${C_RED}"    "${C_RESET}" "$*" >&2; }
dim()     { printf '  %s%s%s\n'         "${C_DIM}"   "$*" "${C_RESET}"; }

# ----- iface enumeration --------------------------------------------------
# Parses `ip -o -4 addr show` output into TAB-separated rows:
#     <iface>\t<type>\t<ip>\t<cidr>
# where type ∈ {ethernet, wifi, virtual, other}. Loopback is skipped.
PARSE_IP_AWK='
{
    iface = $2; cidr = $4
    if (iface == "lo") next
    ip = cidr; sub("/.*", "", ip)
    type = "other"
    if (iface ~ /^(wl|wlan)/) type = "wifi"
    else if (iface ~ /^(docker|br-|virbr|veth|tun|tap|wg|cni|flannel)/) type = "virtual"
    else if (iface ~ /^(en|eth|eno|enp|enx)/) type = "ethernet"
    printf "%s\t%s\t%s\t%s\n", iface, type, ip, cidr
}'

list_local_ipv4() {
    ip -o -4 addr show 2>/dev/null | awk "${PARSE_IP_AWK}"
}

list_pc2_ipv4() {
    # Capture ssh's stdout + stderr to a tempfile so we can keep ssh's
    # exit status separate from awk's (a pipeline would mask it even
    # under pipefail because ssh's stderr is what tells us auth failed).
    local raw
    raw="$(ssh -o StrictHostKeyChecking=accept-new \
              -o ConnectTimeout=5 \
              -o BatchMode=yes \
              "${PC2_USER}@${PC2_HOST}" \
              'ip -o -4 addr show 2>/dev/null' 2>&1)"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        err "ssh ${PC2_USER}@${PC2_HOST} failed (exit ${rc}). Last output:"
        printf '%s\n' "${raw}" | sed 's/^/        /' >&2
        err "  Hint: run \`ssh-copy-id ${PC2_USER}@${PC2_HOST}\` once to install"
        err "  a passwordless key (the PC2 'run' account password is in the AgiBot docs)."
        return 1
    fi
    printf '%s\n' "${raw}" | awk "${PARSE_IP_AWK}"
}

# Render a single TSV stream (iface, type, ip, cidr) as a colorized table.
render_table() {
    local label="$1"
    printf '  %-16s %-9s %-18s %s\n' "INTERFACE" "TYPE" "IPV4" "CIDR"
    printf '  %-16s %-9s %-18s %s\n' "----------------" "---------" "------------------" "----"
    awk -v G="${C_GREEN}" -v Y="${C_YELLOW}" -v D="${C_DIM}" -v R="${C_RESET}" '
    BEGIN { FS = "\t" }
    {
        iface=$1; type=$2; ip=$3; cidr=$4
        color = D
        if (type == "ethernet") color = G
        else if (type == "wifi") color = Y
        printf "  %s%-16s %-9s %-18s %s%s\n", color, iface, type, ip, cidr, R
    }'
}

# Return the /24 prefix of an IPv4 address (e.g. A.B.C.2 -> A.B.C).
prefix24() {
    awk -F. '{print $1"."$2"."$3}' <<< "$1"
}

# ----- gather -------------------------------------------------------------

section "1. Laptop interfaces (this host)"
LAPTOP_TSV="$(list_local_ipv4)"
if [[ -z "${LAPTOP_TSV}" ]]; then
    err "no IPv4 interfaces detected on the laptop (?!)"
    exit 1
fi
render_table "laptop" <<< "${LAPTOP_TSV}"

section "2. PC2 interfaces (${PC2_USER}@${PC2_HOST})"
if ! PC2_TSV="$(list_pc2_ipv4)"; then
    err "could not enumerate PC2 interfaces (see above)."
    err "  Make sure (a) the laptop is on the wire and (b) ssh keys are"
    err "  installed (see hint above). Set PC2_HOST or pass --pc2-host."
    exit 1
fi
if [[ -z "${PC2_TSV}" ]]; then
    err "ssh succeeded but PC2 reported no IPv4 interfaces (?!)."
    err "  Check that 'ip' (iproute2) is installed on PC2."
    exit 1
fi
render_table "pc2" <<< "${PC2_TSV}"

# ----- match candidates ---------------------------------------------------

section "3. Candidate routes (laptop ↔ PC2, matched by /24 prefix)"
candidates_tsv=""   # rows: type, laptop_ip, pc2_ip, prefix
while IFS=$'\t' read -r l_iface l_type l_ip l_cidr; do
    [[ "${l_type}" == "virtual" || "${l_type}" == "other" ]] && continue
    l_pfx="$(prefix24 "${l_ip}")"
    while IFS=$'\t' read -r p_iface p_type p_ip p_cidr; do
        [[ "${p_type}" == "virtual" || "${p_type}" == "other" ]] && continue
        p_pfx="$(prefix24 "${p_ip}")"
        if [[ "${l_pfx}" == "${p_pfx}" ]]; then
            # Pair them by what link type they share.
            route_type="cross"
            if [[ "${l_type}" == "ethernet" && "${p_type}" == "ethernet" ]]; then
                route_type="wired"
            elif [[ "${l_type}" == "wifi" && "${p_type}" == "wifi" ]]; then
                route_type="wifi"
            fi
            candidates_tsv+="${route_type}"$'\t'"${l_ip}"$'\t'"${p_ip}"$'\t'"${l_iface}->${p_iface}"$'\t'"${l_pfx}.0/24"$'\n'
        fi
    done <<< "${PC2_TSV}"
done <<< "${LAPTOP_TSV}"

if [[ -z "${candidates_tsv}" ]]; then
    warn "no laptop/PC2 IP pair shares a /24. The laptop cannot reach PC2"
    warn "directly on any L3-shared subnet right now. Check WiFi association,"
    warn "VLAN settings, or pass --pc2-host with the actual PC2 address."
else
    printf '  %-7s %-18s %-18s %-22s %s\n' "ROUTE" "LAPTOP_IP" "PC2_IP" "INTERFACE_PAIR" "SUBNET"
    printf '  %-7s %-18s %-18s %-22s %s\n' "-------" "------------------" "------------------" "----------------------" "------------"
    while IFS=$'\t' read -r rtype lip pip pair subnet; do
        [[ -z "${rtype}" ]] && continue
        case "${rtype}" in
            wired) color="${C_GREEN}" ;;
            wifi)  color="${C_YELLOW}" ;;
            *)     color="${C_DIM}" ;;
        esac
        printf '  %s%-7s %-18s %-18s %-22s %s%s\n' \
            "${color}" "${rtype}" "${lip}" "${pip}" "${pair}" "${subnet}" "${C_RESET}"
    done <<< "${candidates_tsv}"
fi

# ----- probe --------------------------------------------------------------

if [[ "${PROBE}" -eq 1 && -n "${candidates_tsv}" ]]; then
    section "4. Reachability probe (laptop -> PC2 IP, 1× ping each)"
    while IFS=$'\t' read -r rtype lip pip pair subnet; do
        [[ -z "${rtype}" ]] && continue
        if ping -c 1 -W 1 -q -I "${lip}" "${pip}" >/dev/null 2>&1; then
            ok "${rtype}: laptop ${lip} -> PC2 ${pip} reachable"
        else
            warn "${rtype}: laptop ${lip} -> PC2 ${pip} NOT reachable from this iface"
        fi
    done <<< "${candidates_tsv}"

    # Probe PC1 from the laptop (informational; PC1 is normally talked to
    # from PC2 over SDK ethernet, but the laptop can still ping it when
    # they share a wire).
    if [[ -z "${PC1_HOST}" ]]; then
        dim "PC1_HOST not set (--pc1-host <PC1_IP>); skipping the PC1 probe."
    elif ping -c 1 -W 1 -q "${PC1_HOST}" >/dev/null 2>&1; then
        ok "PC1 ${PC1_HOST} reachable from the laptop"
    else
        dim "PC1 ${PC1_HOST} not reachable from the laptop (expected over WiFi;"
        dim "  PC1 is normally talked to from PC2 over the SDK ethernet)."
    fi
fi

# ----- emit env blocks ----------------------------------------------------

section "5. Copy/paste env blocks for x2_pc2_daemons.sh"

emit_env_block() {
    local label="$1"   # "wired" or "wifi"
    local rtype="$2"
    local lip pip
    lip="$(awk -v t="${rtype}" -F'\t' '$1==t {print $2; exit}' <<< "${candidates_tsv}")"
    pip="$(awk -v t="${rtype}" -F'\t' '$1==t {print $3; exit}' <<< "${candidates_tsv}")"
    if [[ -z "${lip}" || -z "${pip}" ]]; then
        dim "(no ${label} candidate detected; skipping)"
        return 1
    fi
    printf '  %s# ~/.x2/env.%s -- source before x2_pc2_daemons.sh / run_x2_quest3_planner_stack.sh%s\n' \
        "${C_DIM}" "${label}" "${C_RESET}"
    printf '  export PC2_HOST=%s\n' "${pip}"
    printf '  export LAPTOP_HOST=%s\n' "${lip}"
    if [[ -n "${PC1_HOST}" ]]; then
        printf '  export PC1_HOST=%s\n' "${PC1_HOST}"
    else
        printf '  # export PC1_HOST=<PC1_IP>   # optional (pc2_preflight.sh sections 2 + 6); pass --pc1-host to fill in\n'
    fi
    return 0
}

printf '\n'
emit_env_block "wired" "wired" || true
printf '\n'
emit_env_block "wifi" "wifi"   || true

# Optionally drop the env files to disk for the operator.
if [[ -n "${WRITE_ENV_DIR}" ]]; then
    mkdir -p "${WRITE_ENV_DIR}"
    write_env_file() {
        local label="$1" rtype="$2" path="$3"
        local lip pip
        lip="$(awk -v t="${rtype}" -F'\t' '$1==t {print $2; exit}' <<< "${candidates_tsv}")"
        pip="$(awk -v t="${rtype}" -F'\t' '$1==t {print $3; exit}' <<< "${candidates_tsv}")"
        if [[ -z "${lip}" || -z "${pip}" ]]; then
            warn "skipping ${path}: no ${label} candidate detected"
            return 0
        fi
        cat > "${path}" <<EOF
# Generated by gear_sonic_deploy/scripts/x2_discover_network.sh
# on $(date -u +%Y-%m-%dT%H:%M:%SZ). Source this before
# x2_pc2_daemons.sh / run_x2_quest3_planner_stack.sh when the laptop
# is connected to PC2 via ${label}.
export PC2_HOST=${pip}
export LAPTOP_HOST=${lip}
$(if [[ -n "${PC1_HOST}" ]]; then echo "export PC1_HOST=${PC1_HOST}"; else echo "# export PC1_HOST=<PC1_IP>   # optional; pass --pc1-host to fill in"; fi)
EOF
        ok "wrote ${path}"
    }
    section "6. Writing env files to ${WRITE_ENV_DIR}/"
    write_env_file "wired" "wired" "${WRITE_ENV_DIR}/env.wired"
    write_env_file "wifi"  "wifi"  "${WRITE_ENV_DIR}/env.wifi"
fi

# Exit non-zero if NO route is usable so this is safe to chain in a script.
if [[ -z "${candidates_tsv}" ]]; then
    exit 2
fi
exit 0
