#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pi_bin="${PI_ORACLE_BIN:-pi}"
path="${PATH:-/usr/bin:/bin}"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

version="$(env -i PATH="$path" HOME="$tmp_dir/home" PI_CODING_AGENT_DIR="$tmp_dir/agent" "$pi_bin" --version)"
if [[ "$version" != "1.3.0" ]]; then
    printf 'Expected pinned Pi oracle 1.3.0, got %s\n' "$version" >&2
    exit 1
fi
fixture_dir="$repo_root/crates/mu-e2e/fixtures/pi-v${version}"
mkdir -p "$fixture_dir"

capture() {
    local name="$1"
    shift
    local status
    set +e
    env -i PATH="$path" HOME="$tmp_dir/home" PI_CODING_AGENT_DIR="$tmp_dir/agent" PI_OFFLINE=1 \
        "$pi_bin" "$@" >"$fixture_dir/$name.stdout" 2>"$fixture_dir/$name.stderr"
    status=$?
    set -e
    printf '%s\n' "$status" >"$fixture_dir/$name.exit"
}

capture version --version
capture short-version -v
capture help --help
capture unknown-flag --definitely-not-a-pi-flag
capture invalid-mode --mode unsupported
capture print-empty --print </dev/null

cat >"$fixture_dir/metadata.json" <<EOF
{
  "oracle": "@earendil-works/pi-coding-agent",
  "version": "$version",
  "pinnedReferenceRevision": "698a2d23b86a5045dd54554354a5e0cb7e466795",
  "binarySource": "selected by PI_ORACLE_BIN (or pi on PATH); verified by --version only",
  "environment": "isolated HOME and PI_CODING_AGENT_DIR; inherited environment cleared; PI_OFFLINE=1",
  "cases": ["--version", "-v", "--help", "--definitely-not-a-pi-flag", "--mode unsupported", "--print with empty stdin"]
}
EOF
printf 'Recorded Pi %s CLI outputs in %s\n' "$version" "$fixture_dir"
