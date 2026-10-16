#!/usr/bin/env bash
set -euo pipefail

pi_bin="${PI_ORACLE_BIN:-pi}"
vhs_bin="${VHS_BIN:-vhs}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
output_dir="$repo_root/crates/mu-e2e/fixtures/pi-vhs-v1.3.0"
tape="$output_dir/pi-session.tape"

command -v "$vhs_bin" >/dev/null || { echo "VHS is required (brew install vhs)" >&2; exit 1; }
if [[ ! -x "$pi_bin" ]]; then
  pi_bin="$(command -v "$pi_bin" || true)"
fi
[[ -n "$pi_bin" && -x "$pi_bin" ]] || { echo "Pi not found: ${PI_ORACLE_BIN:-pi}" >&2; exit 1; }
version="$("$pi_bin" --version)"
[[ "$version" == "1.3.0" ]] || { echo "expected Pi 1.3.0, got $version" >&2; exit 1; }

mkdir -p "$output_dir"
sandbox="$(mktemp -d)"
trap 'rm -rf "$sandbox"' EXIT
home="$sandbox/home"
agent_dir="$sandbox/agent"
project_dir="$sandbox/project"
mkdir -p "$home" "$agent_dir" "$project_dir"

cat > "$agent_dir/settings.json" <<'JSON'
{"defaultProvider":"oracle","defaultModel":"alpha","hideThinkingBlock":false}
JSON
cat > "$agent_dir/models.json" <<'JSON'
{"providers":{"oracle":{"baseUrl":"http://127.0.0.1:9/v1","api":"openai-completions","apiKey":"offline-test-key","models":[{"id":"alpha","name":"Alpha Model","contextWindow":32768,"maxTokens":4096},{"id":"beta","name":"Beta Model","contextWindow":32768,"maxTokens":4096}]}}}
JSON

pi_bin="$(cd "$(dirname "$pi_bin")" && pwd)/$(basename "$pi_bin")"
pi_name="$(basename "$pi_bin")"
pi_path="$(dirname "$pi_bin"):$PATH"
cd "$repo_root"
env -i \
  PATH="$pi_path" \
  HOME="$home" \
  USER="$(id -un)" \
  LOGNAME="$(id -un)" \
  TERM=xterm-256color \
  COLORTERM=truecolor \
  PI_ORACLE_BIN="$pi_name" \
  PI_VHS_PROJECT_DIR="$project_dir" \
  PI_CODING_AGENT_DIR="$agent_dir" \
  PI_CODING_AGENT_SESSION_DIR="$sandbox/sessions" \
  PI_OFFLINE=1 \
  PI_TELEMETRY=0 \
  "$vhs_bin" "$tape"

cat > "$output_dir/metadata.json" <<JSON
{
  "oracle": "Pi interactive TUI captured by VHS",
  "piVersion": "$version",
  "binarySource": "PI_ORACLE_BIN (version checked; source revision not independently established)",
  "vhsVersion": "$("$vhs_bin" --version)",
  "scenario": "startup, slash-menu model filter, composer typing and clearing; no prompt submitted",
  "environment": "isolated HOME and PI_CODING_AGENT_DIR; PI_OFFLINE=1; provider points to localhost:9; no live model or network calls",
  "artifacts": ["pi-session.gif", "pi-session.ascii"]
}
JSON

echo "Recorded Pi VHS oracle to $output_dir"
