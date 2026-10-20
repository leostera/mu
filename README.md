# mu

A Rust port of the core of Pi, with Deno Core embedding for JavaScript/TypeScript extension compatibility.

## Current status

Mu now starts a fullscreen Ratatui/Crossterm TUI with a scrollable Pi-style chat document above a fixed command/composer/status dock. The header scrolls with the transcript; manual scroll position is preserved while responses stream, with follow-to-end behavior when the user is at the bottom. The live provider implementation supports the Cloudflare OpenCode Pi plugin (`opencode-cf`) with OpenAI Responses models: it discovers the gateway, obtains a Cloudflare Access token from `cloudflared` (or `OPENCODE_CF_TOKEN`), streams text, and writes each turn to a new Pi v3 JSONL session under the same `~/.pi/agent/sessions/<cwd>/` layout. `mu --demo` starts the same UI with a deterministic no-network backend.

This remains an early vertical slice, not a drop-in coding agent. Mu exposes Pi-shaped `read`, `write`, `edit`, and `bash` tools to the OpenAI Responses tool loop, and discovers Pi-style `SKILL.md` resources under `~/.pi/agent/skills`, `~/.agents/skills`, `.pi/skills`, and `.agents/skills`. Skill metadata enters the system prompt; the model reads a skill file when needed. Mu can also connect to stdio and Streamable HTTP MCP servers from `~/.pi/agent/mcp.json`, negotiate MCP, discover tools, and route calls through the agent loop. It honors Pi's `direct`, `hidden`, `codemode`, and `deferred` exposure settings, including per-tool overrides. Direct tools are model-callable; codemode-exposed tools are callable from JavaScript running in the `codemode` tool via the `codemode` crate and embedded Deno Core. Deferred MCP tools can be searched with a BM25-ranked `tool_search` and activated for the next model turn; codemode scripts use the same ranker through `searchTools()`. MCP resource/template tools, OAuth/provider-token authentication, reconnect/lazy lifecycle, dynamic server registration, and `/mcp` reload/connect/disconnect management remain unimplemented; `/mcp status` and `/mcp tools` are available. Trusted project MCP configuration is merged from `.pi/mcp.json` using existing Pi decisions in `trust.json`; Mu does not yet provide an interactive trust prompt. Built-in file/shell tools and MCP server processes run with the current user's OS permissions and are **not sandboxed**. When active, codemode can invoke every non-hidden built-in/MCP tool; Mu removes the crate's `fetch`, `env`, and `Deno` globals to match Pi's no-network/no-filesystem script surface. The current Deno Core wrapper does not yet enforce Pi's execution time or memory limits, so only run trusted workloads and MCP configurations. Session resume, cloning, JSONL import/export, and standalone escaped HTML transcript export are available; branch navigation/tree selection, compaction, the rest of Pi's tools/config surface, and extension API compatibility remain incomplete. The separate Deno extension smoke test only establishes that embedded V8 can evaluate JavaScript. The Pi reference is pinned as a git submodule at [`reference/pi`](reference/pi), commit `698a2d23b86a5045dd54554354a5e0cb7e466795`; initialize it with `git submodule update --init --recursive`.

Mu reads `defaultProvider` and `defaultModel` from Pi's global `settings.json`, with `.pi/settings.json` in the current project taking precedence. Configure MCP servers using the shared `mcpServers` shape in `<PI_CODING_AGENT_DIR>/mcp.json` (default `~/.pi/agent/mcp.json`), for example:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "exposure": "direct"
    }
  }
}
```

Mu reads this global file and merges `.pi/mcp.json` for trusted projects. Project-only entries replace global entries; project overrides are restricted to `enabled`, `exposure`, and `toolExposure`. Mu honors existing trust decisions in `<PI_CODING_AGENT_DIR>/trust.json`, but has no interactive trust prompt. Enabled stdio and Streamable HTTP servers connect when a live agent starts. Pi's default `codemode` exposure is supported: server tools are callable by scripts passed to Mu's `codemode` tool. Set `exposure` to `direct` when you want the model to call a server tool directly, and set top-level `autoEnableCodemode` to `false` to disable codemode. Set `PI_CODING_AGENT_DIR` to use a non-default Pi agent directory. The current live provider adapter is specific to the default Cloudflare OpenCode gateway. A PTY integration test uses `portable-pty` and `vt100` to drive the real binary with the offline demo backend, so normal tests need no gateway credentials or model calls.

## Development

```sh
cargo test --workspace
cargo run                 # interactive Pi-configured session
cargo run -- --demo       # offline TUI preview with a deterministic fake backend
cargo run -- "Hello"     # one prompt, then exit
```

## Pi behavior oracle

The CLI oracle under [`crates/mu-e2e/fixtures`](crates/mu-e2e/fixtures) records actual Pi 1.3.0 output for version/help, invalid arguments and modes, and empty print-mode input. These cases run with an isolated home, cleared environment, and `PI_OFFLINE=1`; they do not make model calls. The regular CLI tests check Mu against the shared behavior contracts and the recorded observations.

To refresh the fixtures from a Pi executable, then run the opt-in live-oracle test:

```sh
PI_ORACLE_BIN=/path/to/pi scripts/record-pi-oracle.sh
PI_ORACLE_BIN=/path/to/pi cargo test -p mu-cli --test pi_cli_oracle \
  live_pi_executable_matches_the_recorded_baseline -- --ignored --exact
```

The recorded binary is identified by its reported version; the test does not independently prove its build came from the pinned submodule revision. Help text is captured as a reference rather than compared byte-for-byte with Mu, since the products have different names and implemented feature sets.

The UI oracle contains **100 independent PTY sessions** in [`pi-ui-v1.3.0`](crates/mu-e2e/fixtures/pi-ui-v1.3.0): 20 each for Ctrl+P model cycling, slash-command filtering, composer growth/shrink (including multiline and Unicode input), terminal resizing, and scrolling a populated transcript. They use isolated Pi settings and a dummy local model configuration; no prompt is submitted and no model/network call is made. To regenerate them (about two minutes):

```sh
PI_ORACLE_BIN=/path/to/pi cargo test -p mu-cli --test pi_ui_oracle \
  record_100_live_pi_ui_sessions -- --ignored --exact --nocapture
```

For a full animated terminal recording, the VHS oracle in [`pi-vhs-v1.3.0`](crates/mu-e2e/fixtures/pi-vhs-v1.3.0) contains both a GIF of Pi's actual terminal and `.ascii` screen-buffer snapshots. VHS drives Pi in a real PTY and captures the rendered terminal; the scripted scenario covers startup, slash-command filtering, and composer interactions without submitting a prompt or calling a model. Install VHS with `brew install vhs`, then regenerate with:

```sh
PI_ORACLE_BIN=/path/to/pi scripts/record-pi-vhs-oracle.sh
```

The ordinary `pi_vhs_oracle` test validates the checked-in recording and snapshots without requiring VHS or a Pi installation.

See [`docs/architecture.md`](docs/architecture.md) for the porting boundaries and testing strategy.
