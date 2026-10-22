# Codemode and MCP permission boundary

What a codemode script, an MCP server and a built-in tool can and cannot do in Mu today.

## Codemode scripts

- Run in an embedded V8 isolate (`codemode` crate on `deno_core`), not in a subprocess.
- **Not available to scripts:** `fetch`, `env` and `Deno` are set to `undefined` before the script
  runs (`crates/mu-core/src/codemode.rs`; asserted by `script_*` tests that check
  `typeof fetch/env/Deno`). Scripts have no direct file, network or process access.
- **Available:** `tools.<name>(args)` for every built-in and MCP tool the agent exposes, `text`,
  `image` (placeholder only), `exit`, `store`/`load` (session-persisted key/value), `searchTools`,
  `describeTool`, `ALL_TOOLS`.
- **Nested tool calls are not sandboxed.** A script calling `tools.bash(...)` runs a real shell in
  the project directory with the user's permissions, exactly as if the model called `bash`.
  The boundary is therefore "scripts can do whatever the exposed tools can", not stronger.
- **No quotas yet:** execution time and memory are not limited (open TODO item). A script that
  loops forever blocks until the turn is cancelled.

## MCP servers

- stdio servers are spawned as the user's own child processes with the configured command, args
  and `env`; HTTP servers receive the configured headers. Servers are fully trusted: they run with
  the user's permissions and nothing restricts their file or network access.
- Project-level `.pi/mcp.json` and `.pi/settings.json` are honoured only for a trusted project
  (`trust.json`, `defaultProjectTrust`, `--approve` / `--no-approve`); untrusted projects fall back
  to global configuration. Malformed trust data fails closed for project settings. MCP project
  overrides remain limited to safe fields; `/settings` currently writes only global settings.
- Tool results are text, structured content, and images; large text is truncated in the middle
  and saved to a temp file. Resource tools expose only servers that advertised `resources`.

## Built-in tools

`read`, `write`, `edit`, `bash`, `grep`, `find`, `ls` act on the real filesystem with the user's
permissions. There is no approval prompt (`--approve` is project trust, as in Pi) and no sandbox.

## Accepted risk

Mu matches Pi here: tool, MCP and codemode execution is unsandboxed. The only technical barrier is
that codemode scripts reach the host solely through `tools.*`.
