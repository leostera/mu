# mu

A Rust port of the core of Pi, with Deno Core embedding for JavaScript/TypeScript extension compatibility.

## Current status

Mu now starts a minimal interactive text session using the Pi default provider/model. The first provider implementation supports the Cloudflare OpenCode Pi plugin (`opencode-cf`) with OpenAI Responses models: it discovers the gateway, obtains a Cloudflare Access token from `cloudflared` (or `OPENCODE_CF_TOKEN`), streams text, and writes each turn to a new Pi v3 JSONL session under the same `~/.pi/agent/sessions/<cwd>/` layout.

This is an initial vertical slice, not yet a drop-in coding agent: tool execution, session resume/branching, full Pi provider/model config, and extension API compatibility are not implemented. The Deno smoke test only establishes that embedded V8 can evaluate JavaScript. The Pi reference is pinned as a git submodule at [`reference/pi`](reference/pi), commit `698a2d23b86a5045dd54554354a5e0cb7e466795`; initialize it with `git submodule update --init --recursive`.

Mu reads `defaultProvider` and `defaultModel` from Pi's global `settings.json`, with `.pi/settings.json` in the current project taking precedence. Set `PI_CODING_AGENT_DIR` to use a non-default Pi agent directory. The current provider adapter is specific to the default Cloudflare OpenCode gateway.

## Development

```sh
cargo test --workspace
cargo run                 # interactive session; /exit to quit
cargo run -- "Hello"     # one prompt, then exit
```

See [`docs/architecture.md`](docs/architecture.md) for the porting boundaries and testing strategy.
