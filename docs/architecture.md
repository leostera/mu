# Port architecture

## Upstream reference

The reference is `earendil-works/pi`, tracked as a submodule at `reference/pi`. The checked-out upstream revision for this bootstrap is `698a2d23b86a5045dd54554354a5e0cb7e466795` (2026-10-16). Pi is an MIT-licensed TypeScript monorepo. Its coding-agent package is layered over separate `agent`, `ai`, `tui`, `mcp`, `telemetry`, and support packages.

Useful entry points in that revision:

- `packages/coding-agent/src/core/agent-session.ts`: session orchestration, message/event coordination, retries, compaction, and tool integration.
- `packages/coding-agent/src/core/session-manager.ts`: append-only JSONL session entries and branch/tree projection.
- `packages/coding-agent/src/core/extensions/{types,runner,loader}.ts`: extension API contract, lifecycle, and loading.
- `packages/agent/src`: provider-neutral agent loop and tool execution.
- `packages/ai/src`: providers, model/message types, streaming and compatibility.
- `packages/coding-agent/docs/extensions.md` and `sdk.md`: extension and session contracts.

The Rust port should reproduce observable behavior and file/API contracts, not translate TypeScript files mechanically. Keep upstream paths and a pinned revision visible during conformance work.

## Workspace boundaries

- `mu-core`: agent loop and orchestration; no terminal or V8 dependencies.
- `mu-session`: Pi-compatible session data model, JSONL codec, and later tree/persistence operations.
- `mu-runtime-deno`: one embedded Deno Core isolate per extension runtime, host operations, module resolution, permissions, and JS/TS plugin loading.
- `mu-cli`: mode selection and process boundary; interactive TUI can be its own crate once the interface is known.
- `mu-e2e`: shared deterministic fixtures and workflow-level tests.

Likely later extractions include provider/model APIs, session persistence, tool execution, resource discovery, terminal UI, and protocol adapters (RPC/MCP). Do not split these into crates prematurely; use module boundaries until dependencies and public APIs stabilize.

## Plugin compatibility is a first-class constraint

Extensions are not just scripts. Pi extensions import packages and register commands, tools, providers, event listeners, and UI integrations through a broad `ExtensionAPI`; they rely on ordered async lifecycle events, session context, tool validation, and cleanup. Compatibility therefore needs:

1. A versioned host API exposed to JavaScript, preserving Pi's extension registration and event semantics.
2. A loader that handles JS and TS entry points, local/package imports, and the same resource discovery rules.
3. Explicit capability/permission policy. Upstream extensions run with process permissions, so isolation must not silently break trusted extensions or imply a sandbox that does not exist.
4. Contract tests that run existing upstream extensions against the Rust host, including failure, cancellation, reload, and shutdown cases.

Deno Core supplies V8 and low-level runtime primitives, not the complete Deno CLI resolver/compiler/permissions stack. TypeScript transpilation, npm/package resolution, and Node-compatible APIs need intentional choices and tests.

## Testing approach

- Unit tests for parsing, state transitions, session branching, prompt construction, and tool result invariants.
- Golden/contract tests against upstream session JSONL and extension API behavior.
- Deterministic provider fakes; no API keys, live endpoints, or paid requests in normal CI.
- Deno integration tests for JS execution, module imports, TypeScript transpilation, host operations, errors, cancellation, isolate teardown, and extension reload.
- End-to-end CLI tests that exercise prompt -> model/tool events -> persisted session -> resume, with extensions in the loop.
- Optional live-provider evals kept separate from required CI.

`mu-session` parses and serializes Pi session JSONL records while retaining unknown fields, migrates legacy v1/v2 records to v3, constructs linked v3 message entries, and atomically persists sessions; fixture tests cover the pinned Pi data. `mu-core` and `mu-cli` provide a Pi-configured OpenCode Responses path, a Ratatui/Crossterm TUI, and a deterministic offline demo backend. The TUI boundary is split into terminal protocol lifecycle (`tui::terminal`), pure viewport geometry (`tui::layout`), and UI state components (`Composer` and `Transcript`); terminal setup tracks acquired modes and rolls them back in reverse order, including on partial setup failure. Bracketed paste is negotiated at entry and normalized into editor text. The composer now follows Pi's multiline cursor/history rules (visual-row navigation, sticky columns, draft restoration), key editing actions, grapheme-aware movement/deletion, submit/newline split, and Ctrl+C clear/double-press exit behavior. Ratatui's backend owns screen rendering while Crossterm normalizes terminal input events.

TUI tests cover component state, in-memory rendering, protocol rollback, and a PTY-driven real-binary turn/paste/history flow using `portable-pty` and `vt100`; live provider calls remain outside CI. Tools and skills are supported, but custom keybinding configuration, undo/kill-ring/autocomplete, large-paste markers, session resume/branching, the full provider/model catalog, richer terminal capability queries, a distinct regular-screen renderer, and extension API parity remain incomplete.
