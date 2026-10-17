# Mu parity TODO

Known behavior gaps against pinned Pi revision `698a2d23b86a5045dd54554354a5e0cb7e466795` (Pi 1.3.0). This is a living inventory of gaps we've identified, not yet a proof that every Pi behavior has been inventoried. Add Pi source references and deterministic Rust tests as items are investigated and ported; don't mark an item done on build success alone.

## Agent loop and AI/provider behavior

- [ ] Add deterministic provider fakes and Pi-derived tests for complete agent-loop turns, streaming, errors, retries, cancellation, and tool results.
- [ ] Match Pi retry policy and user-visible retry/abort behavior.
- [ ] Handle incomplete, interrupted, refused, and length-truncated provider outputs; do not execute truncated tool arguments.
- [ ] Match Pi steering/follow-up interleavings at request, tool-call, error, and cancellation boundaries; current queue is FIFO per lane, but broader interleaving tests are needed.
- [ ] Add finish-turn/prepare-request/prepare-next-turn hooks and their ordering/cancellation semantics.
- [ ] Match tool approval, validation, before/after tool hooks, progress updates, and tool termination behavior.
- [ ] Match Pi's per-tool sequential execution mode while retaining parallel execution for independent calls. Mu currently serializes an entire batch when it contains stateful `tool_search` or `codemode`.
- [ ] Add other Pi-supported provider transports, authentication, request/response formats, streaming events, usage/cost handling, and multimodal content. Mu currently targets the configured OpenCode Cloudflare Responses path.
- [ ] Implement provider-token/OAuth credential lookup and login/logout flows.

## CLI and session behavior

- [ ] Implement session discovery, resume, fork/clone, branch navigation, import/export, tree selection, and their command-line forms.
- [ ] Port Pi session compaction, branch selection, summaries, and context projection behavior end to end.
- [ ] Resume persisted codemode store and MCP tool activation from an actual loaded session/branch.

## Built-in tools and skills

- [ ] Compare each built-in tool's remaining schema, validation, filesystem/shell behavior, output limits, and safety/approval semantics against Pi; add edge-case fixtures and tests.
- [ ] Complete skill activation/command behavior, argument handling, and parity for discovery precedence, frontmatter, and content injection.
- [ ] Match tool-output limits, content-block conversion, and large-output behavior.

## MCP and codemode

- [ ] OAuth and provider-token authentication for MCP servers.
- [ ] MCP resources, resource templates, and resource reads.
- [ ] Lazy/background server lifecycle, retries/reconnect, notifications, dynamic capability/tool registration, server logs, and clean shutdown.
- [ ] Full `/mcp` management UI/commands (Mu currently has a status view only).
- [ ] Project trust confirmation/prompt and persistent decisions; currently trusted Pi decisions are read but no prompt is shown.
- [ ] Preserve and render all MCP content blocks, including images/resources and structured content, in model output and codemode.
- [ ] Enforce Deno/codemode execution time and memory quotas; verify the effective sandbox and permission boundary.
- [ ] Complete codemode output APIs (notably image output), nested tool events/hooks, and Pi-compatible error/output behavior.

## Extensions and runtime

- [ ] Implement Pi-compatible extension discovery/loading, TypeScript transpilation, module resolution, relative/npm imports, and reload/unload lifecycle.
- [ ] Implement the ExtensionAPI surface: ordered async hooks/events, cancellation, commands, tools, providers, settings, UI dialogs, widgets, custom renderers, and cleanup.
- [ ] Match extension host APIs and permissions deliberately; add deterministic compatibility fixtures for real Pi extensions.

## TUI experience

- [ ] Replace remaining plain-text rendering with Pi-compatible Markdown, transformers, custom message/entry renderers, and per-tool renderers.
- [ ] Add expandable/collapsible tool output and richer progress/status rendering; Mu currently truncates tool output in the transcript.
- [ ] Implement provider-driven autocomplete for slash commands, paths, skills, and extension-defined suggestions; support selection/copy and image paste.
- [ ] Implement all supported slash-command UI flows, dialogs, overlays, settings, and extension-provided components; keep autocomplete/menu entries honest about what works.
- [ ] Add transcript search, mouse selection/copy, click/drag routing, OSC 8 link activation, scrollbar interaction, wheel acceleration/chaining, and Pi-style jump navigation.
- [ ] Support Pi regular/main-screen mode and switching between it and fullscreen; Mu currently always uses the alternate screen.
- [ ] Port terminal input parsing and negotiation: Kitty keyboard protocol/fallback, focus and mouse modes, terminal capabilities/colors, cursor/IME handling, synchronized output, progress/program status, and image protocols as appropriate.
- [ ] Add theme, header/footer, and layout customization through the extension/component system.

## Verification and inventory closure

- [ ] For every implemented item, add deterministic Rust tests against pinned-Pi behavior, including edge/error cases and offline fakes/local servers.
- [ ] Continue source-by-source inventory of Pi's CLI, `agent`, `ai`, `coding-agent`, `mcp`, `codemode`, and `tui` packages; add newly discovered behaviors here.
- [ ] Keep `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, and `git diff --check` passing; commit each coherent slice.
- [ ] Claim parity only after every in-scope behavior is implemented and tested or explicitly accepted as an exception.
