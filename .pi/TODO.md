# Mu parity TODO

Known behavior gaps against pinned Pi revision `698a2d23b86a5045dd54554354a5e0cb7e466795` (Pi 1.3.0). The built-in tool surface has been source-audited; see `personal/pi/rust-port/tool-parity/` in the private `ai-research` repository for the full behavioral outline and source map. Other Pi surfaces remain a living, non-exhaustive inventory. Add deterministic Rust tests as items are ported; don't mark an item done on build success alone.

## Agent loop and AI/provider behavior

- [x] Continue tool-call rounds until the model stops or the request is cancelled; Mu no longer applies a fixed 8-round limit.
- [ ] Handle interrupted/refused provider outputs and all incomplete response cases. Mu now recognizes `response.incomplete` and rejects tool calls when `incomplete_details.reason` is `max_output_tokens`, returning Pi's retry-the-call error as a tool result.
- [ ] Match Pi steering/follow-up interleavings at request, tool-call, error, and cancellation boundaries; current queue is FIFO per lane, but broader interleaving tests are needed.
- [ ] Add finish-turn/prepare-request/prepare-next-turn hooks and their ordering/cancellation semantics.
- [ ] Match tool approval, validation, before/after tool hooks, progress updates, and tool termination behavior.
- [ ] Match Pi's per-tool sequential execution mode while retaining parallel execution for independent calls. Mu currently serializes an entire batch when it contains stateful `tool_search` or `codemode`.
- [ ] Add other Pi-supported provider transports, authentication, request/response formats, streaming events, usage/cost handling, and multimodal content. Mu currently targets the configured OpenCode Cloudflare Responses path.
- [ ] Implement provider-token/OAuth credential lookup and login/logout flows.

## CLI and session behavior

- [ ] Implement remaining session workflows and CLI forms: Pi-style searchable session picker/previews, cross-project session-ID lookup/fork prompt, clone, branch navigation/tree selection, and import/export. Mu now supports a numbered `--resume` picker, `--continue`, `--session <path|local-id-prefix>`, exact `--session-id` (opens or creates), `--fork <path|local-id-prefix>`, custom `--session-dir`, `--name`, and in-memory `--no-session`; loaded/forked files preserve active-branch history and MCP/codemode state.
- [ ] Port Pi session compaction, branch selection, summaries, and context projection behavior end to end.

## Built-in tools and skills

### Shared tool contract and activation

- [ ] Finish the built-in result contract: Mu now carries plain text plus optional `details`, `structured_content`, and `is_error`; still add ordered text/image content blocks, progress/cancellation, and render state, then preserve all fields through agent history and Pi session JSONL.
- [ ] Complete built-in activation: all eight Pi tool schemas, including PowerShell, are exposed and the name/schema spec is active; still implement `defaultTools`/platform selection semantics and port default-tools-setting and full invalid-input cases.
- [ ] Implement Pi tool loadouts (coding/read-only/all), custom/MCP tool allowlists, explicit-option precedence, reload/resume restoration, and platform-specific Bash/PowerShell choice. Mu now merges settings-layer `defaultTools`, defaults to Pi's four coding tools, and applies `--tools` name/pattern allowlists or ordered `+name`/`-name` modifiers to available built-in and MCP definitions. `--no-tools` disables built-in and MCP calls; `--no-builtin-tools` retains MCP, codemode, and tool-search tools.
- [ ] Complete strict-sampling parity: Mu now sends `strict: true` and converts optional properties to required nullable schemas for Pi's `read`, `bash`, `edit`, and `write` tools; still add provider capability detection/fallback, explicit opt-outs, and exact unsupported-schema behavior. Built-in tool snippets and available guidelines are now selected into request instructions; still port extension/MCP contributions and cwd-from-execution-context behavior.

### Paths and file tools

- [ ] Port remaining `path-utils.ts`: tool paths now resolve absolute/relative inputs, `~`, `@`, Unicode spaces, and file URLs; `read` retries macOS NFD, curly-quote, and screenshot AM/PM variants. Still port canonicalization/revisions, local-path checks, Windows shell paths, cwd-relative formatting, and exact utility tests.
- [ ] Complete `read` text parity: Mu now handles byte/line caps, offsets/limits, fractional numeric slice semantics, out-of-range errors, continuation text, first-line-too-long fallback, and Pi-shaped truncation details; empty files, trailing newlines, and zero-limit boundaries now have explicit parity tests. Continue with exact boundary semantics and a table-driven port of all Pi truncation cases.
- [ ] Implement `read` image parity: MIME detection by file magic, JPEG/PNG/GIF/WebP/BMP support, BMP conversion, unsupported/corrupt image behavior, resize/model limits, vision-capability handling, and ordered image/text blocks.
- [ ] Complete `write` behavior/error parity: success text now uses the requested path; writes and edits share canonical-path async serialization, including symlink aliases. Still port Pi-specific filesystem errors and deterministic same-path/queued-cancellation tests.
- [ ] Finish `edit` argument/error parity: Mu now accepts JSON-stringified arrays, object-shaped edits, mixed legacy fields, and reports Pi-shaped ENOENT/EACCES codes; still port the remaining exact per-case errors and no-op failure semantics.
- [x] Implement `edit` fuzzy matching (NFKC, trailing whitespace, smart quotes/dashes, special spaces), normalized duplicate detection, untouched-line preservation, and exact-first matching; deterministic unit tests cover each category.
- [ ] Finish `edit` output/error parity: execution now returns unified patch, collapsed display diff, first changed line, and preserves no-partial-write behavior; still add non-mutating diff preview, Pi-shaped ENOENT/EACCES/custom-operation errors, and applicable-patch/failure edge tests.

### Shell tools

- [ ] Complete Bash parity: Mu now returns Pi-style nonzero and signal/null-exit results, strips ANSI/binary controls, U+FFF9–U+FFFB, and carriage returns, captures partial output on timeout, terminates the Unix command process group on timeout, uses a 1 MiB-capped structured output, validates/enforces optional timeouts, and tail-truncates model-facing output to Pi's 2,000-line/50 KiB limits while persisting full output and metadata; still add streamed UTF-8 updates, progress coalescing, structured timeout/abort results, and cancellation-time process-tree cleanup.
- [ ] Match remaining Bash lifecycle behavior: Mu now follows Pi's Unix shell order (`/bin/bash`, `bash` on PATH, then `sh`) and discovers Git Bash/`bash.exe` on Windows, injects `PI_SESSION_ID`, `PI_SESSION_FILE`, `PI_PROVIDER`, and `PI_MODEL` into direct/parallel/codemode shell executions, preserves partial output on timeout, kills Unix process groups on timeout, and returns Pi's signal/null-exit error without structured content; still port abort output/partial results, invalid/missing cwd, spawn errors, Windows process-tree termination, cancellation-time cleanup, inherited-stdio late-child cleanup, explicit shellPath/WSL transport, environment opt-out, reasoning level, command prefix, and spawn hooks.
- [ ] Complete PowerShell parity: Mu now exposes the tool, prefers `pwsh.exe` over `powershell.exe` on PATH, matches Pi's missing-executable error and no-profile/non-interactive/bypass flags, sets process-local UTF-8 output, and shares structured shell results; discovery preference/error has platform-independent tests, but Windows execution and remaining timeout, cancellation, hooks, and process lifecycle behavior still need coverage.
- [ ] Port the Bash/PowerShell lifecycle, truncation, progress, shell-selection, output persistence, and Windows tests using injectable operations/fakes where possible; keep live local process tests small and platform-gated.

### Search and listing tools

- [ ] Complete `grep` parity: Mu now supports regex/literal search, ignore-case, glob, `--`-safe patterns, `.gitignore`/hidden traversal, nested-repository ignore boundaries, sorted file traversal, context, exact next-match limit detection, JavaScript fractional/negative limit semantics, match/line truncation details, and Pi-shaped byte-truncation metadata; align remaining ripgrep edge behavior, execution-context cwd, and cancellation.
- [ ] Complete `find` parity: Mu now supports hidden files, `.gitignore`, nested-repository ignore boundaries, basic glob/path matching, normalized cwd paths, relative POSIX output, Pi-shaped result-limit and byte-truncation details, deterministic filename ordering matching fd's default sort, and symlink path formatting; still align remaining nested-repository/glob edge cases, execution-context cwd, and cancellation.
- [ ] Complete `ls` parity: Mu now lists dotfiles, marks directories, sorts entries, resolves normalized cwd paths, and returns entry-limit and Pi-shaped byte-truncation details; add execution-context cwd override, cancellation, and remaining error/path semantics.
- [ ] Use injectable filesystem/search operations for deterministic tests; do not make required parity tests depend on downloads, network access, or installed `rg`/`fd` binaries.
- [ ] Port every relevant case in pinned `tools.test.ts`, `edit-tool-legacy-input.test.ts`, `path-utils.test.ts`, `default-tools-setting.test.ts`, `builtin-tool-strict-mode.test.ts`, `tool-system-prompt-contributions.test.ts`, `tool-result-images.test.ts`, `powershell-tool.test.ts`, plus focused Bash/process/renderer suites. Activate ignored Mu specs only when implementation and assertions match.

### Skills

- [ ] Complete skill activation/command behavior, argument handling, and parity for discovery precedence, frontmatter, and content injection.

## MCP and codemode

- [ ] OAuth and provider-token authentication for MCP servers.
- [ ] MCP resources, resource templates, and resource reads.
- [ ] Finish MCP lifecycle parity: retries/reconnect, notifications, dynamic capability/tool registration, server logs, and clean shutdown. Connections now start in the background and configured servers initialize concurrently; the first prompt waits at most 10 seconds for startup (Pi waits only for direct-exposure servers).
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

- [ ] Replace remaining plain-text rendering with Pi-compatible Markdown, transformers, custom message/entry renderers, and per-tool renderers. Mu now renders successful `SKILL.md` reads as compact `[skill] <name>` entries, hides their contents by default, and toggles tool output with Ctrl+O; other built-ins still use generic rendering.
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
