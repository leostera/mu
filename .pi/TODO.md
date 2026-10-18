# Mu parity TODO

Known behavior gaps against pinned Pi revision `698a2d23b86a5045dd54554354a5e0cb7e466795` (Pi 1.3.0). The built-in tool surface has been source-audited; see `personal/pi/rust-port/tool-parity/` in the private `ai-research` repository for the full behavioral outline and source map. Other Pi surfaces remain a living, non-exhaustive inventory. Add deterministic Rust tests as items are ported; don't mark an item done on build success alone.

## Agent loop and AI/provider behavior

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

### Shared tool contract and activation

- [ ] Finish the built-in result contract: Mu now carries plain text plus optional `details`, `structured_content`, and `is_error`; still add ordered text/image content blocks, progress/cancellation, and render state, then preserve all fields through agent history and Pi session JSONL.
- [ ] Expose all eight Pi built-ins (`read`, `write`, `edit`, `bash`, `grep`, `find`, `ls`, `powershell`) with Pi-compatible JSON schemas; port `default-tools-setting.test.ts` and schema/invalid-input cases.
- [ ] Implement Pi tool loadouts (coding/read-only/all), `defaultTools`, `+name`/`-name` modifiers, explicit-option precedence, custom-tool activation, reload/resume restoration, and platform-specific Bash/PowerShell choice.
- [ ] Port strict-sampling metadata and opt-outs for the matching built-ins; keep optional schema fields optional. Port per-tool prompt snippets/guidelines and cwd-from-execution-context behavior.

### Paths and file tools

- [ ] Port remaining `path-utils.ts`: tool paths now resolve absolute/relative inputs, `~`, `@`, Unicode spaces, and file URLs; `read` retries macOS NFD, curly-quote, and screenshot AM/PM variants. Still port canonicalization/revisions, local-path checks, Windows shell paths, cwd-relative formatting, and exact utility tests.
- [ ] Complete `read` text parity: Mu now handles byte/line caps, offsets/limits, out-of-range errors, continuation text, first-line-too-long fallback, and Pi-shaped truncation details; continue matching exact boundary/empty-file semantics and add a table-driven port of all Pi truncation cases.
- [ ] Implement `read` image parity: MIME detection by file magic, JPEG/PNG/GIF/WebP/BMP support, BMP conversion, unsupported/corrupt image behavior, resize/model limits, vision-capability handling, and ordered image/text blocks.
- [ ] Complete `write` behavior and success/error messages; share cancellation-safe canonical per-path mutation serialization with `edit` (same-path order, distinct-path concurrency, symlink aliases, queued/in-flight aborts).
- [ ] Finish `edit` argument/error parity: Mu now accepts JSON-stringified arrays, object-shaped edits, and mixed legacy fields; still port exact per-case Pi errors and no-op failure semantics.
- [x] Implement `edit` fuzzy matching (NFKC, trailing whitespace, smart quotes/dashes, special spaces), normalized duplicate detection, untouched-line preservation, and exact-first matching; deterministic unit tests cover each category.
- [ ] Finish `edit` output/error parity: execution now returns unified patch, collapsed display diff, first changed line, and preserves no-partial-write behavior; still add non-mutating diff preview, Pi-shaped ENOENT/EACCES/custom-operation errors, and applicable-patch/failure edge tests.

### Shell tools

- [ ] Complete Bash parity: Mu now returns Pi-style nonzero error results and 1 MiB-capped structured output with exit code/duration; still add streamed UTF-8 output, model-facing line/byte truncation and full-output persistence/path, progress coalescing, timeout, cancellation, and the remaining process lifecycle.
- [ ] Match Bash timeout/abort output, invalid/missing cwd, spawn errors, signal/null-exit handling, process-tree termination, inherited-stdio late-child cleanup, shell path/transport/WSL behavior, command prefix, environment exposure, and spawn hooks.
- [ ] Implement the PowerShell tool on the shared shell contract, including process-local UTF-8 setup and Windows execution-policy behavior.
- [ ] Port the Bash/PowerShell lifecycle, truncation, progress, shell-selection, output persistence, and Windows tests using injectable operations/fakes where possible; keep live local process tests small and platform-gated.

### Search and listing tools

- [ ] Complete `grep` parity: Mu now supports regex/literal search, ignore-case, glob, `--`-safe patterns, `.gitignore`/hidden traversal, context, exact next-match limit detection, and basic line/byte truncation; align ripgrep edge behavior, structured truncation details, execution-context cwd, and cancellation.
- [ ] Complete `find` parity: Mu now supports hidden files, `.gitignore`, basic glob/path matching, relative POSIX output, and result/byte limits; align fd's nested-repository and glob semantics, deterministic ordering, structured limit/truncation details, execution-context cwd, and cancellation.
- [ ] Complete `ls` parity: Mu now lists dotfiles, marks directories, sorts entries, and handles basic limits/errors; add Pi's structured truncation/entry-limit details, execution-context cwd override, cancellation, and remaining error/path semantics.
- [ ] Use injectable filesystem/search operations for deterministic tests; do not make required parity tests depend on downloads, network access, or installed `rg`/`fd` binaries.
- [ ] Port every relevant case in pinned `tools.test.ts`, `edit-tool-legacy-input.test.ts`, `path-utils.test.ts`, `default-tools-setting.test.ts`, `builtin-tool-strict-mode.test.ts`, `tool-system-prompt-contributions.test.ts`, `tool-result-images.test.ts`, `powershell-tool.test.ts`, plus focused Bash/process/renderer suites. Activate ignored Mu specs only when implementation and assertions match.

### Skills

- [ ] Complete skill activation/command behavior, argument handling, and parity for discovery precedence, frontmatter, and content injection.

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
