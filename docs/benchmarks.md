# Benchmarks

Hot-path benchmarks live in `crates/mu-bench` (fixtures in `src/lib.rs`, Criterion groups in
`benches/hot_paths.rs`). They record numbers; nothing gates on them.

```sh
cargo bench -p mu-bench                       # full run
cargo bench -p mu-bench -- --test            # smoke: run each benchmark once
cargo bench -p mu-bench -- tool_search       # one group
```

Crate-private paths (provider decoders, `tool_search` ranking, MCP output limiting) are reached
through `mu_core::bench_internals`, compiled only with mu-core's `bench-internals` feature.

## Baseline (2026-10-24, before the refactor)

Apple M4 Pro, rustc 1.99.0, `--warm-up-time 1 --measurement-time 2`; median estimates.
Workspace at `081d6a5`: 527 tests passing, Clippy clean with `-D warnings`, incremental
`cargo build --workspace --all-targets` after touching `mu-session`: 54.8 s wall.

| Benchmark | Median |
| --- | --- |
| `session/parse_jsonl/50` | 863.35 µs |
| `session/to_jsonl/50` | 602.02 µs |
| `session/active_branch/50` | 30.599 µs |
| `session/parse_jsonl/500` | 9.3479 ms |
| `session/to_jsonl/500` | 7.7732 ms |
| `session/active_branch/500` | 313.84 µs |
| `provider_stream/sse_frames_1k_chunks` | 551.42 µs |
| `provider_stream/completions_decode_2000` | 1.0275 ms |
| `provider_stream/anthropic_decode_2000` | 658.10 µs |
| `edit_diff/exact_2000_lines` | 5.1046 ms |
| `edit_diff/fuzzy_2000_lines` | 13.022 ms |
| `markdown/render/40` | 2.7757 ms |
| `markdown/render/120` | 2.6398 ms |
| `markdown/wrap_spans_10k_chars_80_cols` | 427.15 µs |
| `tool_search/rank/50` | 1.2532 ms |
| `tool_search/rank/500` | 43.976 ms |
| `compaction/estimate_tokens_1000_entries` | 907.26 µs |
| `compaction/find_cut_point_1000_entries` | 374.15 µs |
| `mcp_output/limit/16384` | 567.89 ns |
| `mcp_output/limit/1048576` | 480.99 µs |

Notable: `tool_search/rank/500` (~44 ms) grows far faster than linearly from 50 tools (~1.3 ms);
fuzzy edit matching costs ~2.5x an exact edit.
