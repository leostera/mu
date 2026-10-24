//! Hot-path benchmarks. Run with `cargo bench -p mu-bench`; numbers are recorded, not gated.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mu_bench as fixtures;
use mu_cli::tui::markdown::{MarkdownOptions, MarkdownStyle, render_markdown, wrap_spans};
use mu_cli::tui::theme::Theme;
use mu_core::{bench_internals, compaction, sse::SseFrames, tools};
use mu_session::SessionDocument;
use ratatui::text::Span;

fn session_jsonl(c: &mut Criterion) {
    let mut group = c.benchmark_group("session");
    for turns in [50, 500] {
        let session = fixtures::session(turns);
        let jsonl = session.to_jsonl().unwrap();
        group.throughput(Throughput::Bytes(jsonl.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("parse_jsonl", turns),
            &jsonl,
            |b, jsonl| b.iter(|| SessionDocument::parse_jsonl(black_box(jsonl)).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("to_jsonl", turns),
            &session,
            |b, session| b.iter(|| black_box(session).to_jsonl().unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("active_branch", turns),
            &session,
            |b, session| b.iter(|| black_box(session).active_branch_entries().len()),
        );
    }
    group.finish();
}

fn provider_streams(c: &mut Criterion) {
    let mut group = c.benchmark_group("provider_stream");
    let stream = fixtures::completions_sse(2_000);
    group.throughput(Throughput::Bytes(stream.len() as u64));
    // Network chunks rarely align with frames; 1 KiB chunks exercise split terminators.
    group.bench_function("sse_frames_1k_chunks", |b| {
        b.iter(|| {
            let mut frames = SseFrames::default();
            let mut count = 0;
            for chunk in black_box(&stream).chunks(1024) {
                frames.push(chunk);
                while frames.next_data().is_some() {
                    count += 1;
                }
            }
            count
        })
    });
    let chunks = {
        let mut frames = SseFrames::default();
        frames.push(&stream);
        std::iter::from_fn(|| frames.next_data())
            .filter(|data| data != "[DONE]")
            .map(|data| serde_json::from_str(&data).unwrap())
            .collect::<Vec<_>>()
    };
    group.bench_function("completions_decode_2000", |b| {
        b.iter(|| bench_internals::decode_completions_chunks(black_box(&chunks)))
    });
    let events = fixtures::anthropic_events(2_000);
    group.bench_function("anthropic_decode_2000", |b| {
        b.iter(|| bench_internals::decode_anthropic_events(black_box(&events)))
    });
    group.finish();
}

fn edit_tool(c: &mut Criterion) {
    let dir = std::env::temp_dir().join(format!("mu-bench-edit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("file.rs"), fixtures::source_file(2_000)).unwrap();
    let edit = |old: &str, new: &str| tools::TextEdit {
        old_text: old.into(),
        new_text: new.into(),
    };
    let exact = [edit(
        "    let value_1500 = compute(1500);",
        "    let value_1500 = compute(0);",
    )];
    // Straight quotes against the file's curly ones force the fuzzy-normalized match.
    let fuzzy = [edit(
        "compute(1500); // \"note\" 1500",
        "compute(0); // \"note\" 1500",
    )];
    for edits in [&exact, &fuzzy] {
        let diff = tools::compute_edit_diff(&dir, "file.rs", edits);
        assert!(diff.get("error").is_none(), "benchmark edit must apply: {diff}");
    }
    let mut group = c.benchmark_group("edit_diff");
    group.bench_function("exact_2000_lines", |b| {
        b.iter(|| tools::compute_edit_diff(&dir, "file.rs", black_box(&exact)))
    });
    group.bench_function("fuzzy_2000_lines", |b| {
        b.iter(|| tools::compute_edit_diff(&dir, "file.rs", black_box(&fuzzy)))
    });
    group.finish();
    std::fs::remove_dir_all(dir).unwrap();
}

fn markdown(c: &mut Criterion) {
    let theme = Theme::dark();
    let style = MarkdownStyle::from_theme(&theme);
    let options = MarkdownOptions::default();
    let document = fixtures::markdown_document(40);
    let mut group = c.benchmark_group("markdown");
    group.throughput(Throughput::Bytes(document.len() as u64));
    for width in [40, 120] {
        group.bench_with_input(BenchmarkId::new("render", width), &width, |b, &width| {
            b.iter(|| render_markdown(black_box(&document), width, &style, &options).len())
        });
    }
    let line = vec![Span::raw("word ".repeat(2_000))];
    group.bench_function("wrap_spans_10k_chars_80_cols", |b| {
        b.iter(|| wrap_spans(black_box(&line), 80).len())
    });
    group.finish();
}

fn tool_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("tool_search");
    for count in [50, 500] {
        let tools = fixtures::tool_definitions(count);
        group.bench_with_input(BenchmarkId::new("rank", count), &tools, |b, tools| {
            b.iter(|| bench_internals::rank_tools(black_box("search records project"), tools, 8))
        });
    }
    group.finish();
}

fn compaction_paths(c: &mut Criterion) {
    let entries = fixtures::session_entries(&fixtures::session(500));
    let mut group = c.benchmark_group("compaction");
    group.bench_function("estimate_tokens_1000_entries", |b| {
        b.iter(|| {
            black_box(&entries)
                .iter()
                .filter_map(|entry| entry.get("message"))
                .map(compaction::estimate_tokens)
                .sum::<u64>()
        })
    });
    group.bench_function("find_cut_point_1000_entries", |b| {
        b.iter(|| compaction::find_cut_point(black_box(&entries), 0, entries.len(), 20_000))
    });
    group.finish();
}

fn mcp_output(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcp_output");
    for size in [16 * 1024, 1024 * 1024] {
        let text = "line of tool output with some width\n".repeat(size / 36);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::new("limit", size), &text, |b, text| {
            b.iter(|| bench_internals::limit_mcp_text(black_box(text)).len())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    session_jsonl,
    provider_streams,
    edit_tool,
    markdown,
    tool_search,
    compaction_paths,
    mcp_output
);
criterion_main!(benches);
