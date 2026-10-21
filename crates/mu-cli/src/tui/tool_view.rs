//! Tool call and result boxes, following Pi's `ToolExecutionComponent` and the built-in
//! renderers in `core/tools/renderers/`.
//!
//! Syntax highlighting of `read`/`write` bodies, images, and live "Elapsed" timers are not
//! implemented; results use the tool-output colors.

use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};
use serde_json::Value;
use similar::{ChangeTag, TextDiff};
use unicode_width::UnicodeWidthStr;

use super::{markdown::wrap_spans, theme::Theme};

/// What the renderer needs to know about one tool execution.
#[derive(Clone, Debug)]
pub struct ToolView<'a> {
    pub name: &'a str,
    pub arguments: &'a Value,
    /// `None` while the tool is still running.
    pub output: Option<&'a str>,
    pub details: Option<&'a Value>,
    pub is_error: bool,
    pub duration_ms: Option<u64>,
    pub expanded: bool,
    pub home: Option<&'a str>,
}

type Spans = Vec<Span<'static>>;

const OUTPUT_PAD: usize = 1;
const FALLBACK_PREVIEW_LINES: usize = 10;
const BASH_PREVIEW_LINES: usize = 5;
const COLLAPSED_ARGS_CHARS: usize = 100;
const DEFAULT_MAX_BYTES: u64 = 50 * 1024;
const DEFAULT_MAX_LINES: u64 = 2000;

/// The full box: a spacer row, then padded content on the pending/success/error background.
pub fn tool_lines(view: &ToolView<'_>, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let background = if view.output.is_none() {
        theme.color("toolPendingBg")
    } else if view.is_error {
        theme.color("toolErrorBg")
    } else {
        theme.color("toolSuccessBg")
    };
    let inner = width.saturating_sub(OUTPUT_PAD * 2).max(1);
    let mut content = call_lines(view, theme);
    if view.output.is_some() {
        content.extend(result_lines(view, inner, theme));
    }
    let rows = content
        .into_iter()
        .flat_map(|line| wrap_spans(&line, inner))
        .collect::<Vec<_>>();

    let filled = |spans: Spans| {
        let used: usize = spans.iter().map(|span| span.content.width()).sum();
        let mut row = vec![Span::raw(" ".repeat(OUTPUT_PAD))];
        row.extend(spans);
        row.push(Span::raw(
            " ".repeat(width.saturating_sub(used + OUTPUT_PAD)),
        ));
        Line::from(row).style(Style::default().bg(background))
    };
    let mut lines = vec![Line::from("")];
    lines.push(filled(Vec::new()));
    lines.extend(rows.into_iter().map(filled));
    lines.push(filled(Vec::new()));
    lines
}

fn call_lines(view: &ToolView<'_>, theme: &Theme) -> Vec<Spans> {
    match view.name {
        "bash" => vec![shell_call("$", view.arguments, theme)],
        "powershell" => vec![shell_call("PS>", view.arguments, theme)],
        "read" => read_call(view, theme),
        "write" => write_call(view, theme),
        "edit" => edit_call(view, theme),
        "grep" => vec![grep_call(view.arguments, view.home, theme)],
        "find" => vec![find_call(view.arguments, view.home, theme)],
        "ls" => vec![ls_call(view.arguments, view.home, theme)],
        name => generic_call(name, view.arguments, view.expanded, theme),
    }
}

fn result_lines(view: &ToolView<'_>, width: usize, theme: &Theme) -> Vec<Spans> {
    let output = sanitize_output(view.output.unwrap_or_default());
    match view.name {
        "bash" | "powershell" => shell_result(view, &output, width, theme),
        "read" => read_result(view, &output, theme),
        "write" => error_only(view, &output, theme),
        "edit" => edit_result(view, &output, theme),
        "grep" => listing_result(view, &output, 15, theme, grep_warnings),
        "find" => listing_result(view, &output, 20, theme, find_warnings),
        "ls" => listing_result(view, &output, 20, theme, ls_warnings),
        _ => generic_result(view, &output, theme),
    }
}

// ---------------------------------------------------------------------------
// Shared formatting (`render-utils.ts`)
// ---------------------------------------------------------------------------

fn bold_title(text: impl Into<String>, theme: &Theme) -> Span<'static> {
    Span::styled(
        text.into(),
        theme.fg("toolTitle").add_modifier(Modifier::BOLD),
    )
}

fn muted(text: impl Into<String>, theme: &Theme) -> Span<'static> {
    Span::styled(text.into(), theme.fg("muted"))
}

fn warning(text: impl Into<String>, theme: &Theme) -> Span<'static> {
    Span::styled(text.into(), theme.fg("warning"))
}

/// Strip ANSI sequences and carriage returns from tool text, as Pi's `getTextOutput` does.
fn sanitize_output(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\u{1b}' => {
                if chars.peek() == Some(&'[') {
                    chars.next();
                    for next in chars.by_ref() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                }
            }
            '\r' => {}
            other => output.push(other),
        }
    }
    output
}

fn replace_tabs(text: &str) -> String {
    text.replace('\t', "   ")
}

pub fn shorten_path(path: &str, home: Option<&str>) -> String {
    match home {
        Some(home) if !home.is_empty() && path.starts_with(home) => {
            format!("~{}", &path[home.len()..])
        }
        _ => path.to_owned(),
    }
}

/// `str()` from render-utils: strings pass, missing/null is empty, anything else is invalid.
fn string_arg<'a>(arguments: &'a Value, keys: &[&str]) -> Option<&'a str> {
    for key in keys {
        match arguments.get(*key) {
            Some(Value::String(text)) => return Some(text),
            Some(Value::Null) | None => {}
            Some(_) => return None,
        }
    }
    Some("")
}

fn invalid_arg(theme: &Theme) -> Span<'static> {
    Span::styled("[invalid arg]", theme.fg("error"))
}

fn tool_path(
    raw: Option<&str>,
    home: Option<&str>,
    theme: &Theme,
    fallback: Option<&str>,
) -> Span<'static> {
    match raw {
        None => invalid_arg(theme),
        Some(path) => match (path.is_empty(), fallback) {
            (false, _) => Span::styled(shorten_path(path, home), theme.fg("accent")),
            (true, Some(fallback)) => {
                Span::styled(shorten_path(fallback, home), theme.fg("accent"))
            }
            (true, None) => Span::styled("...", theme.fg("toolOutput")),
        },
    }
}

fn limit_arg(arguments: &Value) -> Option<String> {
    match arguments.get("limit") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(other) => Some(other.to_string()),
    }
}

pub fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

pub fn format_tool_duration(ms: u64) -> String {
    let seconds = ms as f64 / 1000.0;
    if seconds < 60.0 {
        // JavaScript's toFixed rounds exact ties (x.25, x.75) up; Rust rounds them to even.
        if ms % 500 == 250 {
            let tenths = ms / 100 + 1;
            return format!("{}.{}s", tenths / 10, tenths % 10);
        }
        return format!("{seconds:.1}s");
    }
    let total = seconds.floor() as u64;
    let (minutes, remainder) = (total / 60, total % 60);
    if minutes < 60 {
        format!("{minutes}m {remainder}s")
    } else {
        format!("{}h {}m {}s", minutes / 60, minutes % 60, remainder)
    }
}

fn expand_hint(theme: &Theme) -> Vec<Span<'static>> {
    vec![
        Span::styled("ctrl+o", theme.fg("dim")),
        muted(" to expand", theme),
    ]
}

fn more_lines_hint(remaining: usize, theme: &Theme) -> Spans {
    let mut spans = vec![
        muted(format!("... ({remaining} more lines,"), theme),
        Span::raw(" "),
    ];
    spans.extend(expand_hint(theme));
    spans.push(muted(")", theme));
    spans
}

fn text_lines(text: &str, style: Style) -> Vec<Spans> {
    text.split('\n')
        .map(|line| vec![Span::styled(replace_tabs(line), style)])
        .collect()
}

fn trim_trailing_empty(mut lines: Vec<String>) -> Vec<String> {
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    lines
}

fn timer_line(view: &ToolView<'_>, theme: &Theme) -> Option<Spans> {
    view.duration_ms
        .map(|ms| vec![muted(format!("Took {}", format_tool_duration(ms)), theme)])
}

fn truncation_details<'a>(view: &ToolView<'a>) -> Option<&'a Value> {
    view.details
        .and_then(|details| details.get("truncation"))
        .filter(|truncation| truncation["truncated"] == true)
}

fn byte_limit(truncation: &Value) -> u64 {
    truncation["maxBytes"].as_u64().unwrap_or(DEFAULT_MAX_BYTES)
}

// ---------------------------------------------------------------------------
// Generic (extension and MCP tools)
// ---------------------------------------------------------------------------

fn generic_call(name: &str, arguments: &Value, expanded: bool, theme: &Theme) -> Vec<Spans> {
    let header = bold_title(name, theme);
    let entries: Vec<(String, Value)> = match arguments {
        Value::Null => return vec![vec![header]],
        Value::Object(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        other => vec![("args".to_owned(), other.clone())],
    };
    if entries.is_empty() {
        return vec![vec![header]];
    }
    if expanded {
        let mut lines = vec![vec![header]];
        for (key, value) in entries {
            let text = match &value {
                Value::String(text) => text.clone(),
                other => serde_json::to_string_pretty(other).unwrap_or_default(),
            };
            let text = replace_tabs(&text)
                .replace('\r', "")
                .replace('\n', "\n    ");
            for line in format!("  {key}: {text}").split('\n') {
                lines.push(vec![muted(line.to_owned(), theme)]);
            }
        }
        return lines;
    }
    let pairs = entries
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(" ");
    let preview = if pairs.chars().count() > COLLAPSED_ARGS_CHARS {
        format!(
            "{}...",
            pairs
                .chars()
                .take(COLLAPSED_ARGS_CHARS - 3)
                .collect::<String>()
        )
    } else {
        pairs
    };
    vec![vec![header, Span::raw(" "), muted(preview, theme)]]
}

fn generic_result(view: &ToolView<'_>, output: &str, theme: &Theme) -> Vec<Spans> {
    if output.is_empty() {
        return Vec::new();
    }
    let lines = output.split('\n').collect::<Vec<_>>();
    let shown = if view.expanded {
        lines.len()
    } else {
        lines.len().min(FALLBACK_PREVIEW_LINES)
    };
    let mut rendered = lines[..shown]
        .iter()
        .map(|line| vec![Span::styled((*line).to_owned(), theme.fg("toolOutput"))])
        .collect::<Vec<_>>();
    if lines.len() > shown {
        rendered.push(more_lines_hint(lines.len() - shown, theme));
    }
    rendered
}

// ---------------------------------------------------------------------------
// bash / powershell
// ---------------------------------------------------------------------------

fn shell_call(prompt: &str, arguments: &Value, theme: &Theme) -> Spans {
    let command = string_arg(arguments, &["command"]);
    let display = match command {
        None => return vec![invalid_arg(theme)],
        Some("") => "...".to_owned(),
        Some(command) => command.to_owned(),
    };
    let title_style = theme.fg("toolTitle").add_modifier(Modifier::BOLD);
    let mut spans = vec![if command == Some("") {
        Span::styled(format!("{prompt} "), title_style)
    } else {
        Span::styled(format!("{prompt} {display}"), title_style)
    }];
    if command == Some("") {
        spans.push(Span::styled("...", theme.fg("toolOutput")));
    }
    if let Some(timeout) = arguments
        .get("timeout")
        .and_then(Value::as_f64)
        .filter(|t| *t != 0.0)
    {
        spans.push(muted(
            format!(" (timeout {}s)", trim_number(timeout)),
            theme,
        ));
    }
    spans
}

fn trim_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}

fn shell_result(view: &ToolView<'_>, output: &str, width: usize, theme: &Theme) -> Vec<Spans> {
    let mut lines: Vec<Spans> = Vec::new();
    let mut output = output.trim().to_owned();
    let truncation = truncation_details(view);
    let full_output_path = view
        .details
        .and_then(|details| details.get("fullOutputPath"))
        .and_then(Value::as_str);
    if let (Some(path), Some(_)) = (full_output_path, truncation) {
        if output.ends_with(']') {
            if let Some(footer) = output.rfind("\n\n[") {
                if output[footer..].contains(path) {
                    output = output[..footer].trim_end().to_owned();
                }
            }
        }
    }
    if !output.is_empty() {
        let styled = text_lines(&output, theme.fg("toolOutput"));
        lines.push(Vec::new());
        if view.expanded {
            lines.extend(styled);
        } else {
            let visual = styled
                .iter()
                .flat_map(|line| wrap_spans(line, width))
                .collect::<Vec<_>>();
            if visual.len() > BASH_PREVIEW_LINES {
                let hidden = visual.len() - BASH_PREVIEW_LINES;
                let mut hint = vec![
                    muted(format!("... ({hidden} earlier lines,"), theme),
                    Span::raw(" "),
                ];
                hint.extend(expand_hint(theme));
                hint.push(muted(")", theme));
                lines.push(hint);
                lines.extend(visual.into_iter().skip(hidden));
            } else {
                lines.extend(visual);
            }
        }
    }
    if truncation.is_some() || full_output_path.is_some() {
        let mut warnings = Vec::new();
        if let Some(path) = full_output_path {
            warnings.push(format!("Full output: {path}"));
        }
        if let Some(truncation) = truncation {
            if truncation["truncatedBy"] == "lines" {
                warnings.push(format!(
                    "Truncated: showing {} of {} lines",
                    truncation["outputLines"], truncation["totalLines"]
                ));
            } else {
                warnings.push(format!(
                    "Truncated: {} lines shown ({} limit)",
                    truncation["outputLines"],
                    format_size(byte_limit(truncation))
                ));
            }
        }
        lines.push(Vec::new());
        lines.push(vec![warning(format!("[{}]", warnings.join(". ")), theme)]);
    }
    if let Some(timer) = timer_line(view, theme) {
        lines.push(Vec::new());
        lines.push(timer);
    }
    lines
}

// ---------------------------------------------------------------------------
// read / write / edit
// ---------------------------------------------------------------------------

fn read_call(view: &ToolView<'_>, theme: &Theme) -> Vec<Spans> {
    let raw_path = string_arg(view.arguments, &["file_path", "path"]);
    if !view.expanded {
        if let Some(path) = raw_path.filter(|p| !p.is_empty()) {
            if let Some(compact) = compact_read(view, path, theme) {
                return vec![compact];
            }
        }
    }
    let mut spans = vec![
        bold_title("read", theme),
        Span::raw(" "),
        tool_path(raw_path, view.home, theme, None),
    ];
    spans.extend(read_range(view.arguments, theme));
    vec![spans]
}

fn parse_line_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64().filter(|n| n.is_finite()),
        Value::String(text) if !text.trim().is_empty() => {
            text.trim().parse::<f64>().ok().filter(|n| n.is_finite())
        }
        _ => None,
    }
}

fn read_range(arguments: &Value, theme: &Theme) -> Option<Span<'static>> {
    let offset = arguments.get("offset").filter(|v| !v.is_null());
    let limit = arguments.get("limit").filter(|v| !v.is_null());
    if offset.is_none() && limit.is_none() {
        return None;
    }
    let start = match offset {
        None => 1.0,
        Some(value) => parse_line_number(value)?,
    };
    let range = match limit {
        None => format!(":{}", trim_number(start)),
        Some(value) => {
            let end = start + parse_line_number(value)? - 1.0;
            format!(
                ":{}{}",
                trim_number(start),
                if end != 0.0 {
                    format!("-{}", trim_number(end))
                } else {
                    String::new()
                }
            )
        }
    };
    Some(warning(range, theme))
}

fn compact_read(view: &ToolView<'_>, path: &str, theme: &Theme) -> Option<Spans> {
    let file_name = path.rsplit('/').next().unwrap_or(path);
    let hint = {
        let mut spans = vec![Span::styled(" (", theme.fg("dim"))];
        spans.push(Span::styled("ctrl+o to expand)", theme.fg("dim")));
        spans
    };
    if file_name == "SKILL.md" {
        let label = path
            .trim_end_matches("/SKILL.md")
            .rsplit('/')
            .next()
            .filter(|label| !label.is_empty())
            .unwrap_or("SKILL.md");
        let mut spans = vec![
            Span::styled(
                "[skill] ",
                theme.fg("customMessageLabel").add_modifier(Modifier::BOLD),
            ),
            Span::styled(label.to_owned(), theme.fg("customMessageText")),
        ];
        spans.extend(read_range(view.arguments, theme));
        spans.extend(hint);
        return Some(spans);
    }
    const RESOURCE_FILES: [&str; 5] = [
        "AGENTS.override.md",
        "AGENTS.md",
        "AGENTS.MD",
        "CLAUDE.md",
        "CLAUDE.MD",
    ];
    if RESOURCE_FILES.contains(&file_name) {
        let mut spans = vec![
            bold_title("read resource", theme),
            Span::raw(" "),
            Span::styled(shorten_path(path, view.home), theme.fg("accent")),
        ];
        spans.extend(read_range(view.arguments, theme));
        spans.extend(hint);
        return Some(spans);
    }
    None
}

fn read_result(view: &ToolView<'_>, output: &str, theme: &Theme) -> Vec<Spans> {
    if !view.expanded && !view.is_error {
        return Vec::new();
    }
    let lines = trim_trailing_empty(output.split('\n').map(str::to_owned).collect());
    let max = if view.expanded { lines.len() } else { 10 };
    let mut rendered = vec![Vec::new()];
    rendered.extend(
        lines[..lines.len().min(max)]
            .iter()
            .map(|line| vec![Span::styled(replace_tabs(line), theme.fg("toolOutput"))]),
    );
    if lines.len() > max {
        rendered.push(more_lines_hint(lines.len() - max, theme));
    }
    if let Some(truncation) = truncation_details(view) {
        let text = if truncation["firstLineExceedsLimit"] == true {
            format!(
                "[First line exceeds {} limit]",
                format_size(byte_limit(truncation))
            )
        } else if truncation["truncatedBy"] == "lines" {
            format!(
                "[Truncated: showing {} of {} lines ({} line limit)]",
                truncation["outputLines"],
                truncation["totalLines"],
                truncation["maxLines"].as_u64().unwrap_or(DEFAULT_MAX_LINES)
            )
        } else {
            format!(
                "[Truncated: {} lines shown ({} limit)]",
                truncation["outputLines"],
                format_size(byte_limit(truncation))
            )
        };
        rendered.push(vec![warning(text, theme)]);
    }
    rendered
}

fn write_call(view: &ToolView<'_>, theme: &Theme) -> Vec<Spans> {
    let raw_path = string_arg(view.arguments, &["file_path", "path"]);
    let content = string_arg(view.arguments, &["content"]);
    let mut lines = vec![vec![
        bold_title("write", theme),
        Span::raw(" "),
        tool_path(raw_path, view.home, theme, None),
    ]];
    match content {
        None => {
            lines.push(Vec::new());
            lines.push(vec![Span::styled(
                "[invalid content arg - expected string]",
                theme.fg("error"),
            )]);
        }
        Some("") => {}
        Some(content) => {
            let all = trim_trailing_empty(
                content
                    .replace('\r', "")
                    .split('\n')
                    .map(str::to_owned)
                    .collect(),
            );
            let total = all.len();
            let max = if view.expanded { total } else { 10 };
            lines.push(Vec::new());
            lines.extend(
                all[..total.min(max)]
                    .iter()
                    .map(|line| vec![Span::styled(replace_tabs(line), theme.fg("toolOutput"))]),
            );
            if total > max {
                let mut hint = vec![
                    muted(
                        format!("... ({} more lines, {total} total,", total - max),
                        theme,
                    ),
                    Span::raw(" "),
                ];
                hint.extend(expand_hint(theme));
                hint.push(muted(")", theme));
                lines.push(hint);
            }
        }
    }
    lines
}

fn error_only(view: &ToolView<'_>, output: &str, theme: &Theme) -> Vec<Spans> {
    if !view.is_error || output.is_empty() {
        return Vec::new();
    }
    let mut lines = vec![Vec::new()];
    lines.extend(text_lines(output, theme.fg("error")));
    lines
}

fn edit_call(view: &ToolView<'_>, theme: &Theme) -> Vec<Spans> {
    let raw_path = string_arg(view.arguments, &["file_path", "path"]);
    vec![vec![
        bold_title("edit", theme),
        Span::raw(" "),
        tool_path(raw_path, view.home, theme, None),
    ]]
}

fn edit_result(view: &ToolView<'_>, output: &str, theme: &Theme) -> Vec<Spans> {
    if view.is_error {
        if output.is_empty() {
            return Vec::new();
        }
        let mut lines = vec![Vec::new()];
        lines.extend(text_lines(output, theme.fg("error")));
        return lines;
    }
    let Some(diff) = view
        .details
        .and_then(|d| d.get("diff"))
        .and_then(Value::as_str)
    else {
        return Vec::new();
    };
    let mut lines = vec![Vec::new()];
    lines.extend(render_diff(diff, theme));
    lines
}

// ---------------------------------------------------------------------------
// grep / find / ls
// ---------------------------------------------------------------------------

fn grep_call(arguments: &Value, home: Option<&str>, theme: &Theme) -> Spans {
    let pattern = string_arg(arguments, &["pattern"]);
    let path = string_arg(arguments, &["path"])
        .map(|raw| shorten_path(if raw.is_empty() { "." } else { raw }, home));
    let glob = string_arg(arguments, &["glob"]).unwrap_or("");
    let mut spans = vec![bold_title("grep", theme), Span::raw(" ")];
    spans.push(match pattern {
        None => invalid_arg(theme),
        Some(pattern) => Span::styled(format!("/{pattern}/"), theme.fg("accent")),
    });
    spans.push(Span::styled(" in ", theme.fg("toolOutput")));
    spans.push(match path {
        None => invalid_arg(theme),
        Some(path) => Span::styled(path, theme.fg("toolOutput")),
    });
    if !glob.is_empty() {
        spans.push(Span::styled(format!(" ({glob})"), theme.fg("toolOutput")));
    }
    if let Some(limit) = limit_arg(arguments) {
        spans.push(Span::styled(
            format!(" limit {limit}"),
            theme.fg("toolOutput"),
        ));
    }
    spans
}

fn find_call(arguments: &Value, home: Option<&str>, theme: &Theme) -> Spans {
    let pattern = string_arg(arguments, &["pattern"]);
    let path = string_arg(arguments, &["path"])
        .map(|raw| shorten_path(if raw.is_empty() { "." } else { raw }, home));
    let mut spans = vec![bold_title("find", theme), Span::raw(" ")];
    spans.push(match pattern {
        None => invalid_arg(theme),
        Some(pattern) => Span::styled(pattern.to_owned(), theme.fg("accent")),
    });
    spans.push(Span::styled(" in ", theme.fg("toolOutput")));
    spans.push(match path {
        None => invalid_arg(theme),
        Some(path) => Span::styled(path, theme.fg("toolOutput")),
    });
    if let Some(limit) = limit_arg(arguments) {
        spans.push(Span::styled(
            format!(" (limit {limit})"),
            theme.fg("toolOutput"),
        ));
    }
    spans
}

fn ls_call(arguments: &Value, home: Option<&str>, theme: &Theme) -> Spans {
    let mut spans = vec![
        bold_title("ls", theme),
        Span::raw(" "),
        tool_path(string_arg(arguments, &["path"]), home, theme, Some(".")),
    ];
    if let Some(limit) = limit_arg(arguments) {
        spans.push(Span::styled(
            format!(" (limit {limit})"),
            theme.fg("toolOutput"),
        ));
    }
    spans
}

fn listing_result(
    view: &ToolView<'_>,
    output: &str,
    collapsed_lines: usize,
    theme: &Theme,
    warnings: fn(&ToolView<'_>) -> Vec<String>,
) -> Vec<Spans> {
    let output = output.trim();
    let mut rendered = Vec::new();
    if !output.is_empty() {
        let lines = output.split('\n').collect::<Vec<_>>();
        let max = if view.expanded {
            lines.len()
        } else {
            lines.len().min(collapsed_lines)
        };
        rendered.push(Vec::new());
        rendered.extend(
            lines[..max]
                .iter()
                .map(|line| vec![Span::styled((*line).to_owned(), theme.fg("toolOutput"))]),
        );
        if lines.len() > max {
            rendered.push(more_lines_hint(lines.len() - max, theme));
        }
    }
    let warnings = warnings(view);
    if !warnings.is_empty() {
        rendered.push(vec![warning(
            format!("[Truncated: {}]", warnings.join(", ")),
            theme,
        )]);
    }
    rendered
}

fn detail_number(view: &ToolView<'_>, key: &str) -> Option<String> {
    view.details
        .and_then(|details| details.get(key))
        .filter(|value| js_truthy(value))
        .map(|value| value.to_string())
}

/// JavaScript truthiness, which Pi's renderers rely on for limit details.
fn js_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0 && !n.is_nan()),
        Value::String(text) => !text.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

fn grep_warnings(view: &ToolView<'_>) -> Vec<String> {
    let mut warnings = Vec::new();
    if let Some(limit) = detail_number(view, "matchLimitReached") {
        warnings.push(format!("{limit} matches limit"));
    }
    if let Some(truncation) = truncation_details(view) {
        warnings.push(format!("{} limit", format_size(byte_limit(truncation))));
    }
    if view.details.is_some_and(|d| d["linesTruncated"] == true) {
        warnings.push("some lines truncated".into());
    }
    warnings
}

fn find_warnings(view: &ToolView<'_>) -> Vec<String> {
    let mut warnings = Vec::new();
    if let Some(limit) = detail_number(view, "resultLimitReached") {
        warnings.push(format!("{limit} results limit"));
    }
    if let Some(truncation) = truncation_details(view) {
        warnings.push(format!("{} limit", format_size(byte_limit(truncation))));
    }
    warnings
}

fn ls_warnings(view: &ToolView<'_>) -> Vec<String> {
    let mut warnings = Vec::new();
    if let Some(limit) = detail_number(view, "entryLimitReached") {
        warnings.push(format!("{limit} entries limit"));
    }
    if let Some(truncation) = truncation_details(view) {
        warnings.push(format!("{} limit", format_size(byte_limit(truncation))));
    }
    warnings
}

// ---------------------------------------------------------------------------
// Diff (`components/diff.ts`)
// ---------------------------------------------------------------------------

struct DiffLine<'a> {
    prefix: char,
    line_number: &'a str,
    content: &'a str,
}

/// Parse `+123 content`, `-123 content`, or ` 123 content`.
fn parse_diff_line(line: &str) -> Option<DiffLine<'_>> {
    let mut chars = line.char_indices();
    let (_, prefix) = chars.next()?;
    if !matches!(prefix, '+' | '-' | ' ') {
        return None;
    }
    let rest = &line[prefix.len_utf8()..];
    let digits_end = rest
        .char_indices()
        .find(|(_, ch)| !(ch.is_whitespace() || ch.is_ascii_digit()))
        .map_or(rest.len(), |(index, _)| index);
    // The number field is spaces then digits; the content follows one separating space.
    let number_field = &rest[..digits_end];
    let split = number_field.trim_end().len();
    let content_start = rest[split..]
        .find(' ')
        .map_or(split, |offset| split + offset)
        + 1;
    Some(DiffLine {
        prefix,
        line_number: &rest[..split],
        content: rest.get(content_start..).unwrap_or(""),
    })
}

/// Word-level diff with inverse video on changed parts; leading indentation stays plain.
fn intra_line_diff(old: &str, new: &str, base_removed: Style, base_added: Style) -> (Spans, Spans) {
    let diff = TextDiff::from_words(old, new);
    let (mut removed, mut added): (Spans, Spans) = (Vec::new(), Vec::new());
    let (mut first_removed, mut first_added) = (true, true);
    let inverse = Modifier::REVERSED;
    for change in diff.iter_all_changes() {
        let text = change.value();
        match change.tag() {
            ChangeTag::Delete => {
                let mut value = text;
                if first_removed {
                    let indent = value.len() - value.trim_start().len();
                    removed.push(Span::styled(value[..indent].to_owned(), base_removed));
                    value = &value[indent..];
                    first_removed = false;
                }
                if !value.is_empty() {
                    removed.push(Span::styled(
                        value.to_owned(),
                        base_removed.add_modifier(inverse),
                    ));
                }
            }
            ChangeTag::Insert => {
                let mut value = text;
                if first_added {
                    let indent = value.len() - value.trim_start().len();
                    added.push(Span::styled(value[..indent].to_owned(), base_added));
                    value = &value[indent..];
                    first_added = false;
                }
                if !value.is_empty() {
                    added.push(Span::styled(
                        value.to_owned(),
                        base_added.add_modifier(inverse),
                    ));
                }
            }
            ChangeTag::Equal => {
                removed.push(Span::styled(text.to_owned(), base_removed));
                added.push(Span::styled(text.to_owned(), base_added));
            }
        }
    }
    (removed, added)
}

/// Render a diff string with colored lines and intra-line highlighting for single-line edits.
pub fn render_diff(diff: &str, theme: &Theme) -> Vec<Spans> {
    let context = theme.fg("toolDiffContext");
    let removed_style = theme.fg("toolDiffRemoved");
    let added_style = theme.fg("toolDiffAdded");
    let lines = diff.split('\n').collect::<Vec<_>>();
    let mut rendered = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let Some(parsed) = parse_diff_line(lines[index]) else {
            rendered.push(vec![Span::styled(lines[index].to_owned(), context)]);
            index += 1;
            continue;
        };
        match parsed.prefix {
            '-' => {
                let mut removed = Vec::new();
                while let Some(line) = lines
                    .get(index)
                    .and_then(|l| parse_diff_line(l))
                    .filter(|l| l.prefix == '-')
                {
                    removed.push((line.line_number, line.content));
                    index += 1;
                }
                let mut added = Vec::new();
                while let Some(line) = lines
                    .get(index)
                    .and_then(|l| parse_diff_line(l))
                    .filter(|l| l.prefix == '+')
                {
                    added.push((line.line_number, line.content));
                    index += 1;
                }
                if removed.len() == 1 && added.len() == 1 {
                    let (old, new) = (replace_tabs(removed[0].1), replace_tabs(added[0].1));
                    let (removed_spans, added_spans) =
                        intra_line_diff(&old, &new, removed_style, added_style);
                    let mut removed_row =
                        vec![Span::styled(format!("-{} ", removed[0].0), removed_style)];
                    removed_row.extend(removed_spans);
                    let mut added_row =
                        vec![Span::styled(format!("+{} ", added[0].0), added_style)];
                    added_row.extend(added_spans);
                    rendered.push(removed_row);
                    rendered.push(added_row);
                } else {
                    for (number, content) in removed {
                        rendered.push(vec![Span::styled(
                            format!("-{number} {}", replace_tabs(content)),
                            removed_style,
                        )]);
                    }
                    for (number, content) in added {
                        rendered.push(vec![Span::styled(
                            format!("+{number} {}", replace_tabs(content)),
                            added_style,
                        )]);
                    }
                }
            }
            '+' => {
                rendered.push(vec![Span::styled(
                    format!("+{} {}", parsed.line_number, replace_tabs(parsed.content)),
                    added_style,
                )]);
                index += 1;
            }
            _ => {
                rendered.push(vec![Span::styled(
                    format!(" {} {}", parsed.line_number, replace_tabs(parsed.content)),
                    context,
                )]);
                index += 1;
            }
        }
    }
    rendered
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn theme() -> Theme {
        Theme::dark()
    }

    fn view<'a>(name: &'a str, arguments: &'a Value, output: Option<&'a str>) -> ToolView<'a> {
        ToolView {
            name,
            arguments,
            output,
            details: None,
            is_error: false,
            duration_ms: None,
            expanded: false,
            home: Some("/home/me"),
        }
    }

    fn plain(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    fn body(lines: &[Line<'_>]) -> Vec<String> {
        // Drop the spacer row, top/bottom padding, and the one-column margins.
        let rows = plain(lines);
        rows[2..rows.len() - 1]
            .iter()
            .map(|row| row.trim().to_owned())
            .collect()
    }

    #[test]
    fn pending_success_and_error_use_the_pi_background_tokens() {
        let arguments = json!({"command":"ls"});
        let theme = theme();
        for (output, is_error, token) in [
            (None, false, "toolPendingBg"),
            (Some("ok"), false, "toolSuccessBg"),
            (Some("boom"), true, "toolErrorBg"),
        ] {
            let mut tool = view("bash", &arguments, output);
            tool.is_error = is_error;
            let lines = tool_lines(&tool, 40, &theme);
            assert!(
                lines[0].spans.is_empty() && lines[0].style.bg.is_none(),
                "spacer row has no background"
            );
            assert!(
                lines[1..]
                    .iter()
                    .all(|line| line.style.bg == Some(theme.color(token)))
            );
            assert!(
                lines[1..].iter().all(|line| line
                    .spans
                    .iter()
                    .map(|s| s.content.width())
                    .sum::<usize>()
                    == 40)
            );
        }
    }

    #[test]
    fn bash_keeps_the_last_five_visual_lines_with_an_earlier_lines_hint() {
        let arguments = json!({"command":"seq 1 8","timeout":30});
        let output = (1..=8)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let tool = view("bash", &arguments, Some(&output));
        assert_eq!(
            body(&tool_lines(&tool, 60, &theme())),
            [
                "$ seq 1 8 (timeout 30s)",
                "",
                "... (3 earlier lines, ctrl+o to expand)",
                "4",
                "5",
                "6",
                "7",
                "8"
            ]
        );
        let mut expanded = tool.clone();
        expanded.expanded = true;
        let rows = body(&tool_lines(&expanded, 60, &theme()));
        assert_eq!(rows.len(), 2 + 8, "{rows:?}");
    }

    #[test]
    fn bash_reports_truncation_full_output_path_and_duration() {
        let arguments = json!({"command":"cat big"});
        let details = json!({
            "truncation":{"truncated":true,"truncatedBy":"lines","outputLines":2000,"totalLines":5000},
            "fullOutputPath":"/tmp/pi-bash-1.log"
        });
        let mut tool = view(
            "bash",
            &arguments,
            Some("tail\n\n[Showing lines 4001-5000 of 5000. Full output: /tmp/pi-bash-1.log]"),
        );
        tool.details = Some(&details);
        tool.duration_ms = Some(4_250);
        let rows = body(&tool_lines(&tool, 100, &theme()));
        assert!(
            rows.contains(
                &"[Full output: /tmp/pi-bash-1.log. Truncated: showing 2000 of 5000 lines]"
                    .to_owned()
            ),
            "{rows:?}"
        );
        assert_eq!(rows.last().unwrap(), "Took 4.3s");
        assert!(
            !rows.iter().any(|row| row.contains("Showing lines 4001")),
            "footer is stripped"
        );
    }

    #[test]
    fn read_collapses_to_the_call_line_and_shows_ranges_and_compact_skills() {
        let arguments = json!({"path":"/home/me/src/main.rs","offset":10,"limit":5});
        let tool = view("read", &arguments, Some("fn main() {}"));
        assert_eq!(
            body(&tool_lines(&tool, 60, &theme())),
            ["read ~/src/main.rs:10-14"]
        );

        let skill = json!({"path":"/home/me/.pi/skills/rust/SKILL.md"});
        let tool = view("read", &skill, Some("secret instructions"));
        assert_eq!(
            body(&tool_lines(&tool, 80, &theme())),
            ["[skill] rust (ctrl+o to expand)"]
        );
        let mut expanded = tool.clone();
        expanded.expanded = true;
        let rows = body(&tool_lines(&expanded, 80, &theme()));
        assert_eq!(rows[0], "read ~/.pi/skills/rust/SKILL.md");
        assert!(rows.contains(&"secret instructions".to_owned()));
    }

    #[test]
    fn read_expanded_notes_truncation_details() {
        let arguments = json!({"path":"a.txt"});
        let details = json!({"truncation":{"truncated":true,"truncatedBy":"lines","outputLines":2000,"totalLines":2500,"maxLines":2000}});
        let mut tool = view("read", &arguments, Some("x"));
        tool.expanded = true;
        tool.details = Some(&details);
        let rows = body(&tool_lines(&tool, 100, &theme()));
        assert!(
            rows.contains(&"[Truncated: showing 2000 of 2500 lines (2000 line limit)]".to_owned()),
            "{rows:?}"
        );
    }

    #[test]
    fn write_previews_ten_lines_with_a_total_count_and_errors_show_below() {
        let content = (1..=12)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let arguments = json!({"path":"out.txt","content":content});
        let tool = view("write", &arguments, Some("Successfully wrote to out.txt"));
        let rows = body(&tool_lines(&tool, 80, &theme()));
        assert_eq!(rows[0], "write out.txt");
        assert_eq!(rows[2], "line 1");
        assert_eq!(
            rows.last().unwrap(),
            "... (2 more lines, 12 total, ctrl+o to expand)"
        );

        let mut failed = view("write", &arguments, Some("EACCES: permission denied"));
        failed.is_error = true;
        assert!(
            body(&tool_lines(&failed, 80, &theme()))
                .contains(&"EACCES: permission denied".to_owned())
        );
    }

    #[test]
    fn edit_renders_the_unified_diff_with_intra_line_highlighting() {
        let arguments = json!({"path":"a.rs","edits":[{"oldText":"x","newText":"y"}]});
        let details = json!({"diff":" 1 keep\n-2 let a = 1;\n+2 let a = 2;\n 3 end"});
        let mut tool = view(
            "edit",
            &arguments,
            Some("Successfully replaced 1 block(s) in a.rs."),
        );
        tool.details = Some(&details);
        let lines = tool_lines(&tool, 60, &theme());
        assert_eq!(
            body(&lines),
            [
                "edit a.rs",
                "",
                "1 keep",
                "-2 let a = 1;",
                "+2 let a = 2;",
                "3 end"
            ]
        );
        let removed = lines
            .iter()
            .find(|line| plain(std::slice::from_ref(line))[0].contains("-2"))
            .unwrap();
        let changed = removed
            .spans
            .iter()
            .find(|span| span.content == "1;")
            .or_else(|| removed.spans.iter().find(|s| s.content.contains('1')));
        assert!(
            changed
                .unwrap()
                .style
                .add_modifier
                .contains(Modifier::REVERSED)
        );
    }

    #[test]
    fn multi_line_edits_skip_intra_line_diffing() {
        let rows = render_diff("-1 a\n-2 b\n+1 c", &theme());
        assert!(
            rows.iter()
                .flatten()
                .all(|span| !span.style.add_modifier.contains(Modifier::REVERSED))
        );
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn search_tools_show_titles_previews_and_limit_warnings() {
        let arguments = json!({"pattern":"todo","path":"/home/me/proj","glob":"*.rs","limit":5});
        let details = json!({"matchLimitReached":5,"linesTruncated":true});
        let output = (1..=20)
            .map(|n| format!("f.rs:{n}: todo"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut tool = view("grep", &arguments, Some(&output));
        tool.details = Some(&details);
        let rows = body(&tool_lines(&tool, 100, &theme()));
        assert_eq!(rows[0], "grep /todo/ in ~/proj (*.rs) limit 5");
        assert_eq!(rows[2], "f.rs:1: todo");
        assert!(rows.contains(&"... (5 more lines, ctrl+o to expand)".to_owned()));
        assert_eq!(
            rows.last().unwrap(),
            "[Truncated: 5 matches limit, some lines truncated]"
        );

        let find = json!({"pattern":"*.rs","limit":3});
        assert_eq!(
            body(&tool_lines(&view("find", &find, None), 60, &theme()))[0],
            "find *.rs in . (limit 3)"
        );
        let ls = json!({});
        assert_eq!(
            body(&tool_lines(&view("ls", &ls, None), 60, &theme()))[0],
            "ls ."
        );
    }

    #[test]
    fn invalid_arguments_show_the_invalid_arg_marker() {
        let arguments = json!({"command": 7});
        assert_eq!(
            body(&tool_lines(&view("bash", &arguments, None), 60, &theme()))[0],
            "[invalid arg]"
        );
        let arguments = json!({"path": 3});
        assert_eq!(
            body(&tool_lines(&view("read", &arguments, None), 60, &theme()))[0],
            "read [invalid arg]"
        );
    }

    #[test]
    fn unknown_tools_use_the_generic_call_and_ten_line_result_preview() {
        let arguments = json!({"query":"hello","limit":3});
        let output = (1..=12)
            .map(|n| format!("row {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let tool = view("mcp__search__find", &arguments, Some(&output));
        let rows = body(&tool_lines(&tool, 100, &theme()));
        assert_eq!(rows[0], "mcp__search__find query=\"hello\" limit=3");
        assert_eq!(rows.last().unwrap(), "... (2 more lines, ctrl+o to expand)");
        let mut expanded = tool.clone();
        expanded.expanded = true;
        let rows = body(&tool_lines(&expanded, 100, &theme()));
        assert_eq!(rows[0], "mcp__search__find");
        assert!(
            rows.contains(&"limit: 3".to_owned()) || rows.iter().any(|r| r.trim() == "limit: 3")
        );
    }

    #[test]
    fn durations_and_sizes_use_pi_formats() {
        assert_eq!(format_tool_duration(4_200), "4.2s");
        assert_eq!(format_tool_duration(4_250), "4.3s");
        assert_eq!(format_tool_duration(1_150), "1.1s");
        assert_eq!(format_tool_duration(90_000), "1m 30s");
        assert_eq!(format_tool_duration(7_384_000), "2h 3m 4s");
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(51_200), "50.0KB");
        assert_eq!(format_size(3 * 1024 * 1024), "3.0MB");
    }
}
