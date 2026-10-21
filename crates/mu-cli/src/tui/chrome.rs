//! Fixed screen furniture from Pi's interactive mode: startup header, editor frame, slash-command
//! list, and footer. Each function returns plain `Line`s so layout stays a separate concern.

use mu_core::FooterStats;
use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};
use std::borrow::Cow;
use unicode_width::UnicodeWidthStr;

use super::{fuzzy::fuzzy_filter, markdown::wrap_spans, theme::Theme};

// ---------------------------------------------------------------------------
// Startup header
// ---------------------------------------------------------------------------

/// The startup block above the first message (logo, key hints, tagline).
///
/// Each paragraph is wrapped inside one column of padding, like Pi's padded `Text` components.
pub fn header_lines(version: &str, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let accent = theme.fg("accent").add_modifier(Modifier::BOLD);
    let dim = theme.fg("dim");
    let muted = theme.fg("muted");
    let inner = width.saturating_sub(2).max(1);
    let paragraph = |spans: Vec<Span<'static>>| -> Vec<Line<'static>> {
        wrap_spans(&spans, inner)
            .into_iter()
            .map(|row| {
                let mut line = vec![Span::raw(" ")];
                line.extend(row);
                Line::from(line)
            })
            .collect()
    };
    let mut lines = vec![Line::from("")];
    lines.extend(paragraph(vec![
        Span::styled("▛▜▌ ", accent),
        Span::styled(format!(" v{version}"), dim),
    ]));
    let mut hints = vec![Span::styled("▙▟ ▐", accent), Span::raw(" ")];
    for (index, (key, description)) in [
        ("escape", "interrupt"),
        ("ctrl+c/ctrl+d", "clear/exit"),
        ("/", "commands"),
        ("!", "bash"),
        ("ctrl+o", "more"),
    ]
    .into_iter()
    .enumerate()
    {
        if index > 0 {
            hints.push(Span::styled(" · ", muted));
        }
        hints.push(Span::styled(key, dim));
        hints.push(Span::styled(format!(" {description}"), muted));
    }
    lines.extend(paragraph(hints));
    lines.extend(paragraph(vec![Span::styled(
        "Press ctrl+o to show full startup help and loaded resources.",
        dim,
    )]));
    lines.push(Line::from(""));
    lines.extend(paragraph(vec![Span::styled(
        "Mu can explain its own features and look up its docs. Ask it how to use or extend Mu.",
        muted,
    )]));
    lines.push(Line::from(""));
    lines
}

// ---------------------------------------------------------------------------
// Editor frame
// ---------------------------------------------------------------------------

/// `─── ↑ 3 more ───` style border with a centered scroll label (`createScrollBorder`).
pub fn scroll_border(arrow: char, hidden: usize, width: usize) -> String {
    let label = format!(" {arrow} {hidden} more ");
    let label_width = label.width();
    if label_width + 2 <= width {
        let left = (width - label_width) / 2;
        return format!(
            "{}{label}{}",
            "─".repeat(left),
            "─".repeat(width - left - label_width)
        );
    }
    let indicator = format!("─── {arrow} {hidden} more ");
    let indicator_width = indicator.width();
    if indicator_width <= width {
        return format!("{indicator}{}", "─".repeat(width - indicator_width));
    }
    let ellipsis = "...".chars().take(width).collect::<String>();
    let keep = width.saturating_sub(ellipsis.width());
    let mut shown = String::new();
    for ch in indicator.chars() {
        if shown.width() + ch.to_string().width() > keep {
            break;
        }
        shown.push(ch);
    }
    format!("{shown}{ellipsis}")
}

/// Maximum editor text rows: 30% of the terminal, at least 5.
pub fn editor_max_rows(terminal_rows: u16) -> usize {
    (usize::from(terminal_rows) * 3 / 10).max(5)
}

/// Rendered editor: border, visible text rows, border.
pub struct EditorFrame {
    pub lines: Vec<Line<'static>>,
    /// Cursor row (0 = top border) and column.
    pub cursor: (usize, usize),
}

/// Frame the editor's already-wrapped `rows`, scrolling to keep `cursor_row` visible.
pub fn editor_frame(
    rows: &[String],
    cursor: (usize, usize),
    scroll_offset: &mut usize,
    width: usize,
    terminal_rows: u16,
    theme: &Theme,
) -> EditorFrame {
    let max_visible = editor_max_rows(terminal_rows);
    let (cursor_row, cursor_column) = cursor;
    if cursor_row < *scroll_offset {
        *scroll_offset = cursor_row;
    } else if cursor_row >= *scroll_offset + max_visible {
        *scroll_offset = cursor_row + 1 - max_visible;
    }
    *scroll_offset = (*scroll_offset).min(rows.len().saturating_sub(max_visible));
    let visible = rows
        .iter()
        .skip(*scroll_offset)
        .take(max_visible)
        .collect::<Vec<_>>();
    let border = theme.fg("borderMuted");
    let border_line = |arrow: char, hidden: usize| {
        let text = if hidden > 0 {
            scroll_border(arrow, hidden, width)
        } else {
            "─".repeat(width)
        };
        Line::from(Span::styled(text, border))
    };
    let mut lines = vec![border_line('↑', *scroll_offset)];
    lines.extend(visible.iter().map(|row| Line::from(row.to_string())));
    lines.push(border_line(
        '↓',
        rows.len().saturating_sub(*scroll_offset + visible.len()),
    ));
    EditorFrame {
        lines,
        cursor: (1 + cursor_row - *scroll_offset, cursor_column),
    }
}

// ---------------------------------------------------------------------------
// Slash commands and select list
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlashCommand {
    pub name: Cow<'static, str>,
    pub description: Cow<'static, str>,
    pub argument_hint: Option<Cow<'static, str>>,
}

impl SlashCommand {
    /// A prompt template or skill command discovered at runtime.
    pub fn dynamic(name: String, description: String, argument_hint: Option<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            argument_hint: argument_hint.map(Cow::Owned),
        }
    }

    pub fn is_builtin(&self) -> bool {
        SLASH_COMMANDS
            .iter()
            .any(|builtin| builtin.name == self.name)
    }
}

const fn command(name: &'static str, description: &'static str) -> SlashCommand {
    SlashCommand {
        name: Cow::Borrowed(name),
        description: Cow::Borrowed(description),
        argument_hint: None,
    }
}

const fn command_with(
    name: &'static str,
    description: &'static str,
    hint: &'static str,
) -> SlashCommand {
    SlashCommand {
        name: Cow::Borrowed(name),
        description: Cow::Borrowed(description),
        argument_hint: Some(Cow::Borrowed(hint)),
    }
}

/// Pi's built-in slash commands, in menu order (`BUILTIN_SLASH_COMMANDS`).
pub const SLASH_COMMANDS: [SlashCommand; 24] = [
    command("settings", "Open settings menu"),
    command_with(
        "model",
        "Select model (opens selector UI)",
        "<provider/model>",
    ),
    command("tree", "Navigate session tree (switch branches)"),
    command_with("thinking", "Set thinking level", "<level>"),
    command("scoped-models", "Enable/disable models for Ctrl+P cycling"),
    command(
        "export",
        "Export session (HTML default, or specify path: .html/.jsonl)",
    ),
    command("import", "Import and resume a session from a JSONL file"),
    command(
        "share",
        "Share session via a Radius artifact or a GitHub gist",
    ),
    command_with("bug", "Report a bug to the Pi developers", "<description>"),
    command("copy", "Copy last agent message to clipboard"),
    command("name", "Set session display name"),
    command("session", "Show session info and stats"),
    command("changelog", "Show changelog entries"),
    command("hotkeys", "Show all keyboard shortcuts"),
    command("fork", "Create a new fork from a previous user message"),
    command(
        "clone",
        "Duplicate the current session at the current position",
    ),
    command("trust", "Save project trust decision for future sessions"),
    command_with("login", "Configure provider authentication", "<provider>"),
    command("logout", "Remove provider authentication"),
    command("new", "Start a new session"),
    command("compact", "Manually compact the session context"),
    command("resume", "Resume a different session"),
    command(
        "reload",
        "Reload keybindings, extensions, skills, prompts, themes, and context files",
    ),
    command("quit", "Quit Mu"),
];

/// Commands matching the text after `/` (everything when empty), best fuzzy match first.
pub fn matching_commands(query: &str, extra: &[SlashCommand]) -> Vec<SlashCommand> {
    let all = SLASH_COMMANDS
        .iter()
        .chain(extra)
        .cloned()
        .collect::<Vec<_>>();
    fuzzy_filter(&all, query, |command| command.name.to_string())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectItem {
    pub label: String,
    pub description: Option<String>,
}

impl From<&SlashCommand> for SelectItem {
    fn from(command: &SlashCommand) -> Self {
        let description = match &command.argument_hint {
            Some(hint) => format!("{hint} — {}", command.description),
            None => command.description.to_string(),
        };
        Self {
            label: command.name.to_string(),
            description: Some(description),
        }
    }
}

/// Pi's default `autocompleteMaxVisible`.
pub const DEFAULT_MAX_VISIBLE: usize = 5;
const PRIMARY_COLUMN_GAP: usize = 2;
const MIN_PRIMARY_COLUMN: usize = 12;
const MAX_PRIMARY_COLUMN: usize = 32;
const MIN_DESCRIPTION_WIDTH: usize = 10;

fn truncate_to_width(text: &str, max: usize) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        if out.width() + ch.to_string().width() > max {
            break;
        }
        out.push(ch);
    }
    out
}

/// Port of `SelectList.render` with the autocomplete layout (`12..=32` primary column).
pub fn select_list_lines(
    items: &[SelectItem],
    selected: usize,
    width: usize,
    max_visible: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let max_visible = max_visible.max(1);
    if items.is_empty() {
        return vec![Line::from(Span::styled(
            "  No matching commands",
            theme.fg("warning"),
        ))];
    }
    let widest = items
        .iter()
        .map(|item| item.label.width() + PRIMARY_COLUMN_GAP)
        .max()
        .unwrap_or(0);
    let primary_width = widest.clamp(MIN_PRIMARY_COLUMN, MAX_PRIMARY_COLUMN);
    let start = selected
        .saturating_sub(max_visible / 2)
        .min(items.len().saturating_sub(max_visible));
    let start = if items.len() <= max_visible { 0 } else { start };
    let end = (start + max_visible).min(items.len());
    let accent = theme.fg("accent");
    let description_style = theme.fg("muted");
    let mut lines = Vec::new();
    for (offset, item) in items[start..end].iter().enumerate() {
        let is_selected = start + offset == selected;
        let prefix = if is_selected { "→ " } else { "  " };
        let value_style = if is_selected {
            accent
        } else {
            Style::default()
        };
        let mut spans = Vec::new();
        let description = item
            .description
            .as_deref()
            .map(|text| text.split_whitespace().collect::<Vec<_>>().join(" "));
        let mut rendered_description = false;
        if let Some(description) = description.filter(|_| width > 40) {
            let effective = primary_width
                .min(width.saturating_sub(prefix.width() + 4))
                .max(1);
            let max_primary = effective.saturating_sub(PRIMARY_COLUMN_GAP).max(1);
            let value = truncate_to_width(&item.label, max_primary);
            let spacing = effective.saturating_sub(value.width()).max(1);
            let remaining = width.saturating_sub(prefix.width() + value.width() + spacing + 2);
            if remaining > MIN_DESCRIPTION_WIDTH {
                spans.push(Span::styled(format!("{prefix}{value}"), value_style));
                spans.push(Span::styled(
                    format!(
                        "{}{}",
                        " ".repeat(spacing),
                        truncate_to_width(&description, remaining)
                    ),
                    if is_selected {
                        accent
                    } else {
                        description_style
                    },
                ));
                rendered_description = true;
            }
        }
        if !rendered_description {
            let max = width.saturating_sub(prefix.width() + 2);
            spans.push(Span::styled(
                format!("{prefix}{}", truncate_to_width(&item.label, max)),
                value_style,
            ));
        }
        lines.push(Line::from(spans));
    }
    if start > 0 || end < items.len() {
        let text = format!("  ({}/{})", selected + 1, items.len());
        lines.push(Line::from(Span::styled(
            truncate_to_width(&text, width.saturating_sub(2)),
            theme.fg("muted"),
        )));
    }
    lines
}

// ---------------------------------------------------------------------------
// Footer
// ---------------------------------------------------------------------------

/// `1234` → `1.2k`, `12345` → `12k`, `1234567` → `1.2M` (`formatTokens`).
pub fn format_tokens(count: u64) -> String {
    if count < 1000 {
        count.to_string()
    } else if count < 10_000 {
        format!("{:.1}k", count as f64 / 1000.0)
    } else if count < 1_000_000 {
        format!("{}k", (count as f64 / 1000.0).round() as u64)
    } else if count < 10_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else {
        format!("{}M", (count as f64 / 1_000_000.0).round() as u64)
    }
}

/// `~`-relative working directory (`formatCwdForFooter`).
pub fn format_cwd_for_footer(cwd: &str, home: Option<&str>) -> String {
    let Some(home) = home.filter(|home| !home.is_empty()) else {
        return cwd.to_owned();
    };
    let home = home.trim_end_matches('/');
    match cwd.strip_prefix(home) {
        Some("") => "~".to_owned(),
        Some(rest) if rest.starts_with('/') => format!("~{rest}"),
        _ => cwd.to_owned(),
    }
}

pub struct FooterInput<'a> {
    pub cwd: &'a str,
    pub home: Option<&'a str>,
    pub git_branch: Option<&'a str>,
    pub session_name: Option<&'a str>,
    pub stats: &'a FooterStats,
    pub model_id: &'a str,
    pub auto_compact: bool,
}

/// The cwd line and the stats line (`FooterComponent.render`).
pub fn footer_lines(input: &FooterInput<'_>, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let dim = theme.fg("dim");
    let mut pwd = format_cwd_for_footer(input.cwd, input.home);
    if let Some(branch) = input.git_branch {
        pwd = format!("{pwd} ({branch})");
    }
    if let Some(name) = input.session_name {
        pwd = format!("{pwd} • {name}");
    }

    let stats = input.stats;
    let mut parts = Vec::new();
    if stats.input > 0 {
        parts.push(format!("↑{}", format_tokens(stats.input)));
    }
    if stats.output > 0 {
        parts.push(format!("↓{}", format_tokens(stats.output)));
    }
    if stats.cache_read > 0 {
        parts.push(format!("R{}", format_tokens(stats.cache_read)));
    }
    if stats.cache_write > 0 {
        parts.push(format!("W{}", format_tokens(stats.cache_write)));
    }
    if stats.cost > 0.0 {
        parts.push(format!("${:.3}", stats.cost));
    }
    let percent = match (stats.context_tokens, stats.context_window) {
        (Some(tokens), window) if window > 0 => tokens as f64 / window as f64 * 100.0,
        _ => 0.0,
    };
    let auto = if input.auto_compact { " (auto)" } else { "" };
    let context = format!(
        "{percent:.1}%/{}{auto}",
        format_tokens(stats.context_window)
    );
    let context_style = if percent > 90.0 {
        theme.fg("error")
    } else if percent > 70.0 {
        theme.fg("warning")
    } else {
        dim
    };

    let left_plain = parts.join(" ");
    let left_width = if parts.is_empty() {
        0
    } else {
        left_plain.width() + 1
    } + context.width();
    let right = input.model_id;
    let mut spans = Vec::new();
    if !parts.is_empty() {
        spans.push(Span::styled(format!("{left_plain} "), dim));
    }
    spans.push(Span::styled(context, context_style));
    if left_width + 2 + right.width() <= width {
        spans.push(Span::styled(
            format!("{}{right}", " ".repeat(width - left_width - right.width())),
            dim,
        ));
    } else {
        let available = width.saturating_sub(left_width + 2);
        if available > 0 {
            let truncated = truncate_to_width(right, available);
            spans.push(Span::styled(
                format!(
                    "{}{truncated}",
                    " ".repeat(width.saturating_sub(left_width + truncated.width()))
                ),
                dim,
            ));
        }
    }
    let pwd_line = if pwd.width() > width {
        format!("{}...", truncate_to_width(&pwd, width.saturating_sub(3)))
    } else {
        pwd
    };
    vec![Line::from(Span::styled(pwd_line, dim)), Line::from(spans)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    #[test]
    fn select_list_honors_max_visible() {
        let theme = Theme::dark();
        let items = (0..10)
            .map(|i| SelectItem {
                label: format!("item{i}"),
                description: None,
            })
            .collect::<Vec<_>>();
        let rows = |max| select_list_lines(&items, 0, 80, max, &theme).len();
        // Visible rows plus Pi's trailing "(n/total)" position line.
        assert_eq!(rows(3), 4);
        assert_eq!(rows(8), 9);
        assert_eq!(rows(0), rows(1), "zero is clamped to one row");
    }

    #[test]
    fn token_counts_and_cwd_use_pi_formats() {
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1_234), "1.2k");
        assert_eq!(format_tokens(32_768), "33k");
        assert_eq!(format_tokens(1_500_000), "1.5M");
        assert_eq!(
            format_cwd_for_footer("/home/me/project", Some("/home/me")),
            "~/project"
        );
        assert_eq!(format_cwd_for_footer("/home/me", Some("/home/me")), "~");
        assert_eq!(format_cwd_for_footer("/other", Some("/home/me")), "/other");
    }

    #[test]
    fn scroll_border_centers_the_label_and_degrades_on_narrow_widths() {
        let wide = scroll_border('↑', 12, 100);
        assert_eq!(wide.width(), 100);
        assert!(wide.contains(" ↑ 12 more "));
        assert!(wide.find('↑').unwrap() > 40);
        assert_eq!(scroll_border('↓', 3, 14), "── ↓ 3 more ──");
        assert_eq!(scroll_border('↓', 3, 6).width(), 6);
    }

    #[test]
    fn slash_list_pads_the_name_column_and_shows_a_counter() {
        let theme = Theme::dark();
        let items = matching_commands("", &[])
            .iter()
            .map(SelectItem::from)
            .collect::<Vec<_>>();
        let lines = select_list_lines(&items, 0, 100, DEFAULT_MAX_VISIBLE, &theme);
        assert_eq!(text(&lines[0]), "→ settings       Open settings menu");
        assert_eq!(
            text(&lines[1]),
            "  model          <provider/model> — Select model (opens selector UI)"
        );
        assert_eq!(text(&lines[5]), "  (1/24)");
        assert_eq!(
            text(&select_list_lines(&[], 0, 100, DEFAULT_MAX_VISIBLE, &theme)[0]),
            "  No matching commands"
        );
    }

    #[test]
    fn footer_pads_between_context_and_model_and_shows_totals() {
        let stats = FooterStats {
            input: 30,
            output: 30,
            context_tokens: Some(2),
            context_window: 32_768,
            ..FooterStats::default()
        };
        let lines = footer_lines(
            &FooterInput {
                cwd: "/p",
                home: None,
                git_branch: None,
                session_name: Some("demo"),
                stats: &stats,
                model_id: "alpha",
                auto_compact: true,
            },
            60,
            &Theme::dark(),
        );
        assert_eq!(text(&lines[0]), "/p • demo");
        let stats_line = text(&lines[1]);
        assert!(
            stats_line.starts_with("↑30 ↓30 0.0%/33k (auto)"),
            "{stats_line}"
        );
        assert!(stats_line.ends_with("alpha") && stats_line.width() == 60);
    }

    #[test]
    fn editor_frame_scrolls_to_the_cursor_and_labels_hidden_rows() {
        let rows = (0..20).map(|n| format!("row {n}")).collect::<Vec<_>>();
        let mut offset = 0;
        let frame = editor_frame(&rows, (19, 5), &mut offset, 40, 30, &Theme::dark());
        assert_eq!(frame.lines.len(), 2 + 9);
        assert!(text(&frame.lines[0]).contains("↑ 11 more"));
        assert_eq!(text(&frame.lines[1]), "row 11");
        assert_eq!(text(frame.lines.last().unwrap()), "─".repeat(40));
        assert_eq!(frame.cursor, (9, 5));
    }
}
