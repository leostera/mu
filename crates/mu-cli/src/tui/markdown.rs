//! Markdown to styled terminal lines, following `packages/tui/src/components/markdown.ts`.
//!
//! Parsing is delegated to `pulldown-cmark`; this module only turns its events into Pi's
//! visual layout (heading prefixes, code fences, list markers, bordered quotes and tables).
//! LaTeX, mermaid, syntax highlighting, and OSC 8 hyperlinks are not implemented.

use std::ops::Range;

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use super::theme::Theme;

/// Token styles used by the renderer (`MarkdownTheme` in Pi).
#[derive(Clone, Debug)]
pub struct MarkdownStyle {
    pub heading: Style,
    pub link: Style,
    pub link_url: Style,
    pub code: Style,
    pub code_block: Style,
    pub code_block_border: Style,
    pub quote: Style,
    pub quote_border: Style,
    pub hr: Style,
    pub list_bullet: Style,
    pub code_block_indent: String,
}

impl MarkdownStyle {
    pub fn from_theme(theme: &Theme) -> Self {
        Self {
            heading: theme.fg("mdHeading"),
            link: theme.fg("mdLink"),
            link_url: theme.fg("mdLinkUrl"),
            code: theme.fg("mdCode"),
            code_block: theme.fg("mdCodeBlock"),
            code_block_border: theme.fg("mdCodeBlockBorder"),
            quote: theme.fg("mdQuote"),
            quote_border: theme.fg("mdQuoteBorder"),
            hr: theme.fg("mdHr"),
            list_bullet: theme.fg("mdListBullet"),
            code_block_indent: "  ".into(),
        }
    }
}

/// Per-component options (`Markdown` constructor arguments in Pi).
#[derive(Clone, Debug, Default)]
pub struct MarkdownOptions {
    pub padding_x: usize,
    pub padding_y: usize,
    /// Base text style (foreground and decorations) applied to plain text.
    pub text: Style,
    /// Background filled across the full width, including padding.
    pub background: Option<Color>,
    /// Keep the source's list markers (`1)`, `*`, `+`) instead of normalizing them.
    pub preserve_list_markers: bool,
}

/// Render `text` into wrapped lines; with a background every line is `width` columns wide.
pub fn render_markdown(
    text: &str,
    width: usize,
    style: &MarkdownStyle,
    options: &MarkdownOptions,
) -> Vec<Line<'static>> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let content_width = width.saturating_sub(options.padding_x * 2).max(1);
    let normalized = text.replace('\t', "   ");
    let blocks = parse_blocks(&normalized);
    let renderer = Renderer { style, options };
    let rendered = renderer
        .render_blocks(&blocks, content_width, &Context::plain(options.text))
        .into_iter()
        .flat_map(|line| wrap_spans(&line, content_width));

    let margin = " ".repeat(options.padding_x);
    let finish = |mut spans: Vec<Span<'static>>| {
        spans.insert(0, Span::raw(margin.clone()));
        spans.push(Span::raw(margin.clone()));
        let used: usize = spans.iter().map(|span| span.content.width()).sum();
        let mut line = Line::from(spans);
        if let Some(background) = options.background {
            line.spans
                .push(Span::raw(" ".repeat(width.saturating_sub(used))));
            line = line.style(Style::default().bg(background));
        }
        line
    };
    let padding = |count: usize| {
        (0..count)
            .map(|_| {
                finish(vec![Span::raw(
                    " ".repeat(width.saturating_sub(options.padding_x * 2)),
                )])
            })
            .collect::<Vec<_>>()
    };
    let mut lines = padding(options.padding_y);
    lines.extend(rendered.map(finish));
    lines.extend(padding(options.padding_y));
    lines
}

// ---------------------------------------------------------------------------
// Block tree
// ---------------------------------------------------------------------------

/// Inline run: text plus the marks that applied to it.
type Inline = Vec<Segment>;

#[derive(Clone, Debug, PartialEq, Eq)]
enum SegmentKind {
    Text,
    Code,
    Link,
    LinkUrl,
}

#[derive(Clone, Debug)]
struct Segment {
    text: String,
    kind: SegmentKind,
    bold: bool,
    italic: bool,
    strike: bool,
}

#[derive(Debug)]
enum BlockKind {
    Heading { level: u8, inline: Inline },
    Paragraph(Inline),
    Code { language: String, text: String },
    List(List),
    Quote(Vec<Block>),
    Rule,
    Table(Table),
    Html(String),
}

#[derive(Debug)]
struct Block {
    kind: BlockKind,
    range: Range<usize>,
    /// The source had a blank line before the next sibling (marked's `space` token).
    blank_after: bool,
}

#[derive(Debug)]
struct List {
    ordered: bool,
    start: u64,
    loose: bool,
    items: Vec<Item>,
}

#[derive(Debug)]
struct Item {
    blocks: Vec<Block>,
    marker: Option<String>,
    task: Option<bool>,
}

#[derive(Debug)]
struct Table {
    header: Vec<Inline>,
    rows: Vec<Vec<Inline>>,
    raw: String,
}

type Events<'a> = Vec<(Event<'a>, Range<usize>)>;

struct BlockParser<'a> {
    source: &'a str,
    events: Events<'a>,
    cursor: usize,
}

fn parse_blocks(source: &str) -> Vec<Block> {
    let options =
        Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    let events = Parser::new_ext(source, options)
        .into_offset_iter()
        .collect();
    BlockParser {
        source,
        events,
        cursor: 0,
    }
    .sequence(None)
}

impl<'a> BlockParser<'a> {
    fn peek(&self) -> Option<&Event<'a>> {
        self.events.get(self.cursor).map(|(event, _)| event)
    }

    /// Parse sibling blocks up to (and consuming) `until`, or to the end of the stream.
    fn sequence(&mut self, until: Option<TagEnd>) -> Vec<Block> {
        let mut blocks = Vec::new();
        while let Some(event) = self.peek().cloned() {
            let range = self.events[self.cursor].1.clone();
            if let Event::End(end) = &event {
                if Some(*end) == until {
                    self.cursor += 1;
                }
                break;
            }
            if is_inline_event(&event) {
                // Tight list items carry inline content without a paragraph wrapper.
                let inline = self.inline(None);
                blocks.push(Block {
                    kind: BlockKind::Paragraph(inline),
                    range,
                    blank_after: false,
                });
                continue;
            }
            self.cursor += 1;
            let kind = match event {
                Event::Start(Tag::Paragraph) => {
                    BlockKind::Paragraph(self.inline(Some(TagEnd::Paragraph)))
                }
                Event::Start(Tag::Heading { level, .. }) => BlockKind::Heading {
                    level: heading_depth(level),
                    inline: self.inline(Some(TagEnd::Heading(level))),
                },
                Event::Start(Tag::CodeBlock(kind)) => self.code_block(&kind),
                Event::Start(Tag::BlockQuote(kind)) => {
                    BlockKind::Quote(self.sequence(Some(TagEnd::BlockQuote(kind))))
                }
                Event::Start(Tag::List(start)) => BlockKind::List(self.list(start)),
                Event::Start(Tag::Table(_)) => BlockKind::Table(self.table(&range)),
                Event::Rule => BlockKind::Rule,
                Event::Html(_) | Event::Start(Tag::HtmlBlock) => {
                    self.skip_html_block();
                    BlockKind::Html(self.source[range.clone()].trim().to_owned())
                }
                _ => continue,
            };
            blocks.push(Block {
                kind,
                range,
                blank_after: false,
            });
        }
        mark_blank_gaps(self.source, &mut blocks);
        blocks
    }

    fn skip_html_block(&mut self) {
        while let Some(event) = self.peek() {
            let finished = matches!(event, Event::End(TagEnd::HtmlBlock));
            if !matches!(event, Event::Html(_) | Event::End(TagEnd::HtmlBlock)) {
                break;
            }
            self.cursor += 1;
            if finished {
                break;
            }
        }
    }

    fn code_block(&mut self, kind: &CodeBlockKind<'_>) -> BlockKind {
        let language = match kind {
            CodeBlockKind::Fenced(info) => info.split_whitespace().next().unwrap_or("").to_owned(),
            CodeBlockKind::Indented => String::new(),
        };
        let mut text = String::new();
        while let Some((event, _)) = self.events.get(self.cursor) {
            self.cursor += 1;
            match event {
                Event::Text(chunk) => text.push_str(chunk),
                Event::End(TagEnd::CodeBlock) => break,
                _ => {}
            }
        }
        let text = text.strip_suffix('\n').unwrap_or(&text).to_owned();
        BlockKind::Code { language, text }
    }

    fn list(&mut self, start: Option<u64>) -> List {
        let mut items = Vec::new();
        let mut loose = false;
        while let Some(event) = self.peek().cloned() {
            let range = self.events[self.cursor].1.clone();
            self.cursor += 1;
            match event {
                Event::End(TagEnd::List(_)) => break,
                Event::Start(Tag::Item) => {
                    let task = match self.peek() {
                        Some(Event::TaskListMarker(checked)) => Some(*checked),
                        _ => None,
                    };
                    if task.is_some() {
                        self.cursor += 1;
                    }
                    // Loose items wrap their text in paragraphs; tight items carry bare inline text.
                    loose |= matches!(self.peek(), Some(Event::Start(Tag::Paragraph)));
                    let blocks = self.sequence(Some(TagEnd::Item));
                    items.push(Item {
                        blocks,
                        marker: source_marker(&self.source[range.start..]),
                        task,
                    });
                }
                _ => {}
            }
        }
        List {
            ordered: start.is_some(),
            start: start.unwrap_or(1),
            loose,
            items,
        }
    }

    fn table(&mut self, range: &Range<usize>) -> Table {
        let mut header = Vec::new();
        let mut rows: Vec<Vec<Inline>> = Vec::new();
        while let Some(event) = self.peek().cloned() {
            self.cursor += 1;
            match event {
                Event::End(TagEnd::Table) => break,
                Event::Start(Tag::TableHead) => header = self.cells(TagEnd::TableHead),
                Event::Start(Tag::TableRow) => rows.push(self.cells(TagEnd::TableRow)),
                _ => {}
            }
        }
        Table {
            header,
            rows,
            raw: self.source[range.clone()].to_owned(),
        }
    }

    fn cells(&mut self, until: TagEnd) -> Vec<Inline> {
        let mut cells = Vec::new();
        while let Some(event) = self.peek().cloned() {
            self.cursor += 1;
            match event {
                Event::End(end) if end == until => break,
                Event::Start(Tag::TableCell) => cells.push(self.inline(Some(TagEnd::TableCell))),
                _ => {}
            }
        }
        cells
    }

    /// Collect an inline run. With `until` the closing tag is consumed; without it the run
    /// ends at the first non-inline event.
    fn inline(&mut self, until: Option<TagEnd>) -> Inline {
        let mut segments = Vec::new();
        let (mut bold, mut italic, mut strike) = (0usize, 0usize, 0usize);
        let mut links: Vec<(String, usize)> = Vec::new();
        while let Some(event) = self.peek().cloned() {
            if let Event::End(end) = &event {
                if Some(*end) == until {
                    self.cursor += 1;
                    break;
                }
                if !matches!(
                    end,
                    TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough | TagEnd::Link
                ) {
                    break;
                }
            } else if until.is_none() && !is_inline_event(&event) {
                break;
            }
            self.cursor += 1;
            let mark = |text: String, kind: SegmentKind| Segment {
                text,
                kind,
                bold: bold > 0,
                italic: italic > 0,
                strike: strike > 0,
            };
            match event {
                Event::Text(text) => {
                    let kind = if links.is_empty() {
                        SegmentKind::Text
                    } else {
                        SegmentKind::Link
                    };
                    segments.push(mark(text.into_string(), kind));
                }
                Event::Code(text) => segments.push(mark(text.into_string(), SegmentKind::Code)),
                Event::SoftBreak | Event::HardBreak => {
                    segments.push(mark("\n".into(), SegmentKind::Text));
                }
                Event::InlineHtml(html) | Event::Html(html) => {
                    segments.push(mark(html.into_string(), SegmentKind::Text));
                }
                Event::Start(Tag::Emphasis) => italic += 1,
                Event::End(TagEnd::Emphasis) => italic = italic.saturating_sub(1),
                Event::Start(Tag::Strong) => bold += 1,
                Event::End(TagEnd::Strong) => bold = bold.saturating_sub(1),
                Event::Start(Tag::Strikethrough) => strike += 1,
                Event::End(TagEnd::Strikethrough) => strike = strike.saturating_sub(1),
                Event::Start(Tag::Link { dest_url, .. }) => {
                    links.push((dest_url.into_string(), segments.len()));
                }
                Event::End(TagEnd::Link) => {
                    if let Some((url, first)) = links.pop() {
                        let text = segments[first..]
                            .iter()
                            .map(|segment| segment.text.as_str())
                            .collect::<String>();
                        let comparable = url.strip_prefix("mailto:").unwrap_or(&url);
                        if text != url && text != comparable {
                            segments.push(mark(format!(" ({url})"), SegmentKind::LinkUrl));
                        }
                    }
                }
                Event::Start(Tag::Image { dest_url, .. }) => {
                    segments.push(mark(format!("![]({dest_url})"), SegmentKind::Text));
                }
                _ => {}
            }
        }
        segments
    }
}

fn is_inline_event(event: &Event<'_>) -> bool {
    matches!(
        event,
        Event::Text(_)
            | Event::Code(_)
            | Event::SoftBreak
            | Event::HardBreak
            | Event::InlineHtml(_)
            | Event::Start(
                Tag::Emphasis
                    | Tag::Strong
                    | Tag::Strikethrough
                    | Tag::Link { .. }
                    | Tag::Image { .. }
            )
    )
}

fn heading_depth(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

/// The marker (`-`, `1.`, `2)`) that opens a list item in the source.
fn source_marker(item_source: &str) -> Option<String> {
    let trimmed = item_source.trim_start_matches(' ');
    let digits = trimmed.chars().take_while(char::is_ascii_digit).count();
    if digits > 0 {
        let delimiter = trimmed
            .chars()
            .nth(digits)
            .filter(|c| matches!(c, '.' | ')'))?;
        return Some(format!("{}{delimiter} ", &trimmed[..digits]));
    }
    trimmed
        .chars()
        .next()
        .filter(|c| matches!(c, '-' | '+' | '*'))
        .map(|bullet| format!("{bullet} "))
}

/// A blank line follows a block when two newlines (ignoring trailing spaces) precede the next one.
fn mark_blank_gaps(source: &str, blocks: &mut [Block]) {
    for index in 0..blocks.len().saturating_sub(1) {
        let (start, next) = (blocks[index].range.start, blocks[index + 1].range.start);
        let between = source[start.min(next)..next].trim_end_matches([' ', '\t']);
        blocks[index].blank_after = between.chars().rev().take_while(|c| *c == '\n').count() >= 2;
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Style context for inline text (Pi's `InlineStyleContext`).
#[derive(Clone)]
struct Context {
    text: Style,
}

impl Context {
    fn plain(text: Style) -> Self {
        Self { text }
    }
}

struct Renderer<'a> {
    style: &'a MarkdownStyle,
    options: &'a MarkdownOptions,
}

type Spans = Vec<Span<'static>>;

impl Renderer<'_> {
    fn render_blocks(&self, blocks: &[Block], width: usize, context: &Context) -> Vec<Spans> {
        let mut lines = Vec::new();
        for (index, block) in blocks.iter().enumerate() {
            lines.extend(self.render_block(block, width, context));
            if let Some(next) = blocks.get(index + 1) {
                if self.blank_between(block, next) {
                    lines.push(Vec::new());
                }
            }
        }
        lines
    }

    /// marked adds spacing after most blocks unless a `space` token follows; a paragraph
    /// directly followed by a list gets none.
    fn blank_between(&self, block: &Block, next: &Block) -> bool {
        match (&block.kind, &next.kind) {
            (BlockKind::List(_), _) => block.blank_after,
            (BlockKind::Paragraph(_), BlockKind::List(_)) => block.blank_after,
            _ => true,
        }
    }

    fn render_block(&self, block: &Block, width: usize, context: &Context) -> Vec<Spans> {
        match &block.kind {
            BlockKind::Heading { level, inline } => {
                let mut heading_style = self.style.heading.add_modifier(Modifier::BOLD);
                if *level == 1 {
                    heading_style = heading_style.add_modifier(Modifier::UNDERLINED);
                }
                let heading_context = Context::plain(heading_style);
                let mut spans = Vec::new();
                if *level >= 3 {
                    spans.push(Span::styled(
                        format!("{} ", "#".repeat(usize::from(*level))),
                        heading_style,
                    ));
                }
                spans.extend(self.inline_spans(inline, &heading_context));
                split_lines(spans)
            }
            BlockKind::Paragraph(inline) => split_lines(self.inline_spans(inline, context)),
            BlockKind::Code { language, text } => {
                let mut lines = vec![vec![Span::styled(
                    format!("```{language}"),
                    self.style.code_block_border,
                )]];
                for line in text.split('\n') {
                    lines.push(vec![
                        Span::raw(self.style.code_block_indent.clone()),
                        Span::styled(line.to_owned(), self.style.code_block),
                    ]);
                }
                lines.push(vec![Span::styled("```", self.style.code_block_border)]);
                lines
            }
            BlockKind::List(list) => self.render_list(list, 0, width, context),
            BlockKind::Table(table) => self.render_table(table, width, context),
            BlockKind::Quote(blocks) => self.render_quote(blocks, width),
            BlockKind::Rule => vec![vec![Span::styled("─".repeat(width.min(80)), self.style.hr)]],
            BlockKind::Html(html) => split_lines(vec![Span::styled(html.clone(), context.text)]),
        }
    }

    fn render_quote(&self, blocks: &[Block], width: usize) -> Vec<Spans> {
        let quote_style = self.style.quote.add_modifier(Modifier::ITALIC);
        let inner_width = width.saturating_sub(2).max(1);
        // The default message style does not apply inside blockquotes.
        let mut inner = self.render_blocks(blocks, inner_width, &Context::plain(Style::default()));
        while inner.last().is_some_and(Vec::is_empty) {
            inner.pop();
        }
        let mut lines = Vec::new();
        for line in inner {
            let styled = line
                .into_iter()
                .map(|span| Span::styled(span.content, quote_style.patch(span.style)))
                .collect::<Spans>();
            for wrapped in wrap_spans(&styled, inner_width) {
                let mut row = vec![Span::styled("│ ", self.style.quote_border)];
                row.extend(wrapped);
                lines.push(row);
            }
        }
        lines
    }

    fn render_list(
        &self,
        list: &List,
        depth: usize,
        width: usize,
        context: &Context,
    ) -> Vec<Spans> {
        let indent = "    ".repeat(depth);
        let mut lines = Vec::new();
        for (index, item) in list.items.iter().enumerate() {
            let bullet = match (
                list.ordered,
                self.options.preserve_list_markers,
                &item.marker,
            ) {
                (true, true, Some(marker)) | (false, true, Some(marker)) => marker.clone(),
                (true, _, _) => format!("{}. ", list.start + index as u64),
                (false, _, _) => "- ".to_owned(),
            };
            let task = item
                .task
                .map(|checked| format!("[{}] ", if checked { "x" } else { " " }))
                .unwrap_or_default();
            let marker = format!("{bullet}{task}");
            let first_prefix = vec![
                Span::raw(indent.clone()),
                Span::styled(marker.clone(), self.style.list_bullet),
            ];
            let continuation = format!("{indent}{}", " ".repeat(marker.width()));
            let item_width = width.saturating_sub(indent.width() + marker.width()).max(1);
            let mut rendered_any = false;
            for block in &item.blocks {
                if let BlockKind::List(nested) = &block.kind {
                    lines.extend(self.render_list(nested, depth + 1, width, context));
                    rendered_any = true;
                    continue;
                }
                for line in self.render_block(block, item_width, context) {
                    for wrapped in wrap_spans(&line, item_width) {
                        let mut row = if rendered_any {
                            vec![Span::raw(continuation.clone())]
                        } else {
                            first_prefix.clone()
                        };
                        row.extend(wrapped);
                        lines.push(row);
                        rendered_any = true;
                    }
                }
            }
            if !rendered_any {
                lines.push(first_prefix);
            }
            if list.loose && index + 1 < list.items.len() {
                lines.push(Vec::new());
            }
        }
        lines
    }

    fn render_table(&self, table: &Table, width: usize, context: &Context) -> Vec<Spans> {
        let columns = table.header.len();
        if columns == 0 {
            return Vec::new();
        }
        let overhead = 3 * columns + 1;
        let available = width.saturating_sub(overhead);
        if available < columns {
            // Too narrow for a stable table: fall back to the raw markdown.
            return table
                .raw
                .lines()
                .flat_map(|line| wrap_spans(&[Span::styled(line.to_owned(), context.text)], width))
                .collect();
        }

        let render = |cell: &Inline| self.inline_spans(cell, context);
        let header: Vec<Spans> = table.header.iter().map(render).collect();
        let rows: Vec<Vec<Spans>> = table
            .rows
            .iter()
            .map(|row| row.iter().map(render).collect())
            .collect();

        let max_unbroken_word = 30;
        let mut natural = vec![0usize; columns];
        let mut min_words = vec![1usize; columns];
        let measure = |cell: &Spans| (spans_width(cell), longest_word(cell, max_unbroken_word));
        for (column, cell) in header.iter().enumerate() {
            let (cell_width, word) = measure(cell);
            natural[column] = cell_width;
            min_words[column] = word.max(1);
        }
        for row in &rows {
            for (column, cell) in row.iter().enumerate().take(columns) {
                let (cell_width, word) = measure(cell);
                natural[column] = natural[column].max(cell_width);
                min_words[column] = min_words[column].max(word);
            }
        }
        let column_widths = allocate_columns(&natural, &min_words, available, overhead, width);

        let rule = |left: &str, mid: &str, right: &str| {
            let cells = column_widths
                .iter()
                .map(|w| "─".repeat(*w))
                .collect::<Vec<_>>();
            vec![Span::raw(format!(
                "{left}─{}─{right}",
                cells.join(&format!("─{mid}─"))
            ))]
        };
        let mut lines = vec![rule("┌", "┬", "┐")];
        let emit_row = |cells: &[Spans], bold: bool, lines: &mut Vec<Spans>| {
            let wrapped: Vec<Vec<Spans>> = column_widths
                .iter()
                .enumerate()
                .map(|(column, column_width)| {
                    wrap_spans(
                        cells.get(column).map_or(&[][..], Vec::as_slice),
                        *column_width,
                    )
                })
                .collect();
            let height = wrapped.iter().map(Vec::len).max().unwrap_or(1).max(1);
            for line_index in 0..height {
                let mut row = vec![Span::raw("│ ")];
                for (column, cell_lines) in wrapped.iter().enumerate() {
                    let cell = cell_lines.get(line_index).cloned().unwrap_or_default();
                    let used = spans_width(&cell);
                    row.extend(cell.into_iter().map(|span| {
                        if bold {
                            Span::styled(span.content, span.style.add_modifier(Modifier::BOLD))
                        } else {
                            span
                        }
                    }));
                    row.push(Span::raw(
                        " ".repeat(column_widths[column].saturating_sub(used)),
                    ));
                    row.push(Span::raw(if column + 1 < columns {
                        " │ "
                    } else {
                        " │"
                    }));
                }
                lines.push(row);
            }
        };
        emit_row(&header, true, &mut lines);
        lines.push(rule("├", "┼", "┤"));
        for (index, row) in rows.iter().enumerate() {
            emit_row(row, false, &mut lines);
            if index + 1 < rows.len() {
                lines.push(rule("├", "┼", "┤"));
            }
        }
        lines.push(rule("└", "┴", "┘"));
        lines
    }

    fn inline_spans(&self, inline: &Inline, context: &Context) -> Spans {
        inline
            .iter()
            .map(|segment| {
                let mut style = match segment.kind {
                    SegmentKind::Text => context.text,
                    SegmentKind::Code => self.style.code,
                    SegmentKind::Link => self.style.link.add_modifier(Modifier::UNDERLINED),
                    SegmentKind::LinkUrl => self.style.link_url,
                };
                if segment.kind != SegmentKind::Code {
                    if segment.bold {
                        style = style.add_modifier(Modifier::BOLD);
                    }
                    if segment.italic {
                        style = style.add_modifier(Modifier::ITALIC);
                    }
                    if segment.strike {
                        style = style.add_modifier(Modifier::CROSSED_OUT);
                    }
                }
                Span::styled(segment.text.clone(), style)
            })
            .collect()
    }
}

/// Split spans containing `\n` into separate logical lines.
fn split_lines(spans: Spans) -> Vec<Spans> {
    let mut lines = vec![Vec::new()];
    for span in spans {
        let mut parts = span.content.split('\n');
        if let Some(first) = parts.next() {
            if !first.is_empty() {
                lines
                    .last_mut()
                    .expect("one line")
                    .push(Span::styled(first.to_owned(), span.style));
            }
        }
        for part in parts {
            lines.push(Vec::new());
            if !part.is_empty() {
                lines
                    .last_mut()
                    .expect("one line")
                    .push(Span::styled(part.to_owned(), span.style));
            }
        }
    }
    lines
}

fn spans_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(|span| span.content.width()).sum()
}

fn longest_word(spans: &[Span<'_>], max: usize) -> usize {
    let text = spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();
    text.split_whitespace()
        .map(UnicodeWidthStr::width)
        .max()
        .unwrap_or(0)
        .min(max)
}

/// Pi's table column allocation: natural widths when they fit, otherwise shrink toward each
/// column's longest word.
fn allocate_columns(
    natural: &[usize],
    min_words: &[usize],
    available: usize,
    overhead: usize,
    width: usize,
) -> Vec<usize> {
    let columns = natural.len();
    let mut min_columns = min_words.to_vec();
    let mut min_total: usize = min_columns.iter().sum();
    if min_total > available {
        min_columns = vec![1; columns];
        let remaining = available.saturating_sub(columns);
        if remaining > 0 {
            let weights = min_words
                .iter()
                .map(|w| w.saturating_sub(1))
                .collect::<Vec<_>>();
            let total_weight: usize = weights.iter().sum();
            let growth = weights
                .iter()
                .map(|w| (w * remaining).checked_div(total_weight).unwrap_or(0))
                .collect::<Vec<_>>();
            for (min, grow) in min_columns.iter_mut().zip(&growth) {
                *min += grow;
            }
            let mut leftover = remaining - growth.iter().sum::<usize>();
            for min in &mut min_columns {
                if leftover == 0 {
                    break;
                }
                *min += 1;
                leftover -= 1;
            }
        }
        min_total = min_columns.iter().sum();
    }
    let natural_total: usize = natural.iter().sum::<usize>() + overhead;
    if natural_total <= width {
        return natural
            .iter()
            .zip(&min_columns)
            .map(|(n, m)| *n.max(m))
            .collect();
    }
    let potential: usize = natural
        .iter()
        .zip(&min_columns)
        .map(|(n, m)| n.saturating_sub(*m))
        .sum();
    let extra = available.saturating_sub(min_total);
    let mut widths = min_columns
        .iter()
        .zip(natural)
        .map(|(min, nat)| {
            let delta = nat.saturating_sub(*min);
            min + (delta * extra).checked_div(potential).unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let mut remaining = available.saturating_sub(widths.iter().sum::<usize>());
    while remaining > 0 {
        let mut grew = false;
        for column in 0..columns {
            if remaining > 0 && widths[column] < natural[column] {
                widths[column] += 1;
                remaining -= 1;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    widths
}

// ---------------------------------------------------------------------------
// Wrapping
// ---------------------------------------------------------------------------

/// Greedy word wrap that keeps span styles. Whitespace at a wrap point is dropped; a word wider
/// than the line breaks at grapheme boundaries.
pub fn wrap_spans(line: &[Span<'static>], width: usize) -> Vec<Spans> {
    let width = width.max(1);
    let mut rows: Vec<Spans> = vec![Vec::new()];
    let mut row_width = 0usize;
    let mut pending_space: Vec<Span<'static>> = Vec::new();
    let mut pending_width = 0usize;

    let push = |rows: &mut Vec<Spans>, row_width: &mut usize, text: &str, style: Style| {
        if text.is_empty() {
            return;
        }
        let row = rows.last_mut().expect("at least one row");
        match row.last_mut() {
            Some(last) if last.style == style => last.content.to_mut().push_str(text),
            _ => row.push(Span::styled(text.to_owned(), style)),
        }
        *row_width += text.width();
    };

    for (span_index, span) in line.iter().enumerate() {
        for (token_index, word) in split_keeping_whitespace(&span.content)
            .into_iter()
            .enumerate()
        {
            // Pi's wrapper tokenizes ANSI-escaped text: a space right after a style change carries
            // the escape code, so it is not whitespace there and is never dropped at a wrap.
            let sticky_space =
                token_index == 0 && (span_index > 0 || span.style != Style::default());
            let is_space = word.chars().all(char::is_whitespace) && !sticky_space;
            let word_width = word.width();
            if is_space {
                if row_width == 0
                    && rows.len() == 1
                    && pending_space.is_empty()
                    && rows[0].is_empty()
                {
                    // Leading indentation of the logical line is preserved.
                    push(&mut rows, &mut row_width, word, span.style);
                } else {
                    pending_space.push(Span::styled(word.to_owned(), span.style));
                    pending_width += word_width;
                }
                continue;
            }
            if row_width + pending_width + word_width <= width {
                for space in pending_space.drain(..) {
                    push(&mut rows, &mut row_width, &space.content, space.style);
                }
                pending_width = 0;
                push(&mut rows, &mut row_width, word, span.style);
                continue;
            }
            // Needs a new row; pending whitespace is dropped.
            pending_space.clear();
            pending_width = 0;
            if row_width > 0 {
                rows.push(Vec::new());
                row_width = 0;
            }
            if word_width <= width {
                push(&mut rows, &mut row_width, word, span.style);
                continue;
            }
            for grapheme in word.graphemes(true) {
                let grapheme_width = grapheme.width();
                if row_width + grapheme_width > width && row_width > 0 {
                    rows.push(Vec::new());
                    row_width = 0;
                }
                push(&mut rows, &mut row_width, grapheme, span.style);
            }
        }
    }
    for row in &mut rows {
        trim_row_end(row);
    }
    rows
}

/// Remove trailing whitespace from the last spans of a row.
fn trim_row_end(row: &mut Spans) {
    while let Some(last) = row.last_mut() {
        let trimmed = last.content.trim_end().to_owned();
        if trimmed.is_empty() {
            row.pop();
        } else {
            last.content = trimmed.into();
            break;
        }
    }
}

fn split_keeping_whitespace(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_space: Option<bool> = None;
    for (index, ch) in text.char_indices() {
        let space = ch.is_whitespace();
        if in_space.is_some_and(|previous| previous != space) {
            parts.push(&text[start..index]);
            start = index;
        }
        in_space = Some(space);
    }
    if start < text.len() {
        parts.push(&text[start..]);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn render(text: &str, width: usize) -> Vec<String> {
        let style = MarkdownStyle::from_theme(&Theme::dark());
        plain(&render_markdown(
            text,
            width,
            &style,
            &MarkdownOptions::default(),
        ))
    }

    #[test]
    fn headings_use_pi_prefixes_and_blank_lines_between_blocks() {
        assert_eq!(
            render("# One\n\n## Two\n\n### Three\n\nbody", 40),
            ["One", "", "Two", "", "### Three", "", "body"]
        );
    }

    #[test]
    fn paragraph_directly_before_a_list_has_no_blank_line() {
        assert_eq!(render("intro\n- a\n- b", 40), ["intro", "- a", "- b"]);
        assert_eq!(render("intro\n\n- a\n- b", 40), ["intro", "", "- a", "- b"]);
    }

    #[test]
    fn code_blocks_keep_fences_language_and_indent() {
        assert_eq!(
            render("```rust\nfn main() {}\n```\nafter", 40),
            ["```rust", "  fn main() {}", "```", "", "after"]
        );
    }

    #[test]
    fn lists_number_nest_and_render_task_markers() {
        assert_eq!(
            render("3. three\n4. four\n   - nested\n- [x] done\n- [ ] todo", 40),
            [
                "3. three",
                "4. four",
                "    - nested",
                "- [x] done",
                "- [ ] todo"
            ]
        );
    }

    #[test]
    fn preserved_ordered_markers_follow_the_source() {
        let style = MarkdownStyle::from_theme(&Theme::dark());
        let options = MarkdownOptions {
            preserve_list_markers: true,
            ..MarkdownOptions::default()
        };
        assert_eq!(
            plain(&render_markdown(
                "1) one\n2) two\n\n* star",
                40,
                &style,
                &options
            )),
            ["1) one", "2) two", "", "* star"]
        );
    }

    #[test]
    fn blockquotes_are_bordered_and_wrap_inside_the_border() {
        assert_eq!(
            render("> quoted text that wraps around", 14),
            ["│ quoted text", "│ that wraps", "│ around"]
        );
    }

    #[test]
    fn links_show_urls_unless_the_text_matches() {
        assert_eq!(
            render("[docs](https://example.com) and <https://example.com>", 80),
            ["docs (https://example.com) and https://example.com"]
        );
    }

    #[test]
    fn inline_marks_carry_modifiers_and_soft_breaks_stay_line_breaks() {
        let style = MarkdownStyle::from_theme(&Theme::dark());
        let lines = render_markdown(
            "**bold** _it_ `code`\nnext",
            40,
            &style,
            &MarkdownOptions::default(),
        );
        assert_eq!(plain(&lines), ["bold it code", "next"]);
        let bold = lines[0]
            .spans
            .iter()
            .find(|span| span.content == "bold")
            .unwrap();
        assert!(bold.style.add_modifier.contains(Modifier::BOLD));
        let italic = lines[0]
            .spans
            .iter()
            .find(|span| span.content == "it")
            .unwrap();
        assert!(italic.style.add_modifier.contains(Modifier::ITALIC));
        let code = lines[0]
            .spans
            .iter()
            .find(|span| span.content == "code")
            .unwrap();
        assert_eq!(code.style.fg, style.code.fg);
    }

    #[test]
    fn tables_render_borders_and_fall_back_when_too_narrow() {
        assert_eq!(
            render("| a | bb |\n|---|----|\n| 1 | 2 |", 40),
            [
                "┌───┬────┐",
                "│ a │ bb │",
                "├───┼────┤",
                "│ 1 │ 2  │",
                "└───┴────┘"
            ]
        );
        assert_eq!(render("| a | b |\n|---|---|\n| 1 | 2 |", 4)[0], "| a");
    }

    #[test]
    fn background_and_padding_fill_every_row_to_the_full_width() {
        let style = MarkdownStyle::from_theme(&Theme::dark());
        let options = MarkdownOptions {
            padding_x: 1,
            padding_y: 1,
            background: Some(Color::Blue),
            ..MarkdownOptions::default()
        };
        let lines = render_markdown("hi", 8, &style, &options);
        assert_eq!(plain(&lines), ["        ", " hi     ", "        "]);
        assert!(lines.iter().all(|line| line.style.bg == Some(Color::Blue)));
    }

    #[test]
    fn long_words_break_at_grapheme_boundaries_and_wrapping_keeps_styles() {
        assert_eq!(render("abcdefghij", 4), ["abcd", "efgh", "ij"]);
        let bold = Style::default().add_modifier(Modifier::BOLD);
        let rows = wrap_spans(
            &[Span::styled("aa bb ".to_owned(), bold), Span::raw("cc")],
            5,
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0].style, bold);
    }

    #[test]
    fn empty_text_renders_nothing() {
        assert!(render("   \n", 10).is_empty());
    }
}
