mod chrome;
pub(crate) mod fuzzy;
mod markdown;
mod terminal;
mod theme;
pub use theme::Theme;
mod tool_view;

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    future::Future,
    io,
    ops::{Deref, DerefMut},
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseEventKind};
use mu_core::{
    AgentError, AgentEvent, AgentPromptQueue, FooterStats, MuAgent, PromptBehavior,
    SessionTranscriptRole,
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};
use tokio::{
    sync::{Mutex, mpsc},
    task::AbortHandle,
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use self::{
    markdown::{MarkdownOptions, MarkdownStyle, render_markdown},
    tool_view::{ToolView, tool_lines},
};

#[derive(Clone)]
struct Message {
    role: Role,
    text: String,
    thinking: String,
    tools: Vec<ToolActivity>,
    error: bool,
}

#[derive(Clone)]
struct ToolActivity {
    call_id: String,
    name: String,
    arguments: serde_json::Value,
    output: Option<String>,
    error: bool,
    details: Option<serde_json::Value>,
    duration_ms: Option<u64>,
}

#[derive(Clone, Copy)]
enum Role {
    User,
    Assistant,
}

enum UiEvent {
    Agent(AgentEvent),
    Finished(Result<(), String>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TranscriptPosition {
    FollowEnd,
    Offset(u16),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TranscriptScroll {
    position: TranscriptPosition,
    max_offset: u16,
    /// Last time the offset moved; Pi's `auto` scrollbar fades one second later.
    last_activity: Option<Instant>,
    viewport_height: u16,
}

/// Rows a page scroll keeps from the previous page (`PAGE_SCROLL_OVERLAP`).
const PAGE_SCROLL_OVERLAP: u16 = 4;
const SCROLLBAR_HIDE_DELAY: Duration = Duration::from_millis(1000);

impl Default for TranscriptScroll {
    fn default() -> Self {
        Self {
            position: TranscriptPosition::FollowEnd,
            max_offset: 0,
            last_activity: None,
            viewport_height: 0,
        }
    }
}

impl TranscriptScroll {
    fn offset(self) -> u16 {
        match self.position {
            TranscriptPosition::FollowEnd => self.max_offset,
            TranscriptPosition::Offset(offset) => offset.min(self.max_offset),
        }
    }

    fn is_following_end(self) -> bool {
        self.position == TranscriptPosition::FollowEnd
    }

    fn set_max_offset(&mut self, max_offset: u16) {
        self.max_offset = max_offset;
        if max_offset == 0 {
            self.position = TranscriptPosition::FollowEnd;
        } else if let TranscriptPosition::Offset(offset) = self.position {
            self.position = TranscriptPosition::Offset(offset.min(max_offset));
        }
    }

    fn page_rows(&self) -> i32 {
        i32::from(
            self.viewport_height
                .saturating_sub(PAGE_SCROLL_OVERLAP)
                .max(1),
        )
    }

    fn scrollbar_visible(&self) -> bool {
        self.last_activity
            .is_some_and(|at| at.elapsed() < SCROLLBAR_HIDE_DELAY)
    }

    fn scroll_by(&mut self, lines: i32) {
        let next = (i32::from(self.offset()) + lines).clamp(0, i32::from(self.max_offset)) as u16;
        if next != self.offset() {
            self.last_activity = Some(Instant::now());
        }
        self.position = if next == self.max_offset {
            TranscriptPosition::FollowEnd
        } else {
            TranscriptPosition::Offset(next)
        };
    }

    fn scroll_to_start(&mut self) {
        if self.offset() != 0 {
            self.last_activity = Some(Instant::now());
        }
        self.position = if self.max_offset == 0 {
            TranscriptPosition::FollowEnd
        } else {
            TranscriptPosition::Offset(0)
        };
    }

    fn scroll_to_end(&mut self) {
        if !self.is_following_end() {
            self.last_activity = Some(Instant::now());
        }
        self.position = TranscriptPosition::FollowEnd;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ComposerAction {
    OpenCommandMenu,
    ExecuteCommand,
    Submit,
    FollowUp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct VisualLine {
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ComposerActionKind {
    Kill,
    TypeWord,
    Yank,
}

#[derive(Clone, Default)]
struct ComposerDraft {
    input: Vec<char>,
    cursor: usize,
}

#[derive(Clone, Default)]
struct ComposerSnapshot {
    input: Vec<char>,
    cursor: usize,
    pastes: BTreeMap<usize, String>,
    paste_counter: usize,
}

#[derive(Default)]
struct Composer {
    input: Vec<char>,
    cursor: usize,
    prompt_history: Vec<String>,
    history_index: Option<usize>,
    history_draft: Option<ComposerDraft>,
    preferred_column: Option<usize>,
    visual_width: usize,
    undo_stack: Vec<ComposerSnapshot>,
    pastes: BTreeMap<usize, String>,
    paste_counter: usize,
    kill_ring: Vec<String>,
    last_action: Option<ComposerActionKind>,
    last_yanked: String,
    /// First visible editor row (keeps the cursor on screen like Pi's editor).
    scroll_offset: usize,
}

impl Composer {
    fn input_text(&self) -> String {
        self.input.iter().collect()
    }

    fn expanded_text(&self) -> String {
        let text = self.input_text();
        expand_paste_markers(&text, &self.pastes)
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ComposerAction> {
        match key.code {
            KeyCode::Char('/') if self.input.is_empty() => {
                self.insert_char('/');
                Some(ComposerAction::OpenCommandMenu)
            }
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
                self.insert_char('\n');
                None
            }
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => {
                Some(ComposerAction::FollowUp)
            }
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::CONTROL) => None,
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.insert_char('\n');
                None
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.delete_to_line_start();
                None
            }
            KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.delete_to_line_end();
                None
            }
            KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.delete_word_backward();
                None
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.delete_forward();
                None
            }
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.move_to_line_start();
                None
            }
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.move_to_line_end();
                None
            }
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.move_left();
                None
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.move_right();
                None
            }
            KeyCode::Char('y') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.yank();
                None
            }
            KeyCode::Char('y') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.yank_pop();
                None
            }
            KeyCode::Char('-') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.undo();
                None
            }
            KeyCode::Enter
                if !key.modifiers.contains(KeyModifiers::SHIFT)
                    && self.cursor > 0
                    && self.input.get(self.cursor - 1) == Some(&'\\') =>
            {
                self.delete_backward();
                self.insert_char('\n');
                None
            }
            KeyCode::Enter if self.input_text().starts_with('/') => {
                Some(ComposerAction::ExecuteCommand)
            }
            KeyCode::Enter => Some(ComposerAction::Submit),
            KeyCode::Esc => {
                self.clear();
                None
            }
            KeyCode::Char(ch)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.insert_char(ch);
                None
            }
            KeyCode::Backspace if key.modifiers.contains(KeyModifiers::ALT) => {
                self.delete_word_backward();
                None
            }
            KeyCode::Backspace if self.cursor > 0 => {
                self.delete_backward();
                None
            }
            KeyCode::Delete if key.modifiers.contains(KeyModifiers::ALT) => {
                self.delete_word_forward();
                None
            }
            KeyCode::Delete if self.cursor < self.input.len() => {
                self.delete_forward();
                None
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.delete_word_forward();
                None
            }
            KeyCode::Left
                if key
                    .modifiers
                    .intersects(KeyModifiers::ALT | KeyModifiers::CONTROL) =>
            {
                self.move_word_left();
                None
            }
            KeyCode::Right
                if key
                    .modifiers
                    .intersects(KeyModifiers::ALT | KeyModifiers::CONTROL) =>
            {
                self.move_word_right();
                None
            }
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.move_word_left();
                None
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::ALT) => {
                self.move_word_right();
                None
            }
            KeyCode::Left => {
                self.move_left();
                None
            }
            KeyCode::Right => {
                self.move_right();
                None
            }
            KeyCode::Home => {
                self.move_to_line_start();
                None
            }
            KeyCode::End => {
                self.move_to_line_end();
                None
            }
            KeyCode::Up => {
                self.move_up();
                None
            }
            KeyCode::Down => {
                self.move_down();
                None
            }
            _ => None,
        }
    }

    fn insert_char(&mut self, ch: char) {
        self.exit_history_browsing();
        if ch.is_whitespace() || self.last_action != Some(ComposerActionKind::TypeWord) {
            self.push_undo_snapshot();
        }
        self.last_action = (ch != '\n').then_some(ComposerActionKind::TypeWord);
        self.input.insert(self.cursor, ch);
        self.cursor += 1;
        self.preferred_column = None;
    }

    fn insert_text(&mut self, text: &str) {
        let normalized = text
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .replace('\t', "    ");
        let mut pasted: Vec<char> = normalized
            .chars()
            .filter(|character| *character == '\n' || *character >= ' ')
            .collect();
        let is_path = pasted
            .first()
            .is_some_and(|character| matches!(character, '/' | '~' | '.'));
        let follows_word = self
            .input
            .get(self.cursor.saturating_sub(1))
            .is_some_and(|character| character.is_ascii_alphanumeric() || *character == '_');
        if is_path && follows_word {
            pasted.insert(0, ' ');
        }
        if pasted.is_empty() {
            return;
        }
        self.exit_history_browsing();
        self.push_undo_snapshot();
        let pasted_text: String = pasted.iter().collect();
        let line_count = pasted_text.split('\n').count();
        let character_count = pasted_text.encode_utf16().count();
        if line_count > 10 || character_count > 1000 {
            self.paste_counter += 1;
            self.pastes.insert(self.paste_counter, pasted_text);
            let marker = if line_count > 10 {
                format!("[paste #{} +{line_count} lines]", self.paste_counter)
            } else {
                format!("[paste #{} {character_count} chars]", self.paste_counter)
            };
            self.insert_raw(&marker);
        } else {
            self.insert_raw(&pasted_text);
        }
        self.last_action = None;
        self.preferred_column = None;
    }

    fn clear(&mut self) {
        if !self.input.is_empty() {
            self.push_undo_snapshot();
        }
        self.input.clear();
        self.cursor = 0;
        self.exit_history_browsing();
        self.preferred_column = None;
        self.last_action = None;
        self.pastes.clear();
        self.paste_counter = 0;
    }

    fn push_undo_snapshot(&mut self) {
        self.undo_stack.push(ComposerSnapshot {
            input: self.input.clone(),
            cursor: self.cursor,
            pastes: self.pastes.clone(),
            paste_counter: self.paste_counter,
        });
        if self.undo_stack.len() > 100 {
            self.undo_stack.remove(0);
        }
    }

    fn undo(&mut self) {
        let Some(snapshot) = self.undo_stack.pop() else {
            return;
        };
        self.input = snapshot.input;
        self.cursor = snapshot.cursor.min(self.input.len());
        self.pastes = snapshot.pastes;
        self.paste_counter = snapshot.paste_counter;
        self.exit_history_browsing();
        self.preferred_column = None;
        self.last_action = None;
    }

    fn push_kill(&mut self, text: String, prepend: bool) {
        if text.is_empty() {
            return;
        }
        if self.last_action == Some(ComposerActionKind::Kill) {
            if let Some(previous) = self.kill_ring.last_mut() {
                if prepend {
                    previous.insert_str(0, &text);
                } else {
                    previous.push_str(&text);
                }
                return;
            }
        }
        self.kill_ring.push(text);
        if self.kill_ring.len() > 100 {
            self.kill_ring.remove(0);
        }
        self.last_action = Some(ComposerActionKind::Kill);
    }

    fn yank(&mut self) {
        let Some(text) = self.kill_ring.last().cloned() else {
            return;
        };
        self.push_undo_snapshot();
        self.insert_raw(&text);
        self.last_yanked = text;
        self.last_action = Some(ComposerActionKind::Yank);
    }

    fn yank_pop(&mut self) {
        if self.last_action != Some(ComposerActionKind::Yank) || self.kill_ring.len() < 2 {
            return;
        }
        self.push_undo_snapshot();
        let start = self.cursor.saturating_sub(self.last_yanked.chars().count());
        self.input.drain(start..self.cursor);
        self.cursor = start;
        self.kill_ring.rotate_right(1);
        if let Some(text) = self.kill_ring.last().cloned() {
            self.insert_raw(&text);
            self.last_yanked = text;
            self.last_action = Some(ComposerActionKind::Yank);
        }
    }

    fn insert_raw(&mut self, text: &str) {
        let inserted: Vec<char> = text.chars().collect();
        let count = inserted.len();
        self.input.splice(self.cursor..self.cursor, inserted);
        self.cursor += count;
        self.exit_history_browsing();
        self.preferred_column = None;
    }

    fn add_to_history(&mut self, prompt: &str) {
        let prompt = prompt.trim();
        if prompt.is_empty()
            || self
                .prompt_history
                .last()
                .is_some_and(|last| last == prompt)
        {
            return;
        }
        self.prompt_history.push(prompt.to_owned());
        if self.prompt_history.len() > 100 {
            self.prompt_history.remove(0);
        }
    }

    fn line_bounds(&self) -> (usize, usize) {
        let start = self.input[..self.cursor]
            .iter()
            .rposition(|ch| *ch == '\n')
            .map_or(0, |index| index + 1);
        let end = self.input[self.cursor..]
            .iter()
            .position(|ch| *ch == '\n')
            .map_or(self.input.len(), |index| self.cursor + index);
        (start, end)
    }

    fn move_to_line_start(&mut self) {
        self.exit_history_browsing();
        self.cursor = self.line_bounds().0;
        self.preferred_column = None;
        self.last_action = None;
    }

    fn move_to_line_end(&mut self) {
        self.exit_history_browsing();
        self.cursor = self.line_bounds().1;
        self.preferred_column = None;
        self.last_action = None;
    }

    fn move_left(&mut self) {
        self.exit_history_browsing();
        if let Some(cursor) = self.previous_grapheme_start() {
            self.cursor = cursor;
        }
        self.preferred_column = None;
        self.last_action = None;
    }

    fn move_right(&mut self) {
        self.exit_history_browsing();
        if let Some(cursor) = self.next_grapheme_end() {
            self.cursor = cursor;
        }
        self.preferred_column = None;
        self.last_action = None;
    }

    fn previous_grapheme_start(&self) -> Option<usize> {
        let prefix: String = self.input[..self.cursor].iter().collect();
        if let Some((start, _)) = valid_paste_marker_before(&prefix, &self.pastes) {
            return Some(start);
        }
        prefix
            .graphemes(true)
            .next_back()
            .map(|grapheme| self.cursor - grapheme.chars().count())
    }

    fn next_grapheme_end(&self) -> Option<usize> {
        let suffix: String = self.input[self.cursor..].iter().collect();
        if let Some((_, length)) = valid_paste_marker_at_start(&suffix, &self.pastes) {
            return Some(self.cursor + length);
        }
        suffix
            .graphemes(true)
            .next()
            .map(|grapheme| self.cursor + grapheme.chars().count())
    }

    fn removed_paste_id(&self, start: usize, end: usize) -> Option<usize> {
        let marker: String = self.input[start..end].iter().collect();
        valid_paste_marker_at_start(&marker, &self.pastes)
            .filter(|(_, length)| *length == marker.chars().count())
            .map(|(id, _)| id)
    }

    fn remove_paste(&mut self, id: usize) {
        self.pastes.remove(&id);
        let before_cursor: String = self.input[..self.cursor].iter().collect();
        let text = self.input_text();
        let rewritten = renumber_paste_markers(&text, id);
        let rewritten_prefix = renumber_paste_markers(&before_cursor, id);
        let old_pastes = std::mem::take(&mut self.pastes);
        for (old_id, content) in old_pastes {
            self.pastes
                .insert(if old_id > id { old_id - 1 } else { old_id }, content);
        }
        self.input = rewritten.chars().collect();
        self.cursor = rewritten_prefix.chars().count().min(self.input.len());
        self.paste_counter = self.pastes.keys().next_back().copied().unwrap_or(0);
    }

    fn delete_backward(&mut self) {
        if let Some(start) = self.previous_grapheme_start() {
            let paste_id = self.removed_paste_id(start, self.cursor);
            self.push_undo_snapshot();
            self.exit_history_browsing();
            self.input.drain(start..self.cursor);
            self.cursor = start;
            if let Some(id) = paste_id {
                self.remove_paste(id);
            }
            self.preferred_column = None;
            self.last_action = None;
        }
    }

    fn move_word_left(&mut self) {
        self.exit_history_browsing();
        let line_start = self.line_bounds().0;
        if self.cursor == line_start && line_start > 0 {
            self.cursor -= 1;
        } else {
            while self.cursor > line_start && self.input[self.cursor - 1].is_whitespace() {
                self.cursor -= 1;
            }
            while self.cursor > line_start && !self.input[self.cursor - 1].is_whitespace() {
                self.cursor -= 1;
            }
        }
        self.preferred_column = None;
        self.last_action = None;
    }

    fn move_word_right(&mut self) {
        self.exit_history_browsing();
        let line_end = self.line_bounds().1;
        if self.cursor == line_end && line_end < self.input.len() {
            self.cursor += 1;
        } else {
            while self.cursor < line_end && !self.input[self.cursor].is_whitespace() {
                self.cursor += 1;
            }
            while self.cursor < line_end && self.input[self.cursor].is_whitespace() {
                self.cursor += 1;
            }
        }
        self.preferred_column = None;
        self.last_action = None;
    }

    fn set_visual_width(&mut self, width: usize) {
        self.visual_width = width.max(1);
    }

    fn visual_lines(&self) -> Vec<VisualLine> {
        let width = self.visual_width.max(1);
        let text = self.input_text();
        let mut lines = Vec::new();
        let mut start = 0;
        let mut column = 0;

        for (byte_index, grapheme) in text.grapheme_indices(true) {
            let index = text[..byte_index].chars().count();
            let grapheme_chars = grapheme.chars().count();
            if grapheme == "\n" {
                lines.push(VisualLine { start, end: index });
                start = index + grapheme_chars;
                column = 0;
                continue;
            }

            let grapheme_width = UnicodeWidthStr::width(grapheme);
            if column > 0 && column + grapheme_width > width {
                lines.push(VisualLine { start, end: index });
                start = index;
                column = 0;
            }
            column += grapheme_width;
        }
        lines.push(VisualLine {
            start,
            end: self.input.len(),
        });
        lines
    }

    /// Wrapped editor rows as text, plus the cursor's `(row, column)`.
    fn layout_rows(&self) -> (Vec<String>, (usize, usize)) {
        let lines = self.visual_lines();
        let rows = lines
            .iter()
            .map(|line| self.input[line.start..line.end].iter().collect::<String>())
            .collect();
        (rows, self.visual_cursor(&lines))
    }

    fn visual_cursor(&self, lines: &[VisualLine]) -> (usize, usize) {
        let line_index = lines
            .iter()
            .enumerate()
            .rev()
            .find(|(_, line)| self.cursor >= line.start && self.cursor <= line.end)
            .map_or(0, |(index, _)| index);
        let line = lines[line_index];
        let text: String = self.input[line.start..self.cursor.min(line.end)]
            .iter()
            .collect();
        let column = text.graphemes(true).map(UnicodeWidthStr::width).sum();
        (line_index, column)
    }

    fn cursor_at_visual_column(&self, line: VisualLine, target_column: usize) -> usize {
        let text: String = self.input[line.start..line.end].iter().collect();
        let mut column = 0;
        let mut character_offset = 0;
        for grapheme in text.graphemes(true) {
            let grapheme_width = UnicodeWidthStr::width(grapheme);
            if column + grapheme_width > target_column {
                break;
            }
            column += grapheme_width;
            character_offset += grapheme.chars().count();
            if column >= target_column {
                return line.start + character_offset;
            }
        }
        line.start + character_offset
    }

    fn move_up(&mut self) {
        let lines = self.visual_lines();
        let (line_index, column) = self.visual_cursor(&lines);
        if line_index == 0 {
            if self.input.is_empty() || self.history_index.is_some() || column == 0 {
                self.history_up();
            } else {
                self.move_to_line_start();
            }
        } else {
            self.move_vertical(&lines, line_index - 1, column);
        }
    }

    fn move_down(&mut self) {
        let lines = self.visual_lines();
        let (line_index, column) = self.visual_cursor(&lines);
        if line_index + 1 == lines.len() {
            if self.history_index.is_some() {
                self.history_down();
            } else {
                self.move_to_line_end();
            }
        } else {
            self.move_vertical(&lines, line_index + 1, column);
        }
    }

    fn move_vertical(&mut self, lines: &[VisualLine], target_line: usize, column: usize) {
        self.exit_history_browsing();
        self.cursor = self.cursor_at_visual_column(lines[target_line], column);
        self.preferred_column = Some(self.preferred_column.unwrap_or(column));
    }

    fn delete_to_line_start(&mut self) {
        let line_start = self.line_bounds().0;
        let start = if self.cursor == line_start && line_start > 0 {
            line_start - 1
        } else {
            line_start
        };
        if start == self.cursor {
            return;
        }
        let killed = self.input[start..self.cursor].iter().collect();
        self.push_undo_snapshot();
        self.exit_history_browsing();
        self.input.drain(start..self.cursor);
        self.cursor = start;
        self.push_kill(killed, true);
        self.preferred_column = None;
    }

    fn delete_to_line_end(&mut self) {
        let line_end = self.line_bounds().1;
        let end = if self.cursor == line_end && line_end < self.input.len() {
            line_end + 1
        } else {
            line_end
        };
        if self.cursor == end {
            return;
        }
        let killed = self.input[self.cursor..end].iter().collect();
        self.push_undo_snapshot();
        self.exit_history_browsing();
        self.input.drain(self.cursor..end);
        self.push_kill(killed, false);
        self.preferred_column = None;
    }

    fn delete_word_backward(&mut self) {
        let line_start = self.line_bounds().0;
        if self.cursor == 0 {
            return;
        }
        let start = if self.cursor == line_start {
            self.cursor - 1
        } else {
            let mut start = self.cursor;
            while start > line_start && self.input[start - 1].is_whitespace() {
                start -= 1;
            }
            while start > line_start && !self.input[start - 1].is_whitespace() {
                start -= 1;
            }
            start
        };
        let killed = self.input[start..self.cursor].iter().collect();
        self.push_undo_snapshot();
        self.exit_history_browsing();
        self.input.drain(start..self.cursor);
        self.cursor = start;
        self.push_kill(killed, true);
        self.preferred_column = None;
    }

    fn delete_word_forward(&mut self) {
        let line_end = self.line_bounds().1;
        if self.cursor == self.input.len() {
            return;
        }
        let end = if self.cursor == line_end {
            self.cursor + 1
        } else {
            let mut end = self.cursor;
            while end < line_end && self.input[end].is_whitespace() {
                end += 1;
            }
            while end < line_end && !self.input[end].is_whitespace() {
                end += 1;
            }
            end
        };
        let killed = self.input[self.cursor..end].iter().collect();
        self.push_undo_snapshot();
        self.exit_history_browsing();
        self.input.drain(self.cursor..end);
        self.push_kill(killed, false);
        self.preferred_column = None;
    }

    fn delete_forward(&mut self) {
        if let Some(end) = self.next_grapheme_end() {
            let paste_id = self.removed_paste_id(self.cursor, end);
            self.push_undo_snapshot();
            self.exit_history_browsing();
            self.input.drain(self.cursor..end);
            if let Some(id) = paste_id {
                self.remove_paste(id);
            }
            self.preferred_column = None;
            self.last_action = None;
        }
    }

    fn history_up(&mut self) {
        if self.prompt_history.is_empty() {
            return;
        }
        let index = match self.history_index {
            Some(0) => return,
            Some(index) => index - 1,
            None => {
                self.history_draft = Some(ComposerDraft {
                    input: self.input.clone(),
                    cursor: self.cursor,
                });
                self.prompt_history.len() - 1
            }
        };
        self.history_index = Some(index);
        self.input = self.prompt_history[index].chars().collect();
        self.cursor = 0;
        self.preferred_column = None;
    }

    fn history_down(&mut self) {
        let Some(index) = self.history_index else {
            return;
        };
        if index + 1 < self.prompt_history.len() {
            let next = index + 1;
            self.history_index = Some(next);
            self.input = self.prompt_history[next].chars().collect();
            self.cursor = self.input.len();
        } else {
            self.history_index = None;
            let draft = self.history_draft.take().unwrap_or_default();
            self.input = draft.input;
            self.cursor = draft.cursor.min(self.input.len());
        }
        self.preferred_column = None;
    }

    fn exit_history_browsing(&mut self) {
        self.history_index = None;
        self.history_draft = None;
    }
}

fn parse_paste_marker(text: &str) -> Option<(usize, usize)> {
    let suffix = text.strip_prefix("[paste #")?;
    let close = suffix.find(']')?;
    let marker_body = &suffix[..close];
    let id_end = marker_body.find(|character: char| !character.is_ascii_digit())?;
    let id = marker_body[..id_end].parse().ok()?;
    let description = &marker_body[id_end..];
    if !description.is_empty()
        && !(description.starts_with(" +") && description.ends_with(" lines")
            || description.starts_with(' ') && description.ends_with(" chars"))
    {
        return None;
    }
    let marker_bytes = "[paste #".len() + close + 1;
    Some((id, marker_bytes))
}

fn valid_paste_marker_at_start(
    text: &str,
    pastes: &BTreeMap<usize, String>,
) -> Option<(usize, usize)> {
    let (id, bytes) = parse_paste_marker(text)?;
    pastes
        .contains_key(&id)
        .then_some((id, text[..bytes].chars().count()))
}

fn valid_paste_marker_before(
    text: &str,
    pastes: &BTreeMap<usize, String>,
) -> Option<(usize, usize)> {
    let start = text.rfind("[paste #")?;
    let (id, length) = valid_paste_marker_at_start(&text[start..], pastes)?;
    if length != text[start..].chars().count() {
        return None;
    }
    Some((text[..start].chars().count(), id))
}

fn expand_paste_markers(text: &str, pastes: &BTreeMap<usize, String>) -> String {
    let mut output = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("[paste #") {
        output.push_str(&remaining[..start]);
        let candidate = &remaining[start..];
        if let Some((id, bytes)) = parse_paste_marker(candidate) {
            if let Some(content) = pastes.get(&id) {
                output.push_str(content);
                remaining = &candidate[bytes..];
                continue;
            }
        }
        output.push_str("[paste #");
        remaining = &candidate["[paste #".len()..];
    }
    output.push_str(remaining);
    output
}

fn renumber_paste_markers(text: &str, removed_id: usize) -> String {
    let mut output = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("[paste #") {
        output.push_str(&remaining[..start]);
        let candidate = &remaining[start..];
        if let Some((id, bytes)) = parse_paste_marker(candidate) {
            if id > removed_id {
                output.push_str(&candidate[.."[paste #".len()]);
                output.push_str(&(id - 1).to_string());
                output.push_str(&candidate["[paste #".len() + id.to_string().len()..bytes]);
            } else {
                output.push_str(&candidate[..bytes]);
            }
            remaining = &candidate[bytes..];
        } else {
            output.push_str("[paste #");
            remaining = &candidate["[paste #".len()..];
        }
    }
    output.push_str(remaining);
    output
}

#[derive(Default)]
struct Transcript {
    messages: Vec<Message>,
    scroll: TranscriptScroll,
}

impl Deref for Transcript {
    type Target = Vec<Message>;

    fn deref(&self) -> &Self::Target {
        &self.messages
    }
}

impl DerefMut for Transcript {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.messages
    }
}

struct App {
    composer: Composer,
    model: String,
    models: Vec<String>,
    provider: String,
    session_path: String,
    messages: Transcript,
    command_menu_open: bool,
    command_selection: usize,
    model_selector_open: bool,
    /// Generic selector overlay (thinking level, fork point); Enter runs `command <value>`.
    picker: Option<Picker>,
    /// Prompt-template and skill commands offered after the built-ins in the slash menu.
    extra_commands: Vec<chrome::SlashCommand>,
    model_selection: usize,
    loading: bool,
    quit: bool,
    last_ctrl_c: Option<Instant>,
    request: Option<AbortHandle>,
    retry_status: Option<String>,
    prompt_queue: AgentPromptQueue,
    queued_prompts: VecDeque<(String, PromptBehavior)>,
    hide_thinking: bool,
    tools_expanded: bool,
    theme: Theme,
    home: Option<String>,
    stats: FooterStats,
    cwd: String,
    session_name: Option<String>,
    model_names: HashMap<String, String>,
    /// Status lines (`Switched to …`) with the message count they follow.
    statuses: Vec<(usize, String)>,
}

impl Deref for App {
    type Target = Composer;

    fn deref(&self) -> &Self::Target {
        &self.composer
    }
}

impl DerefMut for App {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.composer
    }
}

impl App {
    fn new(agent: &MuAgent) -> Self {
        let mut app = Self::from_metadata(
            agent.model().to_owned(),
            agent.provider().to_owned(),
            agent.session_path().display().to_string(),
        );
        app.apply_agent_metadata(agent);
        app
    }

    fn apply_agent_metadata(&mut self, agent: &MuAgent) {
        self.model = agent.model().to_owned();
        self.provider = agent.provider().to_owned();
        self.session_path = agent.session_path().display().to_string();
        self.prompt_queue = agent.prompt_queue();
        self.hide_thinking = agent.hide_thinking_block();
        if let Some(theme) = agent
            .theme_name()
            .and_then(|name| Theme::resolve(&name, agent.agent_dir()).ok())
        {
            self.theme = theme;
        }
        self.models = agent.model_choices().to_vec();
        self.model_names = self
            .models
            .iter()
            .map(|model| (model.clone(), agent.model_display_name(model)))
            .collect();
        self.cwd = agent.cwd().display().to_string();
        self.extra_commands = agent
            .dynamic_slash_commands()
            .into_iter()
            .map(|(name, description, hint)| chrome::SlashCommand::dynamic(name, description, hint))
            .collect();
        self.refresh_stats(agent);
        self.messages.messages = agent
            .session_transcript()
            .into_iter()
            .map(|message| Message {
                role: match message.role {
                    SessionTranscriptRole::User => Role::User,
                    SessionTranscriptRole::Assistant => Role::Assistant,
                },
                text: message.text,
                thinking: message.thinking,
                tools: message
                    .tools
                    .into_iter()
                    .map(|tool| ToolActivity {
                        call_id: tool.call_id,
                        name: tool.name,
                        arguments: tool.arguments,
                        output: tool.output,
                        error: tool.is_error,
                        details: None,
                        duration_ms: None,
                    })
                    .collect(),
                error: false,
            })
            .collect();
        self.messages.scroll.scroll_to_end();
        self.model_selection = self
            .models
            .iter()
            .position(|model| model == agent.model())
            .unwrap_or(0);
    }

    fn starting() -> Self {
        let mut app = Self::from_metadata(
            "Starting…".into(),
            "mu".into(),
            "Loading settings and session…".into(),
        );
        app.loading = true;
        app.retry_status = Some("Starting Mu…".into());
        app
    }

    fn from_metadata(model: String, provider: String, session_path: String) -> Self {
        let models = vec![model.clone()];
        Self {
            model,
            models,
            provider,
            composer: Composer::default(),
            session_path,
            messages: Transcript::default(),
            command_menu_open: false,
            command_selection: 0,
            model_selector_open: false,
            picker: None,
            extra_commands: Vec::new(),
            model_selection: 0,
            loading: false,
            quit: false,
            last_ctrl_c: None,
            request: None,
            retry_status: None,
            prompt_queue: AgentPromptQueue::default(),
            queued_prompts: VecDeque::new(),
            hide_thinking: false,
            tools_expanded: false,
            theme: Theme::dark(),
            home: std::env::var("HOME").ok(),
            stats: FooterStats::default(),
            cwd: std::env::current_dir()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
            session_name: None,
            model_names: HashMap::new(),
            statuses: Vec::new(),
        }
    }

    fn refresh_stats(&mut self, agent: &MuAgent) {
        self.stats = agent.footer_stats();
        self.session_name = agent.session_name().map(str::to_owned);
    }

    /// Show a status line after the current messages, replacing a trailing status.
    fn show_status(&mut self, text: String) {
        let position = self.messages.len();
        match self.statuses.last_mut() {
            Some((at, existing)) if *at == position => *existing = text,
            _ => self.statuses.push((position, text)),
        }
    }

    fn input_text(&self) -> String {
        self.composer.input_text()
    }

    fn queue_prompt(&mut self, behavior: PromptBehavior) {
        let prompt = self.composer.expanded_text().trim().to_owned();
        if prompt.is_empty() {
            return;
        }
        self.composer.add_to_history(&prompt);
        self.composer.clear();
        self.composer.undo_stack.clear();
        self.prompt_queue.enqueue(prompt.clone(), behavior);
        self.queued_prompts.push_back((prompt, behavior));
        self.messages.scroll.scroll_to_end();
    }

    fn submit(&mut self, agent: Arc<Mutex<MuAgent>>, tx: mpsc::UnboundedSender<UiEvent>) {
        let prompt = self.composer.expanded_text().trim().to_owned();
        if prompt.is_empty() || self.loading {
            return;
        }
        self.composer.add_to_history(&prompt);
        if let Some(command) = prompt.strip_prefix('!') {
            // `!cmd` and `!!cmd` run a shell command locally; the output is shown, not sent to the model.
            let command = command
                .strip_prefix('!')
                .unwrap_or(command)
                .trim()
                .to_owned();
            self.composer.clear();
            self.run_shell_command(command, agent);
            return;
        }
        self.submit_prompt(prompt, agent, tx);
    }

    fn run_shell_command(&mut self, command: String, agent: Arc<Mutex<MuAgent>>) {
        if command.is_empty() {
            return;
        }
        let arguments = serde_json::json!({ "command": command });
        let outcome = agent.try_lock().ok().map(|agent| {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(agent.run_local_tool("bash", &arguments))
            })
        });
        let (output, error) = match outcome {
            Some(Ok(result)) => (result.content, result.is_error),
            Some(Err(error)) => (error.to_string(), true),
            None => ("Mu is busy.".to_owned(), true),
        };
        self.messages.push(Message {
            role: Role::Assistant,
            text: String::new(),
            thinking: String::new(),
            tools: vec![ToolActivity {
                call_id: "local-shell".into(),
                name: "bash".into(),
                arguments,
                output: Some(output),
                error,
                details: None,
                duration_ms: None,
            }],
            error: false,
        });
        self.messages.scroll.scroll_to_end();
    }

    fn submit_queued_prompt(
        &mut self,
        prompt: String,
        behavior: PromptBehavior,
        agent: Arc<Mutex<MuAgent>>,
        tx: mpsc::UnboundedSender<UiEvent>,
    ) {
        if let Some(index) = self
            .queued_prompts
            .iter()
            .position(|(queued, queued_behavior)| queued == &prompt && *queued_behavior == behavior)
        {
            self.queued_prompts.remove(index);
        }
        self.submit_prompt(prompt, agent, tx);
    }

    fn submit_prompt(
        &mut self,
        prompt: String,
        agent: Arc<Mutex<MuAgent>>,
        tx: mpsc::UnboundedSender<UiEvent>,
    ) {
        self.composer.clear();
        self.composer.undo_stack.clear();
        self.messages.push(Message {
            role: Role::User,
            text: prompt.clone(),
            thinking: String::new(),
            tools: Vec::new(),
            error: false,
        });
        self.messages.push(Message {
            role: Role::Assistant,
            text: String::new(),
            thinking: String::new(),
            tools: Vec::new(),
            error: false,
        });
        self.loading = true;
        self.messages.scroll.scroll_to_end();
        let task = tokio::spawn(async move {
            let mut on_event = |event| {
                let _ = tx.send(UiEvent::Agent(event));
            };
            let result = agent
                .lock()
                .await
                .submit(prompt, &mut on_event)
                .await
                .map_err(|e| e.to_string());
            let _ = tx.send(UiEvent::Finished(result));
        });
        self.request = Some(task.abort_handle());
    }

    fn apply(&mut self, event: UiEvent) {
        match event {
            UiEvent::Agent(AgentEvent::QueuedPromptStarted { prompt, behavior }) => {
                if let Some(index) =
                    self.queued_prompts
                        .iter()
                        .position(|(queued, queued_behavior)| {
                            queued == &prompt && queued_behavior == &behavior
                        })
                {
                    self.queued_prompts.remove(index);
                }
                self.messages.push(Message {
                    role: Role::User,
                    text: prompt,
                    thinking: String::new(),
                    tools: Vec::new(),
                    error: false,
                });
                self.messages.push(Message {
                    role: Role::Assistant,
                    text: String::new(),
                    thinking: String::new(),
                    tools: Vec::new(),
                    error: false,
                });
            }
            UiEvent::Agent(event) => {
                let Some(message) = self.messages.last_mut() else {
                    return;
                };
                match event {
                    AgentEvent::TextDelta(delta) => {
                        self.retry_status = None;
                        message.text.push_str(&delta);
                    }
                    AgentEvent::ThinkingDelta(delta) => {
                        self.retry_status = None;
                        message.thinking.push_str(&delta);
                    }
                    AgentEvent::RetryScheduled {
                        attempt,
                        max_attempts,
                        delay_ms,
                        error,
                    } => {
                        message.text.clear();
                        message.thinking.clear();
                        message.tools.clear();
                        self.retry_status = Some(format!(
                            "Retry {attempt}/{max_attempts} in {:.1}s: {error}",
                            delay_ms as f64 / 1000.0
                        ));
                    }
                    AgentEvent::RetryAttemptStarted { attempt } => {
                        self.retry_status = Some(format!("Retry attempt {attempt} started…"));
                    }
                    AgentEvent::AutoCompactionStarted => {
                        self.retry_status = Some("Context is full; compacting…".into());
                    }
                    AgentEvent::SkillCommandWarning(warning) => {
                        self.retry_status = Some(warning);
                    }
                    AgentEvent::ToolStarted {
                        call_id,
                        name,
                        arguments,
                    } => {
                        self.retry_status = None;
                        message.tools.push(ToolActivity {
                            call_id,
                            name,
                            arguments,
                            output: None,
                            error: false,
                            details: None,
                            duration_ms: None,
                        });
                    }
                    AgentEvent::QueuedPromptStarted { .. } => unreachable!("handled above"),
                    AgentEvent::ToolFinished {
                        call_id,
                        output,
                        is_error,
                        details,
                        duration_ms,
                    } => {
                        self.retry_status = None;
                        if let Some(tool) = message
                            .tools
                            .iter_mut()
                            .find(|tool| tool.call_id == call_id)
                        {
                            tool.output = Some(output);
                            tool.error = is_error;
                            tool.details = details;
                            tool.duration_ms = duration_ms;
                        }
                    }
                }
                // New output follows the end only if the user has not scrolled away.
            }
            UiEvent::Finished(Ok(())) => {
                self.retry_status = None;
                self.loading = false;
                self.request = None;
            }
            UiEvent::Finished(Err(error)) => {
                self.retry_status = None;
                self.loading = false;
                self.request = None;
                if let Some(message) = self.messages.last_mut() {
                    if !message.text.is_empty() {
                        message.text.push_str("\n\n");
                    }
                    message.text.push_str(&format!("Error: {error}"));
                    message.error = true;
                }
                // Keep the user's scroll position when an error arrives.
            }
        }
    }

    fn handle_ctrl_c(&mut self) {
        let now = Instant::now();
        if self
            .last_ctrl_c
            .is_some_and(|last| now.duration_since(last) < Duration::from_millis(500))
        {
            self.quit = true;
        } else {
            self.composer.clear();
            self.last_ctrl_c = Some(now);
        }
    }

    fn cancel(&mut self) {
        self.prompt_queue.clear();
        self.queued_prompts.clear();
        if let Some(request) = self.request.take() {
            request.abort();
        }
        self.retry_status = None;
        self.loading = false;
        if let Some(message) = self.messages.last_mut() {
            message.text.push_str("\n\n(interrupted)");
            message.error = true;
        }
    }
}

type StartupFuture = Pin<Box<dyn Future<Output = Result<MuAgent, AgentError>> + Send>>;

pub async fn run_starting<F>(
    startup: F,
    initial_prompt: Option<String>,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Future<Output = Result<MuAgent, AgentError>> + Send + 'static,
{
    run_loop(None, Some(Box::pin(startup)), initial_prompt).await
}

async fn run_loop(
    initial_agent: Option<MuAgent>,
    startup: Option<StartupFuture>,
    mut initial_prompt: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let _terminal_session = terminal::TerminalSession::enter()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let mut app = initial_agent.as_ref().map_or_else(App::starting, App::new);
    let mut agent = initial_agent.map(|agent| Arc::new(Mutex::new(agent)));
    let (tx, mut rx) = mpsc::unbounded_channel();

    // Draw the startup shell before doing configuration, session, provider, or skill setup.
    terminal.draw(|frame| render(frame, &mut app))?;
    let mut startup_task = startup.map(tokio::spawn);
    loop {
        terminal.draw(|frame| render(frame, &mut app))?;
        if app.quit {
            break;
        }
        tokio::select! {
            // Redraw periodically so time-based UI (the fading scrollbar) updates.
            () = tokio::time::sleep(Duration::from_millis(100)) => {},
            result = async {
                startup_task
                    .as_mut()
                    .expect("startup branch only runs while initialization is pending")
                    .await
            }, if agent.is_none() && startup_task.is_some() => {
                startup_task = None;
                let loaded_agent = match result {
                    Ok(Ok(agent)) => agent,
                    Ok(Err(error)) => return Err(Box::new(error)),
                    Err(error) => return Err(Box::new(error)),
                };
                app.apply_agent_metadata(&loaded_agent);
                app.loading = false;
                app.retry_status = None;
                let loaded_agent = Arc::new(Mutex::new(loaded_agent));
                agent = Some(Arc::clone(&loaded_agent));
                if let Some(prompt) = initial_prompt.take().filter(|prompt| !prompt.trim().is_empty()) {
                    app.composer.insert_text(&prompt);
                    app.submit(loaded_agent, tx.clone());
                }
            },
            event = next_event() => if let Some(event) = event {
                match event {
                    Event::Key(key) => {
                        if let Some(agent) = agent.as_ref() {
                            handle_key(key, &mut app, Arc::clone(agent), tx.clone());
                        } else {
                            handle_startup_key(key, &mut app);
                        }
                    },
                    Event::Mouse(mouse) => handle_mouse(mouse.kind, &mut app),
                    Event::Paste(text) if !app.model_selector_open && agent.is_some() => {
                        app.composer.insert_text(&text);
                        app.command_selection = 0;
                    },
                    Event::Resize(_, _) => {},
                    _ => {},
                }
            },
            event = rx.recv(), if app.loading && agent.is_some() => if let Some(event) = event {
                let finished = matches!(&event, UiEvent::Finished(_));
                app.apply(event);
                if finished {
                    if let Some(Ok(agent)) = agent.as_ref().map(|agent| agent.try_lock()) {
                        app.refresh_stats(&agent);
                    }
                    if let (Some(agent), Some((prompt, behavior))) =
                        (agent.as_ref(), app.prompt_queue.take_next())
                    {
                        app.submit_queued_prompt(prompt, behavior, Arc::clone(agent), tx.clone());
                    }
                }
            },
        }
    }
    if let Some(startup_task) = startup_task {
        startup_task.abort();
    }
    Ok(())
}

fn handle_startup_key(key: KeyEvent, app: &mut App) {
    if key.code == KeyCode::Esc
        || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
        || (key.code == KeyCode::Char('d') && key.modifiers.contains(KeyModifiers::CONTROL))
    {
        app.quit = true;
    }
}

async fn next_event() -> Option<Event> {
    loop {
        if event::poll(Duration::ZERO).ok()? {
            return event::read().ok();
        }
        tokio::time::sleep(Duration::from_millis(24)).await;
    }
}

fn handle_key(
    key: KeyEvent,
    app: &mut App,
    agent: Arc<Mutex<MuAgent>>,
    tx: mpsc::UnboundedSender<UiEvent>,
) {
    if key.code == KeyCode::Char('t') && key.modifiers.contains(KeyModifiers::CONTROL) {
        app.hide_thinking = !app.hide_thinking;
        return;
    }
    if key.code == KeyCode::BackTab && !app.loading {
        // Shift+Tab cycles the thinking level (Pi's `app.thinking.cycle`).
        if let Ok(mut agent) = agent.try_lock() {
            let level = agent.thinking().next();
            agent.set_thinking(level);
            app.show_status(format!("Thinking level: {}", level.name()));
        }
        return;
    }
    if key.code == KeyCode::Char('o') && key.modifiers.contains(KeyModifiers::CONTROL) {
        app.tools_expanded = !app.tools_expanded;
        return;
    }
    if app.loading {
        if key.code == KeyCode::Esc {
            app.cancel();
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            app.cancel();
            app.quit = true;
            return;
        }
        match app.composer.handle_key(key) {
            Some(ComposerAction::Submit | ComposerAction::ExecuteCommand) => {
                app.queue_prompt(PromptBehavior::Steer)
            }
            Some(ComposerAction::FollowUp) => app.queue_prompt(PromptBehavior::FollowUp),
            Some(ComposerAction::OpenCommandMenu) => app.command_menu_open = false,
            None => {}
        }
        return;
    }
    if app.picker.is_some() {
        handle_picker(key, app, &agent);
        return;
    }
    if app.model_selector_open {
        handle_model_selector(key, app, &agent);
        return;
    }
    if app.command_menu_open {
        handle_command_menu(key, app, &agent);
        return;
    }
    if key.code == KeyCode::Char('p') && key.modifiers.contains(KeyModifiers::CONTROL) {
        cycle_model(app, &agent);
        return;
    }
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        app.handle_ctrl_c();
        return;
    }
    if key.code == KeyCode::Char('d')
        && key.modifiers.contains(KeyModifiers::CONTROL)
        && app.input.is_empty()
    {
        app.quit = true;
        return;
    }
    match key.code {
        KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.messages.scroll.scroll_to_start();
            return;
        }
        KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.messages.scroll.scroll_to_end();
            return;
        }
        KeyCode::Up if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.composer.history_up();
            return;
        }
        KeyCode::Down if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.composer.history_down();
            return;
        }
        KeyCode::PageUp => {
            app.messages
                .scroll
                .scroll_by(-app.messages.scroll.page_rows());
            return;
        }
        KeyCode::PageDown => {
            app.messages
                .scroll
                .scroll_by(app.messages.scroll.page_rows());
            return;
        }
        _ => {}
    }
    match app.composer.handle_key(key) {
        Some(ComposerAction::OpenCommandMenu) => {
            app.command_menu_open = true;
            app.command_selection = 0;
        }
        Some(ComposerAction::ExecuteCommand) => execute_command(app, &agent),
        Some(ComposerAction::Submit | ComposerAction::FollowUp) => app.submit(agent, tx),
        None => {}
    }
}

fn handle_command_menu(key: KeyEvent, app: &mut App, agent: &Arc<Mutex<MuAgent>>) {
    let commands = filtered_commands(app);
    match key.code {
        KeyCode::Esc => app.command_menu_open = false,
        KeyCode::Up if !commands.is_empty() => {
            app.command_selection = app
                .command_selection
                .checked_sub(1)
                .unwrap_or(commands.len() - 1);
        }
        KeyCode::Down if !commands.is_empty() => {
            app.command_selection = (app.command_selection + 1) % commands.len();
        }
        KeyCode::Enter => {
            if let Some(command) = commands.get(app.command_selection) {
                if command.is_builtin() {
                    app.input = format!("/{}", command.name).chars().collect();
                    app.cursor = app.input.len();
                    app.command_menu_open = false;
                    execute_command(app, agent);
                } else {
                    // Templates and skills take arguments: complete into the editor instead of running.
                    app.input = format!("/{} ", command.name).chars().collect();
                    app.cursor = app.input.len();
                    app.command_menu_open = false;
                }
            }
        }
        KeyCode::Backspace if app.cursor > 1 => {
            app.cursor -= 1;
            let cursor = app.cursor;
            app.input.remove(cursor);
            app.command_selection = 0;
        }
        KeyCode::Char(ch)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            insert_char(app, ch);
            app.command_selection = 0;
        }
        _ => {}
    }
}

/// A list overlay whose chosen row re-runs a slash command with that row's value as its argument.
struct Picker {
    command: &'static str,
    items: Vec<chrome::SelectItem>,
    values: Vec<String>,
    selection: usize,
}

fn handle_picker(key: KeyEvent, app: &mut App, agent: &Arc<Mutex<MuAgent>>) {
    let Some(picker) = app.picker.as_mut() else {
        return;
    };
    match key.code {
        KeyCode::Esc => app.picker = None,
        KeyCode::Up => picker.selection = picker.selection.saturating_sub(1),
        KeyCode::Down => picker.selection = (picker.selection + 1).min(picker.items.len() - 1),
        KeyCode::Enter => {
            let line = format!("/{} {}", picker.command, picker.values[picker.selection]);
            app.picker = None;
            app.composer.insert_text(&line);
            execute_command(app, agent);
        }
        _ => {}
    }
}

fn handle_model_selector(key: KeyEvent, app: &mut App, agent: &Arc<Mutex<MuAgent>>) {
    match key.code {
        KeyCode::Esc => app.model_selector_open = false,
        KeyCode::Up => app.model_selection = app.model_selection.saturating_sub(1),
        KeyCode::Down if !app.models.is_empty() => {
            app.model_selection = (app.model_selection + 1).min(app.models.len() - 1)
        }
        KeyCode::Enter => {
            if let Some(model) = app.models.get(app.model_selection).cloned() {
                let selected = agent
                    .try_lock()
                    .ok()
                    .and_then(|mut agent| agent.select_model(&model).ok())
                    .unwrap_or(false);
                if selected {
                    let name = app
                        .model_names
                        .get(&model)
                        .cloned()
                        .unwrap_or_else(|| model.clone());
                    app.model = model;
                    app.show_status(format!("Switched to {name}"));
                }
            }
            app.model_selector_open = false;
        }
        _ => {}
    }
}

fn cycle_model(app: &mut App, agent: &Arc<Mutex<MuAgent>>) {
    let Ok(mut agent) = agent.try_lock() else {
        return;
    };
    let scope = agent.cycle_models();
    if scope.len() < 2 {
        return;
    }
    let position = scope.iter().position(|model| *model == app.model);
    let next = position.map_or(0, |index| (index + 1) % scope.len());
    let model = scope[next].clone();
    if !agent.select_model(&model).unwrap_or(false) {
        return;
    }
    app.model_selection = app.models.iter().position(|m| *m == model).unwrap_or(0);
    let name = app
        .model_names
        .get(&model)
        .cloned()
        .unwrap_or_else(|| model.clone());
    app.model = model;
    app.show_status(format!("Switched to {name}"));
}

fn filtered_commands(app: &App) -> Vec<chrome::SlashCommand> {
    let query: String = app.input.iter().skip(1).collect();
    chrome::matching_commands(
        query.split_whitespace().next().unwrap_or_default(),
        &app.extra_commands,
    )
}

/// The session whose name or id (prefix) matches `query`; names win over id prefixes.
fn find_session<'a>(
    sessions: &'a [mu_core::SessionInfo],
    query: &str,
) -> Option<&'a mu_core::SessionInfo> {
    sessions
        .iter()
        .find(|s| s.name.as_deref() == Some(query))
        .or_else(|| sessions.iter().find(|s| s.id.starts_with(query)))
}

/// Boolean settings the `/settings` overlay can flip: (settings key, label).
const TOGGLE_SETTINGS: &[(&str, &str)] = &[
    ("hideThinkingBlock", "Hide thinking blocks"),
    ("quietStartup", "Quiet startup"),
    ("enableSkillCommands", "Skill commands"),
    ("compaction.enabled", "Auto-compaction"),
    ("retry.enabled", "Auto-retry"),
    ("terminal.showImages", "Show images"),
    ("images.autoResize", "Auto-resize images"),
    ("images.blockImages", "Block images"),
    ("collapseChangelog", "Collapse changelog"),
];

fn settings_picker(settings: &mu_core::settings::Settings) -> Picker {
    let state = |key: &str| settings.get(key).and_then(|v| v.as_bool()).unwrap_or(false);
    Picker {
        command: "settings",
        items: TOGGLE_SETTINGS
            .iter()
            .map(|(key, label)| chrome::SelectItem {
                label: (*label).to_owned(),
                description: Some(if state(key) { "on" } else { "off" }.to_owned()),
            })
            .collect(),
        values: TOGGLE_SETTINGS
            .iter()
            .map(|(key, _)| (*key).to_owned())
            .collect(),
        selection: 0,
    }
}

fn session_picker(sessions: &[mu_core::SessionInfo]) -> Picker {
    Picker {
        command: "resume",
        items: sessions
            .iter()
            .map(|s| chrome::SelectItem {
                label: s
                    .name
                    .clone()
                    .or_else(|| s.preview.clone())
                    .unwrap_or_else(|| "(unnamed)".into()),
                description: Some(s.id.chars().take(8).collect()),
            })
            .collect(),
        values: sessions.iter().map(|s| s.id.clone()).collect(),
        selection: 0,
    }
}

fn session_list_text(sessions: &[mu_core::SessionInfo]) -> String {
    if sessions.is_empty() {
        return "No sessions found for this directory.".into();
    }
    let rows = sessions
        .iter()
        .map(|s| {
            format!(
                "{}  {}{}",
                &s.id[..s.id.len().min(8)],
                s.name.as_deref().unwrap_or("(unnamed)"),
                s.preview
                    .as_deref()
                    .map(|p| format!(" — {p}"))
                    .unwrap_or_default()
            )
        })
        .collect::<Vec<_>>();
    format!("Sessions (use /resume <id|name>):\n{}", rows.join("\n"))
}

/// Text of the newest assistant message that has any (what `/copy` copies).
fn last_assistant_text(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| {
            matches!(message.role, Role::Assistant)
                && !message.error
                && !message.text.trim().is_empty()
        })
        .map(|message| message.text.trim().to_owned())
}

/// Shortcuts Mu implements today (Pi's table lists more; unimplemented keys are omitted).
const HOTKEYS: &str = "**Keyboard Shortcuts**

| Key | Action |
|-----|--------|
| `Enter` | Send message (steer while generating) |
| `Shift+Enter` / `Ctrl+J` | New line |
| `Alt+Enter` | Queue follow-up while generating |
| `Up` / `Down` | Move cursor / browse history |
| `PageUp` / `PageDown` | Scroll by page |
| `Ctrl+Home` / `Ctrl+End` | Scroll to start / end |
| `Ctrl+P` | Cycle models |
| `Shift+Tab` | Cycle thinking level |
| `Ctrl+T` | Toggle thinking block visibility |
| `Ctrl+O` | Toggle tool output expansion |
| `Esc` | Interrupt / clear editor |
| `Ctrl+C` | Clear editor (first) / exit (second) |
| `Ctrl+D` | Exit (when editor is empty) |
| `/` | Slash commands |";

/// `/model <id>` or `/model <provider>/<id>`; a unique id substring also matches.
fn resolve_model_argument(models: &[String], argument: &str) -> Option<String> {
    let wanted = argument
        .rsplit('/')
        .next()
        .unwrap_or(argument)
        .to_lowercase();
    if let Some(exact) = models.iter().find(|m| m.to_lowercase() == wanted) {
        return Some(exact.clone());
    }
    let hits = models
        .iter()
        .filter(|m| m.to_lowercase().contains(&wanted))
        .collect::<Vec<_>>();
    (hits.len() == 1).then(|| hits[0].clone())
}

fn execute_command(app: &mut App, agent: &Arc<Mutex<MuAgent>>) {
    let input = app.input_text();
    let command_line = input.trim_start_matches('/');
    let command = command_line.split_whitespace().next().unwrap_or_default();
    let argument = command_line
        .split_once(' ')
        .map(|(_, argument)| argument.trim())
        .unwrap_or_default();
    app.composer.clear();
    app.command_menu_open = false;
    match command {
        "quit" | "exit" => app.quit = true,
        "clear" | "new" => match agent.try_lock() {
            Ok(mut agent) => match agent.start_new_session() {
                Ok(()) => {
                    app.messages.clear();
                    app.queued_prompts.clear();
                    app.session_path = agent.session_path().display().to_string();
                }
                Err(error) => app.messages.push(Message {
                    role: Role::Assistant,
                    text: format!("Could not start a new session: {error}"),
                    thinking: String::new(),
                    tools: Vec::new(),
                    error: true,
                }),
            },
            Err(_) => app.messages.push(Message {
                role: Role::Assistant,
                text: "Could not start a new session while Mu is busy.".into(),
                thinking: String::new(),
                tools: Vec::new(),
                error: true,
            }),
        },
        "clone" => match agent.try_lock() {
            Ok(mut agent) => match agent.clone_session() {
                Ok(()) => {
                    app.session_path = agent.session_path().display().to_string();
                    app.messages.push(Message {
                        role: Role::Assistant,
                        text: format!("Cloned session to {}", app.session_path),
                        thinking: String::new(),
                        tools: Vec::new(),
                        error: false,
                    });
                }
                Err(error) => app.messages.push(Message {
                    role: Role::Assistant,
                    text: format!("Could not clone session: {error}"),
                    thinking: String::new(),
                    tools: Vec::new(),
                    error: true,
                }),
            },
            Err(_) => app.messages.push(Message {
                role: Role::Assistant,
                text: "Could not clone session while Mu is busy.".into(),
                thinking: String::new(),
                tools: Vec::new(),
                error: true,
            }),
        },
        "model" if !argument.is_empty() => {
            // `<model>:<level>` also sets the thinking level, when the suffix is a valid level.
            let (argument, level) = match argument.rsplit_once(':') {
                Some((base, level)) if !base.is_empty() => match level.parse::<mu_core::ThinkingSetting>() {
                    Ok(level) => (base, Some(level)),
                    Err(_) => (argument, None),
                },
                _ => (argument, None),
            };
            let status = match resolve_model_argument(&app.models, argument) {
                None => format!("No unique model matches \"{argument}\""),
                Some(model) => match agent.try_lock() {
                    Err(_) => "Mu is busy.".to_owned(),
                    Ok(mut agent) => match agent.select_model(&model) {
                        Ok(true) => {
                            app.model_selection =
                                app.models.iter().position(|m| *m == model).unwrap_or(0);
                            app.model = model.clone();
                            match level {
                                Some(level) => {
                                    agent.set_thinking(level);
                                    format!("Switched to {model} (thinking: {})", level.name())
                                }
                                None => format!("Switched to {model}"),
                            }
                        }
                        _ => format!("Could not switch to {model}"),
                    },
                },
            };
            app.show_status(status);
        }
        "model" => {
            app.model_selection = app
                .models
                .iter()
                .position(|model| model == &app.model)
                .unwrap_or(0);
            app.model_selector_open = true;
        }
        "name" => match agent.try_lock() {
            Ok(mut agent) if !argument.is_empty() => match agent.name_session(argument) {
                Ok(()) => app.messages.push(Message {
                    role: Role::Assistant,
                    text: format!("Session named: {}", argument.trim()),
                    thinking: String::new(),
                    tools: Vec::new(),
                    error: false,
                }),
                Err(error) => app.messages.push(Message {
                    role: Role::Assistant,
                    text: format!("Could not name session: {error}"),
                    thinking: String::new(),
                    tools: Vec::new(),
                    error: true,
                }),
            },
            Ok(_) => app.messages.push(Message {
                role: Role::Assistant,
                text: "Usage: /name <session name>".into(),
                thinking: String::new(),
                tools: Vec::new(),
                error: true,
            }),
            Err(_) => app.messages.push(Message {
                role: Role::Assistant,
                text: "Could not name session while Mu is busy.".into(),
                thinking: String::new(),
                tools: Vec::new(),
                error: true,
            }),
        },
        "login" => {
            let status = match (agent.try_lock(), argument.split_once(' ')) {
                (Err(_), _) => "Mu is busy.".to_owned(),
                (Ok(_), None) => {
                    "Usage: /login <provider> <api-key> (OAuth login is not supported yet)".to_owned()
                }
                (Ok(agent), Some((provider, key))) => match agent.login_api_key(provider, key) {
                    // The key is never echoed back into the transcript.
                    Ok(()) => format!("Stored API key for {provider} in auth.json"),
                    Err(error) => format!("Could not log in: {error}"),
                },
            };
            app.show_status(status);
        }
        "logout" => {
            let provider = if argument.is_empty() { None } else { Some(argument.to_owned()) };
            let status = match agent.try_lock() {
                Err(_) => "Mu is busy.".to_owned(),
                Ok(agent) => {
                    let provider = provider.unwrap_or_else(|| agent.provider().to_owned());
                    match agent.logout(&provider) {
                        Ok(true) => format!("Removed credentials for {provider}"),
                        Ok(false) => format!("No stored credentials for {provider}"),
                        Err(error) => format!("Could not log out: {error}"),
                    }
                }
            };
            app.show_status(status);
        }
        "copy" => {
            let status = match last_assistant_text(&app.messages) {
                None => "No agent messages to copy yet.".to_owned(),
                Some(text) => match arboard::Clipboard::new().and_then(|mut c| c.set_text(text)) {
                    Ok(()) => "Copied last agent message to clipboard".to_owned(),
                    Err(error) => format!("Could not copy to clipboard: {error}"),
                },
            };
            app.show_status(status);
        }
        "compact" => {
            let agent = Arc::clone(agent);
            let status = match agent.try_lock() {
                Err(_) => "Mu is busy.".to_owned(),
                Ok(mut agent) => {
                    match tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(agent.compact())
                    }) {
                        Ok(Some(outcome)) => {
                            format!("Compacted context (was ~{} tokens)", outcome.tokens_before)
                        }
                        Ok(None) => "Nothing to compact".to_owned(),
                        Err(error) => format!("Compaction failed: {error}"),
                    }
                }
            };
            app.show_status(status);
        }
        "hotkeys" => app.messages.push(Message {
            role: Role::Assistant,
            text: HOTKEYS.into(),
            thinking: String::new(),
            tools: Vec::new(),
            error: false,
        }),
        "thinking" => {
            let reply = match agent.try_lock() {
                Err(_) => "Mu is busy.".to_owned(),
                Ok(agent) if argument.is_empty() => {
                    let levels = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
                    let current = agent.thinking().name();
                    app.picker = Some(Picker {
                        command: "thinking",
                        items: levels
                            .iter()
                            .map(|level| chrome::SelectItem {
                                label: level.to_string(),
                                description: (*level == current).then(|| "current".to_owned()),
                            })
                            .collect(),
                        values: levels.iter().map(|l| l.to_string()).collect(),
                        selection: levels.iter().position(|l| *l == current).unwrap_or(0),
                    });
                    format!("Thinking level: {current}")
                }
                Ok(mut agent) => match argument.parse::<mu_core::ThinkingSetting>() {
                    Ok(level) => {
                        agent.set_thinking(level);
                        format!("Thinking level: {argument}")
                    }
                    Err(message) => message,
                },
            };
            app.show_status(reply);
        }
        "session" => {
            let text = agent
                .try_lock()
                .map(|agent| agent.session_summary())
                .unwrap_or_else(|_| "Session info is unavailable while Mu is busy.".into());
            app.messages.push(Message {
                role: Role::Assistant,
                text,
                thinking: String::new(),
                tools: Vec::new(),
                error: false,
            });
        }
        "export" => match agent.try_lock() {
            Ok(agent) => {
                let default_path;
                let path = if argument.is_empty() {
                    // Pi's default: an HTML export named after the session, in the working directory.
                    let id = agent.session_info().id;
                    default_path = agent.cwd().join(format!("mu-session-{id}.html"));
                    default_path.as_path()
                } else {
                    std::path::Path::new(argument)
                };
                let (text, error) = match agent.export_session(path) {
                    Ok(()) => (format!("Exported session to {}", path.display()), false),
                    Err(error) => (format!("Could not export session: {error}"), true),
                };
                app.messages.push(Message {
                    role: Role::Assistant,
                    text,
                    thinking: String::new(),
                    tools: Vec::new(),
                    error,
                });
            }
            Err(_) => app.messages.push(Message {
                role: Role::Assistant,
                text: "Could not export session while Mu is busy.".into(),
                thinking: String::new(),
                tools: Vec::new(),
                error: true,
            }),
        },
        "trust" => {
            let decision = match argument {
                "" | "yes" | "on" | "true" => Some(true),
                "no" | "off" | "false" => Some(false),
                _ => None,
            };
            let status = match (decision, agent.try_lock()) {
                (None, _) => "Usage: /trust [yes|no]".to_owned(),
                (_, Err(_)) => "Mu is busy.".to_owned(),
                (Some(trusted), Ok(agent)) => match agent.set_project_trust(trusted) {
                    Ok(()) => format!(
                        "Project {}; applies to the next session",
                        if trusted { "trusted" } else { "marked untrusted" }
                    ),
                    Err(error) => format!("Could not save trust: {error}"),
                },
            };
            app.show_status(status);
        }
        "settings" => {
            let status = match agent.try_lock() {
                Err(_) => "Mu is busy.".to_owned(),
                Ok(agent) if !argument.is_empty() => {
                    match TOGGLE_SETTINGS.iter().find(|(key, _)| *key == argument) {
                        None => format!("Unknown setting \"{argument}\""),
                        Some((key, label)) => match agent.toggle_setting(key) {
                            Ok(on) => {
                                if *key == "hideThinkingBlock" {
                                    app.hide_thinking = on;
                                }
                                format!("{label}: {} (saved to settings.json; most settings apply to new sessions)", if on { "on" } else { "off" })
                            }
                            Err(error) => format!("Could not save setting: {error}"),
                        },
                    }
                }
                Ok(agent) => {
                    let settings = agent.settings();
                    app.picker = Some(settings_picker(&settings));
                    "Settings: Enter toggles the selected option.".to_owned()
                }
            };
            app.show_status(status);
        }
        "scoped-models" => {
            let status = match agent.try_lock() {
                Err(_) => "Mu is busy.".to_owned(),
                Ok(agent) => format!(
                    "Ctrl+P cycles: {}\nScope it with the enabledModels setting or --models.",
                    agent.cycle_models().join(", ")
                ),
            };
            app.show_status(status);
        }
        "reload" => {
            let status = match agent.try_lock() {
                Ok(mut agent) => agent.reload_resources(),
                Err(_) => "Mu is busy.".to_owned(),
            };
            app.show_status(status);
        }
        "changelog" => app.show_status(format!(
            "Mu {} has no bundled changelog; see the repository history for changes.",
            env!("CARGO_PKG_VERSION")
        )),
        "share" | "bug" => app.show_status(format!(
            "/{command} is not supported in Mu: it would upload session data to an external service."
        )),
        "tree" => {
            let status = match agent.try_lock() {
                Err(_) => "Mu is busy.".to_owned(),
                Ok(mut agent) => {
                    let rows = agent.tree_rows();
                    match argument.parse::<usize>() {
                        _ if argument.is_empty() => {
                            if rows.is_empty() {
                                "Session is empty.".to_owned()
                            } else {
                                app.picker = Some(Picker {
                                    command: "tree",
                                    items: rows
                                        .iter()
                                        .map(|row| chrome::SelectItem {
                                            label: format!("{}{}", "  ".repeat(row.depth.min(8)), row.label),
                                            description: match (row.is_leaf, row.on_branch) {
                                                (true, _) => Some("current".to_owned()),
                                                (false, false) => Some("other branch".to_owned()),
                                                _ => None,
                                            },
                                        })
                                        .collect(),
                                    values: (1..=rows.len()).map(|n| n.to_string()).collect(),
                                    selection: rows.iter().position(|r| r.is_leaf).unwrap_or(0),
                                });
                                "Session tree: choose an entry to continue from.".to_owned()
                            }
                        }
                        Ok(number) if (1..=rows.len()).contains(&number) => {
                            match agent.navigate_to(&rows[number - 1].id) {
                                Ok(()) => {
                                    app.apply_agent_metadata(&agent);
                                    format!("Switched to entry {number}")
                                }
                                Err(error) => format!("Could not switch: {error}"),
                            }
                        }
                        _ => format!("No entry {argument}"),
                    }
                }
            };
            app.show_status(status);
        }
        "fork" => {
            let status = match agent.try_lock() {
                Err(_) => "Mu is busy.".to_owned(),
                Ok(mut agent) => {
                    let messages = agent.user_messages();
                    match argument.parse::<usize>() {
                        _ if argument.is_empty() => {
                            let rows = messages
                                .iter()
                                .enumerate()
                                .map(|(i, (_, text))| {
                                    let line = text.lines().next().unwrap_or_default();
                                    format!("{}. {}", i + 1, line.chars().take(60).collect::<String>())
                                })
                                .collect::<Vec<_>>();
                            if rows.is_empty() {
                                "No messages to fork from.".to_owned()
                            } else {
                                app.picker = Some(Picker {
                                    command: "fork",
                                    items: messages
                                        .iter()
                                        .map(|(_, text)| chrome::SelectItem {
                                            label: text.lines().next().unwrap_or_default().chars().take(80).collect(),
                                            description: None,
                                        })
                                        .collect(),
                                    values: (1..=messages.len()).map(|n| n.to_string()).collect(),
                                    selection: messages.len() - 1,
                                });
                                "Fork from which message?".to_owned()
                            }
                        }
                        Ok(number) if (1..=messages.len()).contains(&number) => {
                            match agent.fork_before_user_message(&messages[number - 1].0) {
                                Ok(text) => {
                                    app.queued_prompts.clear();
                                    app.apply_agent_metadata(&agent);
                                    app.composer.insert_text(&text);
                                    format!("Forked before message {number}")
                                }
                                Err(error) => format!("Could not fork: {error}"),
                            }
                        }
                        _ => format!("No message {argument}"),
                    }
                }
            };
            app.show_status(status);
        }
        "resume" => {
            let sessions = mu_core::list_sessions_for_current_cwd(None).unwrap_or_default();
            let text = if argument.is_empty() {
                if sessions.is_empty() {
                    session_list_text(&sessions)
                } else {
                    app.picker = Some(session_picker(&sessions));
                    "Resume which session?".to_owned()
                }
            } else {
                match find_session(&sessions, argument) {
                    None => format!("No session matching \"{argument}\""),
                    Some(info) => match agent.try_lock() {
                        Err(_) => "Mu is busy.".to_owned(),
                        Ok(mut agent) => match agent.import_session(&info.path) {
                            Ok(()) => {
                                app.queued_prompts.clear();
                                app.apply_agent_metadata(&agent);
                                format!("Resumed session {}", info.id)
                            }
                            Err(error) => format!("Could not resume: {error}"),
                        },
                    },
                }
            };
            app.show_status(text);
        }
        "import" => match agent.try_lock() {
            Ok(mut agent) if !argument.is_empty() => {
                let (text, error) = match agent.import_session(std::path::Path::new(argument)) {
                    Ok(()) => {
                        app.queued_prompts.clear();
                        app.session_path = agent.session_path().display().to_string();
                        app.apply_agent_metadata(&agent);
                        (format!("Imported session from {}", argument), false)
                    }
                    Err(error) => (format!("Could not import session: {error}"), true),
                };
                app.messages.push(Message {
                    role: Role::Assistant,
                    text,
                    thinking: String::new(),
                    tools: Vec::new(),
                    error,
                });
            }
            Ok(_) => app.messages.push(Message {
                role: Role::Assistant,
                text: "Usage: /import <path.jsonl>".into(),
                thinking: String::new(),
                tools: Vec::new(),
                error: true,
            }),
            Err(_) => app.messages.push(Message {
                role: Role::Assistant,
                text: "Could not import session while Mu is busy.".into(),
                thinking: String::new(),
                tools: Vec::new(),
                error: true,
            }),
        },
        "mcp" => {
            let result = agent.try_lock().map(|agent| match argument {
                "" | "status" => Ok(agent.mcp_status()),
                "tools" => {
                    let tools = agent.mcp_tools();
                    if tools.is_empty() {
                        Ok("No connected MCP tools. Use /mcp status to inspect servers.".into())
                    } else {
                        Ok(format!("MCP tools:\n{}", tools.join("\n")))
                    }
                }
                _ => Err("Usage: /mcp [status|tools]".to_owned()),
            });
            let (text, error) = match result {
                Ok(Ok(text)) => (text, false),
                Ok(Err(message)) => (message, true),
                Err(_) => ("MCP status is unavailable while Mu is busy.".into(), true),
            };
            app.messages.push(Message {
                role: Role::Assistant,
                text,
                thinking: String::new(),
                tools: Vec::new(),
                error,
            });
        }
        "help" => app.messages.push(Message {
            role: Role::Assistant,
            text: "Commands: /help, /model, /mcp [status|tools], /new, /clear, /clone, /name, /session, /export, /import, /quit"
                .into(),
            thinking: String::new(),
            tools: Vec::new(),
            error: false,
        }),
        "" => {}
        other => app.messages.push(Message {
            role: Role::Assistant,
            text: format!("/{other} is not implemented yet."),
            thinking: String::new(),
            tools: Vec::new(),
            error: false,
        }),
    }
    if let Ok(agent) = agent.try_lock() {
        app.model = agent.model().to_owned();
    }
    app.messages.scroll.scroll_to_end();
}

/// Rows per wheel notch (Pi's `fullscreenWheelScrollLines: auto`).
const WHEEL_SCROLL_LINES: i32 = 1;

fn handle_mouse(kind: MouseEventKind, app: &mut App) {
    match kind {
        MouseEventKind::ScrollUp => app.messages.scroll.scroll_by(-WHEEL_SCROLL_LINES),
        MouseEventKind::ScrollDown => app.messages.scroll.scroll_by(WHEEL_SCROLL_LINES),
        _ => {}
    }
}

fn insert_char(app: &mut App, ch: char) {
    app.composer.insert_char(ch);
}

fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    let width = usize::from(area.width);
    // Pi lays text out one column short of the width so the cursor has room.
    app.composer
        .set_visual_width(width.saturating_sub(1).max(1));
    let (rows, cursor) = app.composer.layout_rows();
    let mut editor_scroll = app.composer.scroll_offset;
    let editor = chrome::editor_frame(
        &rows,
        cursor,
        &mut editor_scroll,
        width,
        area.height,
        &app.theme,
    );
    app.composer.scroll_offset = editor_scroll;

    let menu = command_menu_lines(app, width);
    let footer = chrome::footer_lines(
        &chrome::FooterInput {
            cwd: &app.cwd,
            home: app.home.as_deref(),
            git_branch: None,
            session_name: app.session_name.as_deref(),
            stats: &app.stats,
            model_id: &app.model,
            auto_compact: true,
        },
        width,
        &app.theme,
    );
    let dock_height = u16::try_from(editor.lines.len() + menu.len() + footer.len())
        .unwrap_or(u16::MAX)
        .min(area.height);
    // Pi keeps one blank row between the transcript viewport and the editor.
    let spacer = u16::from(area.height > dock_height);
    let transcript = Rect {
        height: area.height - dock_height - spacer,
        ..area
    };
    render_transcript(frame, transcript, app);

    // The row above the editor shows transient status (retries, startup) when there is any.
    if spacer == 1 {
        let status_row = Rect {
            y: transcript.y + transcript.height,
            height: 1,
            ..area
        };
        let status = app.retry_status.clone().unwrap_or_default();
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!(" {status}").trim_end().to_owned(),
                app.theme.fg("dim"),
            ))),
            status_row,
        );
    }
    let mut dock = editor.lines;
    let editor_top = transcript.y + transcript.height + spacer;
    dock.extend(menu);
    dock.extend(footer);
    let dock_area = Rect {
        y: editor_top,
        height: dock_height,
        ..area
    };
    frame.render_widget(Paragraph::new(dock), dock_area);

    if !app.loading && editor_top + (editor.cursor.0 as u16) < area.y + area.height {
        frame.set_cursor_position(Position {
            x: area.x + (editor.cursor.1.min(width.saturating_sub(1))) as u16,
            y: editor_top + editor.cursor.0 as u16,
        });
    }
}

/// Rows of the slash-command list shown under the editor.
fn command_menu_lines(app: &App, width: usize) -> Vec<Line<'static>> {
    if let Some(picker) = &app.picker {
        return chrome::select_list_lines(&picker.items, picker.selection, width, &app.theme);
    }
    if app.model_selector_open {
        let items = app
            .models
            .iter()
            .map(|model| chrome::SelectItem {
                label: app
                    .model_names
                    .get(model)
                    .cloned()
                    .unwrap_or_else(|| model.clone()),
                description: None,
            })
            .collect::<Vec<_>>();
        return chrome::select_list_lines(&items, app.model_selection, width, &app.theme);
    }
    if !app.command_menu_open {
        return Vec::new();
    }
    let items = filtered_commands(app)
        .iter()
        .map(chrome::SelectItem::from)
        .collect::<Vec<_>>();
    if items.is_empty() {
        return Vec::new();
    }
    chrome::select_list_lines(&items, app.command_selection, width, &app.theme)
}

fn render_transcript(frame: &mut Frame, area: Rect, app: &mut App) {
    if area.width == 0 || area.height == 0 {
        app.messages.scroll.set_max_offset(0);
        return;
    }

    let mut lines = chrome::header_lines(
        env!("CARGO_PKG_VERSION"),
        usize::from(area.width),
        &app.theme,
    );
    push_statuses(&mut lines, &app.statuses, 0, &app.theme);
    for (index, message) in app.messages.iter().enumerate() {
        if matches!(message.role, Role::User) {
            lines.push(Line::from(""));
            lines.extend(user_message_lines(&message.text, area.width, &app.theme));
            push_statuses(&mut lines, &app.statuses, index + 1, &app.theme);
            continue;
        }
        let markdown_style = MarkdownStyle::from_theme(&app.theme);
        let thinking_style = app.theme.fg("thinkingText").add_modifier(Modifier::ITALIC);
        let has_visible_content = !message.text.trim().is_empty()
            || (!app.hide_thinking && !message.thinking.trim().is_empty());
        if has_visible_content {
            lines.push(Line::from(""));
        }
        if !message.thinking.trim().is_empty() {
            if app.hide_thinking {
                lines.push(Line::from(vec![
                    Span::raw(" "),
                    Span::styled("Thinking...", thinking_style),
                ]));
            } else {
                lines.extend(render_markdown(
                    message.thinking.trim(),
                    usize::from(area.width),
                    &markdown_style,
                    &MarkdownOptions {
                        padding_x: 1,
                        text: thinking_style,
                        ..MarkdownOptions::default()
                    },
                ));
            }
            if !message.text.trim().is_empty() {
                lines.push(Line::from(""));
            }
        }
        if message.error {
            lines.push(Line::from(vec![
                Span::raw(" "),
                Span::styled(format!("Error: {}", message.text), app.theme.fg("error")),
            ]));
        } else if !message.text.trim().is_empty() {
            lines.extend(render_markdown(
                message.text.trim(),
                usize::from(area.width),
                &markdown_style,
                &MarkdownOptions {
                    padding_x: 1,
                    text: app.theme.fg("text"),
                    ..MarkdownOptions::default()
                },
            ));
        } else if app.loading && message.thinking.is_empty() && message.tools.is_empty() {
            lines.push(Line::from(vec![
                Span::raw(" "),
                Span::styled("Thinking…", app.theme.fg("muted")),
            ]));
        }
        for tool in &message.tools {
            lines.extend(tool_lines(
                &ToolView {
                    name: &tool.name,
                    arguments: &tool.arguments,
                    output: tool.output.as_deref(),
                    details: tool.details.as_ref(),
                    is_error: tool.error,
                    duration_ms: tool.duration_ms,
                    expanded: app.tools_expanded,
                    home: app.home.as_deref(),
                },
                usize::from(area.width),
                &app.theme,
            ));
        }
        push_statuses(&mut lines, &app.statuses, index + 1, &app.theme);
    }
    for (prompt, behavior) in &app.queued_prompts {
        let label = match behavior {
            PromptBehavior::Steer => "steer",
            PromptBehavior::FollowUp => "follow-up",
        };
        lines.push(Line::from(Span::styled(
            format!("↳ queued {label}: {prompt}"),
            Style::default().fg(Color::Yellow),
        )));
    }
    // Wrap once here so every row (including full-width background rows) is exactly one terminal row.
    let lines = lines
        .into_iter()
        .flat_map(|line| wrap_line(line, usize::from(area.width.max(1))))
        .collect::<Vec<_>>();
    let content_lines = lines.len();
    let paragraph = Paragraph::new(lines);
    let max_scroll =
        u16::try_from(content_lines.saturating_sub(usize::from(area.height))).unwrap_or(u16::MAX);
    app.messages.scroll.set_max_offset(max_scroll);
    app.messages.scroll.viewport_height = area.height;
    let scroll = app.messages.scroll.offset();
    frame.render_widget(paragraph.scroll((scroll, 0)), area);
    if !app.messages.scroll.is_following_end() && area.height > 0 {
        let hint_area = Rect {
            x: area.x,
            y: area.y + area.height - 1,
            width: area.width.saturating_sub(1),
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(" ↓ Jump to latest message · Ctrl+End ")
                .alignment(ratatui::layout::Alignment::Center)
                .style(app.theme.fg("muted")),
            hint_area,
        );
    }
    if content_lines > usize::from(area.height) && app.messages.scroll.scrollbar_visible() {
        paint_scrollbar(frame, area, content_lines, usize::from(scroll), &app.theme);
    }
}

/// Pi's scrollbar: a `│` track with a `┃` thumb painted over the last column.
fn paint_scrollbar(frame: &mut Frame, area: Rect, content: usize, scroll: usize, theme: &Theme) {
    let track = usize::from(area.height);
    let thumb_height = ((track * track) as f64 / content as f64)
        .round()
        .clamp(track.min(2) as f64, track as f64) as usize;
    let max_scroll = content.saturating_sub(track);
    let max_thumb_top = track - thumb_height;
    let thumb_top = if max_scroll == 0 {
        0
    } else {
        (scroll as f64 / max_scroll as f64 * max_thumb_top as f64).round() as usize
    };
    let x = area.x + area.width.saturating_sub(1);
    let buffer = frame.buffer_mut();
    for offset in 0..track {
        let is_thumb = (thumb_top..thumb_top + thumb_height).contains(&offset);
        let (symbol, style) = if is_thumb {
            ("┃", theme.fg("scrollbarThumb"))
        } else {
            ("│", theme.fg("scrollbarTrack"))
        };
        if let Some(cell) = buffer.cell_mut((x, area.y + offset as u16)) {
            cell.set_symbol(symbol).set_style(style);
        }
    }
}

/// Wrap a styled line to `width`, keeping its line style (background) on every row.
fn wrap_line(line: Line<'static>, width: usize) -> Vec<Line<'static>> {
    let used: usize = line.spans.iter().map(|span| span.content.width()).sum();
    if used <= width {
        return vec![line];
    }
    let style = line.style;
    markdown::wrap_spans(&line.spans, width)
        .into_iter()
        .map(|spans| Line::from(spans).style(style))
        .collect()
}

/// Status lines recorded after `position` messages: a spacer and a dim line each.
fn push_statuses(
    lines: &mut Vec<Line<'static>>,
    statuses: &[(usize, String)],
    position: usize,
    theme: &Theme,
) {
    for (_, text) in statuses.iter().filter(|(at, _)| *at == position) {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!(" {text}"),
            theme.fg("dim"),
        )));
    }
}

/// A user message: Markdown on the `userMessageBg` background with one row of padding.
fn user_message_lines(text: &str, width: u16, theme: &Theme) -> Vec<Line<'static>> {
    render_markdown(
        text,
        usize::from(width),
        &MarkdownStyle::from_theme(theme),
        &MarkdownOptions {
            padding_x: 1,
            padding_y: 1,
            text: theme.fg("userMessageText"),
            background: Some(theme.color("userMessageBg")),
            preserve_list_markers: true,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{
        AgentEvent, App, Composer, ComposerAction, Message, Role, ToolActivity, UiEvent,
        execute_command, handle_key, handle_mouse, handle_startup_key, insert_char, render,
    };
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEventKind};
    use mu_core::{MuAgent, PromptBehavior};
    use ratatui::{Terminal, backend::TestBackend};
    use std::sync::Arc;
    use tokio::sync::{Mutex, mpsc};

    #[test]
    fn startup_shell_renders_before_agent_metadata_is_available() {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        let mut app = App::starting();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let buffer = terminal.backend().buffer();
        let screen = (0..24)
            .map(|y| (0..80).map(|x| buffer[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(screen.contains("Starting…"));
        assert!(screen.contains("Starting Mu…"));
        assert!(app.loading);
        assert!(app.messages.is_empty());

        handle_startup_key(
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
            &mut app,
        );
        assert!(app.quit);
    }

    #[test]
    fn skill_command_read_error_is_visible_in_the_tui() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        app.messages.push(Message {
            role: Role::Assistant,
            text: String::new(),
            thinking: String::new(),
            tools: Vec::new(),
            error: false,
        });
        app.apply(UiEvent::Agent(AgentEvent::SkillCommandWarning(
            "Could not read skill 'review': permission denied".into(),
        )));

        let mut terminal = Terminal::new(TestBackend::new(100, 24)).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let buffer = terminal.backend().buffer();
        let screen = (0..24)
            .map(|y| {
                (0..100)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(screen.contains("Could not read skill 'review': permission denied"));
    }

    #[tokio::test]
    async fn retry_status_is_visible_and_cancel_marks_the_turn_interrupted() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        app.loading = true;
        app.messages.push(Message {
            role: Role::Assistant,
            text: "partial response".into(),
            thinking: "partial reasoning".into(),
            tools: Vec::new(),
            error: false,
        });

        app.apply(UiEvent::Agent(AgentEvent::RetryScheduled {
            attempt: 1,
            max_attempts: 3,
            delay_ms: 2_000,
            error: "provider: HTTP 503".into(),
        }));
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let buffer = terminal.backend().buffer();
        let screen = (0..24)
            .map(|y| (0..80).map(|x| buffer[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(screen.contains("Retry 1/3 in 2.0s: provider: HTTP 503"));
        assert!(app.messages.last().unwrap().text.is_empty());
        assert!(app.messages.last().unwrap().thinking.is_empty());

        let task = tokio::spawn(std::future::pending::<()>());
        app.request = Some(task.abort_handle());
        app.cancel();

        assert!(task.await.unwrap_err().is_cancelled());
        assert!(app.retry_status.is_none());
        assert!(app.messages.last().unwrap().text.contains("(interrupted)"));
    }

    #[tokio::test]
    async fn export_command_supports_html_and_jsonl_import_round_trips() {
        let path = std::env::temp_dir().join(format!(
            "mu-tui-session-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let agent = Arc::new(Mutex::new(MuAgent::demo().expect("demo agent")));
        let mut app = App::from_metadata("demo".into(), "mu-demo".into(), "session".into());
        app.input = "/name test session".chars().collect();
        execute_command(&mut app, &agent);
        assert_eq!(
            app.messages.last().unwrap().text,
            "Session named: test session"
        );
        app.input = "/session".chars().collect();
        execute_command(&mut app, &agent);
        assert!(
            app.messages
                .last()
                .unwrap()
                .text
                .contains("Name: test session")
        );
        app.input = "/mcp tools".chars().collect();
        execute_command(&mut app, &agent);
        assert!(
            app.messages
                .last()
                .unwrap()
                .text
                .contains("No connected MCP tools")
        );

        app.input = format!("/export {}", path.display()).chars().collect();
        execute_command(&mut app, &agent);
        assert!(path.is_file());
        assert!(
            app.messages
                .last()
                .unwrap()
                .text
                .starts_with("Exported session to")
        );

        let html_path = path.with_extension("html");
        app.input = format!("/export {}", html_path.display()).chars().collect();
        execute_command(&mut app, &agent);
        assert!(
            std::fs::read_to_string(&html_path)
                .unwrap()
                .starts_with("<!doctype html>")
        );
        std::fs::remove_file(html_path).unwrap();

        app.input = format!("/import {}", path.display()).chars().collect();
        execute_command(&mut app, &agent);
        assert!(
            app.messages
                .last()
                .unwrap()
                .text
                .starts_with("Imported session from")
        );
        assert_eq!(app.session_path, path.display().to_string());
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn ctrl_o_toggles_expanded_tool_output() {
        let agent = Arc::new(Mutex::new(MuAgent::demo().expect("demo agent")));
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = App::from_metadata("demo".into(), "mu-demo".into(), "session".into());

        handle_key(
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            &mut app,
            Arc::clone(&agent),
            tx.clone(),
        );
        assert!(app.tools_expanded);
        assert!(app.input_text().is_empty());

        handle_key(
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            &mut app,
            agent,
            tx,
        );
        assert!(!app.tools_expanded);
    }

    #[tokio::test]
    async fn enter_while_generating_queues_steering_and_alt_enter_queues_follow_up() {
        let agent = Arc::new(Mutex::new(MuAgent::demo().expect("demo agent")));
        let mut app = App::from_metadata("demo".into(), "mu-demo".into(), "session".into());
        app.prompt_queue = agent.lock().await.prompt_queue();
        app.loading = true;
        let (tx, _rx) = mpsc::unbounded_channel();

        app.composer.insert_text("add a test");
        handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut app,
            Arc::clone(&agent),
            tx.clone(),
        );
        assert_eq!(app.input_text(), "");
        assert_eq!(app.queued_prompts.len(), 1);
        assert_eq!(app.queued_prompts[0].0, "add a test");
        assert_eq!(app.queued_prompts[0].1, PromptBehavior::Steer);

        app.composer.insert_text("then summarize");
        handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT),
            &mut app,
            agent,
            tx,
        );
        assert_eq!(app.queued_prompts.len(), 2);
        assert_eq!(app.queued_prompts[1].0, "then summarize");
        assert_eq!(app.queued_prompts[1].1, PromptBehavior::FollowUp);

        app.apply(UiEvent::Agent(AgentEvent::QueuedPromptStarted {
            prompt: "add a test".into(),
            behavior: PromptBehavior::Steer,
        }));
        assert_eq!(app.queued_prompts.len(), 1);
        assert_eq!(app.messages[0].text, "add a test");
        assert!(matches!(app.messages[0].role, Role::User));
    }

    #[test]
    fn resume_matches_names_before_id_prefixes() {
        let info = |id: &str, name: Option<&str>| mu_core::SessionInfo {
            id: id.into(),
            name: name.map(str::to_owned),
            ..mu_core::SessionInfo::default()
        };
        let sessions = vec![info("abc123", None), info("zzz", Some("abc"))];
        assert_eq!(super::find_session(&sessions, "abc").unwrap().id, "zzz");
        assert_eq!(super::find_session(&sessions, "abc1").unwrap().id, "abc123");
        assert!(super::find_session(&sessions, "nope").is_none());
        assert!(super::session_list_text(&sessions).contains("abc123  (unnamed)"));
        assert!(super::session_list_text(&[]).starts_with("No sessions"));
    }

    #[tokio::test]
    async fn share_and_bug_explain_that_uploads_are_unsupported() {
        let agent = Arc::new(Mutex::new(MuAgent::demo().expect("demo agent")));
        let mut app = App::from_metadata("demo".into(), "mu-demo".into(), "session".into());
        for command in ["/share", "/bug broken"] {
            app.input = command.chars().collect();
            execute_command(&mut app, &agent);
            assert!(
                app.statuses
                    .last()
                    .unwrap()
                    .1
                    .contains("not supported in Mu")
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn bang_prefix_runs_a_local_shell_command_and_shows_its_output() {
        let agent = Arc::new(Mutex::new(MuAgent::demo().expect("demo agent")));
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = App::from_metadata("demo".into(), "mu-demo".into(), "session".into());
        app.composer.insert_text("!!echo local-ok");
        app.submit(agent, tx);
        let tool = &app.messages.last().unwrap().tools[0];
        assert_eq!(tool.name, "bash");
        assert!(tool.output.as_deref().unwrap().contains("local-ok"));
        assert!(app.input.is_empty() && !app.loading);
    }

    #[test]
    fn model_argument_matches_by_id_provider_prefix_or_unique_substring() {
        let models = ["mu-demo", "mu-demo-alt", "other"]
            .map(String::from)
            .to_vec();
        let find = |q| super::resolve_model_argument(&models, q);
        assert_eq!(find("mu-demo-alt").as_deref(), Some("mu-demo-alt"));
        assert_eq!(
            find("mu-demo").as_deref(),
            Some("mu-demo"),
            "exact beats substring"
        );
        assert_eq!(find("acme/OTHER").as_deref(), Some("other"));
        assert_eq!(find("alt").as_deref(), Some("mu-demo-alt"));
        assert_eq!(find("mu"), None, "ambiguous");
        assert_eq!(find("zzz"), None);
    }

    #[tokio::test]
    async fn shift_tab_cycles_the_thinking_level() {
        let agent = Arc::new(Mutex::new(MuAgent::demo().expect("demo agent")));
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = App::from_metadata("demo".into(), "mu-demo".into(), "session".into());
        handle_key(
            KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
            &mut app,
            agent.clone(),
            tx,
        );
        assert_eq!(
            agent.lock().await.thinking(),
            mu_core::ThinkingSetting::High
        );
        assert_eq!(app.statuses.last().unwrap().1, "Thinking level: high");
    }

    #[tokio::test]
    async fn thinking_picker_opens_navigates_and_applies_the_chosen_level() {
        let agent = Arc::new(Mutex::new(MuAgent::demo().expect("demo agent")));
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = App::from_metadata("demo".into(), "mu-demo".into(), "session".into());
        app.composer.insert_text("/thinking");
        execute_command(&mut app, &agent);
        assert!(app.picker.is_some());
        let press = |app: &mut App, code| {
            handle_key(
                KeyEvent::new(code, KeyModifiers::NONE),
                app,
                agent.clone(),
                tx.clone(),
            )
        };
        press(&mut app, KeyCode::Up);
        press(&mut app, KeyCode::Up);
        press(&mut app, KeyCode::Enter);
        assert!(app.picker.is_none());
        assert_eq!(agent.lock().await.thinking().name(), "minimal");
        app.composer.insert_text("/thinking");
        execute_command(&mut app, &agent);
        press(&mut app, KeyCode::Esc);
        assert!(app.picker.is_none());
    }

    #[test]
    fn settings_picker_shows_current_values_with_defaults() {
        let dir = std::env::temp_dir().join(format!("mu-settings-ui-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("settings.json"), r#"{"quietStartup":true}"#).unwrap();
        let settings = mu_core::settings::Settings::load(&dir, &dir, true);
        let picker = super::settings_picker(&settings);
        let state = |key: &str| {
            let at = picker.values.iter().position(|v| v == key).unwrap();
            picker.items[at].description.clone().unwrap()
        };
        assert_eq!(state("quietStartup"), "on");
        assert_eq!(state("compaction.enabled"), "on", "Pi default");
        assert_eq!(state("hideThinkingBlock"), "off");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn template_and_skill_commands_follow_builtins_and_complete_into_the_editor() {
        let agent = Arc::new(Mutex::new(MuAgent::demo().expect("demo agent")));
        let mut app = App::from_metadata("demo".into(), "mu-demo".into(), "session".into());
        app.extra_commands = vec![super::chrome::SlashCommand::dynamic(
            "review".into(),
            "Review code".into(),
            Some("<file>".into()),
        )];
        app.input = "/rev".chars().collect();
        app.cursor = app.input.len();
        app.command_menu_open = true;
        let names = super::filtered_commands(&app)
            .iter()
            .map(|c| c.name.to_string())
            .collect::<Vec<_>>();
        assert!(names.contains(&"review".to_owned()));
        let at = names.iter().position(|n| n == "review").unwrap();
        app.command_selection = at;
        super::handle_command_menu(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut app,
            &agent,
        );
        assert_eq!(
            app.input_text(),
            "/review ",
            "completes instead of executing"
        );
        assert!(!app.command_menu_open);
        app.input = "/".chars().collect();
        let all = super::filtered_commands(&app);
        assert!(
            all[..24]
                .iter()
                .all(super::chrome::SlashCommand::is_builtin)
        );
        assert_eq!(all[24].name, "review");
    }

    #[tokio::test]
    async fn login_without_arguments_shows_usage_and_never_writes_credentials() {
        let agent = Arc::new(Mutex::new(MuAgent::demo().expect("demo agent")));
        let mut app = App::from_metadata("demo".into(), "mu-demo".into(), "session".into());
        app.composer.insert_text("/login acme");
        execute_command(&mut app, &agent);
        assert!(
            app.statuses
                .last()
                .unwrap()
                .1
                .starts_with("Usage: /login <provider> <api-key>")
        );
    }

    #[test]
    fn session_picker_lists_names_and_resumes_by_full_id() {
        let info = |id: &str, name: Option<&str>| mu_core::SessionInfo {
            id: id.into(),
            name: name.map(str::to_owned),
            ..Default::default()
        };
        let picker =
            super::session_picker(&[info("abcdef0123456789", Some("work")), info("zzzz", None)]);
        assert_eq!(picker.items[0].label, "work");
        assert_eq!(picker.items[0].description.as_deref(), Some("abcdef01"));
        assert_eq!(picker.items[1].label, "(unnamed)");
        assert_eq!(picker.values[0], "abcdef0123456789");
    }

    #[test]
    fn copy_targets_the_newest_non_empty_assistant_message() {
        let message = |role, text: &str, error| Message {
            role,
            text: text.into(),
            thinking: String::new(),
            tools: vec![],
            error,
        };
        let messages = vec![
            message(Role::Assistant, "first", false),
            message(Role::User, "question", false),
            message(Role::Assistant, "  second  ", false),
            message(Role::Assistant, "", false),
            message(Role::Assistant, "boom", true),
        ];
        assert_eq!(
            super::last_assistant_text(&messages).as_deref(),
            Some("second")
        );
        assert_eq!(super::last_assistant_text(&[]), None);
    }

    #[tokio::test]
    async fn hotkeys_command_lists_implemented_shortcuts() {
        let agent = Arc::new(Mutex::new(MuAgent::demo().expect("demo agent")));
        let mut app = App::from_metadata("demo".into(), "mu-demo".into(), "session".into());
        app.input = "/hotkeys".chars().collect();
        execute_command(&mut app, &agent);
        let text = &app.messages.last().unwrap().text;
        assert!(text.contains("`Ctrl+O` | Toggle tool output expansion"));
    }

    #[tokio::test]
    async fn thinking_command_reports_and_sets_the_level() {
        let agent = Arc::new(Mutex::new(MuAgent::demo().expect("demo agent")));
        let mut app = App::from_metadata("demo".into(), "mu-demo".into(), "session".into());
        app.input = "/thinking high".chars().collect();
        execute_command(&mut app, &agent);
        assert_eq!(app.statuses.last().unwrap().1, "Thinking level: high");
        assert_eq!(
            agent.lock().await.thinking(),
            mu_core::ThinkingSetting::High
        );
        app.input = "/thinking".chars().collect();
        execute_command(&mut app, &agent);
        assert_eq!(app.statuses.last().unwrap().1, "Thinking level: high");
        app.input = "/thinking loud".chars().collect();
        execute_command(&mut app, &agent);
        assert!(
            app.statuses
                .last()
                .unwrap()
                .1
                .starts_with("Invalid thinking level")
        );
    }

    #[test]
    fn renders_pi_style_header_composer_and_footer() {
        let mut app = App::from_metadata(
            "gpt-6-luna".into(),
            "opencode-cf".into(),
            "/tmp/mu-session.jsonl".into(),
        );
        app.messages.push(Message {
            role: Role::User,
            text: "Check this project".into(),
            thinking: String::new(),
            tools: Vec::new(),
            error: false,
        });
        app.messages.push(Message {
            role: Role::Assistant,
            text: "Ready.".into(),
            thinking: String::new(),
            tools: Vec::new(),
            error: false,
        });
        let mut terminal = Terminal::new(TestBackend::new(100, 24)).expect("test terminal");
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render");
        let screen = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(screen.contains("gpt-6-luna"));
        assert!(screen.contains("Check this project"));
        assert!(screen.contains("Ready."));
        // User messages sit on Pi's `userMessageBg` background.
        let buffer = terminal.backend().buffer();
        let user_row = row_with_text(terminal.backend(), "Check this project");
        let background = buffer[(0, u16::try_from(user_row).unwrap())].bg;
        assert_eq!(
            background,
            super::theme::Theme::dark().color("userMessageBg")
        );
        assert!(screen.contains("0.0%/0 (auto)"));
    }

    fn tool_activity(name: &str, arguments: serde_json::Value, output: &str) -> ToolActivity {
        ToolActivity {
            call_id: format!("{name}-call"),
            name: name.into(),
            arguments,
            output: Some(output.into()),
            error: false,
            details: None,
            duration_ms: None,
        }
    }

    fn screen_text(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>()
    }

    #[test]
    fn renders_compact_tool_output_and_expands_with_ctrl_o() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        let long_output = (1..=9)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        app.messages.push(Message {
            role: Role::Assistant,
            text: String::new(),
            thinking: String::new(),
            tools: vec![
                tool_activity(
                    "read",
                    serde_json::json!({"path":"/project/.pi/skills/rust/SKILL.md"}),
                    "specialized instructions",
                ),
                tool_activity("bash", serde_json::json!({"command":"seq 9"}), &long_output),
            ],
            error: false,
        });
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).expect("test terminal");

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let compact = screen_text(&terminal);
        assert!(compact.contains("[skill] rust"));
        assert!(compact.contains("ctrl+o to expand"));
        assert!(compact.contains("$ seq 9"));
        assert!(compact.contains("4 earlier lines"));
        assert!(compact.contains("line 9"));
        assert!(!compact.contains("line 1 "));
        assert!(!compact.contains("specialized instructions"));

        app.tools_expanded = true;
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let expanded = screen_text(&terminal);
        assert!(expanded.contains("SKILL.md"));
        assert!(expanded.contains("specialized instructions"));
        assert!(expanded.contains("line 1"));
        assert!(!expanded.contains("earlier lines"));
    }

    #[test]
    fn renders_streaming_thinking_and_tool_activity_in_the_assistant_turn() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        app.messages.push(Message {
            role: Role::Assistant,
            text: String::new(),
            thinking: String::new(),
            tools: Vec::new(),
            error: false,
        });
        app.loading = true;
        app.apply(UiEvent::Agent(AgentEvent::ThinkingDelta(
            "Inspecting the source".into(),
        )));
        app.apply(UiEvent::Agent(AgentEvent::ToolStarted {
            call_id: "call-1".into(),
            name: "read".into(),
            arguments: serde_json::json!({"path":"src/main.rs"}),
        }));
        app.apply(UiEvent::Agent(AgentEvent::ToolFinished {
            call_id: "call-1".into(),
            output: "fn main() {}".into(),
            is_error: false,
            details: None,
            duration_ms: None,
        }));
        app.apply(UiEvent::Agent(AgentEvent::TextDelta("Done.".into())));

        let mut terminal = Terminal::new(TestBackend::new(100, 24)).expect("test terminal");
        app.hide_thinking = true;
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render hidden thinking");
        let hidden = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(!hidden.contains("Inspecting the source"));

        app.hide_thinking = false;
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render assistant activity");
        let screen = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(screen.contains("Inspecting the source"));
        assert!(screen.contains("read"));
        assert!(screen.contains("src/main.rs"));
        // Pi collapses read results to the call line until expanded.
        assert!(!screen.contains("fn main()"));
        assert!(screen.contains("Done."));
    }

    #[test]
    fn composer_owns_key_to_action_mapping() {
        let mut composer = Composer::default();
        assert_eq!(
            composer.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE)),
            Some(ComposerAction::OpenCommandMenu)
        );
        assert_eq!(
            composer.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            Some(ComposerAction::ExecuteCommand)
        );

        composer.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        composer.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        assert_eq!(
            composer.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT)),
            None
        );
        assert_eq!(composer.input_text(), "x\n");
        assert_eq!(
            composer.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            Some(ComposerAction::Submit)
        );

        composer.clear();
        composer.insert_text("ab");
        composer.cursor = 0;
        composer.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL));
        assert_eq!(composer.input_text(), "b");
    }

    #[test]
    fn plain_enter_after_backslash_inserts_a_newline() {
        let mut composer = Composer::default();
        composer.insert_char('\\');
        composer.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(composer.input_text(), "\n");
    }

    #[test]
    fn undo_coalesces_typed_words_like_pi() {
        let mut composer = Composer::default();
        for character in "ab c".chars() {
            composer.insert_char(character);
        }

        composer.undo();
        assert_eq!(composer.input_text(), "ab");
        composer.undo();
        assert_eq!(composer.input_text(), "");
    }

    #[test]
    fn kill_ring_yank_pop_cycles_to_the_previous_kill() {
        let mut composer = Composer {
            kill_ring: vec!["older kill".into(), "newer kill".into()],
            ..Composer::default()
        };
        composer.yank();
        assert_eq!(composer.input_text(), "newer kill");

        composer.yank_pop();
        assert_eq!(composer.input_text(), "older kill");
    }

    #[test]
    fn ctrl_c_clears_first_and_exits_on_a_second_press() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        app.composer.insert_text("draft");

        app.handle_ctrl_c();
        assert_eq!(app.input_text(), "");
        assert!(!app.quit);

        app.handle_ctrl_c();
        assert!(app.quit);
    }

    #[test]
    fn up_and_down_move_between_lines_before_using_prompt_history() {
        let mut composer = Composer::default();
        composer.set_visual_width(80);
        composer.insert_text("first\nsecond");
        composer.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(composer.cursor, "first".len());

        composer.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(composer.cursor, 0);
        composer.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(composer.cursor, "first\n".len());
    }

    #[test]
    fn vertical_cursor_movement_tracks_wrapped_visual_lines_and_sticky_column() {
        let mut composer = Composer::default();
        composer.set_visual_width(4);
        composer.insert_text("abcdefgh");
        composer.cursor = 6;

        composer.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(composer.cursor, 2);
        composer.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(composer.cursor, 6);
    }

    #[test]
    fn cursor_movement_and_backspace_treat_combining_graphemes_as_one_character() {
        let mut composer = Composer::default();
        composer.insert_text("a👩‍👩‍👧‍👦e\u{301}b");
        composer.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        composer.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));

        assert_eq!(composer.input_text(), "a👩‍👩‍👧‍👦b");
        composer.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(composer.input_text(), "ab");
    }

    #[test]
    fn history_navigation_restores_the_draft_and_cursor() {
        let mut composer = Composer::default();
        composer.add_to_history("older prompt");
        composer.add_to_history("newer prompt");
        composer.insert_text("my draft");
        composer.cursor = 3;

        composer.history_up();
        assert_eq!(composer.input_text(), "newer prompt");
        assert_eq!(composer.cursor, 0);
        composer.history_up();
        assert_eq!(composer.input_text(), "older prompt");
        composer.history_down();
        assert_eq!(composer.input_text(), "newer prompt");
        composer.history_down();
        assert_eq!(composer.input_text(), "my draft");
        assert_eq!(composer.cursor, 3);
    }

    #[test]
    fn pasted_text_is_inserted_at_the_cursor_with_normalized_newlines() {
        let mut composer = super::Composer::default();
        composer.insert_text("ab\r\ncd\ref");
        composer.cursor = 2;
        composer.insert_text("界\n");

        assert_eq!(composer.input_text(), "ab界\n\ncd\nef");
        assert_eq!(composer.cursor, 4);
    }

    #[test]
    fn large_pastes_use_atomic_visible_markers_but_expand_for_submission() {
        let mut composer = Composer::default();
        let pasted = (0..11)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        composer.insert_text(&pasted);

        assert_eq!(composer.input_text(), "[paste #1 +11 lines]");
        assert_eq!(composer.expanded_text(), pasted);
        composer.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        assert_eq!(composer.cursor, 0);
        composer.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        assert_eq!(composer.cursor, composer.input_text().chars().count());
        composer.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(composer.input_text(), "");
        composer.undo();
        assert_eq!(composer.expanded_text(), pasted);
    }

    #[test]
    fn paste_expands_tabs_filters_control_characters_and_separates_paths() {
        let mut composer = Composer::default();
        composer.insert_text("src");
        composer.insert_text("/tmp/file");
        assert_eq!(composer.input_text(), "src /tmp/file");
        composer.clear();
        composer.insert_text("one\ttwo\u{1}");

        assert_eq!(composer.input_text(), "one    two");
    }

    #[test]
    fn supports_multiline_composition_and_prompt_history() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        insert_char(&mut app, 'a');
        insert_char(&mut app, '\n');
        insert_char(&mut app, 'b');
        assert_eq!(app.input_text(), "a\nb");

        app.prompt_history.push("previous prompt".into());
        app.composer.history_up();
        assert_eq!(app.input_text(), "previous prompt");
        app.composer.history_down();
        assert_eq!(app.input_text(), "a\nb");
    }

    #[test]
    fn shift_enter_keeps_a_trailing_blank_line_visible() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("test terminal");
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render empty composer");
        let empty_title_row = border_rows(terminal.backend())[0];

        app.composer
            .handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
        assert_eq!(app.input_text().chars().collect::<Vec<_>>(), vec!['\n']);
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render trailing blank line");

        let multiline_title_row = border_rows(terminal.backend())[0];
        assert_eq!(multiline_title_row + 1, empty_title_row);
        let cursor = terminal.get_cursor_position().unwrap();
        assert_eq!(usize::from(cursor.y), multiline_title_row + 2);
    }

    #[test]
    fn composer_grows_with_multiline_input_and_shrinks_after_deletion() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("test terminal");
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render empty composer");
        let empty_title_row = border_rows(terminal.backend())[0];

        for ch in "first row".chars() {
            insert_char(&mut app, ch);
        }
        insert_char(&mut app, '\n');
        for ch in "second row".chars() {
            insert_char(&mut app, ch);
        }
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render multiline composer");
        let multiline_title_row = border_rows(terminal.backend())[0];
        assert!(multiline_title_row < empty_title_row);
        let screen = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(screen.contains("first row"));
        assert!(screen.contains("second row"));

        app.composer.delete_to_line_start();
        let cursor = app.cursor - 1;
        app.input.remove(cursor);
        app.cursor -= 1;
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render shrunk composer");
        assert_eq!(app.input_text(), "first row");
        assert_eq!(border_rows(terminal.backend())[0], empty_title_row);
    }

    #[test]
    fn mouse_wheel_scrolls_a_long_conversation_and_clamps_at_the_bottom() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        app.messages.extend((0..20).map(|index| Message {
            role: if index % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            },
            text: format!("conversation line {index}"),
            thinking: String::new(),
            tools: Vec::new(),
            error: false,
        }));
        let mut terminal = Terminal::new(TestBackend::new(80, 20)).expect("test terminal");
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render conversation");
        let bottom = app.messages.scroll.offset();
        assert!(bottom > 0);

        handle_mouse(MouseEventKind::ScrollUp, &mut app);
        assert_eq!(app.messages.scroll.offset(), bottom - 1);
        handle_mouse(MouseEventKind::ScrollDown, &mut app);
        assert_eq!(app.messages.scroll.offset(), bottom);
        assert!(app.messages.scroll.is_following_end());
        handle_mouse(MouseEventKind::ScrollDown, &mut app);
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render at bottom");
        assert_eq!(app.messages.scroll.offset(), bottom);
    }

    #[test]
    fn transcript_header_scrolls_with_history_while_dock_stays_visible() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        app.messages.extend((0..20).map(|index| Message {
            role: if index % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            },
            text: format!("conversation line {index}"),
            thinking: String::new(),
            tools: Vec::new(),
            error: false,
        }));
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).expect("test terminal");
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render latest conversation");

        let screen = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(
            !screen.contains("▙▟ ▐"),
            "header should scroll with the document"
        );
        assert!(screen.contains("0.0%/0 (auto)"), "footer remains docked");

        app.messages.scroll.scroll_to_start();
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render transcript start");
        let screen = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(screen.contains("▙▟ ▐"));
        assert!(screen.contains("0.0%/0 (auto)"));
    }

    #[test]
    fn streaming_updates_preserve_a_manually_scrolled_transcript_position() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        app.messages.extend((0..30).map(|index| Message {
            role: if index % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            },
            text: format!("conversation line {index}"),
            thinking: String::new(),
            tools: Vec::new(),
            error: false,
        }));
        app.loading = true;
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).expect("test terminal");
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render initial transcript");

        handle_mouse(MouseEventKind::ScrollUp, &mut app);
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render scrolled transcript");
        let position_before_update = app.messages.scroll.offset();
        assert!(!app.messages.scroll.is_following_end());

        app.apply(UiEvent::Agent(AgentEvent::TextDelta(" more text".into())));
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render streamed update");

        assert_eq!(app.messages.scroll.offset(), position_before_update);
        assert!(!app.messages.scroll.is_following_end());
    }

    #[test]
    fn cursor_is_positioned_in_the_editor_dock() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        app.input = "a\nb".chars().collect();
        app.cursor = app.input.len();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("test terminal");
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render multiline editor");

        terminal
            .backend_mut()
            .assert_cursor_position(ratatui::layout::Position::new(1, 20));
    }

    /// Rows made only of `─` (the editor's borders and scroll labels).
    fn border_rows(backend: &TestBackend) -> Vec<usize> {
        let width = usize::from(backend.buffer().area.width);
        backend
            .buffer()
            .content
            .chunks(width)
            .enumerate()
            .filter(|(_, row)| {
                let text = row.iter().map(|cell| cell.symbol()).collect::<String>();
                text.trim().starts_with('─') && text.trim().ends_with('─')
            })
            .map(|(index, _)| index)
            .collect()
    }

    fn row_with_text(backend: &TestBackend, needle: &str) -> usize {
        let width = usize::from(backend.buffer().area.width);
        backend
            .buffer()
            .content
            .chunks(width)
            .position(|row| {
                row.iter()
                    .map(|cell| cell.symbol())
                    .collect::<String>()
                    .contains(needle)
            })
            .expect("text should be visible")
    }
}
