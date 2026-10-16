mod layout;
mod terminal;

use std::{
    io,
    ops::{Deref, DerefMut},
    sync::Arc,
    time::Duration,
};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseEventKind};
use mu_core::{AgentEvent, MuAgent};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
};
use tokio::{
    sync::{Mutex, mpsc},
    task::AbortHandle,
};
use unicode_width::UnicodeWidthChar;

use self::layout::ChatViewport;

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
}

impl Default for TranscriptScroll {
    fn default() -> Self {
        Self {
            position: TranscriptPosition::FollowEnd,
            max_offset: 0,
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

    fn scroll_by(&mut self, lines: i32) {
        let next = (i32::from(self.offset()) + lines).clamp(0, i32::from(self.max_offset)) as u16;
        self.position = if next == self.max_offset {
            TranscriptPosition::FollowEnd
        } else {
            TranscriptPosition::Offset(next)
        };
    }

    fn scroll_to_start(&mut self) {
        self.position = if self.max_offset == 0 {
            TranscriptPosition::FollowEnd
        } else {
            TranscriptPosition::Offset(0)
        };
    }

    fn scroll_to_end(&mut self) {
        self.position = TranscriptPosition::FollowEnd;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ComposerAction {
    OpenCommandMenu,
    ExecuteCommand,
    Submit,
}

#[derive(Default)]
struct Composer {
    input: Vec<char>,
    cursor: usize,
    prompt_history: Vec<String>,
    history_index: Option<usize>,
}

impl Composer {
    fn input_text(&self) -> String {
        self.input.iter().collect()
    }

    fn handle_key(&mut self, key: KeyEvent) -> Option<ComposerAction> {
        match key.code {
            KeyCode::Char('/') if self.input.is_empty() => {
                self.insert_char('/');
                Some(ComposerAction::OpenCommandMenu)
            }
            KeyCode::Enter
                if key
                    .modifiers
                    .intersects(KeyModifiers::SHIFT | KeyModifiers::CONTROL) =>
            {
                self.insert_char('\n');
                None
            }
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.insert_char('\n');
                None
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.delete_to_line_start();
                None
            }
            KeyCode::Enter if self.input_text().starts_with('/') => {
                Some(ComposerAction::ExecuteCommand)
            }
            KeyCode::Enter => Some(ComposerAction::Submit),
            KeyCode::Esc => {
                self.input.clear();
                self.cursor = 0;
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
            KeyCode::Backspace if self.cursor > 0 => {
                self.cursor -= 1;
                self.input.remove(self.cursor);
                None
            }
            KeyCode::Delete if self.cursor < self.input.len() => {
                self.input.remove(self.cursor);
                None
            }
            KeyCode::Left if self.cursor > 0 => {
                self.cursor -= 1;
                None
            }
            KeyCode::Right if self.cursor < self.input.len() => {
                self.cursor += 1;
                None
            }
            KeyCode::Home => {
                self.cursor = self
                    .input
                    .iter()
                    .rposition(|ch| *ch == '\n')
                    .map_or(0, |i| i + 1);
                None
            }
            KeyCode::End => {
                self.cursor = self
                    .input
                    .iter()
                    .enumerate()
                    .skip(self.cursor)
                    .find(|(_, ch)| **ch == '\n')
                    .map_or(self.input.len(), |(i, _)| i);
                None
            }
            KeyCode::Up => {
                self.history_up();
                None
            }
            KeyCode::Down => {
                self.history_down();
                None
            }
            _ => None,
        }
    }

    fn insert_char(&mut self, ch: char) {
        self.input.insert(self.cursor, ch);
        self.cursor += 1;
    }

    fn insert_text(&mut self, text: &str) {
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let pasted: Vec<char> = normalized.chars().collect();
        let inserted = pasted.len();
        self.input.splice(self.cursor..self.cursor, pasted);
        self.cursor += inserted;
    }

    fn delete_to_line_start(&mut self) {
        let start = self.input[..self.cursor]
            .iter()
            .rposition(|ch| *ch == '\n')
            .map_or(0, |index| index + 1);
        self.input.drain(start..self.cursor);
        self.cursor = start;
    }

    fn history_up(&mut self) {
        if self.prompt_history.is_empty() {
            return;
        }
        let index = self
            .history_index
            .unwrap_or(self.prompt_history.len())
            .saturating_sub(1);
        self.history_index = Some(index);
        self.input = self.prompt_history[index].chars().collect();
        self.cursor = self.input.len();
    }

    fn history_down(&mut self) {
        let Some(index) = self.history_index else {
            return;
        };
        if index + 1 < self.prompt_history.len() {
            self.history_index = Some(index + 1);
            self.input = self.prompt_history[index + 1].chars().collect();
        } else {
            self.history_index = None;
            self.input.clear();
        }
        self.cursor = self.input.len();
    }
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
    model_selection: usize,
    loading: bool,
    quit: bool,
    request: Option<AbortHandle>,
    hide_thinking: bool,
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
        app.hide_thinking = agent.hide_thinking_block();
        app.models = agent.model_choices().to_vec();
        app.model_selection = app
            .models
            .iter()
            .position(|model| model == agent.model())
            .unwrap_or(0);
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
            model_selection: 0,
            loading: false,
            quit: false,
            request: None,
            hide_thinking: false,
        }
    }

    fn input_text(&self) -> String {
        self.composer.input_text()
    }

    fn submit(&mut self, agent: Arc<Mutex<MuAgent>>, tx: mpsc::UnboundedSender<UiEvent>) {
        let prompt = self.input_text();
        if prompt.trim().is_empty() || self.loading {
            return;
        }
        self.prompt_history.push(prompt.clone());
        self.history_index = None;
        self.input.clear();
        self.cursor = 0;
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
            UiEvent::Agent(event) => {
                let Some(message) = self.messages.last_mut() else {
                    return;
                };
                match event {
                    AgentEvent::TextDelta(delta) => message.text.push_str(&delta),
                    AgentEvent::ThinkingDelta(delta) => message.thinking.push_str(&delta),
                    AgentEvent::ToolStarted {
                        call_id,
                        name,
                        arguments,
                    } => message.tools.push(ToolActivity {
                        call_id,
                        name,
                        arguments,
                        output: None,
                        error: false,
                    }),
                    AgentEvent::ToolFinished {
                        call_id,
                        output,
                        is_error,
                    } => {
                        if let Some(tool) = message
                            .tools
                            .iter_mut()
                            .find(|tool| tool.call_id == call_id)
                        {
                            tool.output = Some(output);
                            tool.error = is_error;
                        }
                    }
                }
                // New output follows the end only if the user has not scrolled away.
            }
            UiEvent::Finished(Ok(())) => {
                self.loading = false;
                self.request = None;
            }
            UiEvent::Finished(Err(error)) => {
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

    fn cancel(&mut self) {
        if let Some(request) = self.request.take() {
            request.abort();
        }
        self.loading = false;
        if let Some(message) = self.messages.last_mut() {
            message.text.push_str("\n\n(interrupted)");
            message.error = true;
        }
    }
}

pub async fn run(agent: MuAgent) -> Result<(), Box<dyn std::error::Error>> {
    let _terminal_session = terminal::TerminalSession::enter()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let mut app = App::new(&agent);
    let agent = Arc::new(Mutex::new(agent));
    let (tx, mut rx) = mpsc::unbounded_channel();

    loop {
        terminal.draw(|frame| render(frame, &mut app))?;
        if app.quit {
            break;
        }
        tokio::select! {
            event = next_event() => if let Some(event) = event {
                match event {
                    Event::Key(key) => handle_key(key, &mut app, Arc::clone(&agent), tx.clone()),
                    Event::Mouse(mouse) => handle_mouse(mouse.kind, &mut app),
                    Event::Paste(text) if !app.loading && !app.model_selector_open => {
                        app.composer.insert_text(&text);
                        app.command_selection = 0;
                    },
                    Event::Resize(_, _) => {},
                    _ => {},
                }
            },
            event = rx.recv(), if app.loading => if let Some(event) = event { app.apply(event); },
        }
    }
    Ok(())
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
    if app.loading {
        if key.code == KeyCode::Esc {
            app.cancel();
        } else if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            app.cancel();
            app.quit = true;
        }
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
        app.quit = true;
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
            app.messages.scroll.scroll_by(-8);
            return;
        }
        KeyCode::Down if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.messages.scroll.scroll_by(8);
            return;
        }
        KeyCode::PageUp => {
            app.messages.scroll.scroll_by(-12);
            return;
        }
        KeyCode::PageDown => {
            app.messages.scroll.scroll_by(12);
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
        Some(ComposerAction::Submit) => app.submit(agent, tx),
        None => {}
    }
}

fn handle_command_menu(key: KeyEvent, app: &mut App, agent: &Arc<Mutex<MuAgent>>) {
    let commands = filtered_commands(app);
    match key.code {
        KeyCode::Esc => app.command_menu_open = false,
        KeyCode::Up => {
            app.command_selection = app.command_selection.saturating_sub(1);
        }
        KeyCode::Down => {
            if !commands.is_empty() {
                app.command_selection = (app.command_selection + 1).min(commands.len() - 1);
            }
        }
        KeyCode::Enter => {
            if let Some((name, _)) = commands.get(app.command_selection) {
                app.input = format!("/{name}").chars().collect();
                app.cursor = app.input.len();
                app.command_menu_open = false;
                execute_command(app, agent);
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
                    app.model = model;
                }
            }
            app.model_selector_open = false;
        }
        _ => {}
    }
}

fn cycle_model(app: &mut App, agent: &Arc<Mutex<MuAgent>>) {
    if app.models.len() < 2 {
        return;
    }
    let next = (app.model_selection + 1) % app.models.len();
    let Some(model) = app.models.get(next).cloned() else {
        return;
    };
    let selected = agent
        .try_lock()
        .ok()
        .and_then(|mut agent| agent.select_model(&model).ok())
        .unwrap_or(false);
    if selected {
        app.model_selection = next;
        app.model = model;
    }
}

const COMMANDS: &[(&str, &str)] = &[
    ("settings", "Open settings menu"),
    ("model", "Select model (opens selector UI)"),
    ("tree", "Navigate session tree (switch branches)"),
    ("thinking", "Set thinking level"),
    ("scoped-models", "Enable/disable models for Ctrl+P cycling"),
    ("session", "Show session information"),
    ("resume", "Resume a previous session"),
    ("new", "Start a new session"),
    ("fork", "Fork the current session"),
    ("branch", "Branch from an earlier message"),
    ("compact", "Compact the conversation"),
    ("clear", "Clear the conversation"),
    ("export", "Export the session"),
    ("copy", "Copy the last assistant response"),
    ("name", "Set the session name"),
    ("help", "Show help"),
    ("hotkeys", "Show keyboard shortcuts"),
    ("tools", "List available tools"),
    ("login", "Configure provider authentication"),
    ("logout", "Remove provider authentication"),
    ("quit", "Exit Mu"),
];

fn filtered_commands(app: &App) -> Vec<&'static (&'static str, &'static str)> {
    let query: String = app.input.iter().skip(1).collect();
    COMMANDS
        .iter()
        .filter(|(name, _)| name.starts_with(&query))
        .collect()
}

fn execute_command(app: &mut App, agent: &Arc<Mutex<MuAgent>>) {
    let input = app.input_text();
    let command = input
        .trim_start_matches('/')
        .split_whitespace()
        .next()
        .unwrap_or_default();
    app.input.clear();
    app.cursor = 0;
    app.command_menu_open = false;
    match command {
        "quit" | "exit" => app.quit = true,
        "clear" | "new" => app.messages.clear(),
        "model" => {
            app.model_selection = app
                .models
                .iter()
                .position(|model| model == &app.model)
                .unwrap_or(0);
            app.model_selector_open = true;
        }
        "help" => app.messages.push(Message {
            role: Role::Assistant,
            text: "Commands: /help, /model, /new, /clear, /quit".into(),
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

fn handle_mouse(kind: MouseEventKind, app: &mut App) {
    match kind {
        MouseEventKind::ScrollUp => app.messages.scroll.scroll_by(-3),
        MouseEventKind::ScrollDown => app.messages.scroll.scroll_by(3),
        _ => {}
    }
}

fn insert_char(app: &mut App, ch: char) {
    app.composer.insert_char(ch);
}

fn render(frame: &mut Frame, app: &mut App) {
    let input_text = app.input_text();
    let editor_width = frame.area().width.saturating_sub(2).max(1);
    let editor_lines = Paragraph::new(input_text.clone())
        .wrap(Wrap { trim: false })
        .line_count(editor_width)
        .clamp(1, 8) as u16;
    let menu_rows = if app.command_menu_open || app.model_selector_open {
        let item_count = if app.model_selector_open {
            app.models.len()
        } else {
            filtered_commands(app).len()
        };
        item_count.clamp(1, 6) as u16
    } else {
        0
    };
    let viewport = ChatViewport::new(frame.area(), menu_rows, editor_lines);

    render_transcript(frame, viewport.transcript, app);
    render_command_menu(frame, viewport.command_menu, app);
    render_editor(frame, viewport.editor, app, &input_text);
    render_footer(frame, viewport.footer, app);

    if !app.loading && viewport.editor.width > 2 && viewport.editor.height > 2 {
        let inner = Block::default()
            .borders(Borders::TOP | Borders::BOTTOM)
            .inner(viewport.editor);
        let cursor = editor_cursor_position(&app.input, app.cursor, inner);
        frame.set_cursor_position(cursor);
    }
}

fn header_lines(app: &App) -> Vec<Line<'static>> {
    let version = env!("CARGO_PKG_VERSION");
    let title = Line::from(vec![
        Span::styled(
            "▞ mu ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(version, Style::default().fg(Color::DarkGray)),
        Span::styled("  ·  ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            app.model.clone(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  {}", app.provider),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(
            if app.loading {
                "  ● thinking"
            } else {
                "  ● ready"
            },
            Style::default().fg(if app.loading {
                Color::Yellow
            } else {
                Color::Green
            }),
        ),
    ]);
    let help = Line::from(Span::styled(
        format!(
            "Ctrl+P model · Ctrl+T thinking {} · / commands · PgUp/PgDn or wheel · Esc clear · Ctrl+C exit",
            if app.hide_thinking {
                "hidden"
            } else {
                "visible"
            }
        ),
        Style::default().fg(Color::DarkGray),
    ));
    vec![title, help]
}

fn render_transcript(frame: &mut Frame, area: Rect, app: &mut App) {
    if area.width == 0 || area.height == 0 {
        app.messages.scroll.set_max_offset(0);
        return;
    }

    let mut lines = header_lines(app);
    lines.push(Line::from(""));
    if app.messages.is_empty() {
        lines.push(Line::from(Span::styled(
            "Mu can chat with your Pi default model and saves turns as Pi sessions.",
            Style::default().fg(Color::Gray),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "[Session]",
            Style::default().fg(Color::Yellow),
        )));
        lines.push(Line::from(Span::styled(
            format!("  provider: {}", app.provider),
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(Span::styled(
            format!("  model: {}", app.model),
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "[Runtime]",
            Style::default().fg(Color::Yellow),
        )));
        lines.push(Line::from(Span::styled(
            "  Chat, Pi session persistence, tools, and skill discovery are active.",
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Ask Mu how to help with your project.",
            Style::default().fg(Color::Gray),
        )));
    }
    for (index, message) in app.messages.iter().enumerate() {
        if index > 0 {
            lines.push(Line::from(""));
        }
        let (label, color) = match message.role {
            Role::User => ("› you", Color::Cyan),
            Role::Assistant => (
                "◆ mu",
                if message.error {
                    Color::Red
                } else {
                    Color::Green
                },
            ),
        };
        lines.push(Line::from(Span::styled(
            label,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )));
        if !app.hide_thinking && !message.thinking.is_empty() {
            lines.push(Line::from(Span::styled(
                "  ◇ thinking",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::ITALIC),
            )));
            for line in message.thinking.lines() {
                lines.push(Line::from(vec![
                    Span::styled("  │ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        line.to_owned(),
                        Style::default()
                            .fg(Color::Gray)
                            .add_modifier(Modifier::ITALIC),
                    ),
                ]));
            }
        }
        for tool in &message.tools {
            let status = if tool.error {
                "failed"
            } else if tool.output.is_some() {
                "done"
            } else {
                "running"
            };
            let color = if tool.error {
                Color::Red
            } else if tool.output.is_some() {
                Color::Green
            } else {
                Color::Yellow
            };
            lines.push(Line::from(vec![
                Span::styled("  ⚙ ", Style::default().fg(color)),
                Span::styled(
                    tool.name.clone(),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" {} · {status}", tool_description(tool)),
                    Style::default().fg(Color::Gray),
                ),
            ]));
            if let Some(output) = &tool.output {
                for line in output.lines().take(4) {
                    let shown = if line.chars().count() > 180 {
                        format!("{}…", line.chars().take(180).collect::<String>())
                    } else {
                        line.to_owned()
                    };
                    lines.push(Line::from(vec![
                        Span::styled("    ", Style::default()),
                        Span::styled(
                            shown,
                            Style::default().fg(if tool.error {
                                Color::Red
                            } else {
                                Color::DarkGray
                            }),
                        ),
                    ]));
                }
                if output.lines().count() > 4 {
                    lines.push(Line::from(Span::styled(
                        "    … output truncated",
                        Style::default().fg(Color::DarkGray),
                    )));
                }
            }
        }
        let content = if message.text.is_empty()
            && app.loading
            && message.thinking.is_empty()
            && message.tools.is_empty()
        {
            "Thinking…"
        } else {
            &message.text
        };
        for line in content.split('\n') {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    line.to_owned(),
                    Style::default().fg(if message.error {
                        Color::Red
                    } else {
                        Color::White
                    }),
                ),
            ]));
        }
    }
    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    let content_lines = paragraph.line_count(area.width.max(1));
    let max_scroll =
        u16::try_from(content_lines.saturating_sub(usize::from(area.height))).unwrap_or(u16::MAX);
    app.messages.scroll.set_max_offset(max_scroll);
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
            Paragraph::new("↓ Jump to latest message · Ctrl+End")
                .style(Style::default().fg(Color::DarkGray)),
            hint_area,
        );
    }
    if content_lines > usize::from(area.height) {
        let mut state = ScrollbarState::new(content_lines)
            .position(usize::from(scroll))
            .viewport_content_length(usize::from(area.height));
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .thumb_symbol("┃")
                .track_symbol(Some("│"))
                .begin_symbol(None)
                .end_symbol(None),
            area,
            &mut state,
        );
    }
}

fn tool_description(tool: &ToolActivity) -> String {
    tool.arguments
        .get("command")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            tool.arguments
                .get("path")
                .and_then(serde_json::Value::as_str)
        })
        .map(|value| {
            let trimmed: String = value.chars().take(72).collect();
            if value.chars().count() > 72 {
                format!("{trimmed}…")
            } else {
                trimmed
            }
        })
        .unwrap_or_default()
}

fn render_command_menu(frame: &mut Frame, area: Rect, app: &App) {
    if area.height == 0 {
        return;
    }
    let lines: Vec<Line<'_>> = if app.model_selector_open {
        app.models
            .iter()
            .enumerate()
            .take(usize::from(area.height))
            .map(|(index, model)| {
                let prefix = if index == app.model_selection {
                    "→ "
                } else {
                    "  "
                };
                Line::from(Span::styled(
                    format!("{prefix}{model}"),
                    if index == app.model_selection {
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::Gray)
                    },
                ))
            })
            .collect()
    } else {
        filtered_commands(app)
            .into_iter()
            .enumerate()
            .take(usize::from(area.height))
            .map(|(index, (name, description))| {
                let prefix = if index == app.command_selection {
                    "→ "
                } else {
                    "  "
                };
                Line::from(vec![
                    Span::styled(
                        format!("{prefix}{name:<14}"),
                        if index == app.command_selection {
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::Cyan)
                        },
                    ),
                    Span::styled(*description, Style::default().fg(Color::DarkGray)),
                ])
            })
            .collect()
    };
    frame.render_widget(Paragraph::new(lines), area);
}

fn editor_cursor_position(input: &[char], cursor: usize, inner: Rect) -> Position {
    let width = inner.width.max(1);
    let width_cells = usize::from(width);
    let text: String = input.iter().collect();
    let content_rows = Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .line_count(width);
    let visible_rows = usize::from(inner.height.max(1));
    let scroll = content_rows.saturating_sub(visible_rows);

    let mut row = 0usize;
    let mut column = 0usize;
    for character in input.iter().take(cursor.min(input.len())) {
        if *character == '\n' {
            row += 1;
            column = 0;
            continue;
        }
        let character_width = UnicodeWidthChar::width(*character).unwrap_or(0);
        if character_width > 0 && column + character_width > width_cells {
            row += 1;
            column = 0;
        }
        column += character_width;
        if column >= width_cells {
            row += 1;
            column = 0;
        }
    }

    Position {
        x: inner.x.saturating_add(column.min(width_cells - 1) as u16),
        y: inner
            .y
            .saturating_add(row.saturating_sub(scroll).min(visible_rows - 1) as u16),
    }
}

fn render_editor(frame: &mut Frame, area: Rect, app: &App, input: &str) {
    let title = if app.loading {
        " generating · Esc to interrupt "
    } else {
        " message · Enter send · Shift+Enter/Ctrl+J newline "
    };
    let color = if app.loading {
        Color::Yellow
    } else {
        Color::Cyan
    };
    let block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(Style::default().fg(color))
        .title(Span::styled(
            title,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ));
    let text = if input.is_empty() && !app.loading {
        "Ask Mu to inspect or change something…"
    } else {
        input
    };
    let style = if input.is_empty() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };
    let visible = area.height.saturating_sub(2);
    let content_lines = Paragraph::new(text.to_owned())
        .wrap(Wrap { trim: false })
        .line_count(area.width.saturating_sub(2).max(1));
    let scroll = (content_lines as u16).saturating_sub(visible);
    frame.render_widget(
        Paragraph::new(text.to_owned())
            .style(style)
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        area,
    );
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let cwd = std::env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let session = app
        .session_path
        .rsplit('/')
        .next()
        .unwrap_or(&app.session_path);
    let path = if cwd.chars().count() > 50 {
        format!(
            "…{}",
            cwd.chars()
                .skip(cwd.chars().count() - 49)
                .collect::<String>()
        )
    } else {
        cwd
    };
    let first = Line::from(vec![
        Span::styled(path, Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("  ·  session {session}"),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    let help = if app.loading {
        "Esc interrupt · Ctrl+C cancel and exit"
    } else {
        "Enter send · Shift+Enter/Ctrl+J newline · Ctrl+P model · / commands · Ctrl+C exit"
    };
    frame.render_widget(
        Paragraph::new(vec![
            first,
            Line::from(Span::styled(help, Style::default().fg(Color::DarkGray))),
        ]),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::{
        AgentEvent, App, Composer, ComposerAction, Message, Role, UiEvent, handle_mouse,
        insert_char, render,
    };
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEventKind};
    use ratatui::{Terminal, backend::TestBackend};

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
        assert!(screen.contains("› you"));
        assert!(screen.contains("Ready."));
        assert!(screen.contains("Enter send"));
        assert!(screen.contains("session mu-session.jsonl"));
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
        assert!(screen.contains("thinking"));
        assert!(screen.contains("read"));
        assert!(screen.contains("src/main.rs"));
        assert!(screen.contains("fn main()"));
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
        assert_eq!(app.input_text(), "");
    }

    #[test]
    fn composer_grows_with_multiline_input_and_shrinks_after_deletion() {
        let mut app = App::from_metadata("model".into(), "provider".into(), "session".into());
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("test terminal");
        terminal
            .draw(|frame| render(frame, &mut app))
            .expect("render empty composer");
        let empty_title_row = row_with_text(terminal.backend(), "message · Enter send");

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
        let multiline_title_row = row_with_text(terminal.backend(), "message · Enter send");
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
        assert_eq!(
            row_with_text(terminal.backend(), "message · Enter send"),
            empty_title_row
        );
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
        assert!(app.messages.scroll.offset() < bottom);
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
            !screen.contains("▞ mu"),
            "header should scroll with the document"
        );
        assert!(screen.contains("session session"), "footer remains docked");

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
        assert!(screen.contains("▞ mu"));
        assert!(screen.contains("Enter send"));
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
