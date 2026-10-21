mod tui;

use std::{
    io::{IsTerminal, Read},
    path::PathBuf,
};

use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "mu",
    version,
    disable_version_flag = true,
    about = "A Rust-native coding agent compatible with Pi sessions",
    long_about = "Mu is an early Rust port of Pi's coding-agent core. It reads Pi settings and persists Pi-compatible sessions."
)]
struct Cli {
    /// Print the Mu version.
    #[arg(short = 'v', long, action = clap::ArgAction::SetTrue)]
    version: bool,

    /// Use a deterministic offline backend; no gateway or credentials required.
    #[arg(long)]
    demo: bool,

    /// Process a prompt and exit without opening the interactive TUI.
    #[arg(short = 'p', long)]
    print: bool,

    /// Continue the most recent session for the current working directory.
    #[arg(
        short = 'c',
        long = "continue",
        conflicts_with_all = ["session", "session_id", "resume"]
    )]
    continue_session: bool,

    /// Choose a session from the current working directory to resume.
    #[arg(
        short = 'r',
        long = "resume",
        conflicts_with_all = ["continue_session", "session", "session_id"]
    )]
    resume: bool,

    /// Resume a specific Pi-compatible session file or session ID/prefix.
    #[arg(long, conflicts_with_all = ["continue_session", "session_id"])]
    session: Option<String>,

    /// Open an exact project session ID, creating a new session with that ID if absent.
    #[arg(long, conflicts_with_all = ["continue_session", "session"])]
    session_id: Option<String>,

    /// Override the directory used for session storage and discovery.
    #[arg(long)]
    session_dir: Option<PathBuf>,

    /// Keep the session in memory without writing it to disk.
    #[arg(long)]
    no_session: bool,

    /// Fork a session file or local session ID/prefix into a new session.
    #[arg(
        long,
        conflicts_with_all = ["session", "continue_session", "resume", "no_session"]
    )]
    fork: Option<String>,

    /// Set the session display name.
    #[arg(short = 'n', long)]
    name: Option<String>,

    /// Enable coding, read-only, or all tools; use names/patterns or ordered +name/-name modifiers.
    #[arg(short = 't', long, value_delimiter = ',')]
    tools: Option<Vec<String>>,

    /// Disable tools by name or `*` pattern, after selection.
    #[arg(
        short = 'x',
        long = "exclude-tools",
        value_delimiter = ',',
        value_name = "TOOLS"
    )]
    exclude_tools: Vec<String>,

    /// Disable all tools.
    #[arg(long)]
    no_tools: bool,

    /// Disable built-in tools while leaving MCP tools available.
    #[arg(long)]
    no_builtin_tools: bool,

    /// Load an additional skill Markdown file or directory (repeatable).
    #[arg(long = "skill", value_name = "PATH")]
    skill_paths: Vec<PathBuf>,

    /// Disable automatic skill discovery; explicit --skill paths remain enabled.
    #[arg(long)]
    no_skills: bool,

    /// Replace the default system prompt (text or a file path).
    #[arg(long, value_name = "TEXT")]
    system_prompt: Option<String>,

    /// Append text or file contents to the system prompt (repeatable).
    #[arg(long, value_name = "TEXT")]
    append_system_prompt: Vec<String>,

    /// Do not load AGENTS.md / CLAUDE.md context files.
    #[arg(long)]
    no_context_files: bool,

    /// Do not make startup network requests (Pi's `--offline`).
    #[arg(long)]
    offline: bool,

    /// Do not load extensions (Mu has no extension runtime yet, so this is always the case).
    #[arg(long)]
    no_extensions: bool,

    /// Output mode; JSON and RPC modes are not implemented yet.
    #[arg(long, value_enum, default_value_t = OutputMode::Text)]
    mode: OutputMode,

    /// Send a one-shot prompt instead of opening the interactive TUI.
    #[arg(value_name = "PROMPT", num_args = 1..)]
    prompt: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum OutputMode {
    #[default]
    Text,
    Json,
    Rpc,
}

#[tokio::main]
async fn main() {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let exit_code = if error.use_stderr() { 1 } else { 0 };
            let _ = error.print();
            std::process::exit(exit_code);
        }
    };
    if cli.version {
        println!("mu {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if let Err(error) = run(cli).await {
        eprintln!("mu: {error}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let is_terminal = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    if cli.mode == OutputMode::Rpc {
        return Err(format!("--mode {:?} is not implemented yet", cli.mode).into());
    }
    if cli.resume && !is_terminal {
        return Err("--resume requires an interactive terminal".into());
    }
    let resumed_file = if cli.resume {
        let sessions = mu_core::list_sessions_for_current_cwd(cli.session_dir.as_deref())?;
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        select_session(&sessions, &mut stdin.lock(), &mut stdout.lock())?
    } else {
        None
    };
    if cli.resume && resumed_file.is_none() {
        return Ok(());
    }
    let fork_from = if let Some(source) = cli.fork.as_deref() {
        Some(resolve_session_source(source, cli.session_dir.as_deref())?)
    } else {
        None
    };
    let (session_file, session_selector) = if let Some(path) = resumed_file {
        (Some(path), None)
    } else if cli.fork.is_some() {
        (None, None)
    } else {
        match cli.session {
            Some(selector)
                if selector.contains('/')
                    || selector.contains('\\')
                    || selector.ends_with(".jsonl") =>
            {
                (Some(PathBuf::from(selector)), None)
            }
            Some(selector) => (None, Some(selector)),
            None => (None, None),
        }
    };
    let session_options = mu_core::SessionOptions {
        path: session_file,
        directory: cli.session_dir,
        id: cli.session_id,
        id_prefix: session_selector,
        continue_latest: cli.continue_session,
        no_session: cli.no_session,
        fork_from,
        name: cli.name,
        tools: cli.tools,
        no_tools: cli.no_tools,
        no_builtin_tools: cli.no_builtin_tools,
        skill_paths: cli.skill_paths,
        no_skills: cli.no_skills,
        demo: cli.demo,
        system_prompt: cli.system_prompt,
        append_system_prompt: cli.append_system_prompt,
        no_context_files: cli.no_context_files,
        exclude_tools: cli.exclude_tools,
    };

    if cli.print || !is_terminal {
        let prompt = if cli.prompt.is_empty() {
            read_stdin_prompt(is_terminal)?
        } else {
            cli.prompt.join(" ")
        };
        if prompt.trim().is_empty() {
            return Ok(());
        }
        if cli.mode == OutputMode::Json {
            mu_core::run_prompt_json(prompt, session_options).await?;
        } else {
            mu_core::run_prompt_with_options(prompt, session_options).await?;
        }
        return Ok(());
    }

    let initial_prompt = (!cli.prompt.is_empty()).then(|| cli.prompt.join(" "));
    tui::run_starting(
        async move { mu_core::MuAgent::connect_with_options(session_options).await },
        initial_prompt,
    )
    .await
}

fn resolve_session_source(
    selector: &str,
    session_dir: Option<&std::path::Path>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if selector.contains('/') || selector.contains('\\') || selector.ends_with(".jsonl") {
        return Ok(PathBuf::from(selector));
    }
    let sessions = mu_core::list_sessions_for_current_cwd(session_dir)?;
    sessions
        .into_iter()
        .find(|session| session.id == selector || session.id.starts_with(selector))
        .map(|session| session.path)
        .ok_or_else(|| {
            format!("No session found matching '{selector}' in this working directory").into()
        })
}

fn select_session(
    sessions: &[mu_core::SessionInfo],
    input: &mut impl std::io::BufRead,
    output: &mut impl std::io::Write,
) -> Result<Option<PathBuf>, std::io::Error> {
    if sessions.is_empty() {
        writeln!(output, "No sessions found for this working directory.")?;
        return Ok(None);
    }
    let mut visible = sessions.iter().collect::<Vec<_>>();
    loop {
        writeln!(output, "Sessions:")?;
        for (index, session) in visible.iter().enumerate() {
            let title = session.name.as_deref().unwrap_or(&session.id);
            writeln!(
                output,
                "  {}. {}  [{}]  {}{}",
                index + 1,
                title,
                session.id,
                session.timestamp,
                session
                    .preview
                    .as_deref()
                    .map(|preview| format!("  — {preview}"))
                    .unwrap_or_default()
            )?;
        }
        write!(
            output,
            "Choose a session (1-{}, search text, or q to cancel): ",
            visible.len()
        )?;
        output.flush()?;

        let mut choice = String::new();
        if input.read_line(&mut choice)? == 0 {
            return Ok(None);
        }
        let choice = choice.trim();
        if choice.eq_ignore_ascii_case("q") || choice.is_empty() {
            return Ok(None);
        }
        if let Some(index) = choice
            .parse::<usize>()
            .ok()
            .filter(|index| (1..=visible.len()).contains(index))
        {
            return Ok(Some(visible[index - 1].path.clone()));
        }

        let query = choice.to_lowercase();
        visible = sessions
            .iter()
            .filter(|session| {
                session.id.to_lowercase().contains(&query)
                    || session
                        .name
                        .as_deref()
                        .is_some_and(|name| name.to_lowercase().contains(&query))
                    || session
                        .preview
                        .as_deref()
                        .is_some_and(|preview| preview.to_lowercase().contains(&query))
            })
            .collect();
        if visible.is_empty() {
            writeln!(
                output,
                "No sessions match '{choice}'. Try another search or q to cancel."
            )?;
        }
    }
}

fn read_stdin_prompt(is_terminal: bool) -> Result<String, std::io::Error> {
    if is_terminal {
        return Ok(String::new());
    }
    let mut prompt = String::new();
    std::io::stdin().read_to_string(&mut prompt)?;
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::{Cli, select_session};
    use clap::{CommandFactory, Parser};
    use std::path::PathBuf;

    #[test]
    fn parses_demo_and_one_shot_prompt_modes() {
        let demo = Cli::try_parse_from(["mu", "--demo"]).expect("parse demo mode");
        assert!(demo.demo);
        assert!(!demo.version);
        assert!(!demo.print);
        assert_eq!(demo.mode, super::OutputMode::Text);
        assert!(demo.prompt.is_empty());

        let prompt =
            Cli::try_parse_from(["mu", "inspect", "this repo"]).expect("parse one-shot prompt");
        assert!(!prompt.demo);
        assert_eq!(prompt.prompt, ["inspect", "this repo"]);

        let print =
            Cli::try_parse_from(["mu", "-p", "inspect", "this repo"]).expect("parse print mode");
        assert!(print.print);
        assert_eq!(print.prompt, ["inspect", "this repo"]);

        let continue_session =
            Cli::try_parse_from(["mu", "-c"]).expect("parse continue-session mode");
        assert!(continue_session.continue_session);
        let resume = Cli::try_parse_from(["mu", "-r"]).expect("parse resume picker");
        assert!(resume.resume);
        assert!(Cli::try_parse_from(["mu", "--resume", "--continue"]).is_err());
        let session = Cli::try_parse_from(["mu", "--session", "/tmp/pi-session.jsonl"])
            .expect("parse explicit session mode");
        assert_eq!(session.session.as_deref(), Some("/tmp/pi-session.jsonl"));
        assert!(
            Cli::try_parse_from(["mu", "--continue", "--session", "/tmp/pi-session.jsonl"])
                .is_err()
        );
        let session_prefix = Cli::try_parse_from(["mu", "--session", "partial-id"])
            .expect("parse session ID prefix");
        assert_eq!(session_prefix.session.as_deref(), Some("partial-id"));
        let session_id = Cli::try_parse_from(["mu", "--session-id", "my-session"])
            .expect("parse exact session ID");
        assert_eq!(session_id.session_id.as_deref(), Some("my-session"));
        let session_dir = Cli::try_parse_from(["mu", "--session-dir", "/tmp/sessions"])
            .expect("parse session directory");
        assert_eq!(
            session_dir.session_dir,
            Some(PathBuf::from("/tmp/sessions"))
        );
        assert!(
            Cli::try_parse_from([
                "mu",
                "--session-id",
                "my-session",
                "--session",
                "partial-id"
            ])
            .is_err()
        );
        let no_session =
            Cli::try_parse_from(["mu", "--no-session"]).expect("parse ephemeral session option");
        assert!(no_session.no_session);
        let fork =
            Cli::try_parse_from(["mu", "--fork", "/tmp/source.jsonl"]).expect("parse session fork");
        assert_eq!(fork.fork.as_deref(), Some("/tmp/source.jsonl"));
        assert!(Cli::try_parse_from(["mu", "--fork", "source", "--continue"]).is_err());
        let named = Cli::try_parse_from(["mu", "-n", "investigate bug"])
            .expect("parse session display name");
        assert_eq!(named.name.as_deref(), Some("investigate bug"));
        let tools = Cli::try_parse_from(["mu", "--tools", "read,+grep,-bash"])
            .expect("parse tool selection");
        assert_eq!(
            tools.tools.unwrap(),
            vec!["read".to_owned(), "+grep".to_owned(), "-bash".to_owned()]
        );
        assert!(
            Cli::try_parse_from(["mu", "--no-tools"])
                .expect("parse disable-tools option")
                .no_tools
        );
        assert!(
            Cli::try_parse_from(["mu", "--no-builtin-tools"])
                .expect("parse disable-builtin-tools option")
                .no_builtin_tools
        );
    }

    #[test]
    fn parses_repeatable_skill_paths_and_disable_discovery() {
        let cli = Cli::try_parse_from([
            "mu",
            "--skill",
            "./first/SKILL.md",
            "--skill",
            "./second",
            "--no-skills",
        ])
        .expect("parse skill options");
        assert_eq!(
            cli.skill_paths,
            vec![PathBuf::from("./first/SKILL.md"), PathBuf::from("./second")]
        );
        assert!(cli.no_skills);
    }

    #[test]
    fn session_picker_selects_one_based_entries_and_can_be_cancelled() {
        let sessions = vec![
            mu_core::SessionInfo {
                id: "first".into(),
                name: Some("First session".into()),
                preview: Some("Refactor the session picker".into()),
                path: PathBuf::from("/tmp/first.jsonl"),
                timestamp: "2025-01-01T00:00:00Z".into(),
            },
            mu_core::SessionInfo {
                id: "second".into(),
                name: None,
                preview: None,
                path: PathBuf::from("/tmp/second.jsonl"),
                timestamp: "2025-01-02T00:00:00Z".into(),
            },
        ];
        let mut output = Vec::new();
        assert_eq!(
            select_session(&sessions, &mut &b"2\n"[..], &mut output).unwrap(),
            Some(PathBuf::from("/tmp/second.jsonl"))
        );
        let mut search_output = Vec::new();
        assert_eq!(
            select_session(&sessions, &mut &b"second\n1\n"[..], &mut search_output).unwrap(),
            Some(PathBuf::from("/tmp/second.jsonl"))
        );
        let search_output = String::from_utf8(search_output).unwrap();
        assert!(search_output.contains("Choose a session"));
        assert!(search_output.contains("second"));
        let mut preview_output = Vec::new();
        assert_eq!(
            select_session(&sessions, &mut &b"picker\n1\n"[..], &mut preview_output).unwrap(),
            Some(PathBuf::from("/tmp/first.jsonl"))
        );
        assert!(
            String::from_utf8(preview_output)
                .unwrap()
                .contains("Refactor the session picker")
        );
        assert!(
            select_session(&sessions, &mut &b"q\n"[..], &mut output)
                .unwrap()
                .is_none()
        );
        assert!(
            select_session(&sessions, &mut &b"7\n"[..], &mut output)
                .unwrap()
                .is_none()
        );
        assert!(
            select_session(&[], &mut &b""[..], &mut output)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn help_mentions_demo_and_prompt_modes() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("--demo"));
        assert!(help.contains("--print"));
        assert!(help.contains("--mode"));
        assert!(help.contains("--continue"));
        assert!(help.contains("--resume"));
        assert!(help.contains("--session"));
        assert!(help.contains("--session-id"));
        assert!(help.contains("--session-dir"));
        assert!(help.contains("--no-session"));
        assert!(help.contains("--fork"));
        assert!(help.contains("--name"));
        assert!(help.contains("--tools"));
        assert!(help.contains("--no-tools"));
        assert!(help.contains("--no-builtin-tools"));
        assert!(help.contains("--skill"));
        assert!(help.contains("--no-skills"));
        assert!(help.contains("PROMPT"));
    }
}
