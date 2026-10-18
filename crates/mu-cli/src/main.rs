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

    /// Run the TUI with a deterministic backend; no gateway or credentials required.
    #[arg(
        long,
        conflicts_with_all = ["print", "prompt", "continue_session", "session"]
    )]
    demo: bool,

    /// Process a prompt and exit without opening the interactive TUI.
    #[arg(short = 'p', long)]
    print: bool,

    /// Continue the most recent session for the current working directory.
    #[arg(short = 'c', long = "continue", conflicts_with = "session")]
    continue_session: bool,

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
    if cli.demo {
        if !is_terminal {
            return Err("--demo requires an interactive terminal".into());
        }
        return tui::run(mu_core::MuAgent::demo()?).await;
    }

    if cli.mode != OutputMode::Text {
        return Err(format!("--mode {:?} is not implemented yet", cli.mode).into());
    }
    let (session_file, session_selector) = match cli.session {
        Some(selector)
            if selector.contains('/')
                || selector.contains('\\')
                || selector.ends_with(".jsonl") =>
        {
            (Some(PathBuf::from(selector)), None)
        }
        Some(selector) => (None, Some(selector)),
        None => (None, None),
    };
    let session_options = mu_core::SessionOptions {
        path: session_file,
        directory: cli.session_dir,
        id: cli.session_id,
        id_prefix: session_selector,
        continue_latest: cli.continue_session,
        no_session: cli.no_session,
    };

    if cli.print || !cli.prompt.is_empty() || !is_terminal {
        let prompt = if cli.prompt.is_empty() {
            read_stdin_prompt(is_terminal)?
        } else {
            cli.prompt.join(" ")
        };
        if prompt.trim().is_empty() {
            return Ok(());
        }
        mu_core::run_prompt_with_options(prompt, session_options).await?;
        return Ok(());
    }

    tui::run_starting(async move { mu_core::MuAgent::connect_with_options(session_options).await })
        .await
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
    use super::Cli;
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
    }

    #[test]
    fn help_mentions_demo_and_prompt_modes() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("--demo"));
        assert!(help.contains("--print"));
        assert!(help.contains("--mode"));
        assert!(help.contains("--continue"));
        assert!(help.contains("--session"));
        assert!(help.contains("--session-id"));
        assert!(help.contains("--session-dir"));
        assert!(help.contains("--no-session"));
        assert!(help.contains("PROMPT"));
    }
}
