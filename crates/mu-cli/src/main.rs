mod tui;

use std::io::{IsTerminal, Read};

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
    #[arg(long, conflicts_with_all = ["print", "prompt"])]
    demo: bool,

    /// Process a prompt and exit without opening the interactive TUI.
    #[arg(short = 'p', long)]
    print: bool,

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

    if cli.print || !cli.prompt.is_empty() || !is_terminal {
        let prompt = if cli.prompt.is_empty() {
            read_stdin_prompt(is_terminal)?
        } else {
            cli.prompt.join(" ")
        };
        if prompt.trim().is_empty() {
            return Ok(());
        }
        mu_core::run_prompt(prompt).await?;
        return Ok(());
    }

    tui::run(mu_core::MuAgent::connect().await?).await
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
    }

    #[test]
    fn help_mentions_demo_and_prompt_modes() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("--demo"));
        assert!(help.contains("--print"));
        assert!(help.contains("--mode"));
        assert!(help.contains("PROMPT"));
    }
}
