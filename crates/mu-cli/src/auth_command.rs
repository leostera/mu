//! `mu auth print-api-key|check --provider <id>` (Pi's `cli/auth-command.ts`, API-key subset).

use std::path::Path;

use mu_core::auth::{CredentialSource, credential_source};

#[derive(Debug, PartialEq, Eq)]
pub enum AuthCommand {
    PrintApiKey { provider: String },
    Check { provider: String, json: bool },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AuthCommandError {
    #[error("Unknown auth command \"{0}\". Use \"mu auth print-api-key\" or \"mu auth check\".")]
    Unknown(String),
    #[error("{0} requires --provider <provider>")]
    MissingProvider(&'static str),
    #[error("Unknown option {0} for \"mu auth\".")]
    UnknownOption(String),
    #[error("Provider \"{0}\" is configured with OAuth, not an API key")]
    OAuthNotKey(String),
    #[error("No usable API key is configured for \"{0}\"")]
    NoKey(String),
}

/// `None` when `args` (after the program name) is not an `auth` command.
pub fn parse(args: &[String]) -> Option<Result<AuthCommand, AuthCommandError>> {
    if args.first().map(String::as_str) != Some("auth") {
        return None;
    }
    let sub = args.get(1).cloned().unwrap_or_default();
    let (name, check) = match sub.as_str() {
        "print-api-key" => ("print-api-key", false),
        "check" => ("check", true),
        _ => return Some(Err(AuthCommandError::Unknown(sub))),
    };
    let (mut provider, mut json) = (None, false);
    let mut rest = args[2..].iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--provider" => provider = rest.next().map(|p| p.trim().to_owned()),
            "--json" if check => json = true,
            other => return Some(Err(AuthCommandError::UnknownOption(other.to_owned()))),
        }
    }
    let Some(provider) = provider.filter(|p| !p.is_empty()) else {
        return Some(Err(AuthCommandError::MissingProvider(name)));
    };
    Some(Ok(if check {
        AuthCommand::Check { provider, json }
    } else {
        AuthCommand::PrintApiKey { provider }
    }))
}

/// Run the command; the returned text is printed to stdout.
pub fn run(command: &AuthCommand, agent_dir: &Path) -> Result<String, AuthCommandError> {
    match command {
        AuthCommand::PrintApiKey { provider } => match credential_source(agent_dir, provider) {
            CredentialSource::AuthJson(key) | CredentialSource::ModelsJson(key) => Ok(key),
            CredentialSource::OAuth => Err(AuthCommandError::OAuthNotKey(provider.clone())),
            CredentialSource::None => Err(AuthCommandError::NoKey(provider.clone())),
        },
        AuthCommand::Check { provider, json } => {
            let (configured, source) = match credential_source(agent_dir, provider) {
                CredentialSource::AuthJson(_) => (true, "auth.json"),
                CredentialSource::OAuth => (true, "oauth"),
                CredentialSource::ModelsJson(_) => (true, "models.json"),
                CredentialSource::None => (false, "none"),
            };
            if !configured {
                return Err(AuthCommandError::NoKey(provider.clone()));
            }
            Ok(if *json {
                serde_json::json!({"provider": provider, "configured": true, "source": source})
                    .to_string()
            } else {
                format!("{provider}: configured ({source})")
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn parses_auth_commands_and_rejects_bad_ones() {
        assert_eq!(parse(&args(&["hello"])), None);
        assert_eq!(
            parse(&args(&["auth", "check", "--provider", "acme", "--json"])),
            Some(Ok(AuthCommand::Check {
                provider: "acme".into(),
                json: true
            }))
        );
        assert_eq!(
            parse(&args(&["auth", "print-api-key"])),
            Some(Err(AuthCommandError::MissingProvider("print-api-key")))
        );
        assert_eq!(
            parse(&args(&["auth", "nope"])),
            Some(Err(AuthCommandError::Unknown("nope".into())))
        );
        assert_eq!(
            parse(&args(&[
                "auth",
                "print-api-key",
                "--provider",
                "a",
                "--json"
            ])),
            Some(Err(AuthCommandError::UnknownOption("--json".into())))
        );
    }

    #[test]
    fn prints_keys_and_reports_missing_or_oauth_credentials() {
        let dir = std::env::temp_dir().join(format!("mu-auth-cmd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("models.json"),
            r#"{"providers":{"acme":{"apiKey":"k1"}}}"#,
        )
        .unwrap();
        let print = |p: &str| run(&AuthCommand::PrintApiKey { provider: p.into() }, &dir);
        assert_eq!(print("acme").unwrap(), "k1");
        assert_eq!(
            print("none").unwrap_err(),
            AuthCommandError::NoKey("none".into())
        );
        let check = run(
            &AuthCommand::Check {
                provider: "acme".into(),
                json: true,
            },
            &dir,
        )
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&check).unwrap();
        assert_eq!(
            parsed,
            serde_json::json!({"provider":"acme","configured":true,"source":"models.json"})
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
