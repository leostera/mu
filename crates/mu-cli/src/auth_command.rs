//! `mu auth print-api-key|check --provider <id>` (Pi's `cli/auth-command.ts`, API-key subset).

use std::path::Path;

use mu_core::auth::{CredentialSource, credential_source};

#[derive(Debug, PartialEq, Eq)]
pub enum AuthCommand {
    PrintApiKey { provider: String },
    Check { provider: String, json: bool },
    Login { provider: String, copy_code: bool },
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
    #[error("{0}")]
    Login(String),
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
        "login" => ("login", false),
        _ => return Some(Err(AuthCommandError::Unknown(sub))),
    };
    let (mut provider, mut json, mut copy_code) = (None, false, false);
    let mut rest = args[2..].iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--provider" => provider = rest.next().map(|p| p.trim().to_owned()),
            "--json" if check => json = true,
            "--copy-code" if name == "login" => copy_code = true,
            other => return Some(Err(AuthCommandError::UnknownOption(other.to_owned()))),
        }
    }
    let Some(provider) = provider.filter(|p| !p.is_empty()) else {
        return Some(Err(AuthCommandError::MissingProvider(name)));
    };
    if name == "login" {
        return Some(Ok(AuthCommand::Login {
            provider,
            copy_code,
        }));
    }
    Some(Ok(if check {
        AuthCommand::Check { provider, json }
    } else {
        AuthCommand::PrintApiKey { provider }
    }))
}

/// Run the command; the returned text is printed to stdout.
pub fn run(command: &AuthCommand, agent_dir: &Path) -> Result<String, AuthCommandError> {
    match command {
        AuthCommand::Login { .. } => unreachable!("login is async; see `login`"),
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

async fn login_openrouter(agent_dir: &Path) -> Result<String, AuthCommandError> {
    use mu_core::oauth::{CallbackServer, Pkce, openrouter};
    let fail = |e: String| AuthCommandError::Login(e);
    let pkce = Pkce::generate();
    // OpenRouter sends no `state`; a random path keeps stray requests from completing the login.
    let path = format!("/oauth/callback/{}", uuid::Uuid::new_v4());
    let server = CallbackServer::bind("127.0.0.1", 0, &path, "127.0.0.1")
        .await
        .map_err(|e| fail(e.to_string()))?;
    eprintln!(
        "Open this URL to sign in to OpenRouter:\n\n{}\n",
        openrouter::authorize(&server.redirect_uri, &pkce)
    );
    let code = server
        .wait_for_code("OpenRouter", None, std::time::Duration::from_secs(300))
        .await
        .map_err(fail)?;
    let credential = openrouter::exchange(&code, &pkce.verifier, openrouter::TOKEN_URL)
        .await
        .map_err(fail)?;
    mu_core::auth::set_credential(&agent_dir.join("auth.json"), "openrouter", credential)
        .map_err(|e| AuthCommandError::Login(e.to_string()))?;
    Ok("Logged in to openrouter; credential saved to auth.json".to_owned())
}

/// `mu auth login --provider anthropic [--copy-code]`: OAuth login stored in `auth.json`.
pub async fn login(
    provider: &str,
    copy_code: bool,
    agent_dir: &Path,
) -> Result<String, AuthCommandError> {
    use mu_core::oauth::{CallbackServer, anthropic};
    if provider == "openrouter" {
        return login_openrouter(agent_dir).await;
    }
    if provider != "anthropic" {
        return Err(AuthCommandError::Login(format!(
            "OAuth login is only implemented for \"anthropic\" and \"openrouter\" (got \"{provider}\"); use /login <provider> <api-key> for API keys"
        )));
    }
    let fail = AuthCommandError::Login;
    let server = if copy_code {
        None
    } else {
        match CallbackServer::bind(
            "127.0.0.1",
            anthropic::CALLBACK_PORT,
            anthropic::CALLBACK_PATH,
            "localhost",
        )
        .await
        {
            Ok(server) => Some(server),
            Err(_) => CallbackServer::bind("127.0.0.1", 0, anthropic::CALLBACK_PATH, "localhost")
                .await
                .ok(),
        }
    };
    let redirect = server
        .as_ref()
        .map_or(anthropic::COPY_CODE_REDIRECT_URI, |s| {
            s.redirect_uri.as_str()
        });
    let attempt = anthropic::Login::start(redirect);
    eprintln!(
        "Open this URL to sign in to Anthropic:\n\n{}\n",
        attempt.url
    );
    eprintln!(
        "Then paste the code or redirect URL here{}:",
        if server.is_some() {
            " (or just finish in the browser)"
        } else {
            ""
        }
    );
    let pasted = tokio::task::spawn_blocking(|| {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).map(|_| line)
    });
    let credential = match server {
        Some(server) => tokio::select! {
            code = server.wait_for_code("Anthropic", Some(attempt.state()), std::time::Duration::from_secs(300)) => {
                attempt.finish_with_code(&code.map_err(fail)?, anthropic::TOKEN_URL).await
            }
            line = pasted => {
                let line = line.map_err(|e| AuthCommandError::Login(e.to_string()))?.map_err(|e| AuthCommandError::Login(e.to_string()))?;
                attempt.finish_with_input(&line, anthropic::TOKEN_URL).await
            }
        },
        None => {
            let line = pasted.await.map_err(|e| AuthCommandError::Login(e.to_string()))?.map_err(|e| AuthCommandError::Login(e.to_string()))?;
            attempt.finish_with_input(&line, anthropic::TOKEN_URL).await
        }
    }
    .map_err(AuthCommandError::Login)?;
    mu_core::auth::set_credential(&agent_dir.join("auth.json"), provider, credential)
        .map_err(|e| AuthCommandError::Login(e.to_string()))?;
    Ok(format!(
        "Logged in to {provider}; credential saved to auth.json"
    ))
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
    fn parses_login_with_the_copy_code_option() {
        assert_eq!(
            parse(&args(&[
                "auth",
                "login",
                "--provider",
                "anthropic",
                "--copy-code"
            ])),
            Some(Ok(AuthCommand::Login {
                provider: "anthropic".into(),
                copy_code: true
            }))
        );
        assert_eq!(
            parse(&args(&["auth", "check", "--provider", "a", "--copy-code"])),
            Some(Err(AuthCommandError::UnknownOption("--copy-code".into())))
        );
    }

    #[tokio::test]
    async fn login_rejects_providers_without_an_oauth_flow() {
        let dir = std::env::temp_dir();
        let error = login("acme", false, &dir).await.unwrap_err();
        assert!(error.to_string().contains("only implemented for"));
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
