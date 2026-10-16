use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    process::Stdio,
};

use futures_util::StreamExt;
use mu_session::{MessageCost, MessageUsage, SessionDocument};
use reqwest::{Client, Url, header};
use serde_json::{Value, json};
use thiserror::Error;

const ORIGIN: &str = "https://opencode.cloudflare.dev";

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("configuration: {0}")]
    Configuration(String),
    #[error("provider: {0}")]
    Provider(String),
    #[error("session: {0}")]
    Session(String),
    #[error("I/O: {0}")]
    Io(#[from] io::Error),
}

struct Config {
    agent_dir: PathBuf,
    cwd: PathBuf,
    provider: String,
    model: String,
}
struct Provider {
    http: Client,
    endpoint: Url,
    token: String,
    model: String,
}
struct Reply {
    text: String,
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    reasoning: Option<u64>,
    total_tokens: u64,
}

fn config() -> Result<Config, AgentError> {
    let cwd = std::env::current_dir()?;
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| AgentError::Configuration("HOME is not set".into()))?;
    let agent_dir = std::env::var_os("PI_CODING_AGENT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".pi/agent"));
    let mut provider = None;
    let mut model = None;
    for path in [
        agent_dir.join("settings.json"),
        cwd.join(".pi/settings.json"),
    ] {
        if !path.exists() {
            continue;
        }
        let value: Value = serde_json::from_str(
            &std::fs::read_to_string(&path)
                .map_err(|e| AgentError::Configuration(format!("{}: {e}", path.display())))?,
        )
        .map_err(|e| AgentError::Configuration(format!("{}: {e}", path.display())))?;
        if let Some(v) = value.get("defaultProvider").and_then(Value::as_str) {
            provider = Some(v.to_owned());
        }
        if let Some(v) = value.get("defaultModel").and_then(Value::as_str) {
            model = Some(v.to_owned());
        }
    }
    Ok(Config {
        agent_dir,
        cwd,
        provider: provider.ok_or_else(|| {
            AgentError::Configuration("Pi settings have no defaultProvider".into())
        })?,
        model: model
            .ok_or_else(|| AgentError::Configuration("Pi settings have no defaultModel".into()))?,
    })
}

async fn token(provider: &str) -> Result<String, AgentError> {
    let env = if provider == "opencode-cf" {
        "OPENCODE_CF_TOKEN".into()
    } else {
        format!("{}_TOKEN", provider.replace('-', "_").to_ascii_uppercase())
    };
    if let Ok(value) = std::env::var(&env) {
        if !value.trim().is_empty() {
            return Ok(value.trim().to_owned());
        }
    }
    let bin = std::env::var_os("CLOUDFLARED_PATH").unwrap_or_else(|| "cloudflared".into());
    let output = tokio::process::Command::new(bin)
        .args(["access", "token", "-app=https://opencode.cloudflare.dev"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
        .map_err(|e| AgentError::Provider(format!("could not run cloudflared: {e}")))?;
    let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !output.status.success() || value.is_empty() || value.to_lowercase().contains("login") {
        return Err(AgentError::Provider(format!(
            "no Cloudflare Access token; run `cloudflared access login -app={ORIGIN}` or set {env}"
        )));
    }
    Ok(value)
}

fn trusted_url(value: &str) -> Result<Url, AgentError> {
    let url =
        Url::parse(value).map_err(|e| AgentError::Provider(format!("invalid gateway URL: {e}")))?;
    let host = url.host_str().unwrap_or_default();
    if url.scheme() != "https"
        || !(host == "opencode.cloudflare.dev" || host.ends_with(".opencode.cloudflare.dev"))
    {
        return Err(AgentError::Provider(format!(
            "refusing untrusted gateway URL `{value}`"
        )));
    }
    Ok(url)
}

async fn connect(config: &Config) -> Result<Provider, AgentError> {
    if !config.provider.starts_with("opencode-cf") {
        return Err(AgentError::Configuration(format!(
            "Pi provider `{}` is unsupported (expected opencode-cf)",
            config.provider
        )));
    }
    let token = token(&config.provider).await?;
    let http = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| AgentError::Provider(format!("HTTP client: {e}")))?;
    let well: Value = http
        .get(format!("{ORIGIN}/.well-known/opencode"))
        .send()
        .await
        .map_err(|e| AgentError::Provider(format!("gateway discovery failed: {e}")))?
        .error_for_status()
        .map_err(|e| AgentError::Provider(format!("gateway discovery failed: {e}")))?
        .json()
        .await
        .map_err(|e| AgentError::Provider(format!("invalid gateway discovery: {e}")))?;
    let remote = well
        .get("remote_config")
        .ok_or_else(|| AgentError::Provider("discovery has no remote_config".into()))?;
    let config_url = remote
        .get("url")
        .and_then(Value::as_str)
        .ok_or_else(|| AgentError::Provider("discovery has no config URL".into()))?;
    let mut request = http.get(trusted_url(config_url)?);
    if let Some(headers) = remote.get("headers").and_then(Value::as_object) {
        for (name, value) in headers {
            if let Some(value) = value.as_str() {
                let value = value
                    .replace("{env:TOKEN}", &token)
                    .replace("${TOKEN}", &token);
                request = request.header(name, value);
            }
        }
    }
    let gateway: Value = request
        .send()
        .await
        .map_err(|e| AgentError::Provider(format!("gateway config fetch failed: {e}")))?
        .error_for_status()
        .map_err(|e| AgentError::Provider(format!("gateway config fetch failed: {e}")))?
        .json()
        .await
        .map_err(|e| AgentError::Provider(format!("invalid gateway config: {e}")))?;
    let providers = gateway
        .get("provider")
        .and_then(Value::as_object)
        .ok_or_else(|| AgentError::Provider("config has no provider catalog".into()))?;
    let mut endpoint = None;
    for (id, p) in providers {
        let Some(m) = p.pointer(&format!("/models/{}", config.model)) else {
            continue;
        };
        let base = m
            .pointer("/provider/api")
            .and_then(Value::as_str)
            .or_else(|| p.pointer("/options/baseURL").and_then(Value::as_str))
            .or_else(|| p.pointer("/options/baseUrl").and_then(Value::as_str))
            .or_else(|| p.get("api").and_then(Value::as_str))
            .ok_or_else(|| AgentError::Provider(format!("provider `{id}` has no API URL")))?;
        if id != "openai"
            && !Url::parse(base)
                .map(|u| u.path().contains("/openai"))
                .unwrap_or(false)
        {
            return Err(AgentError::Provider(format!(
                "model uses unsupported provider `{id}`"
            )));
        }
        let mut url = trusted_url(base)?;
        url.set_path(&format!("{}/responses", url.path().trim_end_matches('/')));
        endpoint = Some(url);
        break;
    }
    let endpoint = endpoint.ok_or_else(|| {
        AgentError::Provider(format!(
            "model `{}` not found in gateway catalog",
            config.model
        ))
    })?;
    Ok(Provider {
        http,
        endpoint,
        token,
        model: config.model.clone(),
    })
}

impl Provider {
    async fn complete(
        &self,
        history: &[(String, String)],
        delta: &mut impl FnMut(&str),
    ) -> Result<Reply, AgentError> {
        let input: Vec<Value> = history.iter().map(|(role, text)| json!({"role": role, "content": [{"type": if role == "assistant" {"output_text"} else {"input_text"}, "text": text}]})).collect();
        let response = self.http.post(self.endpoint.clone()).bearer_auth(&self.token)
            .header("cf-access-token", &self.token).header("x-requested-with", "xmlhttprequest")
            .header(header::ACCEPT, "text/event-stream")
            .json(&json!({"model":self.model,"instructions":"You are Mu, a helpful coding assistant. Be clear, practical, and concise.","input":input,"stream":true,"store":false,"include":["reasoning.encrypted_content"]}))
            .send().await.map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(AgentError::Provider(format!(
                "model request returned HTTP {}",
                response.status()
            )));
        }
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        let mut text = String::new();
        let (mut input_tokens, mut output_tokens) = (0, 0);
        let (mut cache_read, mut cache_write, mut reasoning, mut total_tokens) = (0, 0, None, 0);
        while let Some(chunk) = stream.next().await {
            bytes.extend_from_slice(
                &chunk.map_err(|e| AgentError::Provider(format!("model stream failed: {e}")))?,
            );
            while let Some(pos) = bytes.windows(2).position(|w| w == b"\n\n") {
                let frame: Vec<_> = bytes.drain(..pos + 2).collect();
                let frame = String::from_utf8_lossy(&frame);
                let Some(data) = frame.lines().find_map(|line| line.strip_prefix("data:")) else {
                    continue;
                };
                if let Ok(event) = serde_json::from_str::<Value>(data.trim()) {
                    match event.get("type").and_then(Value::as_str).unwrap_or("") {
                        "response.output_text.delta" => {
                            if let Some(s) = event.get("delta").and_then(Value::as_str) {
                                text.push_str(s);
                                delta(s);
                            }
                        }
                        "response.completed" => {
                            let usage = event.pointer("/response/usage");
                            let raw_input = usage
                                .and_then(|value| value.get("input_tokens"))
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            cache_read = usage
                                .and_then(|value| {
                                    value.pointer("/input_tokens_details/cached_tokens")
                                })
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            cache_write = usage
                                .and_then(|value| {
                                    value.pointer("/input_tokens_details/cache_write_tokens")
                                })
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            input_tokens = raw_input.saturating_sub(cache_read + cache_write);
                            output_tokens = usage
                                .and_then(|value| value.get("output_tokens"))
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            reasoning = usage
                                .and_then(|value| {
                                    value.pointer("/output_tokens_details/reasoning_tokens")
                                })
                                .and_then(Value::as_u64);
                            total_tokens = usage
                                .and_then(|value| value.get("total_tokens"))
                                .and_then(Value::as_u64)
                                .unwrap_or(raw_input + output_tokens);
                        }
                        "response.failed" | "error" => {
                            return Err(AgentError::Provider(
                                event
                                    .pointer("/response/error/message")
                                    .or_else(|| event.pointer("/error/message"))
                                    .and_then(Value::as_str)
                                    .unwrap_or("response failed")
                                    .to_owned(),
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }
        if text.is_empty() {
            return Err(AgentError::Provider(
                "stream ended without response text".into(),
            ));
        }
        Ok(Reply {
            text,
            input: input_tokens,
            output: output_tokens,
            cache_read,
            cache_write,
            reasoning,
            total_tokens,
        })
    }
}

fn session_path(c: &Config) -> PathBuf {
    let cwd = c
        .cwd
        .to_string_lossy()
        .trim_start_matches(['/', '\\'])
        .replace(['/', '\\', ':'], "-");
    let dir = c.agent_dir.join("sessions").join(format!("--{cwd}--"));
    let ts = chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
        .replace([':', '.'], "-");
    dir.join(format!("{ts}_{}.jsonl", uuid::Uuid::new_v4()))
}

fn usage(r: &Reply) -> MessageUsage {
    MessageUsage {
        input: r.input,
        output: r.output,
        cache_read: r.cache_read,
        cache_write: r.cache_write,
        reasoning: r.reasoning,
        total_tokens: r.total_tokens,
        cost: MessageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    }
}

pub async fn run_cli() -> Result<(), AgentError> {
    let c = config()?;
    let provider = connect(&c).await?;
    let path = session_path(&c);
    let mut session = SessionDocument::new(c.cwd.to_string_lossy().into_owned());
    let mut history = Vec::new();
    println!(
        "Mu — {} via {} (session {})",
        c.model,
        c.provider,
        path.display()
    );
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        turn(&provider, &mut session, &mut history, &path, args.join(" ")).await?;
        return Ok(());
    }
    let mut line = String::new();
    loop {
        print!("\n> ");
        io::stdout().flush()?;
        line.clear();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let prompt = line.trim();
        if prompt == "/exit" || prompt == "/quit" {
            break;
        }
        if prompt.is_empty() {
            continue;
        }
        turn(
            &provider,
            &mut session,
            &mut history,
            &path,
            prompt.to_owned(),
        )
        .await?;
    }
    Ok(())
}

async fn turn(
    p: &Provider,
    s: &mut SessionDocument,
    h: &mut Vec<(String, String)>,
    path: &Path,
    prompt: String,
) -> Result<(), AgentError> {
    s.append_user_text(prompt.clone());
    s.write_atomic(path)
        .map_err(|e| AgentError::Session(e.to_string()))?;
    h.push(("user".into(), prompt));
    println!();
    let mut output = |text: &str| {
        print!("{text}");
        let _ = io::stdout().flush();
    };
    let reply = p.complete(h, &mut output).await?;
    println!();
    s.append_assistant_text(
        "openai-responses",
        "opencode-cf",
        &p.model,
        reply.text.clone(),
        usage(&reply),
    );
    s.write_atomic(path)
        .map_err(|e| AgentError::Session(e.to_string()))?;
    h.push(("assistant".into(), reply.text));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::trusted_url;
    #[test]
    fn allows_access_gateway_and_rejects_untrusted_hosts() {
        assert!(trusted_url("https://gateway.opencode.cloudflare.dev/openai").is_ok());
        assert!(trusted_url("https://attacker.example/openai").is_err());
    }
}
