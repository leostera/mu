//! OAuth building blocks shared by provider logins (Pi's `ai/src/auth/oauth`): PKCE, authorization
//! URLs, pasted-redirect parsing, a loopback callback server, and token exchange/refresh.
//!
//! Provider-specific flows (client ids, scopes, endpoints) build on these; none are wired to a
//! provider yet.

use std::time::Duration;

use base64::Engine;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

use crate::auth::Credential;

/// PKCE verifier and its S256 challenge (Pi's `generatePKCE`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
}

impl Pkce {
    pub fn generate() -> Self {
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        bytes[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        Self::from_verifier(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
    }

    pub fn from_verifier(verifier: String) -> Self {
        let digest = Sha256::digest(verifier.as_bytes());
        let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
        Self {
            verifier,
            challenge,
        }
    }
}

/// `https://…/authorize?…` with the given query parameters percent-encoded.
pub fn authorize_url(base: &str, params: &[(&str, &str)]) -> Result<String, String> {
    let mut url =
        url::Url::parse(base).map_err(|error| format!("invalid authorize URL: {error}"))?;
    url.query_pairs_mut().extend_pairs(params.iter().copied());
    Ok(url.into())
}

/// What the user pasted instead of completing the browser redirect: a redirect URL,
/// `code#state`, a `code=…&state=…` query, or a bare code.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct AuthorizationInput {
    pub code: Option<String>,
    pub state: Option<String>,
}

pub fn parse_authorization_input(input: &str) -> AuthorizationInput {
    let value = input.trim();
    if value.is_empty() {
        return AuthorizationInput::default();
    }
    let from_pairs = |pairs: url::form_urlencoded::Parse<'_>| {
        let (mut code, mut state) = (None, None);
        for (key, value) in pairs {
            match key.as_ref() {
                "code" => code = Some(value.into_owned()),
                "state" => state = Some(value.into_owned()),
                _ => {}
            }
        }
        AuthorizationInput { code, state }
    };
    if let Ok(url) = url::Url::parse(value) {
        return from_pairs(url.query_pairs());
    }
    if let Some((code, state)) = value.split_once('#') {
        return AuthorizationInput {
            code: Some(code.to_owned()),
            state: Some(state.to_owned()),
        };
    }
    if value.contains("code=") {
        return from_pairs(url::form_urlencoded::parse(value.as_bytes()));
    }
    AuthorizationInput {
        code: Some(value.to_owned()),
        state: None,
    }
}

/// A loopback listener that receives one OAuth redirect.
pub struct CallbackServer {
    listener: TcpListener,
    pub redirect_uri: String,
    path: String,
}

impl CallbackServer {
    /// Bind `host:port` (port 0 picks a free one); `redirect_host` appears in the redirect URI.
    pub async fn bind(
        host: &str,
        port: u16,
        path: &str,
        redirect_host: &str,
    ) -> std::io::Result<Self> {
        let listener = TcpListener::bind((host, port)).await?;
        let port = listener.local_addr()?.port();
        Ok(Self {
            listener,
            redirect_uri: format!("http://{redirect_host}:{port}{path}"),
            path: path.to_owned(),
        })
    }

    /// Wait for the redirect and return its authorization code. A wrong `state`, wrong route or
    /// missing code gets an error page and the wait continues; a provider `error` fails the login.
    pub async fn wait_for_code(
        &self,
        provider: &str,
        expected_state: Option<&str>,
        timeout: Duration,
    ) -> Result<String, String> {
        let mut params = self
            .wait_for_params(provider, expected_state, timeout)
            .await?;
        Ok(params.remove("code").unwrap_or_default())
    }

    /// Like [`Self::wait_for_code`], returning every query parameter of the accepted redirect
    /// (OpenAI's dynamic registration returns the issued `client_id` there).
    pub async fn wait_for_params(
        &self,
        provider: &str,
        expected_state: Option<&str>,
        timeout: Duration,
    ) -> Result<std::collections::HashMap<String, String>, String> {
        tokio::time::timeout(timeout, self.serve(provider, expected_state))
            .await
            .map_err(|_| format!("{provider} sign-in timed out"))?
    }

    async fn serve(
        &self,
        provider: &str,
        expected_state: Option<&str>,
    ) -> Result<std::collections::HashMap<String, String>, String> {
        loop {
            let (mut socket, _) = self.listener.accept().await.map_err(|e| e.to_string())?;
            let mut buffer = vec![0u8; 8192];
            let read = socket.read(&mut buffer).await.map_err(|e| e.to_string())?;
            let request = String::from_utf8_lossy(&buffer[..read]);
            let target = request
                .lines()
                .next()
                .and_then(|line| line.strip_prefix("GET "))
                .and_then(|rest| rest.split(' ').next())
                .unwrap_or_default();
            let url = url::Url::parse(&format!("http://localhost{target}"));
            let respond = |status: &str, body: String| {
                format!(
                    "HTTP/1.1 {status}\r\ncontent-type: text/html; charset=utf-8\r\ncache-control: no-store\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                )
            };
            let Ok(url) = url else {
                let _ = socket
                    .write_all(respond("400 Bad Request", page("Bad request.")).as_bytes())
                    .await;
                continue;
            };
            if url.path() != self.path {
                let _ = socket
                    .write_all(
                        respond("404 Not Found", page("Callback route not found.")).as_bytes(),
                    )
                    .await;
                continue;
            }
            let query = |name: &str| {
                url.query_pairs()
                    .find(|(k, _)| k == name)
                    .map(|(_, v)| v.into_owned())
            };
            if expected_state.is_some_and(|state| query("state").as_deref() != Some(state)) {
                let _ = socket
                    .write_all(respond("400 Bad Request", page("State mismatch.")).as_bytes())
                    .await;
                continue;
            }
            if let Some(error) = query("error") {
                let description = query("error_description").unwrap_or(error);
                let _ = socket
                    .write_all(
                        respond(
                            "400 Bad Request",
                            page(&format!("{provider} authorization failed.")),
                        )
                        .as_bytes(),
                    )
                    .await;
                return Err(format!("{provider} authorization failed: {description}"));
            }
            let Some(code) = query("code") else {
                let _ = socket
                    .write_all(
                        respond("400 Bad Request", page("Missing authorization code.")).as_bytes(),
                    )
                    .await;
                continue;
            };
            let message = format!("Signed in to {provider}. You may now close this page.");
            let _ = socket
                .write_all(respond("200 OK", page(&message)).as_bytes())
                .await;
            let _ = code;
            return Ok(url
                .query_pairs()
                .map(|(k, v)| (k.into_owned(), v.into_owned()))
                .collect());
        }
    }
}

fn page(message: &str) -> String {
    let message = message
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    format!("<!doctype html><meta charset=\"utf-8\"><title>Mu</title><p>{message}</p>")
}

/// Tokens are treated as expired five minutes early, as in Pi.
const EXPIRY_MARGIN_MS: f64 = 5.0 * 60.0 * 1000.0;

fn now_ms() -> f64 {
    chrono::Utc::now().timestamp_millis() as f64
}

/// Turn a token endpoint response into a stored credential.
pub fn credential_from_token_response(body: &Value, now_ms: f64) -> Result<Credential, String> {
    let text = |key: &str| body.get(key).and_then(Value::as_str).map(str::to_owned);
    let access = text("access_token").ok_or("token response has no access_token")?;
    let refresh = text("refresh_token").ok_or("token response has no refresh_token")?;
    let expires_in = body
        .get("expires_in")
        .and_then(Value::as_f64)
        .ok_or("token response has no expires_in")?;
    Ok(Credential::Oauth {
        access,
        refresh,
        expires: now_ms + expires_in * 1000.0 - EXPIRY_MARGIN_MS,
        extra: Default::default(),
    })
}

async fn post_json(url: &str, body: Value) -> Result<Value, String> {
    let response = reqwest::Client::new()
        .post(url)
        .timeout(Duration::from_secs(30))
        .header("accept", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|error| format!("Token request failed. url={url}; {error}"))?;
    let status = response.status();
    let text = response.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!(
            "HTTP request failed. status={}; url={url}; body={text}",
            status.as_u16()
        ));
    }
    serde_json::from_str(&text)
        .map_err(|error| format!("Token response was not JSON. url={url}; body={text}; {error}"))
}

/// Exchange an authorization code (JSON body, as Anthropic's token endpoint expects).
pub async fn exchange_code(
    token_url: &str,
    client_id: &str,
    code: &str,
    state: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<Credential, String> {
    let body = post_json(
        token_url,
        json!({
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "state": state,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }),
    )
    .await?;
    credential_from_token_response(&body, now_ms())
}

/// Refresh an expired credential with its refresh token.
pub async fn refresh(
    token_url: &str,
    client_id: &str,
    refresh_token: &str,
) -> Result<Credential, String> {
    let body = post_json(
        token_url,
        json!({"grant_type": "refresh_token", "client_id": client_id, "refresh_token": refresh_token}),
    )
    .await?;
    credential_from_token_response(&body, now_ms())
}

/// Anthropic (Claude Pro/Max) login, as in Pi's `auth/oauth/anthropic.ts`.
pub mod anthropic {
    use super::*;

    // Pi stores the client id base64-encoded; it is public.
    const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
    const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
    pub const TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
    pub const CALLBACK_PORT: u16 = 53692;
    pub const CALLBACK_PATH: &str = "/callback";
    pub const COPY_CODE_REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
    const SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";

    /// A started login: open `url`, then finish with the redirect or a pasted code.
    pub struct Login {
        pub url: String,
        pub redirect_uri: String,
        pkce: Pkce,
    }

    impl Login {
        pub fn start(redirect_uri: &str) -> Self {
            let pkce = Pkce::generate();
            let url = authorize_url(
                AUTHORIZE_URL,
                &[
                    ("code", "true"),
                    ("client_id", CLIENT_ID),
                    ("response_type", "code"),
                    ("redirect_uri", redirect_uri),
                    ("scope", SCOPES),
                    ("code_challenge", &pkce.challenge),
                    ("code_challenge_method", "S256"),
                    ("state", &pkce.verifier),
                ],
            )
            .expect("static authorize URL parses");
            Self {
                url,
                redirect_uri: redirect_uri.to_owned(),
                pkce,
            }
        }

        /// The `state` the redirect must carry (Pi uses the PKCE verifier).
        pub fn state(&self) -> &str {
            &self.pkce.verifier
        }

        /// Finish with a pasted redirect URL or code; rejects a mismatched state.
        pub async fn finish_with_input(
            &self,
            input: &str,
            token_url: &str,
        ) -> Result<Credential, String> {
            let parsed = parse_authorization_input(input);
            if parsed
                .state
                .as_deref()
                .is_some_and(|state| state != self.state())
            {
                return Err("OAuth state mismatch".into());
            }
            let code = parsed.code.ok_or("Missing authorization code")?;
            self.finish_with_code(&code, token_url).await
        }

        pub async fn finish_with_code(
            &self,
            code: &str,
            token_url: &str,
        ) -> Result<Credential, String> {
            exchange_code(
                token_url,
                CLIENT_ID,
                code,
                self.state(),
                &self.pkce.verifier,
                &self.redirect_uri,
            )
            .await
        }
    }

    pub async fn refresh_token(refresh_token: &str) -> Result<Credential, String> {
        refresh(TOKEN_URL, CLIENT_ID, refresh_token).await
    }
}

/// OpenRouter login: the code is exchanged for a permanent API key (Pi's `openrouter.ts`).
pub mod openrouter {
    use super::*;

    const AUTHORIZE_URL: &str = "https://openrouter.ai/auth";
    pub const TOKEN_URL: &str = "https://openrouter.ai/api/v1/auth/keys";

    /// `callback_url` is the loopback redirect URI; OpenRouter sends no `state`.
    pub fn authorize(callback_url: &str, pkce: &Pkce) -> String {
        authorize_url(
            AUTHORIZE_URL,
            &[
                ("callback_url", callback_url),
                ("code_challenge", &pkce.challenge),
                ("code_challenge_method", "S256"),
            ],
        )
        .expect("static authorize URL parses")
    }

    /// The bare code from a pasted redirect URL, `code=…` query, or the code itself.
    pub fn code_from_input(input: &str) -> Option<String> {
        let parsed = parse_authorization_input(input);
        parsed.code.filter(|code| !code.is_empty())
    }

    pub fn credential_from_key_response(body: &Value) -> Result<Credential, String> {
        let key = body
            .get("key")
            .and_then(Value::as_str)
            .filter(|key| !key.is_empty())
            .ok_or("OpenRouter OAuth response carries no \"key\"")?;
        Ok(Credential::Oauth {
            access: key.to_owned(),
            refresh: String::new(),
            expires: 9_007_199_254_740_991.0,
            extra: Default::default(),
        })
    }

    pub async fn exchange(
        code: &str,
        verifier: &str,
        token_url: &str,
    ) -> Result<Credential, String> {
        let body = post_json(
            token_url,
            json!({"code": code, "code_verifier": verifier, "code_challenge_method": "S256"}),
        )
        .await
        .map_err(|error| format!("OpenRouter OAuth key exchange failed: {error}"))?;
        credential_from_key_response(&body)
    }
}

/// One poll of an RFC 8628 device-code flow.
#[derive(Debug, PartialEq, Eq)]
pub enum DevicePoll<T> {
    Pending,
    /// Poll more slowly; the server may state the new interval in seconds.
    SlowDown(Option<u64>),
    Failed(String),
    Complete(T),
}

/// Timing of a device-code poll loop; [`DevicePollTiming::pi`] has Pi's constants.
#[derive(Debug, Clone, Copy)]
pub struct DevicePollTiming {
    pub interval: Duration,
    pub expires_in: Option<Duration>,
    pub wait_before_first_poll: bool,
    pub minimum_interval: Duration,
    pub slow_down_increment: Duration,
}

impl DevicePollTiming {
    /// `interval_seconds` defaults to 5 (RFC 8628 §3.2); the minimum is 1s; slow_down adds 5s.
    pub fn pi(
        interval_seconds: Option<u64>,
        expires_in_seconds: Option<u64>,
        wait_before_first_poll: bool,
    ) -> Self {
        Self {
            interval: Duration::from_secs(interval_seconds.unwrap_or(5)),
            expires_in: expires_in_seconds.map(Duration::from_secs),
            wait_before_first_poll,
            minimum_interval: Duration::from_secs(1),
            slow_down_increment: Duration::from_secs(5),
        }
    }
}

const DEVICE_TIMEOUT: &str = "Device flow timed out";
const DEVICE_SLOW_DOWN_TIMEOUT: &str = "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again.";

/// Pi's `pollOAuthDeviceCodeFlow`: poll until complete, failed, or the deadline passes.
pub async fn poll_device_code<T, F, Fut>(timing: DevicePollTiming, mut poll: F) -> Result<T, String>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = DevicePoll<T>>,
{
    let deadline = timing
        .expires_in
        .map(|expires| std::time::Instant::now() + expires);
    let remaining = |deadline: Option<std::time::Instant>| {
        deadline.map_or(Duration::MAX, |at| {
            at.saturating_duration_since(std::time::Instant::now())
        })
    };
    let mut interval = timing.interval.max(timing.minimum_interval);
    let mut slow_downs = 0u32;
    if timing.wait_before_first_poll {
        let wait = interval.min(remaining(deadline));
        tokio::time::sleep(wait).await;
    }
    while !remaining(deadline).is_zero() {
        match poll().await {
            DevicePoll::Complete(value) => return Ok(value),
            DevicePoll::Failed(message) => return Err(message),
            DevicePoll::Pending => {}
            DevicePoll::SlowDown(server_interval) => {
                slow_downs += 1;
                interval = match server_interval.filter(|seconds| *seconds > 0) {
                    Some(seconds) => Duration::from_secs(seconds).max(timing.minimum_interval),
                    None => (interval + timing.slow_down_increment).max(timing.minimum_interval),
                };
            }
        }
        let left = remaining(deadline);
        if left.is_zero() {
            break;
        }
        tokio::time::sleep(interval.min(left)).await;
    }
    Err(if slow_downs > 0 {
        DEVICE_SLOW_DOWN_TIMEOUT
    } else {
        DEVICE_TIMEOUT
    }
    .to_owned())
}

/// xAI (Grok / X subscription) device-code login, as in Pi's `xai.ts`.
pub mod xai {
    use super::*;

    const CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
    const SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
    pub const DEVICE_CODE_URL: &str = "https://auth.x.ai/oauth2/device/code";
    pub const TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";

    #[derive(Debug, PartialEq, Eq)]
    pub struct DeviceCode {
        pub device_code: String,
        pub user_code: String,
        /// The URL to show: `verification_uri_complete` when present.
        pub verification_uri: String,
        pub interval_seconds: Option<u64>,
        pub expires_in_seconds: u64,
    }

    /// Only https URLs may be shown or opened (Pi rejects anything else).
    fn https_uri(raw: &str) -> Result<String, String> {
        match url::Url::parse(raw) {
            Ok(url) if url.scheme() == "https" => Ok(url.into()),
            _ => Err("Untrusted verification URI in xAI OAuth response".into()),
        }
    }

    fn required<'a>(body: &'a Value, field: &str) -> Result<&'a str, String> {
        body.get(field)
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| format!("Invalid xAI OAuth response field: {field}"))
    }

    pub fn parse_device_code(body: &Value) -> Result<DeviceCode, String> {
        let expires_in = body
            .get("expires_in")
            .and_then(Value::as_f64)
            .filter(|v| *v > 0.0)
            .ok_or("Invalid xAI OAuth response field: expires_in")? as u64;
        let complete = body
            .get("verification_uri_complete")
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
            .map(https_uri)
            .transpose()?;
        let plain = https_uri(required(body, "verification_uri")?)?;
        Ok(DeviceCode {
            device_code: required(body, "device_code")?.to_owned(),
            user_code: required(body, "user_code")?.to_owned(),
            verification_uri: complete.unwrap_or(plain),
            interval_seconds: body
                .get("interval")
                .and_then(Value::as_u64)
                .filter(|v| *v > 0),
            expires_in_seconds: expires_in,
        })
    }

    /// xAI may omit `refresh_token` on refresh when it is not rotated.
    pub fn credential_from_token_response(
        body: &Value,
        previous_refresh: Option<&str>,
        now_ms: f64,
    ) -> Result<Credential, String> {
        let access = required(body, "access_token")?.to_owned();
        let refresh = match (body.get("refresh_token"), previous_refresh) {
            (None, Some(previous)) => previous.to_owned(),
            _ => required(body, "refresh_token")?.to_owned(),
        };
        let lifetime = match body.get("expires_in") {
            None => 3600.0,
            Some(v) => v
                .as_f64()
                .filter(|v| *v > 0.0)
                .ok_or("Invalid xAI OAuth response field: expires_in")?,
        };
        Ok(Credential::Oauth {
            access,
            refresh,
            expires: now_ms + lifetime * 1000.0 - EXPIRY_MARGIN_MS,
            extra: Default::default(),
        })
    }

    async fn post_form(url: &str, fields: &[(&str, &str)]) -> Result<(bool, u16, Value), String> {
        let response = reqwest::Client::new()
            .post(url)
            .timeout(Duration::from_secs(30))
            .header("accept", "application/json")
            .form(fields)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = response.status();
        let body = response
            .json::<Value>()
            .await
            .map_err(|_| format!("xAI OAuth returned invalid JSON (HTTP {})", status.as_u16()))?;
        Ok((status.is_success(), status.as_u16(), body))
    }

    fn failure(action: &str, status: u16, body: &Value) -> String {
        let text = |k: &str| body.get(k).and_then(Value::as_str).map(str::to_owned);
        let detail = [text("error"), text("error_description")]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(": ");
        let suffix = if detail.is_empty() {
            String::new()
        } else {
            format!(": {detail}")
        };
        format!("xAI OAuth {action} failed (HTTP {status}){suffix}")
    }

    pub async fn request_device_code(url: &str) -> Result<DeviceCode, String> {
        let (ok, status, body) = post_form(
            url,
            &[
                ("client_id", CLIENT_ID),
                ("scope", SCOPE),
                ("referrer", "pi"),
            ],
        )
        .await?;
        if !ok {
            return Err(failure("device authorization", status, &body));
        }
        parse_device_code(&body)
    }

    /// Map one token-endpoint answer to a poll result (Pi's `poll` callback).
    pub fn poll_result(ok: bool, status: u16, body: &Value) -> DevicePoll<Credential> {
        if ok {
            return match credential_from_token_response(body, None, now_ms()) {
                Ok(credential) => DevicePoll::Complete(credential),
                Err(message) => DevicePoll::Failed(message),
            };
        }
        match body.get("error").and_then(Value::as_str) {
            Some("authorization_pending") => DevicePoll::Pending,
            Some("slow_down") => DevicePoll::SlowDown(body.get("interval").and_then(Value::as_u64)),
            Some("access_denied" | "authorization_denied") => {
                DevicePoll::Failed("xAI device authorization was denied".into())
            }
            Some("expired_token") => DevicePoll::Failed("xAI device code expired".into()),
            _ => DevicePoll::Failed(failure("device token polling", status, body)),
        }
    }

    pub async fn poll_for_tokens(
        device: &DeviceCode,
        token_url: &str,
    ) -> Result<Credential, String> {
        let timing = DevicePollTiming::pi(
            device.interval_seconds,
            Some(device.expires_in_seconds),
            true,
        );
        poll_device_code(timing, || async {
            match post_form(
                token_url,
                &[
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ("client_id", CLIENT_ID),
                    ("device_code", &device.device_code),
                ],
            )
            .await
            {
                Ok((ok, status, body)) => poll_result(ok, status, &body),
                Err(message) => DevicePoll::Failed(message),
            }
        })
        .await
    }

    pub async fn refresh_token(refresh: &str) -> Result<Credential, String> {
        let (ok, status, body) = post_form(
            TOKEN_URL,
            &[
                ("grant_type", "refresh_token"),
                ("client_id", CLIENT_ID),
                ("refresh_token", refresh),
            ],
        )
        .await?;
        if !ok {
            return Err(failure("token refresh", status, &body));
        }
        credential_from_token_response(&body, Some(refresh), now_ms())
    }
}

/// Kimi Code (subscription) device-code login, as in Pi's `kimi-coding.ts`.
pub mod kimi {
    use super::*;

    const CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
    pub const DEFAULT_HOST: &str = "https://auth.kimi.com";

    #[derive(Debug, PartialEq, Eq)]
    pub struct DeviceAuthorization {
        pub device_code: String,
        pub user_code: String,
        pub verification_uri_complete: String,
        pub interval_seconds: u64,
        pub expires_in_seconds: u64,
    }

    fn http_url(value: &Value) -> Option<&str> {
        let text = value.as_str().filter(|v| !v.is_empty())?;
        matches!(url::Url::parse(text).ok()?.scheme(), "http" | "https").then_some(text)
    }

    pub fn parse_device_authorization(json: &Value) -> Result<DeviceAuthorization, String> {
        let text = |k: &str| json.get(k).and_then(Value::as_str).map(str::to_owned);
        let (Some(device_code), Some(user_code), Some(_), Some(complete)) = (
            text("device_code"),
            text("user_code"),
            http_url(&json["verification_uri"]),
            http_url(&json["verification_uri_complete"]),
        ) else {
            return Err(format!(
                "Invalid Kimi Code device authorization response: {json}"
            ));
        };
        let positive = |k: &str, default: u64| {
            json.get(k)
                .and_then(Value::as_f64)
                .filter(|v| *v > 0.0)
                .map_or(default, |v| v as u64)
        };
        Ok(DeviceAuthorization {
            device_code,
            user_code,
            verification_uri_complete: complete.to_owned(),
            interval_seconds: positive("interval", 5),
            expires_in_seconds: positive("expires_in", 15 * 60),
        })
    }

    /// Unlike the other providers, Kimi's expiry has no early-refresh margin.
    pub fn credential_from_token(
        json: &Value,
        operation: &str,
        now_ms: f64,
    ) -> Result<Credential, String> {
        let text = |k: &str| {
            json.get(k)
                .and_then(Value::as_str)
                .filter(|v| !v.is_empty())
        };
        let (Some(access), Some(refresh), Some(expires_in)) = (
            text("access_token"),
            text("refresh_token"),
            json.get("expires_in")
                .and_then(Value::as_f64)
                .filter(|v| *v > 0.0),
        ) else {
            return Err(format!(
                "Kimi Code token {operation} response missing fields: {json}"
            ));
        };
        Ok(Credential::Oauth {
            access: access.to_owned(),
            refresh: refresh.to_owned(),
            expires: now_ms + expires_in * 1000.0,
            extra: Default::default(),
        })
    }

    /// One poll of `/api/oauth/token`, as Pi maps it.
    pub fn poll_result(status: u16, json: &Value) -> DevicePoll<Credential> {
        if status >= 500 {
            return DevicePoll::Failed(format!(
                "Kimi Code device token request failed with status {status}: {json}"
            ));
        }
        if (200..300).contains(&status)
            && json.get("access_token").and_then(Value::as_str).is_some()
        {
            return match credential_from_token(json, "poll", now_ms()) {
                Ok(credential) => DevicePoll::Complete(credential),
                Err(message) => DevicePoll::Failed(message),
            };
        }
        let error = json.get("error").and_then(Value::as_str);
        match error {
            Some("authorization_pending") => DevicePoll::Pending,
            Some("slow_down") => DevicePoll::SlowDown(
                json.get("interval")
                    .and_then(Value::as_u64)
                    .filter(|v| *v > 0),
            ),
            Some("expired_token") => DevicePoll::Failed(
                "Kimi Code device authorization expired. Please restart login.".into(),
            ),
            Some("access_denied") => DevicePoll::Failed("Kimi Code login was denied.".into()),
            _ => {
                let description = json
                    .get("error_description")
                    .and_then(Value::as_str)
                    .map(|d| format!(": {d}"))
                    .unwrap_or_default();
                let tail = error
                    .map(|e| format!(": {e}{description}"))
                    .unwrap_or_default();
                DevicePoll::Failed(format!(
                    "Kimi Code device token request failed (status {status}){tail}"
                ))
            }
        }
    }

    async fn post_form(url: &str, fields: &[(&str, &str)]) -> Result<(u16, Value), String> {
        let response = reqwest::Client::new()
            .post(url)
            .timeout(Duration::from_secs(30))
            .header("accept", "application/json")
            .form(fields)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = response.status().as_u16();
        Ok((
            status,
            response.json::<Value>().await.unwrap_or(Value::Null),
        ))
    }

    pub async fn start(host: &str) -> Result<DeviceAuthorization, String> {
        let (status, json) = post_form(
            &format!("{host}/api/oauth/device_authorization"),
            &[("client_id", CLIENT_ID)],
        )
        .await?;
        if !(200..300).contains(&status) {
            return Err(format!(
                "Kimi Code device authorization failed with status {status}: {json}"
            ));
        }
        parse_device_authorization(&json)
    }

    pub async fn poll_for_token(
        host: &str,
        device: &DeviceAuthorization,
    ) -> Result<Credential, String> {
        let timing = DevicePollTiming::pi(
            Some(device.interval_seconds),
            Some(device.expires_in_seconds),
            true,
        );
        poll_device_code(timing, || async {
            match post_form(
                &format!("{host}/api/oauth/token"),
                &[
                    ("client_id", CLIENT_ID),
                    ("device_code", &device.device_code),
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ],
            )
            .await
            {
                Ok((status, json)) => poll_result(status, &json),
                Err(message) => DevicePoll::Failed(message),
            }
        })
        .await
    }

    /// Refresh with Pi's retry policy: 429/5xx and network errors retry (1s, 2s, 4s); 401/403 or
    /// `invalid_grant` mean the stored credential is dead.
    pub async fn refresh_token(host: &str, refresh: &str) -> Result<Credential, String> {
        let mut last = None;
        for attempt in 0..=3u32 {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_secs(1 << (attempt - 1))).await;
            }
            let result = post_form(
                &format!("{host}/api/oauth/token"),
                &[
                    ("client_id", CLIENT_ID),
                    ("grant_type", "refresh_token"),
                    ("refresh_token", refresh),
                ],
            )
            .await;
            let (status, json) = match result {
                Ok(answer) => answer,
                Err(error) => {
                    last = Some(error);
                    continue;
                }
            };
            if (200..300).contains(&status) {
                return credential_from_token(&json, "refresh", now_ms());
            }
            if matches!(status, 401 | 403)
                || json.get("error").and_then(Value::as_str) == Some("invalid_grant")
            {
                let description = json
                    .get("error_description")
                    .and_then(Value::as_str)
                    .map(|d| format!(": {d}"))
                    .unwrap_or_default();
                return Err(format!(
                    "Kimi Code token refresh unauthorized (status {status}){description}"
                ));
            }
            if (status == 429 || status >= 500) && attempt < 3 {
                last = Some(format!(
                    "Kimi Code token refresh failed with status {status}"
                ));
                continue;
            }
            return Err(format!(
                "Kimi Code token refresh failed with status {status}: {json}"
            ));
        }
        Err(last.unwrap_or_else(|| "Kimi Code token refresh failed".into()))
    }
}

/// OpenAI "Sign in with ChatGPT" (Pi's `openai-chatgpt.ts`): a public-client PKCE flow whose
/// client id is issued per login through dynamic registration and returned in the redirect.
pub mod openai_chatgpt {
    use super::*;

    pub const AUTHORIZE_URL: &str = "https://auth.openai.com/api/accounts/authorize";
    pub const TOKEN_URL: &str = "https://auth.openai.com/api/accounts/oauth/token";
    pub const CALLBACK_PORT: u16 = 1455;
    pub const CALLBACK_PATH: &str = "/auth/callback";
    pub const REDIRECT_URI: &str = "http://127.0.0.1:1455/auth/callback";
    const RESOURCE: &str = "https://api.openai.com/v1";
    const DIRECT_TOKEN_SCOPE: &str = "chatgpt.tokens.use.direct";
    const SCOPE: &str =
        "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
    /// Tokens are refreshed this long before they expire.
    const EXPIRY_MARGIN_MS: f64 = 3.0 * 60.0 * 1000.0;

    /// `urn:uuid:<uuid>` identifying this installation.
    pub fn agent_host_id(device_id: &str) -> Result<String, String> {
        let valid = device_id.len() == 36
            && device_id.char_indices().all(|(i, c)| match i {
                8 | 13 | 18 | 23 => c == '-',
                _ => c.is_ascii_hexdigit(),
            });
        valid
            .then(|| format!("urn:uuid:{}", device_id.to_lowercase()))
            .ok_or_else(|| {
                "Sign in with ChatGPT requires a device ID (UUID) for this installation".to_owned()
            })
    }

    /// A started login: open `url`, then finish with the redirect's code and issued client id.
    pub struct Login {
        pub url: String,
        pub state: String,
        pkce: Pkce,
    }

    impl Login {
        pub fn start(device_id: &str, agent_name: &str) -> Result<Self, String> {
            let host = agent_host_id(device_id)?;
            let pkce = Pkce::generate();
            let state = Pkce::generate().verifier;
            let nonce = Pkce::generate().verifier;
            let url = authorize_url(
                AUTHORIZE_URL,
                &[
                    ("client_id", "dynamic_agent_client"),
                    ("agent_name_hint", agent_name),
                    ("ext_agent_host_id", &host),
                    ("response_type", "code"),
                    ("redirect_uri", REDIRECT_URI),
                    ("resource", RESOURCE),
                    ("scope", SCOPE),
                    ("state", &state),
                    ("code_challenge", &pkce.challenge),
                    ("code_challenge_method", "S256"),
                    ("nonce", &nonce),
                ],
            )?;
            Ok(Self { url, state, pkce })
        }

        /// Exchange the redirect's parameters (`code`, `client_id`) for a credential.
        pub async fn finish(
            &self,
            params: &std::collections::HashMap<String, String>,
            token_url: &str,
        ) -> Result<Credential, String> {
            let code = params
                .get("code")
                .filter(|c| !c.is_empty())
                .ok_or("Missing authorization code")?;
            let client_id = params
                .get("client_id")
                .map(|c| c.trim())
                .filter(|c| !c.is_empty())
                .ok_or("OpenAI OAuth registration callback did not contain an issued client ID")?;
            let body = post_form(
                token_url,
                &[
                    ("grant_type", "authorization_code"),
                    ("client_id", client_id),
                    ("code", code),
                    ("code_verifier", &self.pkce.verifier),
                    ("redirect_uri", REDIRECT_URI),
                    ("resource", RESOURCE),
                ],
            )
            .await?;
            if body
                .get("id_token")
                .and_then(Value::as_str)
                .is_none_or(|t| t.trim().is_empty())
            {
                return Err("OpenAI OAuth token response did not contain an ID token".into());
            }
            credential_from_token_response(&body, client_id, now_ms())
        }
    }

    /// The credential keeps the issued `clientId` and granted `scopes` as extra fields.
    pub fn credential_from_token_response(
        body: &Value,
        client_id: &str,
        now_ms: f64,
    ) -> Result<Credential, String> {
        let text = |key: &str| {
            body.get(key)
                .and_then(Value::as_str)
                .filter(|v| !v.trim().is_empty())
                .ok_or_else(|| format!("OpenAI OAuth token response has invalid {key}"))
        };
        let (access, refresh, scope) = (
            text("access_token")?,
            text("refresh_token")?,
            text("scope")?,
        );
        let expires_in = body
            .get("expires_in")
            .and_then(Value::as_f64)
            .filter(|v| *v > 0.0)
            .ok_or("OpenAI OAuth token response has invalid expires_in")?;
        let scopes = scope
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if !scopes.iter().any(|s| s == DIRECT_TOKEN_SCOPE) {
            return Err(format!(
                "OpenAI OAuth grant did not include {DIRECT_TOKEN_SCOPE}"
            ));
        }
        let mut extra = std::collections::BTreeMap::new();
        extra.insert("clientId".to_owned(), json!(client_id));
        extra.insert("scopes".to_owned(), json!(scopes));
        Ok(Credential::Oauth {
            access: access.to_owned(),
            refresh: refresh.to_owned(),
            expires: now_ms + expires_in * 1000.0 - EXPIRY_MARGIN_MS,
            extra,
        })
    }

    async fn post_form(url: &str, fields: &[(&str, &str)]) -> Result<Value, String> {
        let response = reqwest::Client::new()
            .post(url)
            .timeout(Duration::from_secs(30))
            .header("accept", "application/json")
            .form(fields)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = response.status();
        let text = response.text().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!(
                "OpenAI OAuth token request failed ({}): {text}",
                status.as_u16()
            ));
        }
        serde_json::from_str::<Value>(&text)
            .ok()
            .filter(Value::is_object)
            .ok_or_else(|| "OpenAI OAuth token response must be an object".to_owned())
    }

    /// Refresh with the `clientId` stored in the credential.
    pub async fn refresh_token(
        refresh: &str,
        client_id: &str,
        token_url: &str,
    ) -> Result<Credential, String> {
        let body = post_form(
            token_url,
            &[
                ("grant_type", "refresh_token"),
                ("client_id", client_id),
                ("refresh_token", refresh),
                ("resource", RESOURCE),
            ],
        )
        .await?;
        credential_from_token_response(&body, client_id, now_ms())
    }
}

/// Meta Model API login (Pi's `meta.ts`): a device-code login yields an identity token, which is
/// exchanged for a roughly one-day API key. The identity token is stored as `refresh` and the key
/// as `access`, so refreshing means minting a new key; the identity token itself cannot be renewed.
pub mod meta {
    use super::*;

    const CLIENT_ID: &str = "1031625952748946";
    pub const DEVICE_AUTHORIZATION_URL: &str = "https://auth.meta.com/oidc/device/authorization/";
    pub const DEVICE_TOKEN_URL: &str = "https://auth.meta.com/oidc/device/token/";
    pub const API_KEY_MINT_URL: &str = "https://api.meta.ai/muse-code/key";
    const API_KEY_LIFETIME_MS: f64 = 24.0 * 60.0 * 60.0 * 1000.0;

    #[derive(Debug, PartialEq, Eq)]
    pub struct DeviceAuthorization {
        pub device_code: String,
        pub user_code: String,
        pub verification_uri: String,
        pub interval_seconds: Option<u64>,
        pub expires_in_seconds: Option<u64>,
    }

    fn trusted_http_url(value: &Value) -> Option<&str> {
        let text = value.as_str().filter(|v| !v.is_empty())?;
        matches!(url::Url::parse(text).ok()?.scheme(), "http" | "https").then_some(text)
    }

    /// `: <detail>` from the first of Pi's error fields that holds text, else empty.
    fn error_detail(json: &Value) -> String {
        ["error_description", "detail", "message", "error"]
            .iter()
            .find_map(|key| {
                json.get(key)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
            })
            .map(|detail| format!(": {detail}"))
            .unwrap_or_default()
    }

    fn positive(json: &Value, key: &str) -> Option<u64> {
        json.get(key)
            .and_then(Value::as_f64)
            .filter(|v| *v > 0.0)
            .map(|v| v as u64)
    }

    pub fn parse_device_authorization(json: &Value) -> Result<DeviceAuthorization, String> {
        let text = |k: &str| {
            json.get(k)
                .and_then(Value::as_str)
                .filter(|v| !v.is_empty())
                .map(str::to_owned)
        };
        let uri = trusted_http_url(&json["verification_uri_complete"])
            .or_else(|| trusted_http_url(&json["verification_uri"]));
        let (Some(device_code), Some(user_code), Some(uri)) =
            (text("device_code"), text("user_code"), uri)
        else {
            return Err(format!(
                "Invalid Meta device authorization response: {json}"
            ));
        };
        Ok(DeviceAuthorization {
            device_code,
            user_code,
            verification_uri: uri.to_owned(),
            interval_seconds: positive(json, "interval"),
            expires_in_seconds: positive(json, "expires_in"),
        })
    }

    /// One poll of the device-token endpoint; the identity token is the `access_token`.
    pub fn poll_result(status: u16, json: &Value) -> DevicePoll<String> {
        if (200..300).contains(&status)
            && let Some(token) = json
                .get("access_token")
                .and_then(Value::as_str)
                .filter(|t| !t.is_empty())
        {
            return DevicePoll::Complete(token.to_owned());
        }
        match json.get("error").and_then(Value::as_str) {
            Some("authorization_pending") => DevicePoll::Pending,
            Some("slow_down") => DevicePoll::SlowDown(positive(json, "interval")),
            Some("access_denied") => DevicePoll::Failed("Meta login was denied.".into()),
            Some("expired_token") => DevicePoll::Failed(
                "Meta device authorization expired. Please restart login.".into(),
            ),
            _ => DevicePoll::Failed(format!(
                "Meta device token request failed with status {status}{}",
                error_detail(json)
            )),
        }
    }

    /// Turn the mint endpoint's answer into a credential holding the identity token.
    pub fn credential_from_mint(
        identity_token: &str,
        status: u16,
        json: &Value,
        now_ms: f64,
    ) -> Result<Credential, String> {
        if matches!(status, 401 | 403) {
            return Err(format!(
                "Meta session expired (status {status}). Run `mu auth login --provider meta` to sign in again.{}",
                error_detail(json)
            ));
        }
        if !(200..300).contains(&status) {
            return Err(format!(
                "Meta API key mint failed with status {status}{}",
                error_detail(json)
            ));
        }
        let Some(key) = json
            .get("api_key")
            .and_then(Value::as_str)
            .filter(|k| !k.is_empty())
        else {
            let action = trusted_http_url(&json["action_url"])
                .map(|url| format!(" Complete setup at {url}"))
                .unwrap_or_default();
            return Err(format!("Meta did not issue an API key.{action}"));
        };
        Ok(Credential::Oauth {
            access: key.to_owned(),
            refresh: identity_token.to_owned(),
            expires: now_ms + API_KEY_LIFETIME_MS,
            extra: Default::default(),
        })
    }

    async fn request_json(request: reqwest::RequestBuilder) -> Result<(u16, Value), String> {
        let response = request
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = response.status().as_u16();
        Ok((
            status,
            response.json::<Value>().await.unwrap_or(Value::Null),
        ))
    }

    pub async fn start(url: &str) -> Result<DeviceAuthorization, String> {
        let (status, json) = request_json(
            reqwest::Client::new()
                .post(url)
                .header("accept", "application/json")
                .form(&[("client_id", CLIENT_ID)]),
        )
        .await?;
        if !(200..300).contains(&status) {
            return Err(format!(
                "Meta device authorization failed with status {status}{}",
                error_detail(&json)
            ));
        }
        parse_device_authorization(&json)
    }

    pub async fn poll_for_identity_token(
        url: &str,
        device: &DeviceAuthorization,
    ) -> Result<String, String> {
        let timing = DevicePollTiming::pi(device.interval_seconds, device.expires_in_seconds, true);
        poll_device_code(timing, || async {
            let request = reqwest::Client::new()
                .post(url)
                .header("accept", "application/json")
                .form(&[
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ("device_code", device.device_code.as_str()),
                    ("client_id", CLIENT_ID),
                ]);
            match request_json(request).await {
                Ok((status, json)) => poll_result(status, &json),
                Err(message) => DevicePoll::Failed(message),
            }
        })
        .await
    }

    /// Mint a fresh API key from an identity token (login, and every refresh).
    pub async fn mint_api_key(identity_token: &str, url: &str) -> Result<Credential, String> {
        let request = reqwest::Client::new()
            .post(url)
            .header("accept", "application/json")
            .header("x-api-version", "1.0.0")
            .bearer_auth(identity_token)
            .json(&json!({}));
        let (status, json) = request_json(request).await?;
        credential_from_mint(identity_token, status, &json, now_ms())
    }
}

/// GitHub Copilot login (Pi's `github-copilot.ts`): a GitHub device-code login yields a GitHub
/// token (stored as `refresh`), which is exchanged for a short-lived Copilot token (`access`).
/// The model catalog and per-model policy enabling Pi does afterwards are not ported.
pub mod github_copilot {
    use super::*;

    // Pi stores the client id base64-encoded; it is public.
    const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
    const USER_AGENT: &str = "GitHubCopilotChat/0.35.0";

    /// Pi's `normalizeDomain`: a bare host from `company.ghe.com` or `https://company.ghe.com/x`.
    pub fn normalize_domain(input: &str) -> Option<String> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }
        let with_scheme = if trimmed.contains("://") {
            trimmed.to_owned()
        } else {
            format!("https://{trimmed}")
        };
        url::Url::parse(&with_scheme)
            .ok()?
            .host_str()
            .map(str::to_owned)
    }

    pub struct Urls {
        pub device_code: String,
        pub access_token: String,
        pub copilot_token: String,
    }

    pub fn urls(domain: &str) -> Urls {
        Urls {
            device_code: format!("https://{domain}/login/device/code"),
            access_token: format!("https://{domain}/login/oauth/access_token"),
            copilot_token: format!("https://api.{domain}/copilot_internal/v2/token"),
        }
    }

    /// The API base URL: the `proxy-ep` host of the token with `proxy.` turned into `api.`, else
    /// the enterprise endpoint, else the individual one.
    pub fn base_url(token: &str, enterprise_domain: Option<&str>) -> String {
        if let Some(host) = token
            .split(';')
            .find_map(|part| part.strip_prefix("proxy-ep="))
            .filter(|host| !host.is_empty())
        {
            return format!(
                "https://{}",
                host.strip_prefix("proxy.")
                    .map_or_else(|| host.to_owned(), |rest| format!("api.{rest}"))
            );
        }
        match enterprise_domain {
            Some(domain) => format!("https://copilot-api.{domain}"),
            None => "https://api.individual.githubcopilot.com".to_owned(),
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    pub struct DeviceCode {
        pub device_code: String,
        pub user_code: String,
        pub verification_uri: String,
        pub interval_seconds: Option<u64>,
        pub expires_in_seconds: u64,
    }

    pub fn parse_device_code(json: &Value) -> Result<DeviceCode, String> {
        let text = |k: &str| json.get(k).and_then(Value::as_str).map(str::to_owned);
        let (Some(device_code), Some(user_code), Some(uri), Some(expires)) = (
            text("device_code"),
            text("user_code"),
            text("verification_uri"),
            json.get("expires_in").and_then(Value::as_f64),
        ) else {
            return Err("Invalid device code response fields".into());
        };
        let parsed = url::Url::parse(&uri)
            .map_err(|_| "Untrusted verification_uri in device code response".to_owned())?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err("Untrusted verification_uri in device code response".into());
        }
        Ok(DeviceCode {
            device_code,
            user_code,
            verification_uri: parsed.into(),
            interval_seconds: json.get("interval").and_then(Value::as_u64),
            expires_in_seconds: expires as u64,
        })
    }

    /// One poll of GitHub's access-token endpoint; the GitHub token is the result.
    pub fn poll_result(json: &Value) -> DevicePoll<String> {
        if let Some(token) = json.get("access_token").and_then(Value::as_str) {
            return DevicePoll::Complete(token.to_owned());
        }
        let Some(error) = json.get("error").and_then(Value::as_str) else {
            return DevicePoll::Failed("Invalid device token response".into());
        };
        match error {
            "authorization_pending" => DevicePoll::Pending,
            "slow_down" => DevicePoll::SlowDown(json.get("interval").and_then(Value::as_u64)),
            other => {
                let suffix = json
                    .get("error_description")
                    .and_then(Value::as_str)
                    .filter(|d| !d.is_empty())
                    .map(|d| format!(": {d}"))
                    .unwrap_or_default();
                DevicePoll::Failed(format!("Device flow failed: {other}{suffix}"))
            }
        }
    }

    /// The Copilot token response as a credential; the GitHub token is kept as `refresh`.
    pub fn credential_from_copilot_token(
        github_token: &str,
        json: &Value,
        enterprise_domain: Option<&str>,
    ) -> Result<Credential, String> {
        let (Some(token), Some(expires_at)) = (
            json.get("token").and_then(Value::as_str),
            json.get("expires_at").and_then(Value::as_f64),
        ) else {
            return Err("Invalid Copilot token response fields".into());
        };
        let mut extra = std::collections::BTreeMap::new();
        if let Some(domain) = enterprise_domain {
            extra.insert("enterpriseUrl".to_owned(), json!(domain));
        }
        Ok(Credential::Oauth {
            access: token.to_owned(),
            refresh: github_token.to_owned(),
            expires: expires_at * 1000.0 - EXPIRY_MARGIN_MS,
            extra,
        })
    }

    async fn json_request(request: reqwest::RequestBuilder) -> Result<Value, String> {
        let response = request
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = response.status();
        let text = response.text().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!(
                "{} {}: {text}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("")
            ));
        }
        serde_json::from_str(&text).map_err(|e| format!("invalid JSON: {e}"))
    }

    pub async fn start(domain: &str) -> Result<DeviceCode, String> {
        let json = json_request(
            reqwest::Client::new()
                .post(urls(domain).device_code)
                .header("accept", "application/json")
                .header("user-agent", USER_AGENT)
                .form(&[("client_id", CLIENT_ID), ("scope", "read:user")]),
        )
        .await?;
        parse_device_code(&json)
    }

    pub async fn poll_for_github_token(
        domain: &str,
        device: &DeviceCode,
    ) -> Result<String, String> {
        let timing = DevicePollTiming::pi(
            device.interval_seconds,
            Some(device.expires_in_seconds),
            true,
        );
        let url = urls(domain).access_token;
        poll_device_code(timing, || async {
            let request = reqwest::Client::new()
                .post(&url)
                .header("accept", "application/json")
                .header("user-agent", USER_AGENT)
                .form(&[
                    ("client_id", CLIENT_ID),
                    ("device_code", device.device_code.as_str()),
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ]);
            match json_request(request).await {
                Ok(json) => poll_result(&json),
                Err(message) => DevicePoll::Failed(message),
            }
        })
        .await
    }

    /// Exchange a GitHub token for a Copilot token (login, and every refresh).
    pub async fn copilot_token(
        github_token: &str,
        enterprise_domain: Option<&str>,
    ) -> Result<Credential, String> {
        let domain = enterprise_domain.unwrap_or("github.com");
        let json = json_request(
            reqwest::Client::new()
                .get(urls(domain).copilot_token)
                .header("accept", "application/json")
                .header("user-agent", USER_AGENT)
                .header("editor-version", "vscode/1.107.0")
                .header("editor-plugin-version", "copilot-chat/0.35.0")
                .header("copilot-integration-id", "vscode-chat")
                .bearer_auth(github_token),
        )
        .await?;
        credential_from_copilot_token(github_token, &json, enterprise_domain)
    }
}

/// OpenAI Codex (ChatGPT Plus/Pro) browser login (Pi's `openai-codex.ts`, browser method): PKCE
/// against `auth.openai.com` with a callback on port 1455. The device-code method is not ported.
pub mod openai_codex {
    use super::*;

    const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
    const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
    pub const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
    pub const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
    pub const CALLBACK_PORT: u16 = 1455;
    pub const CALLBACK_PATH: &str = "/auth/callback";
    const SCOPE: &str = "openid profile email offline_access";
    const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";

    pub struct Login {
        pub url: String,
        pub state: String,
        pkce: Pkce,
    }

    impl Login {
        pub fn start(originator: &str) -> Self {
            let pkce = Pkce::generate();
            // 16 random bytes as hex, like Pi's `createState`.
            let state = uuid::Uuid::new_v4().simple().to_string();
            let url = authorize_url(
                AUTHORIZE_URL,
                &[
                    ("response_type", "code"),
                    ("client_id", CLIENT_ID),
                    ("redirect_uri", REDIRECT_URI),
                    ("scope", SCOPE),
                    ("code_challenge", &pkce.challenge),
                    ("code_challenge_method", "S256"),
                    ("state", &state),
                    ("id_token_add_organizations", "true"),
                    ("codex_cli_simplified_flow", "true"),
                    ("originator", originator),
                ],
            )
            .expect("static authorize URL parses");
            Self { url, state, pkce }
        }

        /// Finish with a pasted redirect URL or code; a mismatched state is rejected.
        pub async fn finish_with_input(
            &self,
            input: &str,
            token_url: &str,
        ) -> Result<Credential, String> {
            let parsed = parse_authorization_input(input);
            if parsed
                .state
                .as_deref()
                .is_some_and(|state| state != self.state)
            {
                return Err("State mismatch".into());
            }
            self.finish_with_code(&parsed.code.ok_or("Missing authorization code")?, token_url)
                .await
        }

        pub async fn finish_with_code(
            &self,
            code: &str,
            token_url: &str,
        ) -> Result<Credential, String> {
            let body = post_form(
                token_url,
                &[
                    ("grant_type", "authorization_code"),
                    ("client_id", CLIENT_ID),
                    ("code", code),
                    ("code_verifier", &self.pkce.verifier),
                    ("redirect_uri", REDIRECT_URI),
                ],
                "exchange",
            )
            .await?;
            credential_from_token(&body, "exchange", now_ms())
        }
    }

    /// `chatgpt_account_id` from the access token's JWT payload.
    pub fn account_id(access_token: &str) -> Option<String> {
        let payload = access_token
            .split('.')
            .nth(1)
            .filter(|_| access_token.split('.').count() == 3)?;
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(payload.trim_end_matches('='))
            .ok()?;
        let claims: Value = serde_json::from_slice(&bytes).ok()?;
        claims[JWT_CLAIM_PATH]["chatgpt_account_id"]
            .as_str()
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
    }

    /// Expiry has no early margin here, and the credential carries the ChatGPT `accountId`.
    pub fn credential_from_token(
        body: &Value,
        operation: &str,
        now_ms: f64,
    ) -> Result<Credential, String> {
        let (Some(access), Some(refresh), Some(expires_in)) = (
            body.get("access_token")
                .and_then(Value::as_str)
                .filter(|v| !v.is_empty()),
            body.get("refresh_token")
                .and_then(Value::as_str)
                .filter(|v| !v.is_empty()),
            body.get("expires_in").and_then(Value::as_f64),
        ) else {
            return Err(format!(
                "OpenAI Codex token {operation} response missing fields: {body}"
            ));
        };
        let account = account_id(access).ok_or("Failed to extract accountId from token")?;
        let mut extra = std::collections::BTreeMap::new();
        extra.insert("accountId".to_owned(), json!(account));
        Ok(Credential::Oauth {
            access: access.to_owned(),
            refresh: refresh.to_owned(),
            expires: now_ms + expires_in * 1000.0,
            extra,
        })
    }

    async fn post_form(
        url: &str,
        fields: &[(&str, &str)],
        operation: &str,
    ) -> Result<Value, String> {
        let response = reqwest::Client::new()
            .post(url)
            .timeout(Duration::from_secs(30))
            .form(fields)
            .send()
            .await
            .map_err(|e| format!("OpenAI Codex token {operation} error: {e}"))?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(format!(
                "OpenAI Codex token {operation} failed ({}): {text}",
                status.as_u16()
            ));
        }
        serde_json::from_str(&text)
            .map_err(|_| format!("OpenAI Codex token {operation} response missing fields: {text}"))
    }

    pub async fn refresh_token(refresh: &str, token_url: &str) -> Result<Credential, String> {
        let body = post_form(
            token_url,
            &[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh),
                ("client_id", CLIENT_ID),
            ],
            "refresh",
        )
        .await?;
        credential_from_token(&body, "refresh", now_ms())
    }
}

/// Whether a stored OAuth credential needs refreshing.
pub fn is_expired(expires_ms: f64) -> bool {
    now_ms() >= expires_ms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_login_url_carries_pkce_and_state_is_checked() {
        let login = anthropic::Login::start("http://localhost:53692/callback");
        let url = url::Url::parse(&login.url).unwrap();
        let q = |k: &str| {
            url.query_pairs()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v.into_owned())
                .unwrap()
        };
        assert_eq!(url.host_str(), Some("claude.ai"));
        assert_eq!(q("client_id"), "9d1c250a-e61b-44d9-88ed-5944d1962f5e");
        assert_eq!(q("code_challenge_method"), "S256");
        assert_eq!(q("state"), login.state());
        assert_eq!(
            q("code_challenge"),
            Pkce::from_verifier(login.state().to_owned()).challenge
        );
        let mismatch = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(login.finish_with_input("code#wrong", "http://127.0.0.1:1/t"));
        assert_eq!(mismatch.err().as_deref(), Some("OAuth state mismatch"));
    }

    #[test]
    fn openrouter_authorize_url_and_key_response() {
        let pkce = Pkce::from_verifier("v".into());
        let url = openrouter::authorize("http://127.0.0.1:9/oauth/callback/x", &pkce);
        assert!(url.starts_with("https://openrouter.ai/auth?callback_url=http%3A%2F%2F127.0.0.1%3A9%2Foauth%2Fcallback%2Fx&code_challenge="));
        assert_eq!(
            openrouter::code_from_input("http://x/cb?code=abc").as_deref(),
            Some("abc")
        );
        assert_eq!(openrouter::code_from_input("abc").as_deref(), Some("abc"));
        assert_eq!(openrouter::code_from_input(" "), None);
        let credential =
            openrouter::credential_from_key_response(&json!({"key":"sk-or-1"})).unwrap();
        assert!(
            matches!(credential, Credential::Oauth { ref access, ref refresh, .. } if access == "sk-or-1" && refresh.is_empty())
        );
        assert!(openrouter::credential_from_key_response(&json!({"key":""})).is_err());
    }

    fn fast(expires_ms: Option<u64>) -> DevicePollTiming {
        DevicePollTiming {
            interval: Duration::from_millis(5),
            expires_in: expires_ms.map(Duration::from_millis),
            wait_before_first_poll: false,
            minimum_interval: Duration::from_millis(1),
            slow_down_increment: Duration::from_millis(5),
        }
    }

    #[tokio::test]
    async fn device_polling_waits_through_pending_and_slow_down_until_complete() {
        let mut script = vec![
            DevicePoll::Pending,
            DevicePoll::SlowDown(None),
            DevicePoll::SlowDown(Some(1)),
            DevicePoll::Complete("tok"),
        ]
        .into_iter();
        let polls = std::sync::atomic::AtomicUsize::new(0);
        let result = poll_device_code(fast(None), || {
            polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let next = script.next().unwrap();
            async move { next }
        })
        .await;
        assert_eq!(result, Ok("tok"));
        assert_eq!(polls.load(std::sync::atomic::Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn device_polling_reports_failures_and_both_timeout_messages() {
        let failed = poll_device_code::<(), _, _>(fast(None), || async {
            DevicePoll::Failed("denied".into())
        })
        .await;
        assert_eq!(failed, Err("denied".into()));
        let timed_out =
            poll_device_code::<(), _, _>(fast(Some(30)), || async { DevicePoll::Pending }).await;
        assert_eq!(timed_out, Err("Device flow timed out".into()));
        let slowed =
            poll_device_code::<(), _, _>(fast(Some(30)), || async { DevicePoll::SlowDown(None) })
                .await;
        assert!(
            slowed
                .unwrap_err()
                .starts_with("Device flow timed out after one or more slow_down responses")
        );
        assert_eq!(
            DevicePollTiming::pi(None, None, false).interval,
            Duration::from_secs(5)
        );
    }

    #[test]
    fn xai_device_code_parsing_validates_and_prefers_the_complete_uri() {
        let device = xai::parse_device_code(&json!({
            "device_code":"d","user_code":"U-1","verification_uri":"https://x.ai/device",
            "verification_uri_complete":"https://x.ai/device?c=U-1","interval":0,"expires_in":600
        }))
        .unwrap();
        assert_eq!(device.verification_uri, "https://x.ai/device?c=U-1");
        assert_eq!(
            (device.interval_seconds, device.expires_in_seconds),
            (None, 600)
        );
        let insecure = xai::parse_device_code(&json!({
            "device_code":"d","user_code":"u","verification_uri":"http://x.ai/device","expires_in":60}));
        assert_eq!(
            insecure.unwrap_err(),
            "Untrusted verification URI in xAI OAuth response"
        );
        assert!(xai::parse_device_code(&json!({"user_code":"u"})).is_err());
    }

    #[test]
    fn xai_poll_results_and_token_credentials_follow_pi() {
        let err = |e: &str| json!({"error": e});
        assert_eq!(
            xai::poll_result(false, 400, &err("authorization_pending")),
            DevicePoll::Pending
        );
        assert_eq!(
            xai::poll_result(false, 400, &json!({"error":"slow_down","interval":9})),
            DevicePoll::SlowDown(Some(9))
        );
        assert_eq!(
            xai::poll_result(false, 400, &err("access_denied")),
            DevicePoll::Failed("xAI device authorization was denied".into())
        );
        assert_eq!(
            xai::poll_result(false, 400, &err("expired_token")),
            DevicePoll::Failed("xAI device code expired".into())
        );
        assert_eq!(
            xai::poll_result(
                false,
                500,
                &json!({"error":"boom","error_description":"bad"})
            ),
            DevicePoll::Failed(
                "xAI OAuth device token polling failed (HTTP 500): boom: bad".into()
            )
        );
        let kept =
            xai::credential_from_token_response(&json!({"access_token":"a"}), Some("old"), 0.0)
                .unwrap();
        let Credential::Oauth {
            refresh, expires, ..
        } = kept
        else {
            panic!("oauth")
        };
        assert_eq!(refresh, "old");
        assert_eq!(expires, 3_600_000.0 - 300_000.0);
        assert!(
            xai::credential_from_token_response(&json!({"access_token":"a"}), None, 0.0).is_err()
        );
    }

    #[test]
    fn kimi_device_authorization_tokens_and_polls_follow_pi() {
        let device = kimi::parse_device_authorization(&json!({
            "device_code":"d","user_code":"u","verification_uri":"https://k.test/a",
            "verification_uri_complete":"https://k.test/a?c=u"}))
        .unwrap();
        assert_eq!(
            (device.interval_seconds, device.expires_in_seconds),
            (5, 900),
            "Pi defaults"
        );
        assert!(kimi::parse_device_authorization(&json!({"device_code":"d"})).is_err());
        assert!(kimi::parse_device_authorization(&json!({
            "device_code":"d","user_code":"u","verification_uri":"ftp://x","verification_uri_complete":"https://x"})).is_err());
        let Credential::Oauth { expires, .. } = kimi::credential_from_token(
            &json!({"access_token":"a","refresh_token":"r","expires_in":60}),
            "poll",
            1000.0,
        )
        .unwrap() else {
            panic!("oauth")
        };
        assert_eq!(expires, 61_000.0, "no early-refresh margin");
        assert!(
            kimi::credential_from_token(&json!({"access_token":"a"}), "refresh", 0.0)
                .unwrap_err()
                .contains("refresh response missing fields")
        );
        assert_eq!(
            kimi::poll_result(400, &json!({"error":"authorization_pending"})),
            DevicePoll::Pending
        );
        assert_eq!(
            kimi::poll_result(400, &json!({"error":"slow_down","interval":12})),
            DevicePoll::SlowDown(Some(12))
        );
        assert_eq!(
            kimi::poll_result(400, &json!({"error":"access_denied"})),
            DevicePoll::Failed("Kimi Code login was denied.".into())
        );
        assert_eq!(
            kimi::poll_result(400, &json!({"error":"weird","error_description":"why"})),
            DevicePoll::Failed(
                "Kimi Code device token request failed (status 400): weird: why".into()
            )
        );
        assert!(
            matches!(kimi::poll_result(503, &json!({})), DevicePoll::Failed(m) if m.contains("status 503"))
        );
    }

    #[test]
    fn openai_chatgpt_login_url_token_checks_and_credential_extras() {
        assert_eq!(
            openai_chatgpt::agent_host_id("123E4567-E89B-12D3-A456-426614174000").unwrap(),
            "urn:uuid:123e4567-e89b-12d3-a456-426614174000"
        );
        assert!(openai_chatgpt::agent_host_id("not-a-uuid").is_err());
        let login =
            openai_chatgpt::Login::start("123e4567-e89b-12d3-a456-426614174000", "Mu").unwrap();
        let url = url::Url::parse(&login.url).unwrap();
        let q = |k: &str| {
            url.query_pairs()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v.into_owned())
                .unwrap()
        };
        assert_eq!(q("client_id"), "dynamic_agent_client");
        assert_eq!(q("state"), login.state);
        assert_eq!(q("redirect_uri"), "http://127.0.0.1:1455/auth/callback");
        assert!(q("scope").contains("chatgpt.tokens.use.direct"));
        let body = json!({"access_token":"a","refresh_token":"r","expires_in":3600,"scope":"openid chatgpt.tokens.use.direct"});
        let Credential::Oauth { expires, extra, .. } =
            openai_chatgpt::credential_from_token_response(&body, "cid-1", 0.0).unwrap()
        else {
            panic!("oauth")
        };
        assert_eq!(expires, 3_600_000.0 - 180_000.0);
        assert_eq!(extra["clientId"], "cid-1");
        assert_eq!(
            extra["scopes"],
            json!(["openid", "chatgpt.tokens.use.direct"])
        );
        let missing_scope =
            json!({"access_token":"a","refresh_token":"r","expires_in":1,"scope":"openid"});
        assert!(
            openai_chatgpt::credential_from_token_response(&missing_scope, "c", 0.0)
                .unwrap_err()
                .contains("did not include")
        );
    }

    #[test]
    fn meta_device_flow_and_key_minting_follow_pi() {
        let device = meta::parse_device_authorization(&json!({
            "device_code":"d","user_code":"u","verification_uri":"https://m.test/a",
            "verification_uri_complete":"https://m.test/a?c=u","interval":3}))
        .unwrap();
        assert_eq!(device.verification_uri, "https://m.test/a?c=u");
        assert_eq!(
            (device.interval_seconds, device.expires_in_seconds),
            (Some(3), None)
        );
        assert!(meta::parse_device_authorization(&json!({"device_code":"d"})).is_err());
        assert_eq!(
            meta::poll_result(200, &json!({"access_token":"idt"})),
            DevicePoll::Complete("idt".into())
        );
        assert_eq!(
            meta::poll_result(400, &json!({"error":"authorization_pending"})),
            DevicePoll::Pending
        );
        assert_eq!(
            meta::poll_result(400, &json!({"error":"access_denied"})),
            DevicePoll::Failed("Meta login was denied.".into())
        );
        assert_eq!(
            meta::poll_result(500, &json!({"detail":" boom "})),
            DevicePoll::Failed("Meta device token request failed with status 500: boom".into())
        );
        let Credential::Oauth {
            access,
            refresh,
            expires,
            ..
        } = meta::credential_from_mint("idt", 200, &json!({"api_key":"k"}), 1000.0).unwrap()
        else {
            panic!("oauth")
        };
        assert_eq!(
            (access.as_str(), refresh.as_str(), expires),
            ("k", "idt", 1000.0 + 86_400_000.0)
        );
        assert!(
            meta::credential_from_mint("idt", 401, &json!({}), 0.0)
                .unwrap_err()
                .starts_with("Meta session expired (status 401)")
        );
        assert_eq!(
            meta::credential_from_mint(
                "idt",
                200,
                &json!({"action_url":"https://m.test/setup"}),
                0.0
            )
            .unwrap_err(),
            "Meta did not issue an API key. Complete setup at https://m.test/setup"
        );
        assert!(
            meta::credential_from_mint("idt", 500, &json!({"message":"down"}), 0.0)
                .unwrap_err()
                .ends_with("status 500: down")
        );
    }

    #[test]
    fn github_copilot_domains_urls_tokens_and_polls_follow_pi() {
        use github_copilot as gh;
        assert_eq!(
            gh::normalize_domain("company.ghe.com").as_deref(),
            Some("company.ghe.com")
        );
        assert_eq!(
            gh::normalize_domain(" https://company.ghe.com/x ").as_deref(),
            Some("company.ghe.com")
        );
        assert_eq!(gh::normalize_domain("  "), None);
        assert_eq!(
            gh::urls("github.com").copilot_token,
            "https://api.github.com/copilot_internal/v2/token"
        );
        assert_eq!(
            gh::base_url(
                "tid=1;exp=2;proxy-ep=proxy.business.githubcopilot.com;x=y",
                None
            ),
            "https://api.business.githubcopilot.com"
        );
        assert_eq!(
            gh::base_url("plain", Some("c.ghe.com")),
            "https://copilot-api.c.ghe.com"
        );
        assert_eq!(
            gh::base_url("plain", None),
            "https://api.individual.githubcopilot.com"
        );
        let device = gh::parse_device_code(&json!({
            "device_code":"d","user_code":"U","verification_uri":"https://github.com/login/device","expires_in":900})).unwrap();
        assert_eq!(
            (device.interval_seconds, device.expires_in_seconds),
            (None, 900)
        );
        assert!(gh::parse_device_code(&json!({"device_code":"d","user_code":"U","verification_uri":"file:///x","expires_in":1})).unwrap_err().contains("Untrusted"));
        assert_eq!(
            gh::poll_result(&json!({"access_token":"gho"})),
            DevicePoll::Complete("gho".into())
        );
        assert_eq!(
            gh::poll_result(&json!({"error":"authorization_pending"})),
            DevicePoll::Pending
        );
        assert_eq!(
            gh::poll_result(&json!({"error":"slow_down","interval":10})),
            DevicePoll::SlowDown(Some(10))
        );
        assert_eq!(
            gh::poll_result(&json!({"error":"access_denied","error_description":"nope"})),
            DevicePoll::Failed("Device flow failed: access_denied: nope".into())
        );
        assert_eq!(
            gh::poll_result(&json!({})),
            DevicePoll::Failed("Invalid device token response".into())
        );
        let Credential::Oauth {
            access,
            refresh,
            expires,
            extra,
        } = gh::credential_from_copilot_token(
            "gho",
            &json!({"token":"tok","expires_at":1000}),
            Some("c.ghe.com"),
        )
        .unwrap()
        else {
            panic!("oauth")
        };
        assert_eq!(
            (access.as_str(), refresh.as_str(), expires),
            ("tok", "gho", 700_000.0)
        );
        assert_eq!(extra["enterpriseUrl"], "c.ghe.com");
        assert!(gh::credential_from_copilot_token("g", &json!({"token":"t"}), None).is_err());
    }

    #[test]
    fn openai_codex_login_url_jwt_account_id_and_credentials() {
        let login = openai_codex::Login::start("mu");
        let url = url::Url::parse(&login.url).unwrap();
        let q = |k: &str| {
            url.query_pairs()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v.into_owned())
                .unwrap()
        };
        assert_eq!(q("client_id"), "app_EMoamEEZ73f0CkXaXp7hrann");
        assert_eq!(
            (q("state"), q("originator"), q("codex_cli_simplified_flow")),
            (login.state.clone(), "mu".into(), "true".into())
        );
        assert_eq!(login.state.len(), 32);
        let claims = json!({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}});
        let jwt = format!(
            "h.{}.s",
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(claims.to_string())
        );
        assert_eq!(openai_codex::account_id(&jwt).as_deref(), Some("acct-1"));
        assert_eq!(openai_codex::account_id("not-a-jwt"), None);
        assert_eq!(openai_codex::account_id("a.!!.c"), None);
        let Credential::Oauth { expires, extra, .. } = openai_codex::credential_from_token(
            &json!({"access_token": jwt, "refresh_token":"r","expires_in":60}),
            "exchange",
            1000.0,
        )
        .unwrap() else {
            panic!("oauth")
        };
        assert_eq!(
            (expires, extra["accountId"].as_str()),
            (61_000.0, Some("acct-1"))
        );
        assert!(
            openai_codex::credential_from_token(
                &json!({"access_token":"x.y.z","refresh_token":"r","expires_in":1}),
                "refresh",
                0.0
            )
            .unwrap_err()
            .contains("Failed to extract accountId")
        );
        assert!(
            openai_codex::credential_from_token(&json!({}), "refresh", 0.0)
                .unwrap_err()
                .starts_with("OpenAI Codex token refresh response missing fields")
        );
    }

    #[test]
    fn pkce_challenge_is_the_s256_of_the_verifier() {
        // RFC 7636 appendix B vector.
        let pkce = Pkce::from_verifier("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk".into());
        assert_eq!(
            pkce.challenge,
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
        let generated = Pkce::generate();
        assert_eq!(generated.verifier.len(), 43);
        assert_ne!(generated, Pkce::generate());
    }

    #[test]
    fn authorize_url_encodes_parameters() {
        let url = authorize_url(
            "https://x.test/authorize",
            &[("scope", "a b"), ("state", "s&t")],
        )
        .unwrap();
        assert_eq!(url, "https://x.test/authorize?scope=a+b&state=s%26t");
        assert!(authorize_url("nope", &[]).is_err());
    }

    #[test]
    fn pasted_input_is_parsed_in_every_shape_pi_accepts() {
        let both = |code: &str, state: &str| AuthorizationInput {
            code: Some(code.into()),
            state: Some(state.into()),
        };
        assert_eq!(
            parse_authorization_input("http://localhost:1/callback?code=c1&state=s1"),
            both("c1", "s1")
        );
        assert_eq!(parse_authorization_input("c2#s2"), both("c2", "s2"));
        assert_eq!(
            parse_authorization_input("code=c3&state=s3"),
            both("c3", "s3")
        );
        assert_eq!(
            parse_authorization_input(" plain "),
            AuthorizationInput {
                code: Some("plain".into()),
                state: None
            }
        );
        assert_eq!(
            parse_authorization_input("  "),
            AuthorizationInput::default()
        );
    }

    #[test]
    fn token_responses_become_credentials_expiring_five_minutes_early() {
        let credential = credential_from_token_response(
            &json!({"access_token":"a","refresh_token":"r","expires_in":3600}),
            1_000_000.0,
        )
        .unwrap();
        let Credential::Oauth {
            access,
            refresh,
            expires,
            ..
        } = credential
        else {
            panic!("oauth")
        };
        assert_eq!((access.as_str(), refresh.as_str()), ("a", "r"));
        assert_eq!(expires, 1_000_000.0 + 3_600_000.0 - 300_000.0);
        assert!(credential_from_token_response(&json!({"access_token":"a"}), 0.0).is_err());
    }

    async fn get(port: u16, target: &str) -> String {
        let mut socket = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .unwrap();
        socket
            .write_all(format!("GET {target} HTTP/1.1\r\nhost: x\r\n\r\n").as_bytes())
            .await
            .unwrap();
        let mut out = String::new();
        socket.read_to_string(&mut out).await.unwrap();
        out
    }

    #[tokio::test]
    async fn callback_server_rejects_bad_requests_then_returns_the_code() {
        let server = CallbackServer::bind("127.0.0.1", 0, "/callback", "localhost")
            .await
            .unwrap();
        let port = server
            .redirect_uri
            .rsplit(':')
            .next()
            .unwrap()
            .split('/')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            server.redirect_uri.starts_with("http://localhost:")
                && server.redirect_uri.ends_with("/callback")
        );
        let waiter = tokio::spawn(async move {
            server
                .wait_for_code("Acme", Some("st"), Duration::from_secs(5))
                .await
        });
        assert!(get(port, "/other").await.starts_with("HTTP/1.1 404"));
        assert!(
            get(port, "/callback?code=x&state=wrong")
                .await
                .starts_with("HTTP/1.1 400")
        );
        assert!(
            get(port, "/callback?state=st")
                .await
                .contains("Missing authorization code")
        );
        assert!(
            get(port, "/callback?code=good&state=st")
                .await
                .starts_with("HTTP/1.1 200")
        );
        assert_eq!(waiter.await.unwrap(), Ok("good".to_owned()));
    }

    #[tokio::test]
    async fn callback_server_reports_provider_errors_and_times_out() {
        let server = CallbackServer::bind("127.0.0.1", 0, "/cb", "127.0.0.1")
            .await
            .unwrap();
        let port: u16 = server
            .redirect_uri
            .rsplit(':')
            .next()
            .unwrap()
            .split('/')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let waiter = tokio::spawn(async move {
            server
                .wait_for_code("Acme", None, Duration::from_secs(5))
                .await
        });
        get(port, "/cb?error=access_denied&error_description=nope").await;
        assert_eq!(
            waiter.await.unwrap(),
            Err("Acme authorization failed: nope".into())
        );
        let idle = CallbackServer::bind("127.0.0.1", 0, "/cb", "localhost")
            .await
            .unwrap();
        assert_eq!(
            idle.wait_for_code("Acme", None, Duration::from_millis(50))
                .await,
            Err("Acme sign-in timed out".into())
        );
    }

    #[tokio::test]
    async fn code_exchange_and_refresh_post_json_to_the_token_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let mut bodies = Vec::new();
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buffer = vec![0u8; 8192];
                let read = socket.read(&mut buffer).await.unwrap();
                let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
                bodies.push(
                    request
                        .split("\r\n\r\n")
                        .nth(1)
                        .unwrap_or_default()
                        .to_owned(),
                );
                let body = r#"{"access_token":"A","refresh_token":"R","expires_in":60}"#;
                let reply = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(reply.as_bytes()).await.unwrap();
            }
            bodies
        });
        let url = format!("http://127.0.0.1:{port}/token");
        let exchanged = exchange_code(&url, "cid", "the-code", "st", "ver", "http://localhost/cb")
            .await
            .unwrap();
        assert!(matches!(exchanged, Credential::Oauth { ref access, .. } if access == "A"));
        refresh(&url, "cid", "R").await.unwrap();
        let bodies = server.await.unwrap();
        let first: Value = serde_json::from_str(&bodies[0]).unwrap();
        assert_eq!(first["grant_type"], "authorization_code");
        assert_eq!(first["code_verifier"], "ver");
        let second: Value = serde_json::from_str(&bodies[1]).unwrap();
        assert_eq!(
            second,
            json!({"grant_type":"refresh_token","client_id":"cid","refresh_token":"R"})
        );
        assert!(is_expired(0.0) && !is_expired(f64::MAX));
    }
}
