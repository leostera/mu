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
        tokio::time::timeout(timeout, self.serve(provider, expected_state))
            .await
            .map_err(|_| format!("{provider} sign-in timed out"))?
    }

    async fn serve(&self, provider: &str, expected_state: Option<&str>) -> Result<String, String> {
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
            return Ok(code);
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
