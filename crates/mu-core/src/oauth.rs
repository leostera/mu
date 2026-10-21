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

/// Whether a stored OAuth credential needs refreshing.
pub fn is_expired(expires_ms: f64) -> bool {
    now_ms() >= expires_ms
}

#[cfg(test)]
mod tests {
    use super::*;

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
