//! MCP OAuth discovery (Pi's `packages/mcp/src/oauth/discovery.ts`): reading a 401's
//! `WWW-Authenticate` challenge and discovering the authorization server. Client registration and
//! the sign-in flow are not ported yet.

use serde_json::Value;
use url::Url;

/// What a `Bearer`/`DPoP` challenge tells the client.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct OAuthChallenge {
    pub resource_metadata_url: Option<Url>,
    pub scope: Option<String>,
    pub error: Option<String>,
    pub error_description: Option<String>,
}

/// `name=value` or `name="value"` in a challenge; an empty value counts as absent.
fn field(header: &str, name: &str) -> Option<String> {
    let lower = header.to_ascii_lowercase();
    let needle = format!("{name}=");
    let mut from = 0;
    while let Some(found) = lower[from..].find(&needle) {
        let start = from + found;
        // The name must begin the header or follow a comma or whitespace.
        let boundary =
            start == 0 || lower[..start].ends_with(|c: char| c == ',' || c.is_whitespace());
        let rest = &header[start + needle.len()..];
        if boundary {
            let value = match rest.strip_prefix('"') {
                Some(quoted) => quoted.split('"').next().unwrap_or_default(),
                None => rest
                    .split(|c: char| c.is_whitespace() || c == ',')
                    .next()
                    .unwrap_or_default(),
            };
            return (!value.is_empty()).then(|| value.to_owned());
        }
        from = start + needle.len();
    }
    None
}

/// Pi's `parseWwwAuthenticate`: anything but a `Bearer` or `DPoP` challenge is empty.
pub fn parse_www_authenticate(header: Option<&str>) -> OAuthChallenge {
    let Some(header) = header else {
        return OAuthChallenge::default();
    };
    let scheme = header
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if scheme != "bearer" && scheme != "dpop" {
        return OAuthChallenge::default();
    }
    OAuthChallenge {
        resource_metadata_url: field(header, "resource_metadata")
            .and_then(|url| Url::parse(&url).ok()),
        scope: field(header, "scope"),
        error: field(header, "error"),
        error_description: field(header, "error_description"),
    }
}

fn path_suffix(path: &str) -> &str {
    path.strip_suffix('/').unwrap_or(path)
}

fn join(origin: &Url, path: &str) -> Url {
    Url::parse(&format!("{}{path}", origin.origin().ascii_serialization()))
        .expect("origin plus path parses")
}

/// Where to look for protected-resource metadata: an explicit `resource_metadata` URL alone, else
/// the path-specific well-known URL, then (for a non-root path) the root one.
pub fn protected_resource_metadata_urls(server: &Url, explicit: Option<&Url>) -> Vec<Url> {
    if let Some(explicit) = explicit {
        return vec![explicit.clone()];
    }
    let mut urls = vec![join(
        server,
        &format!(
            "/.well-known/oauth-protected-resource{}",
            path_suffix(server.path())
        ),
    )];
    if server.path() != "/" {
        urls.push(join(server, "/.well-known/oauth-protected-resource"));
    }
    urls
}

/// Pi's `buildAuthorizationServerDiscoveryUrls`.
pub fn authorization_server_discovery_urls(issuer: &Url) -> Vec<Url> {
    let path = path_suffix(issuer.path());
    let mut urls = vec![
        join(
            issuer,
            &format!("/.well-known/oauth-authorization-server{path}"),
        ),
        join(issuer, &format!("/.well-known/openid-configuration{path}")),
    ];
    if !path.is_empty() {
        urls.push(join(
            issuer,
            &format!("{path}/.well-known/openid-configuration"),
        ));
    }
    urls
}

/// The endpoints an MCP server's authorization server advertises.
#[derive(Debug, PartialEq, Eq)]
pub struct AuthorizationServer {
    pub issuer: String,
    pub authorization_endpoint: Option<String>,
    pub token_endpoint: Option<String>,
    pub registration_endpoint: Option<String>,
}

/// 4xx and 502 mean "not here", so discovery tries the next candidate URL.
fn is_miss(status: u16) -> bool {
    (400..500).contains(&status) || status == 502
}

async fn fetch_json(client: &reqwest::Client, url: &Url) -> Result<(u16, Value), String> {
    let response = client
        .get(url.clone())
        .header("accept", "application/json")
        .header("mcp-protocol-version", "2025-11-25")
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|error| error.to_string())?;
    let status = response.status().as_u16();
    Ok((
        status,
        response.json::<Value>().await.unwrap_or(Value::Null),
    ))
}

/// Pi's `discoverOAuthServerInfo`: protected-resource metadata names the authorization server
/// (else the server's origin is assumed), whose metadata must state the same issuer.
pub async fn discover_authorization_server(
    client: &reqwest::Client,
    server: &Url,
    challenge: &OAuthChallenge,
) -> Result<AuthorizationServer, String> {
    let mut named = None;
    for url in protected_resource_metadata_urls(server, challenge.resource_metadata_url.as_ref()) {
        match fetch_json(client, &url).await {
            Ok((200..=299, json)) => {
                named = json["authorization_servers"][0].as_str().map(str::to_owned);
                break;
            }
            Ok((status, _)) if is_miss(status) => continue,
            _ => break,
        }
    }
    let issuer_url = match named {
        Some(issuer) => Url::parse(&issuer)
            .map_err(|error| format!("invalid authorization server `{issuer}`: {error}"))?,
        None => join(server, "/"),
    };
    for url in authorization_server_discovery_urls(&issuer_url) {
        let (status, json) = fetch_json(client, &url).await?;
        if is_miss(status) {
            continue;
        }
        if !(200..300).contains(&status) {
            return Err(format!(
                "HTTP {status} loading authorization server metadata from {url}"
            ));
        }
        let issuer = json["issuer"]
            .as_str()
            .ok_or("authorization server metadata has no issuer")?
            .to_owned();
        let trim = |value: &str| value.trim_end_matches('/').to_owned();
        if trim(&issuer) != trim(issuer_url.as_str()) {
            return Err(format!(
                "Authorization server issuer mismatch: expected {}, got {issuer}",
                issuer_url.as_str()
            ));
        }
        let field = |name: &str| json[name].as_str().map(str::to_owned);
        return Ok(AuthorizationServer {
            issuer,
            authorization_endpoint: field("authorization_endpoint"),
            token_endpoint: field("token_endpoint"),
            registration_endpoint: field("registration_endpoint"),
        });
    }
    Err(format!(
        "no authorization server metadata found for {issuer_url}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(text: &str) -> Url {
        Url::parse(text).unwrap()
    }

    #[test]
    fn challenges_are_read_from_bearer_and_dpop_headers_only() {
        let challenge = parse_www_authenticate(Some(
            r#"Bearer realm="x", resource_metadata="https://mcp.test/.well-known/oauth-protected-resource", scope="read write", error=invalid_token, error_description="expired""#,
        ));
        assert_eq!(
            challenge.resource_metadata_url,
            Some(url("https://mcp.test/.well-known/oauth-protected-resource"))
        );
        assert_eq!(challenge.scope.as_deref(), Some("read write"));
        assert_eq!(challenge.error.as_deref(), Some("invalid_token"));
        assert_eq!(challenge.error_description.as_deref(), Some("expired"));
        assert_eq!(
            parse_www_authenticate(Some("Basic realm=x")),
            OAuthChallenge::default()
        );
        assert_eq!(parse_www_authenticate(None), OAuthChallenge::default());
        assert!(parse_www_authenticate(Some("dpop scope=a")).scope.is_some());
        // An empty value is absent, and a field must start at a word boundary.
        let odd = parse_www_authenticate(Some(
            r#"Bearer scope="", xscope=nope, resource_metadata=not-a-url"#,
        ));
        assert_eq!(odd, OAuthChallenge::default());
    }

    #[test]
    fn well_known_urls_follow_the_path_rules() {
        let server = url("https://mcp.test/api/v1/");
        let urls = protected_resource_metadata_urls(&server, None);
        assert_eq!(
            urls.iter().map(Url::as_str).collect::<Vec<_>>(),
            [
                "https://mcp.test/.well-known/oauth-protected-resource/api/v1",
                "https://mcp.test/.well-known/oauth-protected-resource"
            ]
        );
        assert_eq!(
            protected_resource_metadata_urls(&url("https://mcp.test/"), None).len(),
            1
        );
        let explicit = url("https://other.test/meta");
        assert_eq!(
            protected_resource_metadata_urls(&server, Some(&explicit)),
            vec![explicit]
        );
        let issuer = authorization_server_discovery_urls(&url("https://auth.test/tenant1"));
        assert_eq!(
            issuer.iter().map(Url::as_str).collect::<Vec<_>>(),
            [
                "https://auth.test/.well-known/oauth-authorization-server/tenant1",
                "https://auth.test/.well-known/openid-configuration/tenant1",
                "https://auth.test/tenant1/.well-known/openid-configuration"
            ]
        );
    }

    /// Serve JSON documents by path on a free port; unknown paths are 404. `documents` receives
    /// the server's base URL so documents can refer to themselves.
    async fn serve(documents: impl FnOnce(&str) -> Vec<(String, String)>) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let documents = documents(&base);
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let documents = documents.clone();
                tokio::spawn(async move {
                    let mut buffer = vec![0; 4096];
                    let read = socket.read(&mut buffer).await.unwrap_or(0);
                    let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
                    let path = request.split_whitespace().nth(1).unwrap_or("/").to_owned();
                    let reply = match documents.iter().find(|(p, _)| *p == path) {
                        Some((_, body)) => format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        ),
                        None => "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_owned(),
                    };
                    let _ = socket.write_all(reply.as_bytes()).await;
                });
            }
        });
        base
    }

    fn documents(base: &str, issuer_in_metadata: &str) -> Vec<(String, String)> {
        vec![
            (
                "/.well-known/oauth-protected-resource/mcp".into(),
                format!(r#"{{"authorization_servers":["{base}/tenant1"]}}"#),
            ),
            (
                "/.well-known/oauth-authorization-server/tenant1".into(),
                format!(
                    r#"{{"issuer":"{issuer_in_metadata}","authorization_endpoint":"{base}/a","token_endpoint":"{base}/t"}}"#
                ),
            ),
        ]
    }

    #[tokio::test]
    async fn discovery_follows_protected_resource_metadata_to_the_authorization_server() {
        let base = serve(|base| documents(base, &format!("{base}/tenant1"))).await;
        let found = discover_authorization_server(
            &reqwest::Client::new(),
            &url(&format!("{base}/mcp")),
            &OAuthChallenge::default(),
        )
        .await
        .unwrap();
        assert_eq!(found.issuer, format!("{base}/tenant1"));
        assert_eq!(found.token_endpoint, Some(format!("{base}/t")));
        assert_eq!(found.registration_endpoint, None);
    }

    #[tokio::test]
    async fn discovery_refuses_a_metadata_issuer_that_does_not_match() {
        let base = serve(|base| documents(base, "https://evil.test/tenant1")).await;
        let error = discover_authorization_server(
            &reqwest::Client::new(),
            &url(&format!("{base}/mcp")),
            &OAuthChallenge::default(),
        )
        .await
        .unwrap_err();
        assert!(
            error.starts_with("Authorization server issuer mismatch"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn discovery_reports_when_no_metadata_exists() {
        let base = serve(|_| Vec::new()).await;
        let error = discover_authorization_server(
            &reqwest::Client::new(),
            &url(&format!("{base}/mcp")),
            &OAuthChallenge::default(),
        )
        .await
        .unwrap_err();
        assert!(
            error.starts_with("no authorization server metadata found"),
            "{error}"
        );
    }
}
