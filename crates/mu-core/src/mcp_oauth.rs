//! MCP OAuth discovery (Pi's `packages/mcp/src/oauth/discovery.ts`): reading a 401's
//! `WWW-Authenticate` challenge. Metadata discovery, client registration and the sign-in flow are
//! not ported yet.

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
}
