//! AWS Signature Version 4 request signing (hand-rolled over `hmac` + `sha2`).

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

impl AwsCredentials {
    /// `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` from the environment.
    pub(super) fn from_env() -> Option<Self> {
        let get = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        Some(Self {
            access_key_id: get("AWS_ACCESS_KEY_ID")?,
            secret_access_key: get("AWS_SECRET_ACCESS_KEY")?,
            session_token: get("AWS_SESSION_TOKEN"),
        })
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn hmac(key: &[u8], data: &str) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

/// Percent-encode everything except unreserved characters (and `/` when `keep_slash`).
fn encode(value: &str, keep_slash: bool) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b'/' if keep_slash => "/".to_owned(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

pub(super) struct SignedHeaders {
    pub authorization: String,
    pub amz_date: String,
}

/// Sign a request. `headers` are the headers to sign besides `x-amz-date` and the session token
/// (which are added here); names must be lowercase. `amz_date` is `YYYYMMDDTHHMMSSZ`.
#[allow(clippy::too_many_arguments)]
pub(super) fn sign(
    method: &str,
    path: &str,
    query: &str,
    headers: &[(String, String)],
    body: &[u8],
    credentials: &AwsCredentials,
    region: &str,
    service: &str,
    amz_date: &str,
) -> SignedHeaders {
    let mut all = headers.to_vec();
    all.push(("x-amz-date".into(), amz_date.to_owned()));
    if let Some(token) = &credentials.session_token {
        all.push(("x-amz-security-token".into(), token.clone()));
    }
    all.sort();
    let canonical_headers = all
        .iter()
        .map(|(name, value)| format!("{name}:{}\n", value.trim()))
        .collect::<String>();
    let signed_headers = all
        .iter()
        .map(|(n, _)| n.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let canonical_path = if path.is_empty() {
        "/".to_owned()
    } else {
        encode(path, true)
    };
    let canonical_request = format!(
        "{method}\n{canonical_path}\n{query}\n{canonical_headers}\n{signed_headers}\n{}",
        sha256_hex(body)
    );
    let date = &amz_date[..8];
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );
    let key = hmac(
        format!("AWS4{}", credentials.secret_access_key).as_bytes(),
        date,
    );
    let key = hmac(&key, region);
    let key = hmac(&key, service);
    let key = hmac(&key, "aws4_request");
    let signature = hex(&hmac(&key, &string_to_sign));
    SignedHeaders {
        authorization: format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
            credentials.access_key_id
        ),
        amz_date: amz_date.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> AwsCredentials {
        AwsCredentials {
            access_key_id: "AKIDEXAMPLE".into(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(),
            session_token: None,
        }
    }

    /// AWS's published `get-vanilla` test vector.
    #[test]
    fn matches_the_aws_get_vanilla_vector() {
        let signed = sign(
            "GET",
            "/",
            "",
            &[("host".into(), "example.amazonaws.com".into())],
            b"",
            &example(),
            "us-east-1",
            "service",
            "20150830T123600Z",
        );
        assert_eq!(
            signed.authorization,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-date, \
             Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
    }

    #[test]
    fn session_tokens_are_signed_and_paths_are_encoded_per_segment() {
        let mut credentials = example();
        credentials.session_token = Some("tok".into());
        let signed = sign(
            "POST",
            "/model/a:b/converse",
            "",
            &[("host".into(), "h".into())],
            b"{}",
            &credentials,
            "us-east-1",
            "bedrock",
            "20150830T123600Z",
        );
        assert!(
            signed
                .authorization
                .contains("SignedHeaders=host;x-amz-date;x-amz-security-token")
        );
        assert_eq!(encode("/model/a:b/converse", true), "/model/a%3Ab/converse");
    }
}
