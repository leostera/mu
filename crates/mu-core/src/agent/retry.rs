//! Provider retry policy: which failures are retryable and how long to wait.

use serde_json::Value;

use super::AgentError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RetryPolicy {
    pub(super) enabled: bool,
    pub(super) max_retries: u32,
    pub(super) base_delay_ms: u64,
    pub(super) max_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 2_000,
            max_delay_ms: 60_000,
        }
    }
}

pub(super) fn update_retry_policy(retry: &mut RetryPolicy, settings: &Value) {
    let Some(settings) = settings.get("retry").and_then(Value::as_object) else {
        return;
    };
    if let Some(enabled) = settings.get("enabled").and_then(Value::as_bool) {
        retry.enabled = enabled;
    }
    if let Some(max_retries) = settings.get("maxRetries").and_then(Value::as_u64) {
        retry.max_retries = max_retries.min(u32::MAX as u64) as u32;
    }
    if let Some(base_delay_ms) = settings.get("baseDelayMs").and_then(Value::as_u64) {
        retry.base_delay_ms = base_delay_ms;
    }
    if let Some(max_delay_ms) = settings
        .get("maxAgentDelayMs")
        .or_else(|| settings.get("maxDelayMs"))
        .and_then(Value::as_u64)
    {
        retry.max_delay_ms = max_delay_ms;
    }
}

/// `model request returned HTTP <status>` plus the start of the provider's error body, which
/// carries quota and billing reasons the retry policy needs to see.
pub(super) async fn http_failure(response: reqwest::Response) -> AgentError {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    AgentError::Provider(http_failure_message(status, &body))
}

pub(super) fn http_failure_message(status: reqwest::StatusCode, body: &str) -> String {
    let body = body.trim();
    if body.is_empty() {
        return format!("model request returned HTTP {status}");
    }
    let shown = body.chars().take(500).collect::<String>();
    let ellipsis = if body.chars().count() > 500 {
        "…"
    } else {
        ""
    };
    format!("model request returned HTTP {status}: {shown}{ellipsis}")
}

pub(super) fn retryable_provider_error(error: &AgentError) -> bool {
    let AgentError::Provider(message) = error else {
        return false;
    };
    let message = message.to_ascii_lowercase();
    if [
        "gousagelimiterror",
        "freeusagelimiterror",
        "monthly usage limit reached",
        "available balance",
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
        "subscription_sharing_usage_limit_exceeded",
        "usage limit",
    ]
    .iter()
    .any(|phrase| message.contains(phrase))
    {
        return false;
    }
    [
        "429",
        "500",
        "502",
        "503",
        "504",
        "520",
        "524",
        "rate limit",
        "rate-limit",
        "ratelimit",
        "too many requests",
        "overloaded",
        "server_busy",
        "servers are currently busy",
        "currently experiencing high demand",
        "model is at capacity",
        "service unavailable",
        "service is unavailable",
        "temporarily unavailable",
        "server error",
        "internal error",
        "network",
        "connection",
        "fetch failed",
        "eai_again",
        "enotfound",
        "socket hang up",
        "timeout",
        "timed out",
        "upstream",
        "stream failed",
        "stream was interrupted",
        "provider returned error",
        "provider returned an error",
        "request buffer limit while retrying upstream",
        "unable to forward upstream",
        "upstream request failed",
        "upstream_stream_read_error",
        "http/2 stream failed",
        "other side closed",
        "getaddrinfo",
        "socket connection was closed",
        "ended without",
        "stream ended before message_stop",
        "terminal response event",
        "http2 request did not get a response",
        "pending stream has been canceled",
        "retry delay",
        "you can retry your request",
        "try your request again",
        "please retry your request",
        "resourceexhausted",
        "subscription_sharing_usage_unavailable",
        "subscription_sharing_user_unavailable",
        "terminated",
    ]
    .iter()
    .any(|phrase| message.contains(phrase))
}

pub(super) fn retry_delay_ms(policy: RetryPolicy, attempt: u32) -> u64 {
    let exponent = attempt.saturating_sub(1).min(63);
    let multiplier = 1_u64.checked_shl(exponent).unwrap_or(u64::MAX);
    policy
        .base_delay_ms
        .saturating_mul(multiplier)
        .min(policy.max_delay_ms)
}
