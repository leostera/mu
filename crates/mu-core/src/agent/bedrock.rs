//! Amazon Bedrock Converse adapter (`api: "bedrock-converse-stream"` in `models.json`).
//!
//! Uses the non-streaming `converse` endpoint with a Bedrock API key (bearer token). SigV4
//! request signing and the binary event stream (`converse-stream`) are not implemented yet.

use serde_json::{Value, json};

use super::{FunctionCall, Reply};

fn text_of(content: &Value) -> String {
    content
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Converse `messages`; consecutive same-role blocks merge (Converse requires alternation).
pub(super) fn messages(history: &[Value]) -> Vec<Value> {
    let mut out: Vec<(String, Vec<Value>)> = Vec::new();
    let mut push = |role: &str, block: Value| match out.last_mut() {
        Some((last, blocks)) if last == role => blocks.push(block),
        _ => out.push((role.to_owned(), vec![block])),
    };
    for item in history {
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => push(
                "assistant",
                json!({"toolUse":{
                    "toolUseId": item["call_id"],
                    "name": item["name"],
                    "input": item["arguments"].as_str()
                        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                        .unwrap_or_else(|| json!({})),
                }}),
            ),
            Some("function_call_output") => push(
                "user",
                json!({"toolResult":{
                    "toolUseId": item["call_id"],
                    "content": [{"text": item["output"].as_str().unwrap_or_default()}],
                }}),
            ),
            _ => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let text = text_of(&item["content"]);
                if !text.trim().is_empty() {
                    push(role, json!({"text": text}));
                }
            }
        }
    }
    out.into_iter()
        .map(|(role, content)| json!({"role": role, "content": content}))
        .collect()
}

pub(super) fn tool_config(definitions: &[Value]) -> Option<Value> {
    let tools = definitions
        .iter()
        .filter_map(|tool| {
            Some(json!({"toolSpec":{
                "name": tool.get("name")?,
                "description": tool.get("description").cloned().unwrap_or(json!("")),
                "inputSchema": {"json": tool.get("parameters").cloned().unwrap_or(json!({"type":"object"}))},
            }}))
        })
        .collect::<Vec<_>>();
    (!tools.is_empty()).then(|| json!({"tools": tools}))
}

pub(super) fn request_body(system: &str, history: &[Value], definitions: &[Value]) -> Value {
    let mut body = json!({
        "system": [{"text": system}],
        "messages": messages(history),
        "inferenceConfig": {"maxTokens": 16_384},
    });
    if let Some(config) = tool_config(definitions) {
        body["toolConfig"] = config;
    }
    body
}

/// Decode a Converse response body into a [`Reply`].
pub(super) fn parse_response(response: &Value) -> Result<Reply, String> {
    if let Some(message) = response.get("message").and_then(Value::as_str)
        && response.get("output").is_none()
    {
        return Err(message.to_owned());
    }
    let (mut text, mut thinking, mut tool_calls) = (String::new(), String::new(), Vec::new());
    for block in response["output"]["message"]["content"]
        .as_array()
        .into_iter()
        .flatten()
    {
        if let Some(t) = block.get("text").and_then(Value::as_str) {
            text.push_str(t);
        } else if let Some(t) = block["reasoningContent"]["reasoningText"]["text"].as_str() {
            thinking.push_str(t);
        } else if let Some(call) = block.get("toolUse") {
            tool_calls.push(FunctionCall {
                call_id: call["toolUseId"].as_str().unwrap_or_default().to_owned(),
                name: call["name"].as_str().unwrap_or_default().to_owned(),
                arguments: call.get("input").cloned().unwrap_or_else(|| json!({})),
            });
        }
    }
    let usage = &response["usage"];
    let count = |key: &str| usage[key].as_u64().unwrap_or(0);
    let (input, output) = (count("inputTokens"), count("outputTokens"));
    let (cache_read, cache_write) = (
        count("cacheReadInputTokens"),
        count("cacheWriteInputTokens"),
    );
    let truncated = response["stopReason"] == "max_tokens";
    Ok(Reply {
        text,
        thinking,
        thinking_signature: None,
        output_items: Vec::new(),
        tool_calls,
        input,
        output,
        cache_read,
        cache_write,
        reasoning: None,
        total_tokens: usage["totalTokens"]
            .as_u64()
            .unwrap_or(input + output + cache_read + cache_write),
        incomplete_reason: truncated.then(|| "max_output_tokens".to_owned()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_becomes_alternating_converse_messages_with_tool_blocks() {
        let history = vec![
            json!({"role":"user","content":[{"type":"input_text","text":"hi"}]}),
            json!({"type":"function_call","call_id":"c1","name":"read","arguments":"{\"path\":\"a\"}"}),
            json!({"type":"function_call_output","call_id":"c1","output":"file"}),
        ];
        let out = messages(&history);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], json!({"role":"user","content":[{"text":"hi"}]}));
        assert_eq!(
            out[1]["content"][0]["toolUse"]["input"],
            json!({"path":"a"})
        );
        assert_eq!(out[2]["role"], "user");
        assert_eq!(
            out[2]["content"][0]["toolResult"]["content"][0]["text"],
            "file"
        );
    }

    #[test]
    fn tool_config_wraps_schemas_and_is_absent_without_tools() {
        assert_eq!(tool_config(&[]), None);
        let config =
            tool_config(&[json!({"name":"ls","description":"d","parameters":{"type":"object"}})])
                .unwrap();
        assert_eq!(
            config["tools"][0]["toolSpec"]["inputSchema"]["json"],
            json!({"type":"object"})
        );
    }

    #[test]
    fn response_yields_text_thinking_tool_calls_and_usage() {
        let reply = parse_response(&json!({
            "output":{"message":{"role":"assistant","content":[
                {"reasoningContent":{"reasoningText":{"text":"hmm"}}},
                {"text":"ok"},
                {"toolUse":{"toolUseId":"t1","name":"bash","input":{"command":"ls"}}}
            ]}},
            "stopReason":"tool_use",
            "usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15,"cacheReadInputTokens":2}
        }))
        .unwrap();
        assert_eq!(
            (reply.text.as_str(), reply.thinking.as_str()),
            ("ok", "hmm")
        );
        assert_eq!(reply.tool_calls[0].arguments, json!({"command":"ls"}));
        assert_eq!(
            (
                reply.input,
                reply.output,
                reply.cache_read,
                reply.total_tokens
            ),
            (10, 5, 2, 15)
        );
        assert_eq!(reply.incomplete_reason, None);
    }

    #[test]
    fn max_tokens_is_truncation_and_error_bodies_surface() {
        let reply = parse_response(
            &json!({"output":{"message":{"content":[{"text":"x"}]}},"stopReason":"max_tokens"}),
        )
        .unwrap();
        assert_eq!(
            reply.incomplete_reason.as_deref(),
            Some("max_output_tokens")
        );
        assert_eq!(
            parse_response(&json!({"message":"Access denied"}))
                .err()
                .unwrap(),
            "Access denied"
        );
    }
}
