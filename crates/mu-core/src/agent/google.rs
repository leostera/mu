//! Google Generative AI (Gemini) adapter (`api: "google-generative-ai"`).
//!
//! Gemini has no tool-call ids, so ids are synthesized (`call_<n>`) and `functionResponse`
//! parts are matched to calls by name through the history.

use std::collections::HashMap;

use serde_json::{Value, json};

use super::{AgentEvent, FunctionCall, Reply};

fn text_of(content: &Value) -> String {
    content
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn contents(history: &[Value]) -> Vec<Value> {
    let names: HashMap<&str, &str> = history
        .iter()
        .filter(|item| item["type"] == "function_call")
        .filter_map(|item| Some((item["call_id"].as_str()?, item["name"].as_str()?)))
        .collect();
    let mut out: Vec<(&'static str, Vec<Value>)> = Vec::new();
    let mut push = |role: &'static str, part: Value| match out.last_mut() {
        Some((last, parts)) if *last == role => parts.push(part),
        _ => out.push((role, vec![part])),
    };
    for item in history {
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => push(
                "model",
                json!({"functionCall":{
                    "name": item["name"],
                    "args": item["arguments"].as_str()
                        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                        .unwrap_or_else(|| json!({})),
                }}),
            ),
            Some("function_call_output") => push(
                "user",
                json!({"functionResponse":{
                    "name": names.get(item["call_id"].as_str().unwrap_or_default()).copied().unwrap_or("tool"),
                    "response": {"output": item["output"]},
                }}),
            ),
            _ => {
                let role = if item["role"] == "assistant" {
                    "model"
                } else {
                    "user"
                };
                let text = text_of(&item["content"]);
                if !text.is_empty() {
                    push(role, json!({"text": text}));
                }
            }
        }
    }
    out.into_iter()
        .map(|(role, parts)| json!({"role": role, "parts": parts}))
        .collect()
}

/// Gemini rejects `additionalProperties` and `$schema`; strip them recursively.
fn clean_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(key, _)| !matches!(key.as_str(), "additionalProperties" | "$schema"))
                .map(|(key, value)| (key.clone(), clean_schema(value)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(clean_schema).collect()),
        other => other.clone(),
    }
}

pub(super) fn request_body(system: &str, history: &[Value], definitions: &[Value]) -> Value {
    let mut body = json!({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents(history),
    });
    let declarations = definitions
        .iter()
        .filter_map(|tool| {
            Some(json!({
                "name": tool.get("name")?,
                "description": tool.get("description").cloned().unwrap_or(json!("")),
                "parameters": clean_schema(tool.get("parameters").unwrap_or(&json!({"type":"object"}))),
            }))
        })
        .collect::<Vec<_>>();
    if !declarations.is_empty() {
        body["tools"] = json!([{"functionDeclarations": declarations}]);
    }
    body
}

#[derive(Default)]
pub(super) struct Decoder {
    text: String,
    thinking: String,
    calls: Vec<(String, Value)>,
    input: u64,
    output: u64,
    cache_read: u64,
    finish_reason: Option<String>,
}

impl Decoder {
    pub(super) fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }

    pub(super) fn feed(
        &mut self,
        chunk: &Value,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<(), String> {
        if let Some(error) = chunk.get("error") {
            return Err(error["message"]
                .as_str()
                .unwrap_or("unknown error")
                .to_owned());
        }
        if let Some(usage) = chunk.get("usageMetadata") {
            let cached = usage["cachedContentTokenCount"].as_u64().unwrap_or(0);
            self.cache_read = cached;
            self.input = usage["promptTokenCount"]
                .as_u64()
                .unwrap_or(0)
                .saturating_sub(cached);
            self.output = usage["candidatesTokenCount"].as_u64().unwrap_or(0)
                + usage["thoughtsTokenCount"].as_u64().unwrap_or(0);
        }
        let Some(candidate) = chunk["candidates"].get(0) else {
            return Ok(());
        };
        for part in candidate["content"]["parts"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if let Some(call) = part.get("functionCall") {
                self.calls.push((
                    call["name"].as_str().unwrap_or_default().to_owned(),
                    call["args"].clone(),
                ));
            } else if let Some(text) = part["text"].as_str().filter(|t| !t.is_empty()) {
                if part["thought"] == true {
                    self.thinking.push_str(text);
                    on_event(AgentEvent::ThinkingDelta(text.to_owned()));
                } else {
                    self.text.push_str(text);
                    on_event(AgentEvent::TextDelta(text.to_owned()));
                }
            }
        }
        if let Some(reason) = candidate["finishReason"].as_str() {
            self.finish_reason = Some(reason.to_owned());
        }
        Ok(())
    }

    pub(super) fn into_reply(self) -> Reply {
        let truncated = self.finish_reason.as_deref() == Some("MAX_TOKENS");
        let tool_calls = self
            .calls
            .into_iter()
            .enumerate()
            .map(|(index, (name, arguments))| FunctionCall {
                call_id: format!("call_{index}"),
                name,
                arguments: if arguments.is_null() {
                    json!({})
                } else {
                    arguments
                },
            })
            .collect();
        Reply {
            text: self.text,
            thinking: self.thinking,
            thinking_signature: None,
            output_items: Vec::new(),
            tool_calls,
            input: self.input,
            output: self.output,
            cache_read: self.cache_read,
            cache_write: 0,
            reasoning: None,
            total_tokens: self.input + self.output + self.cache_read,
            incomplete_reason: truncated.then(|| "max_output_tokens".to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contents_map_roles_and_match_function_responses_by_name() {
        let history = vec![
            json!({"role":"user","content":[{"type":"input_text","text":"hi"}]}),
            json!({"type":"function_call","call_id":"c","name":"read","arguments":"{\"p\":1}"}),
            json!({"type":"function_call_output","call_id":"c","output":"data"}),
            json!({"role":"assistant","content":[{"type":"output_text","text":"done"}]}),
        ];
        let out = contents(&history);
        assert_eq!(out.len(), 4);
        assert_eq!(out[1]["role"], "model");
        assert_eq!(out[1]["parts"][0]["functionCall"]["args"]["p"], 1);
        assert_eq!(out[2]["parts"][0]["functionResponse"]["name"], "read");
        assert_eq!(out[3]["role"], "model");
    }

    #[test]
    fn schemas_drop_unsupported_keys_and_tools_use_function_declarations() {
        let body = request_body(
            "sys",
            &[],
            &[
                json!({"name":"t","parameters":{"type":"object","additionalProperties":false,"properties":{"a":{"type":"object","additionalProperties":false}}}}),
            ],
        );
        let params = &body["tools"][0]["functionDeclarations"][0]["parameters"];
        assert!(params.get("additionalProperties").is_none());
        assert!(
            params["properties"]["a"]
                .get("additionalProperties")
                .is_none()
        );
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "sys");
    }

    #[test]
    fn decoder_separates_thoughts_collects_calls_and_usage() {
        let mut decoder = Decoder::default();
        let mut events = 0;
        for chunk in [
            json!({"candidates":[{"content":{"parts":[{"text":"hmm","thought":true},{"text":"hi"}]}}]}),
            json!({"candidates":[{"content":{"parts":[{"functionCall":{"name":"ls","args":{"path":"."}}}]},"finishReason":"STOP"}],
                   "usageMetadata":{"promptTokenCount":10,"cachedContentTokenCount":4,"candidatesTokenCount":3}}),
        ] {
            decoder.feed(&chunk, &mut |_| events += 1).unwrap();
        }
        assert!(decoder.finished());
        let reply = decoder.into_reply();
        assert_eq!(
            (reply.text.as_str(), reply.thinking.as_str()),
            ("hi", "hmm")
        );
        assert_eq!(reply.tool_calls[0].call_id, "call_0");
        assert_eq!((reply.input, reply.cache_read, reply.output), (6, 4, 3));
        assert_eq!(events, 2);
    }

    #[test]
    fn max_tokens_is_truncation_and_errors_surface() {
        let mut decoder = Decoder::default();
        decoder
            .feed(
                &json!({"candidates":[{"finishReason":"MAX_TOKENS"}]}),
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(
            decoder.into_reply().incomplete_reason.as_deref(),
            Some("max_output_tokens")
        );
        assert_eq!(
            Decoder::default()
                .feed(&json!({"error":{"message":"quota"}}), &mut |_| {})
                .unwrap_err(),
            "quota"
        );
    }
}
