//! OpenAI Chat Completions adapter (`api: "openai-completions"`), for the many compatible servers.

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

/// Chat messages from Responses-style history, led by the system prompt.
pub(super) fn messages(system: &str, history: &[Value]) -> Vec<Value> {
    let mut out = vec![json!({"role":"system","content":system})];
    for item in history {
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => {
                let call = json!({
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                });
                match out.last_mut() {
                    Some(last)
                        if last["role"] == "assistant" && last.get("tool_calls").is_some() =>
                    {
                        last["tool_calls"].as_array_mut().expect("array").push(call);
                    }
                    _ => out.push(json!({"role":"assistant","content":null,"tool_calls":[call]})),
                }
            }
            Some("function_call_output") => out.push(
                json!({"role":"tool","tool_call_id":item["call_id"],"content":super::output_text(item)}),
            ),
            _ => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let text = text_of(&item["content"]);
                if !text.is_empty() {
                    out.push(json!({"role":role,"content":text}));
                }
            }
        }
    }
    out
}

pub(super) fn tools(definitions: &[Value]) -> Vec<Value> {
    definitions
        .iter()
        .filter_map(|tool| {
            Some(json!({"type":"function","function":{
                "name": tool.get("name")?,
                "description": tool.get("description").cloned().unwrap_or(json!("")),
                "parameters": tool.get("parameters").cloned().unwrap_or(json!({"type":"object"})),
            }}))
        })
        .collect()
}

pub(super) fn request_body(
    model: &str,
    system: &str,
    history: &[Value],
    definitions: &[Value],
    reasoning_effort: Option<&str>,
) -> Value {
    let mut body = json!({
        "model": model,
        "stream": true,
        "stream_options": {"include_usage": true},
        "messages": messages(system, history),
    });
    let tools = tools(definitions);
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    if let Some(effort) = reasoning_effort {
        body["reasoning_effort"] = json!(effort);
    }
    body
}

#[derive(Default)]
pub(super) struct Decoder {
    text: String,
    thinking: String,
    calls: Vec<(String, String, String)>,
    input: u64,
    output: u64,
    cache_read: u64,
    finish_reason: Option<String>,
    done: bool,
}

impl Decoder {
    pub(super) fn finished(&self) -> bool {
        self.done || self.finish_reason.is_some()
    }

    pub(super) fn feed_done(&mut self) {
        self.done = true;
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
        if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
            let cached = usage["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0);
            self.cache_read = cached;
            self.input = usage["prompt_tokens"]
                .as_u64()
                .unwrap_or(0)
                .saturating_sub(cached);
            self.output = usage["completion_tokens"].as_u64().unwrap_or(0);
        }
        let Some(choice) = chunk["choices"].get(0) else {
            return Ok(());
        };
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            self.text.push_str(text);
            on_event(AgentEvent::TextDelta(text.to_owned()));
        }
        // Compatible servers name reasoning text differently.
        for key in ["reasoning_content", "reasoning"] {
            if let Some(text) = delta[key].as_str().filter(|t| !t.is_empty()) {
                self.thinking.push_str(text);
                on_event(AgentEvent::ThinkingDelta(text.to_owned()));
            }
        }
        for call in delta["tool_calls"].as_array().into_iter().flatten() {
            let index = call["index"].as_u64().unwrap_or(0) as usize;
            while self.calls.len() <= index {
                self.calls.push(Default::default());
            }
            let entry = &mut self.calls[index];
            if let Some(id) = call["id"].as_str() {
                entry.0 = id.to_owned();
            }
            if let Some(name) = call["function"]["name"].as_str() {
                entry.1.push_str(name);
            }
            if let Some(arguments) = call["function"]["arguments"].as_str() {
                entry.2.push_str(arguments);
            }
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finish_reason = Some(reason.to_owned());
        }
        Ok(())
    }

    pub(super) fn into_reply(self) -> Reply {
        let truncated = self.finish_reason.as_deref() == Some("length");
        let tool_calls = self
            .calls
            .into_iter()
            .map(|(call_id, name, raw)| FunctionCall {
                call_id,
                name,
                arguments: if raw.is_empty() {
                    json!({})
                } else {
                    serde_json::from_str(&raw).unwrap_or(Value::String(raw))
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
    fn history_becomes_chat_messages_with_grouped_tool_calls() {
        let history = vec![
            json!({"role":"user","content":[{"type":"input_text","text":"hi"}]}),
            json!({"type":"function_call","call_id":"a","name":"read","arguments":"{}"}),
            json!({"type":"function_call","call_id":"b","name":"ls","arguments":"{}"}),
            json!({"type":"function_call_output","call_id":"a","output":"x"}),
        ];
        let out = messages("sys", &history);
        assert_eq!(out[0]["role"], "system");
        assert_eq!(out[2]["tool_calls"].as_array().unwrap().len(), 2);
        assert_eq!(
            out[3],
            json!({"role":"tool","tool_call_id":"a","content":"x"})
        );
        assert_eq!(
            tools(&[json!({"name":"n","parameters":{"type":"object"}})])[0]["function"]["name"],
            "n"
        );
    }

    #[test]
    fn decoder_merges_streamed_tool_call_fragments_and_usage() {
        let mut decoder = Decoder::default();
        let mut events = 0;
        for chunk in [
            json!({"choices":[{"delta":{"reasoning_content":"think"}}]}),
            json!({"choices":[{"delta":{"content":"hello"}}]}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"bash","arguments":"{\"com"}}]}}]}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"mand\":\"ls\"}"}}]},"finish_reason":"tool_calls"}]}),
            json!({"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":2}}}),
        ] {
            decoder.feed(&chunk, &mut |_| events += 1).unwrap();
        }
        assert!(decoder.finished());
        let reply = decoder.into_reply();
        assert_eq!(
            (reply.text.as_str(), reply.thinking.as_str()),
            ("hello", "think")
        );
        assert_eq!(reply.tool_calls[0].arguments, json!({"command":"ls"}));
        assert_eq!(
            (
                reply.input,
                reply.cache_read,
                reply.output,
                reply.total_tokens
            ),
            (10, 2, 4, 16)
        );
        assert_eq!(events, 2);
    }

    #[test]
    fn length_finish_marks_truncation_and_errors_surface() {
        let mut decoder = Decoder::default();
        decoder
            .feed(
                &json!({"choices":[{"delta":{},"finish_reason":"length"}]}),
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(
            decoder.into_reply().incomplete_reason.as_deref(),
            Some("max_output_tokens")
        );
        assert_eq!(
            Decoder::default()
                .feed(&json!({"error":{"message":"rate limited"}}), &mut |_| {})
                .unwrap_err(),
            "rate limited"
        );
    }
}
