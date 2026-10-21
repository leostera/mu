//! Anthropic Messages adapter: request building from Responses-style history and SSE decoding.
//!
//! History items use the OpenAI Responses shape Mu keeps internally (`role` messages,
//! `function_call`, `function_call_output`); they are translated to Anthropic content blocks.

use serde_json::{Value, json};

use super::{AgentEvent, FunctionCall, Reply};

/// Anthropic OAuth (Claude subscription) tokens contain `sk-ant-oat`.
pub(super) fn is_oauth_token(key: &str) -> bool {
    key.contains("sk-ant-oat")
}

/// OAuth requests must identify as Claude Code (Pi's `anthropic-messages.ts`).
pub(super) const CLAUDE_CODE_IDENTITY: &str =
    "You are Claude Code, Anthropic's official CLI for Claude.";

const CLAUDE_CODE_TOOLS: &[&str] = &[
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
];

/// Pi's `toClaudeCodeName`: Claude Code's canonical casing when the name matches case-insensitively.
pub(super) fn to_claude_code_name(name: &str) -> String {
    CLAUDE_CODE_TOOLS
        .iter()
        .find(|tool| tool.eq_ignore_ascii_case(name))
        .map_or_else(|| name.to_owned(), |tool| (*tool).to_owned())
}

/// Pi's `fromClaudeCodeName`: the caller's own tool name for a case-insensitive match.
pub(super) fn from_claude_code_name(name: &str, tools: &[Value]) -> String {
    tools
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .find(|own| own.eq_ignore_ascii_case(name))
        .unwrap_or(name)
        .to_owned()
}

/// Rename tool definitions and past tool calls to Claude Code names (for OAuth requests).
pub(super) fn rename_for_claude_code(
    history: &[Value],
    definitions: &[Value],
) -> (Vec<Value>, Vec<Value>) {
    let rename = |items: &[Value]| {
        items
            .iter()
            .cloned()
            .map(|mut item| {
                if let Some(name) = item
                    .get("name")
                    .and_then(Value::as_str)
                    .map(to_claude_code_name)
                {
                    item["name"] = Value::String(name);
                }
                item
            })
            .collect::<Vec<_>>()
    };
    (rename(history), rename(definitions))
}

fn text_of(content: &Value) -> String {
    content
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Anthropic `messages` array; consecutive same-role blocks merge (the API requires alternation).
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
                json!({
                    "type":"tool_use",
                    "id": item["call_id"],
                    "name": item["name"],
                    "input": item["arguments"].as_str()
                        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                        .unwrap_or_else(|| json!({})),
                }),
            ),
            Some("function_call_output") => push(
                "user",
                json!({"type":"tool_result","tool_use_id":item["call_id"],"content":tool_result_content(&item["output"])}),
            ),
            _ => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let text = text_of(&item["content"]);
                if !text.is_empty() {
                    push(role, json!({"type":"text","text":text}));
                }
            }
        }
    }
    out.into_iter()
        .map(|(role, content)| json!({"role":role,"content":content}))
        .collect()
}

/// Anthropic tool-result content: text, or text and base64 image blocks.
fn tool_result_content(output: &Value) -> Value {
    let Some(parts) = output.as_array() else {
        return output.clone();
    };
    Value::Array(
        parts
            .iter()
            .filter_map(|part| match part["type"].as_str() {
                Some("input_text") => Some(json!({"type":"text","text":part["text"]})),
                Some("input_image") => {
                    let url = part["image_url"].as_str()?;
                    let (header, data) = url.strip_prefix("data:")?.split_once(";base64,")?;
                    Some(json!({"type":"image","source":{"type":"base64","media_type":header,"data":data}}))
                }
                _ => None,
            })
            .collect(),
    )
}

/// Responses-format tool definitions → Anthropic tools.
pub(super) fn tools(definitions: &[Value]) -> Vec<Value> {
    definitions
        .iter()
        .filter_map(|tool| {
            Some(json!({
                "name": tool.get("name")?,
                "description": tool.get("description").cloned().unwrap_or(json!("")),
                "input_schema": tool.get("parameters").cloned().unwrap_or(json!({"type":"object"})),
            }))
        })
        .collect()
}

pub(super) fn request_body(
    model: &str,
    system: &str,
    history: &[Value],
    definitions: &[Value],
    thinking_budget: Option<u64>,
) -> Value {
    let mut body = json!({
        "model": model,
        "max_tokens": 16_384,
        "stream": true,
        "system": system,
        "messages": messages(history),
    });
    let tools = tools(definitions);
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    if let Some(budget) = thinking_budget {
        body["thinking"] = json!({"type":"enabled","budget_tokens":budget});
    }
    body
}

/// Accumulates one streamed response.
#[derive(Default)]
pub(super) struct Decoder {
    text: String,
    thinking: String,
    calls: Vec<(String, String, String)>,
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    stop_reason: Option<String>,
    finished: bool,
}

impl Decoder {
    pub(super) fn finished(&self) -> bool {
        self.finished
    }

    pub(super) fn feed(
        &mut self,
        event: &Value,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<(), String> {
        match event["type"].as_str().unwrap_or_default() {
            "message_start" => {
                let usage = &event["message"]["usage"];
                self.input = usage["input_tokens"].as_u64().unwrap_or(0);
                self.cache_read = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);
                self.cache_write = usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
            }
            "content_block_start" => {
                let block = &event["content_block"];
                if block["type"] == "tool_use" {
                    self.calls.push((
                        block["id"].as_str().unwrap_or_default().to_owned(),
                        block["name"].as_str().unwrap_or_default().to_owned(),
                        String::new(),
                    ));
                }
            }
            "content_block_delta" => {
                let delta = &event["delta"];
                match delta["type"].as_str().unwrap_or_default() {
                    "text_delta" => {
                        let text = delta["text"].as_str().unwrap_or_default();
                        self.text.push_str(text);
                        on_event(AgentEvent::TextDelta(text.to_owned()));
                    }
                    "thinking_delta" => {
                        let text = delta["thinking"].as_str().unwrap_or_default();
                        self.thinking.push_str(text);
                        on_event(AgentEvent::ThinkingDelta(text.to_owned()));
                    }
                    "input_json_delta" => {
                        if let Some(call) = self.calls.last_mut() {
                            call.2
                                .push_str(delta["partial_json"].as_str().unwrap_or_default());
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                self.stop_reason = event["delta"]["stop_reason"].as_str().map(str::to_owned);
                self.output = event["usage"]["output_tokens"]
                    .as_u64()
                    .unwrap_or(self.output);
            }
            "message_stop" => self.finished = true,
            "error" => {
                return Err(format!(
                    "{}: {}",
                    event["error"]["type"].as_str().unwrap_or("error"),
                    event["error"]["message"]
                        .as_str()
                        .unwrap_or("unknown error")
                ));
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn into_reply(self) -> Reply {
        let truncated = self.stop_reason.as_deref() == Some("max_tokens");
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
            cache_write: self.cache_write,
            reasoning: None,
            total_tokens: self.input + self.output + self.cache_read + self.cache_write,
            incomplete_reason: truncated.then(|| "max_output_tokens".to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_code_names_map_both_ways_case_insensitively() {
        assert_eq!(to_claude_code_name("read"), "Read");
        assert_eq!(to_claude_code_name("ls"), "ls");
        let tools = [json!({"name":"bash"}), json!({"name":"custom"})];
        assert_eq!(from_claude_code_name("Bash", &tools), "bash");
        assert_eq!(from_claude_code_name("Unknown", &tools), "Unknown");
        let (history, defs) = rename_for_claude_code(
            &[
                json!({"type":"function_call","name":"edit","call_id":"c"}),
                json!({"role":"user"}),
            ],
            &[json!({"name":"write"})],
        );
        assert_eq!(
            (history[0]["name"].as_str(), defs[0]["name"].as_str()),
            (Some("Edit"), Some("Write"))
        );
        assert!(history[1].get("name").is_none());
    }

    #[test]
    fn oauth_tokens_are_recognised_by_their_prefix() {
        assert!(is_oauth_token("sk-ant-oat01-abc"));
        assert!(!is_oauth_token("sk-ant-api03-abc"));
    }

    #[test]
    fn history_translates_to_alternating_anthropic_messages_with_tool_blocks() {
        let history = vec![
            json!({"role":"user","content":[{"type":"input_text","text":"hi"}]}),
            json!({"role":"assistant","content":[{"type":"output_text","text":"calling"}]}),
            json!({"type":"function_call","call_id":"c1","name":"read","arguments":"{\"path\":\"a\"}"}),
            json!({"type":"function_call_output","call_id":"c1","output":"file"}),
        ];
        let out = messages(&history);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1]["content"][1]["type"], "tool_use");
        assert_eq!(out[1]["content"][1]["input"]["path"], "a");
        assert_eq!(out[2]["role"], "user");
        assert_eq!(out[2]["content"][0]["tool_use_id"], "c1");
    }

    #[test]
    fn image_tool_outputs_become_anthropic_image_blocks() {
        let history = vec![
            json!({"type":"function_call","call_id":"c","name":"read","arguments":"{}"}),
            json!({"type":"function_call_output","call_id":"c","output":[
                {"type":"input_text","text":"Read image file [image/png]"},
                {"type":"input_image","image_url":"data:image/png;base64,QUJD"}]}),
        ];
        let out = messages(&history);
        let content = &out[1]["content"][0]["content"];
        assert_eq!(
            content[0],
            json!({"type":"text","text":"Read image file [image/png]"})
        );
        assert_eq!(
            content[1]["source"],
            json!({"type":"base64","media_type":"image/png","data":"QUJD"})
        );
    }

    #[test]
    fn decoder_assembles_text_thinking_tool_input_and_usage() {
        let mut decoder = Decoder::default();
        let mut deltas = Vec::new();
        for event in [
            json!({"type":"message_start","message":{"usage":{"input_tokens":10,"cache_read_input_tokens":2}}}),
            json!({"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"hm"}}),
            json!({"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}),
            json!({"type":"content_block_start","content_block":{"type":"tool_use","id":"t1","name":"bash"}}),
            json!({"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"command\":"}}),
            json!({"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"\"ls\"}"}}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}),
            json!({"type":"message_stop"}),
        ] {
            decoder.feed(&event, &mut |e| deltas.push(e)).unwrap();
        }
        assert!(decoder.finished());
        let reply = decoder.into_reply();
        assert_eq!((reply.text.as_str(), reply.thinking.as_str()), ("ok", "hm"));
        assert_eq!(reply.tool_calls[0].arguments, json!({"command":"ls"}));
        assert_eq!(
            (
                reply.input,
                reply.output,
                reply.cache_read,
                reply.total_tokens
            ),
            (10, 5, 2, 17)
        );
        assert_eq!(deltas.len(), 2);
        assert_eq!(reply.incomplete_reason, None);
    }

    #[test]
    fn max_tokens_is_reported_as_truncation_and_errors_surface() {
        let mut decoder = Decoder::default();
        decoder
            .feed(
                &json!({"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{}}),
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(
            decoder.into_reply().incomplete_reason.as_deref(),
            Some("max_output_tokens")
        );
        let error = Decoder::default()
            .feed(
                &json!({"type":"error","error":{"type":"overloaded_error","message":"busy"}}),
                &mut |_| {},
            )
            .unwrap_err();
        assert_eq!(error, "overloaded_error: busy");
    }
}
