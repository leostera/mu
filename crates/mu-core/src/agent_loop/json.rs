//! Pi's JSON event wire format (`modes/json-event.ts`): one object per agent event.

use serde_json::{Value, json};

use super::types::{AgentEvent, AgentMessage, AgentToolResult, ContentBlock};

fn message(message: &AgentMessage) -> Value {
    serde_json::to_value(message).unwrap_or(Value::Null)
}

fn result(result: &AgentToolResult) -> Value {
    let mut value = json!({
        "content": serde_json::to_value(&result.content).unwrap_or(Value::Null),
        "details": result.details,
    });
    if let Some(structured) = &result.structured_content {
        value["structuredContent"] = structured.clone();
    }
    value
}

/// Convert an agent event to its JSON wire shape.
///
/// Streaming updates omit the cumulative `partial` snapshot, as Pi does; the final message
/// arrives with `message_end`.
pub fn event_to_json(event: &AgentEvent) -> Value {
    match event {
        AgentEvent::AgentStart => json!({"type":"agent_start"}),
        AgentEvent::AgentEnd { messages } => json!({
            "type":"agent_end",
            "messages": messages.iter().map(message).collect::<Vec<_>>(),
        }),
        AgentEvent::TurnStart => json!({"type":"turn_start"}),
        AgentEvent::TurnEnd {
            message: m,
            tool_results,
        } => json!({
            "type":"turn_end",
            "message": message(m),
            "toolResults": tool_results
                .iter()
                .map(|r| message(&AgentMessage::ToolResult(r.clone())))
                .collect::<Vec<_>>(),
        }),
        AgentEvent::MessageStart { message: m } => {
            json!({"type":"message_start","message":message(m)})
        }
        AgentEvent::MessageEnd { message: m } => json!({"type":"message_end","message":message(m)}),
        AgentEvent::MessageUpdate { message: m, event } => {
            let kind = format!("{event:?}");
            let kind = kind
                .chars()
                .enumerate()
                .flat_map(|(i, c)| {
                    if c.is_uppercase() && i > 0 {
                        vec!['_', c.to_ascii_lowercase()]
                    } else {
                        vec![c.to_ascii_lowercase()]
                    }
                })
                .collect::<String>();
            let mut update = json!({"type": kind});
            if let AgentMessage::Assistant(assistant) = m {
                if let Some(ContentBlock::ToolCall { id, name, .. }) = assistant.content.last() {
                    if kind.starts_with("toolcall") {
                        update["id"] = json!(id);
                        update["toolName"] = json!(name);
                    }
                }
            }
            json!({"type":"message_update","assistantMessageEvent":update})
        }
        AgentEvent::ToolExecutionStart {
            tool_call_id,
            tool_name,
            args,
        } => json!({
            "type":"tool_execution_start","toolCallId":tool_call_id,"toolName":tool_name,"args":args,
        }),
        AgentEvent::ToolExecutionUpdate {
            tool_call_id,
            tool_name,
            args,
            partial_result,
        } => json!({
            "type":"tool_execution_update","toolCallId":tool_call_id,"toolName":tool_name,
            "args":args,"partialResult":result(partial_result),
        }),
        AgentEvent::ToolExecutionEnd {
            tool_call_id,
            tool_name,
            result: r,
            is_error,
            duration_ms,
        } => {
            let mut value = json!({
                "type":"tool_execution_end","toolCallId":tool_call_id,"toolName":tool_name,
                "result":result(r),"isError":is_error,
            });
            if let Some(ms) = duration_ms {
                value["durationMs"] = json!(ms);
            }
            value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_loop::{AssistantMessage, AssistantMessageEventKind, StopReason};

    #[test]
    fn lifecycle_events_use_pi_type_names_and_camel_case_fields() {
        assert_eq!(
            event_to_json(&AgentEvent::AgentStart),
            json!({"type":"agent_start"})
        );
        let end = event_to_json(&AgentEvent::ToolExecutionEnd {
            tool_call_id: "c1".into(),
            tool_name: "bash".into(),
            result: AgentToolResult::text("ok"),
            is_error: false,
            duration_ms: Some(7),
        });
        assert_eq!(end["toolCallId"], "c1");
        assert_eq!(end["durationMs"], 7);
        assert_eq!(end["result"]["content"][0]["text"], "ok");
    }

    #[test]
    fn message_updates_drop_the_cumulative_partial_and_name_the_delta() {
        let partial = AgentMessage::Assistant(AssistantMessage::new(
            vec![ContentBlock::text("hi")],
            StopReason::Stop,
        ));
        let update = event_to_json(&AgentEvent::MessageUpdate {
            message: partial,
            event: AssistantMessageEventKind::TextDelta,
        });
        assert_eq!(update["type"], "message_update");
        assert_eq!(update["assistantMessageEvent"]["type"], "text_delta");
        assert!(update.get("message").is_none());
    }
}
