//! Tool-state replay (`packages/ai/src/utils/transcript.ts` plus `declareToolChanges`).
//!
//! `context.tools` is what the runtime can execute; the transcript's system messages declare
//! what the model may call. Before each request the difference becomes `tools_added` and
//! `tools_removed` on a system message.

use super::types::{AgentContext, AgentMessage, SystemMessage, ToolDeclaration, ToolReference};

/// Tools available after applying every transcript delta in order.
pub fn current_tools(messages: &[AgentMessage]) -> Vec<ToolDeclaration> {
    let mut tools: Vec<ToolDeclaration> = Vec::new();
    for message in messages {
        let AgentMessage::System(system) = message else {
            continue;
        };
        for removed in &system.tools_removed {
            tools.retain(|tool| tool.name != removed.name);
        }
        for added in &system.tools_added {
            match tools.iter_mut().find(|tool| tool.name == added.name) {
                Some(existing) => *existing = added.clone(),
                None => tools.push(added.clone()),
            }
        }
    }
    tools
}

/// Added and removed tools between two complete tool states.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolStateChanges {
    pub tools_added: Vec<ToolDeclaration>,
    pub tools_removed: Vec<ToolReference>,
}

impl ToolStateChanges {
    fn is_empty(&self) -> bool {
        self.tools_added.is_empty() && self.tools_removed.is_empty()
    }
}

/// A changed definition is a removal followed by an addition.
pub fn tool_state_changes(
    previous: &[ToolDeclaration],
    current: &[ToolDeclaration],
) -> ToolStateChanges {
    let find =
        |tools: &[ToolDeclaration], name: &str| tools.iter().find(|t| t.name == name).cloned();
    ToolStateChanges {
        tools_added: current
            .iter()
            .filter(|tool| find(previous, &tool.name).is_none_or(|old| old != **tool))
            .cloned()
            .collect(),
        tools_removed: previous
            .iter()
            .filter(|tool| find(current, &tool.name).is_none_or(|new| new != **tool))
            .map(|tool| ToolReference {
                name: tool.name.clone(),
            })
            .collect(),
    }
}

fn with_tool_changes(message: &SystemMessage, changes: ToolStateChanges) -> SystemMessage {
    SystemMessage {
        tools_added: changes.tools_added,
        tools_removed: changes.tools_removed,
        ..message.clone()
    }
}

/// Insert or rewrite a system message so replaying the transcript yields exactly `context.tools`.
///
/// When a pending system message exists, its tool fields are treated as intent and replaced
/// with the delta between the committed transcript and the executable set. Otherwise a new
/// system message goes before the first non-system pending message.
pub fn declare_tool_changes(
    context: &AgentContext,
    pending: Vec<AgentMessage>,
) -> Vec<AgentMessage> {
    let system_index = pending
        .iter()
        .rposition(|message| matches!(message, AgentMessage::System(_)));
    let baseline: Vec<AgentMessage> = match system_index {
        Some(index) => pending
            .iter()
            .enumerate()
            .map(|(i, message)| match message {
                AgentMessage::System(system) if i == index => {
                    AgentMessage::System(with_tool_changes(system, ToolStateChanges::default()))
                }
                other => other.clone(),
            })
            .collect(),
        None => pending.clone(),
    };
    let committed: Vec<AgentMessage> = context
        .messages
        .iter()
        .chain(baseline.iter())
        .cloned()
        .collect();
    let executable: Vec<ToolDeclaration> = context
        .tools
        .iter()
        .map(|tool| tool.declaration())
        .collect();
    let changes = tool_state_changes(&current_tools(&committed), &executable);

    match system_index {
        Some(index) => {
            let AgentMessage::System(system) = &pending[index] else {
                unreachable!("index was found by matching a system message");
            };
            if changes.is_empty()
                && system.tools_added.is_empty()
                && system.tools_removed.is_empty()
            {
                return pending;
            }
            baseline
                .into_iter()
                .enumerate()
                .map(|(i, message)| match message {
                    AgentMessage::System(base) if i == index => {
                        AgentMessage::System(with_tool_changes(&base, changes.clone()))
                    }
                    other => other,
                })
                .collect()
        }
        None if changes.is_empty() => pending,
        None => {
            let update = AgentMessage::System(with_tool_changes(
                &SystemMessage {
                    timestamp: now_ms(),
                    ..SystemMessage::default()
                },
                changes,
            ));
            let insert_at = pending
                .iter()
                .position(|message| !matches!(message, AgentMessage::System(_)))
                .unwrap_or(pending.len());
            let mut messages = pending;
            messages.insert(insert_at, update);
            messages
        }
    }
}

pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis() as i64)
}
