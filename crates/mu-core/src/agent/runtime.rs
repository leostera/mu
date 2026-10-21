//! Adapters that run [`MuAgent`] turns on Pi's agent loop (`crate::agent_loop`).
//!
//! The loop owns scheduling (steering, follow-ups, tool batches, termination). This module
//! supplies the provider stream function, executable tools, queue hooks, and the persistence
//! and UI-event mapping that used to live inline in `submit`.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex as StdMutex},
};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::{OnceCell, mpsc};
use tokio_util::sync::CancellationToken;

use super::{
    AgentError, AgentEvent, AgentPromptQueue, MuAgent, PromptBehavior, Provider, QueueMode, Reply,
    RetryPolicy, ToolSelection, api_name, expand_skill_prompt, filter_tool_definitions,
    retry_delay_ms, retryable_provider_error, usage,
};
use crate::agent_loop::{
    self as agent_loop, AgentContext, AgentHooks, AgentLoopConfig, AgentLoopTurnUpdate,
    AgentMessage, AgentTool, AgentToolResult, AgentTurnContext, AssistantMessage,
    AssistantMessageEvent, AssistantMessageEventKind, ContentBlock, Model, SharedTool, StopReason,
    StreamFn, StreamRequest, ToolDeclaration, ToolExecutionMode, ToolResultMessage,
    ToolUpdateCallback, UserMessage, current_tools,
};
use mu_session::{AssistantToolResponse, AssistantTurnResponse, SessionDocument, ToolCallRecord};

/// Mutable state shared between a running loop and the agent that started it.
pub(super) struct RunShared {
    pub(super) mcp: crate::mcp::McpRegistry,
    pub(super) active_mcp_tools: HashSet<String>,
    pub(super) codemode_store: Value,
    /// Custom session entries produced by tools, persisted before the next message.
    pub(super) session_entries: Vec<(String, Value)>,
    /// Queue prompts the loop pulled in, waiting for their `message_end`.
    pub(super) delivered_prompts: VecDeque<(String, PromptBehavior)>,
    pub(super) provider_error: Option<AgentError>,
}

pub(super) type Shared = Arc<StdMutex<RunShared>>;

fn lock(shared: &Shared) -> std::sync::MutexGuard<'_, RunShared> {
    shared.lock().unwrap_or_else(|error| error.into_inner())
}

/// What a running loop reports back to the agent task.
pub(super) enum Signal {
    Loop(Box<agent_loop::AgentEvent>),
    Ui(AgentEvent),
}

type Signals = mpsc::UnboundedSender<Signal>;

// ---------------------------------------------------------------------------
// Transcript <-> Responses API conversion
// ---------------------------------------------------------------------------

/// A Responses `function_call_output.output`: plain text, or `input_text`/`input_image` parts
/// when the tool returned images.
fn tool_output_value(blocks: &[ContentBlock]) -> Value {
    if !blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::Image { .. }))
    {
        return Value::String(content_text(blocks));
    }
    Value::Array(
        blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(json!({"type":"input_text","text":text})),
                ContentBlock::Image { data, mime_type } => Some(
                    json!({"type":"input_image","image_url":format!("data:{mime_type};base64,{data}")}),
                ),
                _ => None,
            })
            .collect(),
    )
}

fn content_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build a Responses API `input` array from a transcript.
///
/// Assistant messages replay the provider's own output items when present (reasoning
/// signatures, sibling calls); otherwise they are synthesized from content blocks. Failed
/// or aborted assistant turns are not replayed.
pub(super) fn responses_input(messages: &[AgentMessage]) -> Vec<Value> {
    let mut input = Vec::new();
    for message in messages {
        match message {
            AgentMessage::User(UserMessage { content, .. }) => {
                let blocks = content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => {
                            Some(json!({"type":"input_text","text":text}))
                        }
                        ContentBlock::Image { data, mime_type } => Some(json!({
                            "type":"input_image",
                            "image_url":format!("data:{mime_type};base64,{data}"),
                        })),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if !blocks.is_empty() {
                    input.push(json!({"role":"user","content":blocks}));
                }
            }
            AgentMessage::Assistant(assistant) => {
                if matches!(
                    assistant.stop_reason,
                    StopReason::Error | StopReason::Aborted
                ) {
                    continue;
                }
                match assistant.provider_items.as_deref() {
                    Some(items) if !items.is_empty() => input.extend(items.iter().cloned()),
                    _ => synthesize_assistant_items(assistant, &mut input),
                }
            }
            AgentMessage::ToolResult(result) => input.push(json!({
                "type":"function_call_output",
                "call_id":result.tool_call_id,
                "output":tool_output_value(&result.content),
            })),
            AgentMessage::System(_) | AgentMessage::Custom { .. } => {}
        }
    }
    close_orphaned_calls(input)
}

/// Pi's `transformMessages` second pass: a tool call with no result (an interrupted turn, or a
/// hand-edited session) gets a synthetic error result before the next assistant or user item,
/// so every provider sees a complete call/result pairing.
pub(super) fn close_orphaned_calls(input: Vec<Value>) -> Vec<Value> {
    let mut out = Vec::with_capacity(input.len());
    let mut pending: Vec<String> = Vec::new();
    let flush = |pending: &mut Vec<String>, out: &mut Vec<Value>| {
        for call_id in pending.drain(..) {
            out.push(json!({
                "type":"function_call_output",
                "call_id":call_id,
                "output":"No result provided",
            }));
        }
    };
    for item in input {
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => {
                if let Some(id) = item.get("call_id").and_then(Value::as_str) {
                    pending.push(id.to_owned());
                }
            }
            Some("function_call_output") => {
                let id = item.get("call_id").and_then(Value::as_str);
                pending.retain(|pending_id| Some(pending_id.as_str()) != id);
            }
            // A new message or reasoning item starts the next turn.
            _ => flush(&mut pending, &mut out),
        }
        out.push(item);
    }
    flush(&mut pending, &mut out);
    out
}

const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str =
    "(tool image omitted: model does not support images)";

/// Pi's `downgradeUnsupportedImages` for tool results: for a model without image input, each run
/// of images in a `function_call_output` becomes one placeholder text part.
pub(super) fn downgrade_images(history: &[Value]) -> Vec<Value> {
    history
        .iter()
        .map(|item| {
            let (field, placeholder) = if item.get("role").and_then(Value::as_str) == Some("user") {
                ("content", NON_VISION_USER_IMAGE_PLACEHOLDER)
            } else {
                ("output", NON_VISION_TOOL_IMAGE_PLACEHOLDER)
            };
            let Some(parts) = item.get(field).and_then(Value::as_array) else {
                return item.clone();
            };
            let mut out: Vec<Value> = Vec::new();
            for part in parts {
                if part.get("type").and_then(Value::as_str) == Some("input_image") {
                    let already = out.last().is_some_and(|last| last["text"] == placeholder);
                    if !already {
                        out.push(json!({"type":"input_text","text":placeholder}));
                    }
                } else {
                    out.push(part.clone());
                }
            }
            let mut item = item.clone();
            item[field] = Value::Array(out);
            item
        })
        .collect()
}

fn synthesize_assistant_items(assistant: &AssistantMessage, input: &mut Vec<Value>) {
    let text_blocks = assistant
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(json!({"type":"output_text","text":text})),
            _ => None,
        })
        .collect::<Vec<_>>();
    if !text_blocks.is_empty() {
        input.push(json!({"role":"assistant","content":text_blocks}));
    }
    for block in &assistant.content {
        if let ContentBlock::ToolCall {
            id,
            name,
            arguments,
        } = block
        {
            let arguments = arguments.as_str().map(str::to_owned).unwrap_or_else(|| {
                serde_json::to_string(arguments).unwrap_or_else(|_| "{}".into())
            });
            input.push(json!({
                "type":"function_call",
                "call_id":id,
                "name":name,
                "arguments":arguments,
            }));
        }
    }
}

const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";

/// Reconstruct the active branch of a session as transcript messages.
pub(super) fn session_messages(session: &SessionDocument) -> Vec<AgentMessage> {
    let mut messages = Vec::new();
    let path = super::branch_entry_values(session);
    for entry in crate::compaction::build_context_entries(&path) {
        if entry.get("type").and_then(Value::as_str) == Some("compaction") {
            let summary = entry
                .get("summary")
                .and_then(Value::as_str)
                .unwrap_or_default();
            messages.push(AgentMessage::user_text(
                format!("{COMPACTION_SUMMARY_PREFIX}{summary}{COMPACTION_SUMMARY_SUFFIX}"),
                0,
            ));
            continue;
        }
        if entry.get("type").and_then(Value::as_str) == Some("branch_summary") {
            if let Some(summary) = entry
                .get("summary")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                messages.push(AgentMessage::user_text(
                    format!(
                        "{}{summary}{}",
                        crate::compaction::BRANCH_SUMMARY_PREFIX,
                        crate::compaction::BRANCH_SUMMARY_SUFFIX
                    ),
                    0,
                ));
            }
            continue;
        }
        let Some(message) = entry.get("message").and_then(Value::as_object) else {
            continue;
        };
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let blocks_owned = super::message_blocks(message);
        let blocks = &blocks_owned;
        let text_blocks = || {
            blocks
                .iter()
                .filter_map(|block| block.get("text")?.as_str())
                .map(ContentBlock::text)
                .collect::<Vec<_>>()
        };
        match role {
            "user" => {
                let content = text_blocks();
                if !content.is_empty() {
                    messages.push(AgentMessage::User(UserMessage {
                        content,
                        timestamp: 0,
                    }));
                }
            }
            "assistant" => {
                let mut content = Vec::new();
                for block in blocks {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(text) = block.get("text").and_then(Value::as_str) {
                                content.push(ContentBlock::text(text));
                            }
                        }
                        Some("toolCall") => {
                            let (Some(id), Some(name)) = (
                                block.get("id").and_then(Value::as_str),
                                block.get("name").and_then(Value::as_str),
                            ) else {
                                continue;
                            };
                            content.push(ContentBlock::ToolCall {
                                id: id.to_owned(),
                                name: name.to_owned(),
                                arguments: block.get("arguments").cloned().unwrap_or(Value::Null),
                            });
                        }
                        _ => {}
                    }
                }
                let has_calls = content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolCall { .. }));
                let stop = if has_calls {
                    StopReason::ToolUse
                } else {
                    StopReason::Stop
                };
                messages.push(AgentMessage::Assistant(AssistantMessage::new(
                    content, stop,
                )));
            }
            "toolResult" => {
                let Some(call_id) = message.get("toolCallId").and_then(Value::as_str) else {
                    continue;
                };
                messages.push(AgentMessage::ToolResult(ToolResultMessage {
                    tool_call_id: call_id.to_owned(),
                    tool_name: message
                        .get("toolName")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    content: text_blocks(),
                    details: message.get("details").cloned().unwrap_or(Value::Null),
                    usage: None,
                    is_error: message
                        .get("isError")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    duration_ms: None,
                    timestamp: 0,
                }));
            }
            _ => {}
        }
    }
    messages
}

fn assistant_from_reply(reply: &Reply) -> AssistantMessage {
    let mut content = Vec::new();
    if !reply.thinking.is_empty() {
        content.push(ContentBlock::Thinking {
            thinking: reply.thinking.clone(),
            thinking_signature: reply.thinking_signature.clone(),
        });
    }
    if !reply.text.is_empty() {
        content.push(ContentBlock::text(reply.text.clone()));
    }
    content.extend(reply.tool_calls.iter().map(|call| ContentBlock::ToolCall {
        id: call.call_id.clone(),
        name: call.name.clone(),
        arguments: call.arguments.clone(),
    }));
    let truncated = reply.incomplete_reason.as_deref() == Some("max_output_tokens");
    let stop = if truncated {
        StopReason::Length
    } else if reply.tool_calls.is_empty() {
        StopReason::Stop
    } else {
        StopReason::ToolUse
    };
    let mut message = AssistantMessage::new(content, stop);
    message.provider_items = (!reply.output_items.is_empty()).then(|| reply.output_items.clone());
    message.usage = Some(usage(reply));
    message
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// Tool definitions (Responses format) selected for this agent, before strict-schema rewriting.
pub(super) struct ToolPlan {
    pub(super) enabled_tools: Vec<String>,
    pub(super) no_tools: bool,
    pub(super) selection: Option<ToolSelection>,
    pub(super) excluded: Vec<String>,
}

impl ToolPlan {
    pub(super) fn definitions(
        &self,
        mcp: &crate::mcp::McpRegistry,
        active: &HashSet<String>,
    ) -> Vec<Value> {
        if self.no_tools {
            return Vec::new();
        }
        let all = crate::tools::definitions();
        let mut definitions = self
            .enabled_tools
            .iter()
            .filter_map(|enabled| {
                all.iter()
                    .find(|definition| {
                        definition.get("name").and_then(Value::as_str) == Some(enabled)
                    })
                    .cloned()
            })
            .collect::<Vec<_>>();
        definitions.extend(mcp.definitions_for_model(active));
        if mcp.codemode_enabled() {
            definitions.push(crate::codemode::definition(mcp));
        }
        if mcp.has_deferred_tools() {
            definitions.push(crate::tool_search::definition());
        }
        let mut definitions = match &self.selection {
            Some(selection) => filter_tool_definitions(definitions, selection),
            None => definitions,
        };
        definitions.retain(|definition| {
            let name = definition
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            !self
                .excluded
                .iter()
                .any(|pattern| super::tool_name_matches(pattern, name))
        });
        definitions
    }
}

pub(super) fn declaration_from_definition(definition: &Value) -> Option<ToolDeclaration> {
    let name = definition.get("name")?.as_str()?;
    let constrained = matches!(name, "read" | "bash" | "edit" | "write")
        .then(|| json!({"type":"json_schema","strict":"prefer"}));
    Some(ToolDeclaration {
        name: name.to_owned(),
        description: definition
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        parameters: definition
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type":"object","properties":{}})),
        constrained_sampling: constrained,
    })
}

/// Responses API tool for a declaration; strict mode follows the provider capability.
pub(super) fn request_tool(declaration: &ToolDeclaration, supports_strict: bool) -> Value {
    let definition = json!({
        "type":"function",
        "name":declaration.name,
        "description":declaration.description,
        "parameters":declaration.parameters,
    });
    if supports_strict && declaration.constrained_sampling.is_some() {
        super::mark_strict_builtin_tool(definition)
    } else {
        definition
    }
}

/// Everything a tool needs at execution time.
struct ToolRuntime {
    cwd: PathBuf,
    shell_environment: HashMap<String, String>,
    shared: Shared,
}

struct RuntimeTool {
    declaration: ToolDeclaration,
    runtime: Arc<ToolRuntime>,
}

fn flatten_result(output: String, is_error: bool, details: Option<Value>) -> AgentToolResult {
    flatten_result_with_images(output, Vec::new(), is_error, details)
}

fn flatten_result_with_images(
    output: String,
    images: Vec<crate::tools::ToolImage>,
    is_error: bool,
    details: Option<Value>,
) -> AgentToolResult {
    let mut content = vec![ContentBlock::text(output)];
    // Pi's `normalizeToolResultImages`: shrink or convert images as they enter history.
    for image in images {
        let decoded =
            base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &image.data);
        let processed = decoded.ok().and_then(|bytes| {
            crate::image_process::process_image(&bytes, &image.mime_type, true, Default::default())
                .ok()
        });
        match processed {
            Some(processed) => {
                content.push(ContentBlock::Image {
                    data: processed.data,
                    mime_type: processed.mime_type,
                });
                if !processed.hints.is_empty() {
                    content.push(ContentBlock::text(processed.hints.join("\n")));
                }
            }
            // Keep undecodable images rather than dropping a tool's output.
            None => content.push(ContentBlock::Image {
                data: image.data,
                mime_type: image.mime_type,
            }),
        }
    }
    AgentToolResult {
        content,
        details: details.unwrap_or(Value::Null),
        is_error,
        ..AgentToolResult::default()
    }
}

#[async_trait]
impl AgentTool for RuntimeTool {
    fn declaration(&self) -> ToolDeclaration {
        self.declaration.clone()
    }

    /// `tool_search` and `codemode` read and write shared run state, so they never overlap.
    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        matches!(self.declaration.name.as_str(), "tool_search" | "codemode")
            .then_some(ToolExecutionMode::Sequential)
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        params: Value,
        _signal: &CancellationToken,
        _on_update: ToolUpdateCallback<'_>,
    ) -> Result<AgentToolResult, String> {
        let runtime = &self.runtime;
        match self.declaration.name.as_str() {
            "tool_search" => runtime.tool_search(&params),
            "codemode" => runtime.codemode(&params).await,
            name => runtime.call_independent(name, &params).await,
        }
    }
}

impl ToolRuntime {
    fn tool_search(&self, params: &Value) -> Result<AgentToolResult, String> {
        let query = params
            .get("query")
            .and_then(Value::as_str)
            .filter(|query| !query.trim().is_empty())
            .ok_or_else(|| "tool_search requires a non-empty query".to_owned())?;
        let limit = params
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(8)
            .clamp(1, 30) as usize;
        let mut shared = lock(&self.shared);
        let matches = shared
            .mcp
            .search_tools(query, limit, &shared.active_mcp_tools);
        for tool in &matches {
            if let Some(name) = tool.get("name").and_then(Value::as_str) {
                shared.active_mcp_tools.insert(name.to_owned());
            }
        }
        if matches.is_empty() {
            return Ok(flatten_result(
                "No matching tools found.".into(),
                false,
                None,
            ));
        }
        let loaded = matches
            .iter()
            .map(|tool| {
                let description = tool
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .lines()
                    .next()
                    .unwrap_or_default();
                format!(
                    "- {}: {description}",
                    tool.get("name").and_then(Value::as_str).unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut active = shared.active_mcp_tools.iter().cloned().collect::<Vec<_>>();
        active.sort();
        shared
            .session_entries
            .push(("mcp-tool-loadout".into(), json!({"active":active})));
        Ok(flatten_result(
            format!(
                "Loaded {} tool{}. They are available from your next call:\n{loaded}",
                matches.len(),
                if matches.len() == 1 { "" } else { "s" }
            ),
            false,
            None,
        ))
    }

    async fn codemode(&self, params: &Value) -> Result<AgentToolResult, String> {
        let (mcp, store) = {
            let shared = lock(&self.shared);
            (shared.mcp.clone(), shared.codemode_store.clone())
        };
        let (output, is_error, store_writes) =
            crate::codemode::execute(&self.cwd, &mcp, &store, params, &self.shell_environment)
                .await
                .map_err(|error| error.to_string())?;
        if let Some(writes) = store_writes {
            let mut shared = lock(&self.shared);
            apply_codemode_store_write(&mut shared.codemode_store, &writes);
            shared
                .session_entries
                .push(("codemode-store".into(), writes));
        }
        Ok(flatten_result(output, is_error, None))
    }

    async fn call_independent(
        &self,
        name: &str,
        params: &Value,
    ) -> Result<AgentToolResult, String> {
        let mcp = lock(&self.shared).mcp.clone();
        match mcp.call(name, params).await {
            Some(Ok(result)) => Ok(flatten_result_with_images(
                result.content,
                result.images,
                result.is_error,
                result.structured_content,
            )),
            Some(Err(error)) => Err(error),
            None => crate::tools::execute_with_environment(
                &self.cwd,
                name,
                params,
                &self.shell_environment,
            )
            .await
            .map(|result| {
                flatten_result_with_images(
                    result.content,
                    result.images,
                    result.is_error,
                    result.details.or(result.structured_content),
                )
            })
            .map_err(|error| error.to_string()),
        }
    }
}

/// Merge one `codemode-store` entry (`{set, delete}`) into the store map.
pub(super) fn apply_codemode_store_write(store: &mut Value, data: &Value) {
    let Some(map) = store.as_object_mut() else {
        return;
    };
    if let Some(deleted) = data.get("delete").and_then(Value::as_array) {
        for key in deleted.iter().filter_map(Value::as_str) {
            map.remove(key);
        }
    }
    if let Some(values) = data.get("set").and_then(Value::as_object) {
        map.extend(values.clone());
    }
}

fn build_tools(plan: &ToolPlan, runtime: &Arc<ToolRuntime>) -> Vec<SharedTool> {
    let definitions = {
        let shared = lock(&runtime.shared);
        plan.definitions(&shared.mcp, &shared.active_mcp_tools)
    };
    definitions
        .iter()
        .filter_map(declaration_from_definition)
        .map(|declaration| {
            Arc::new(RuntimeTool {
                declaration,
                runtime: runtime.clone(),
            }) as SharedTool
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

struct RuntimeHooks {
    queue: AgentPromptQueue,
    shared: Shared,
    signals: Signals,
    skills: crate::skills::SkillCatalog,
    skill_commands_enabled: bool,
    plan: ToolPlan,
    runtime: Arc<ToolRuntime>,
    mcp_startup: Option<Arc<OnceCell<crate::mcp::McpRegistry>>>,
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,
}

impl RuntimeHooks {
    fn queued_message(&self, prompt: String, behavior: PromptBehavior) -> AgentMessage {
        let signals = self.signals.clone();
        let prompt = if self.skill_commands_enabled {
            expand_skill_prompt(prompt, &self.skills, &mut |event| {
                let _ = signals.send(Signal::Ui(event));
            })
        } else {
            prompt
        };
        lock(&self.shared)
            .delivered_prompts
            .push_back((prompt.clone(), behavior));
        AgentMessage::user_text(prompt, 0)
    }

    /// MCP servers connect in the background; adopt the registry once it is ready.
    fn adopt_ready_mcp(&self) {
        let Some(registry) = self.mcp_startup.as_ref().and_then(|startup| startup.get()) else {
            return;
        };
        lock(&self.shared).mcp = registry.clone();
    }
}

#[async_trait]
impl AgentHooks for RuntimeHooks {
    async fn get_steering_messages(&self) -> Vec<AgentMessage> {
        self.queue
            .take_steering_batch(self.steering_mode)
            .into_iter()
            .map(|prompt| self.queued_message(prompt, PromptBehavior::Steer))
            .collect()
    }

    async fn get_follow_up_messages(&self) -> Vec<AgentMessage> {
        self.queue
            .take_follow_ups(self.follow_up_mode)
            .into_iter()
            .map(|prompt| self.queued_message(prompt, PromptBehavior::FollowUp))
            .collect()
    }

    /// Refresh the executable tool set (tool_search activations, MCP readiness) per turn.
    async fn prepare_next_turn(&self, turn: &AgentTurnContext) -> Option<AgentLoopTurnUpdate> {
        self.adopt_ready_mcp();
        let tools = build_tools(&self.plan, &self.runtime);
        let declared = |tools: &[SharedTool]| {
            tools
                .iter()
                .map(|tool| tool.declaration())
                .collect::<Vec<_>>()
        };
        if declared(&tools) == declared(&turn.context.tools) {
            return None;
        }
        Some(AgentLoopTurnUpdate {
            context: Some(AgentContext {
                messages: turn.context.messages.clone(),
                tools,
            }),
            ..AgentLoopTurnUpdate::default()
        })
    }
}

// ---------------------------------------------------------------------------
// Provider stream function
// ---------------------------------------------------------------------------

struct RuntimeStream {
    provider: Provider,
    retry: RetryPolicy,
    shared: Shared,
    signals: Signals,
}

fn aborted_message() -> AssistantMessage {
    AssistantMessage {
        error_message: Some("Request was aborted".into()),
        ..AssistantMessage::new(Vec::new(), StopReason::Aborted)
    }
}

impl RuntimeStream {
    fn send_ui(&self, event: AgentEvent) {
        let _ = self.signals.send(Signal::Ui(event));
    }

    async fn complete_with_retry(
        &self,
        history: &[Value],
        tools: &[Value],
        signal: &CancellationToken,
        on_event: &mut (dyn FnMut(AssistantMessageEvent) + Send),
    ) -> Result<Reply, AgentError> {
        let mut attempt = 0;
        loop {
            let mut partial = StreamingPartial::default();
            let result = self
                .provider
                .complete(history, tools, &mut |event| {
                    partial.observe(&event, on_event);
                    self.send_ui(event);
                })
                .await;
            match result {
                Ok(reply) => return Ok(reply),
                Err(error)
                    if self.retry.enabled
                        && attempt < self.retry.max_retries
                        && retryable_provider_error(&error) =>
                {
                    attempt += 1;
                    let delay_ms = retry_delay_ms(self.retry, attempt);
                    self.send_ui(AgentEvent::RetryScheduled {
                        attempt,
                        max_attempts: self.retry.max_retries,
                        delay_ms,
                        error: error.to_string(),
                    });
                    let sleep = tokio::time::sleep(std::time::Duration::from_millis(delay_ms));
                    tokio::select! {
                        () = sleep => {}
                        () = signal.cancelled() => return Err(AgentError::Provider("Request was aborted".into())),
                    }
                    self.send_ui(AgentEvent::RetryAttemptStarted { attempt });
                }
                Err(error) => return Err(error),
            }
        }
    }
}

/// Accumulates streamed deltas into the partial assistant message the loop publishes.
#[derive(Default)]
struct StreamingPartial {
    started: bool,
    thinking: String,
    text: String,
}

impl StreamingPartial {
    fn observe(
        &mut self,
        event: &AgentEvent,
        on_event: &mut (dyn FnMut(AssistantMessageEvent) + Send),
    ) {
        let kind = match event {
            AgentEvent::ThinkingDelta(delta) => {
                self.thinking.push_str(delta);
                AssistantMessageEventKind::ThinkingDelta
            }
            AgentEvent::TextDelta(delta) => {
                self.text.push_str(delta);
                AssistantMessageEventKind::TextDelta
            }
            _ => return,
        };
        if !self.started {
            self.started = true;
            on_event(AssistantMessageEvent::Start {
                partial: self.message(),
            });
        }
        on_event(AssistantMessageEvent::Update {
            kind,
            partial: self.message(),
        });
    }

    fn message(&self) -> AssistantMessage {
        let mut content = Vec::new();
        if !self.thinking.is_empty() {
            content.push(ContentBlock::Thinking {
                thinking: self.thinking.clone(),
                thinking_signature: None,
            });
        }
        if !self.text.is_empty() {
            content.push(ContentBlock::text(self.text.clone()));
        }
        AssistantMessage::new(content, StopReason::Stop)
    }
}

#[async_trait]
impl StreamFn for RuntimeStream {
    async fn stream(
        &self,
        request: StreamRequest,
        signal: &CancellationToken,
        on_event: &mut (dyn FnMut(AssistantMessageEvent) + Send),
    ) -> AssistantMessage {
        let history = responses_input(&request.messages);
        let tools = current_tools(&request.messages)
            .iter()
            .map(|declaration| request_tool(declaration, self.provider.supports_strict_sampling))
            .collect::<Vec<_>>();
        let outcome = tokio::select! {
            biased;
            () = signal.cancelled() => return aborted_message(),
            outcome = self.complete_with_retry(&history, &tools, signal, on_event) => outcome,
        };
        match outcome {
            Ok(reply) => assistant_from_reply(&reply),
            Err(error) => {
                if signal.is_cancelled() {
                    return aborted_message();
                }
                let message = error.to_string();
                lock(&self.shared).provider_error = Some(error);
                AssistantMessage {
                    error_message: Some(message),
                    ..AssistantMessage::new(Vec::new(), StopReason::Error)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MuAgent integration
// ---------------------------------------------------------------------------

fn joined_blocks(blocks: &[ContentBlock], pick: impl Fn(&ContentBlock) -> Option<&str>) -> String {
    blocks.iter().filter_map(pick).collect::<Vec<_>>().join("")
}

fn details_option(details: &Value) -> Option<Value> {
    match details {
        Value::Null => None,
        Value::Object(map) if map.is_empty() => None,
        other => Some(other.clone()),
    }
}

impl MuAgent {
    /// Run one user turn on the agent loop; queued steering and follow-up prompts continue it.
    pub async fn submit_with_cancel(
        &mut self,
        prompt: String,
        on_event: &mut impl FnMut(AgentEvent),
        cancel: &CancellationToken,
    ) -> Result<(), AgentError> {
        let prompt = self.expand_prompt(prompt, on_event);
        self.session.append_user_text(prompt.clone());
        self.save_session()?;
        self.ensure_mcp_ready().await;

        let (sender, mut receiver) = mpsc::unbounded_channel();
        let shared: Shared = Arc::new(StdMutex::new(RunShared {
            mcp: self.mcp.clone(),
            active_mcp_tools: self.active_mcp_tools.clone(),
            codemode_store: self.codemode_store(),
            session_entries: Vec::new(),
            delivered_prompts: VecDeque::new(),
            provider_error: None,
        }));
        let runtime = Arc::new(ToolRuntime {
            cwd: self.cwd.clone(),
            shell_environment: self.shell_environment(),
            shared: shared.clone(),
        });
        let plan = || ToolPlan {
            enabled_tools: self.enabled_tools.clone(),
            no_tools: self.no_tools,
            selection: self.tool_selection.clone(),
            excluded: self.excluded_tools.clone(),
        };
        let tools = build_tools(&plan(), &runtime);
        let hooks = Arc::new(RuntimeHooks {
            queue: self.prompt_queue.clone(),
            shared: shared.clone(),
            signals: sender.clone(),
            skills: self.skills.clone(),
            skill_commands_enabled: super::skill_commands_enabled(&self.agent_dir, &self.cwd),
            plan: plan(),
            runtime: runtime.clone(),
            mcp_startup: self.mcp_startup.clone(),
            steering_mode: super::configured_queue_mode(&self.agent_dir, &self.cwd, "steeringMode"),
            follow_up_mode: super::configured_queue_mode(
                &self.agent_dir,
                &self.cwd,
                "followUpMode",
            ),
        });
        let stream = RuntimeStream {
            provider: self.provider.clone(),
            retry: self.retry,
            shared: shared.clone(),
            signals: sender.clone(),
        };
        let config = AgentLoopConfig::new(
            Model {
                id: self.provider.model.clone(),
                provider: self.provider_id.clone(),
            },
            hooks,
        );
        let context = AgentContext {
            messages: self.transcript.clone(),
            tools,
        };
        let sink_sender = sender.clone();
        let sink: agent_loop::EventSink = Arc::new(move |event| {
            let _ = sink_sender.send(Signal::Loop(Box::new(event)));
        });
        drop(sender);

        let run = agent_loop::run_agent_loop(
            vec![AgentMessage::user_text(prompt, 0)],
            context,
            config,
            sink,
            cancel,
            &stream,
        );
        tokio::pin!(run);
        let mut initial_prompt_seen = false;
        let new_messages = loop {
            tokio::select! {
                biased;
                Some(signal) = receiver.recv() => {
                    self.apply_signal(signal, &shared, &mut initial_prompt_seen, on_event)?;
                }
                messages = &mut run => break messages,
            }
        };
        while let Ok(signal) = receiver.try_recv() {
            self.apply_signal(signal, &shared, &mut initial_prompt_seen, on_event)?;
        }

        let (provider_error, active, mcp) = {
            let mut state = lock(&shared);
            (
                state.provider_error.take(),
                std::mem::take(&mut state.active_mcp_tools),
                state.mcp.clone(),
            )
        };
        self.active_mcp_tools = active;
        self.mcp = mcp;
        let mut transcript_additions = new_messages;
        if provider_error.is_some() {
            // The failed attempt is reported to the caller, not replayed to the model.
            if matches!(
                transcript_additions.last(),
                Some(AgentMessage::Assistant(message)) if message.stop_reason == StopReason::Error
            ) {
                transcript_additions.pop();
            }
        }
        self.transcript.extend(transcript_additions);
        if let Some(error) = provider_error {
            return Err(error);
        }
        // Prompts queued after the loop's last poll start a fresh turn.
        if !cancel.is_cancelled() {
            if let Some((prompt, behavior)) = self.prompt_queue.take_next() {
                on_event(AgentEvent::QueuedPromptStarted {
                    prompt: prompt.clone(),
                    behavior,
                });
                return Box::pin(self.submit_with_cancel(prompt, on_event, cancel)).await;
            }
            if self.needs_auto_compaction() {
                on_event(AgentEvent::AutoCompactionStarted);
                // A failed summary leaves the session untouched; the turn itself succeeded.
                let _ = self.compact().await;
            }
        }
        Ok(())
    }

    /// Persist and relay one event from the running loop.
    fn apply_signal(
        &mut self,
        signal: Signal,
        shared: &Shared,
        initial_prompt_seen: &mut bool,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<(), AgentError> {
        let event = match signal {
            Signal::Ui(event) => {
                on_event(event);
                return Ok(());
            }
            Signal::Loop(event) => *event,
        };
        if let Some(tap) = &self.event_tap {
            tap(&event);
        }
        let pending_entries = std::mem::take(&mut lock(shared).session_entries);
        for (custom_type, data) in pending_entries {
            self.session.append_custom_entry(&custom_type, data);
        }
        match event {
            agent_loop::AgentEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                args,
            } => on_event(AgentEvent::ToolStarted {
                call_id: tool_call_id,
                name: tool_name,
                arguments: args,
            }),
            agent_loop::AgentEvent::ToolExecutionEnd {
                tool_call_id,
                result,
                is_error,
                duration_ms,
                ..
            } => on_event(AgentEvent::ToolFinished {
                call_id: tool_call_id,
                output: content_text(&result.content),
                is_error,
                details: details_option(&result.details),
                duration_ms,
            }),
            agent_loop::AgentEvent::MessageEnd {
                message: AgentMessage::User(user),
            } => {
                if !std::mem::replace(initial_prompt_seen, true) {
                    return Ok(()); // The submitted prompt was persisted before the loop started.
                }
                let text = content_text(&user.content);
                let queued = lock(shared).delivered_prompts.pop_front();
                self.session.append_user_text(text.clone());
                self.save_session()?;
                if let Some((prompt, behavior)) = queued {
                    on_event(AgentEvent::QueuedPromptStarted { prompt, behavior });
                }
            }
            agent_loop::AgentEvent::MessageEnd {
                message: AgentMessage::Assistant(assistant),
            } => self.persist_assistant(&assistant)?,
            agent_loop::AgentEvent::MessageEnd {
                message: AgentMessage::ToolResult(result),
            } => {
                self.session.append_tool_result_with_details(
                    &result.tool_call_id,
                    &result.tool_name,
                    content_text(&result.content),
                    result.is_error,
                    details_option(&result.details),
                );
                self.save_session()?;
            }
            _ => {}
        }
        Ok(())
    }

    fn persist_assistant(&mut self, assistant: &AssistantMessage) -> Result<(), AgentError> {
        if matches!(
            assistant.stop_reason,
            StopReason::Error | StopReason::Aborted
        ) {
            return Ok(());
        }
        let thinking = joined_blocks(&assistant.content, |block| match block {
            ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
            _ => None,
        });
        let thinking_signature = assistant.content.iter().find_map(|block| match block {
            ContentBlock::Thinking {
                thinking_signature, ..
            } => thinking_signature.clone(),
            _ => None,
        });
        let text = joined_blocks(&assistant.content, |block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        });
        let calls = assistant
            .tool_calls()
            .into_iter()
            .map(|call| ToolCallRecord {
                id: call.id,
                name: call.name,
                arguments: call.arguments,
            })
            .collect::<Vec<_>>();
        let api = api_name(&self.provider.transport);
        let usage = assistant.usage.clone().unwrap_or_default();
        if calls.is_empty() {
            self.session.append_assistant_response(
                api,
                &self.provider_id,
                &self.provider.model,
                AssistantTurnResponse {
                    thinking,
                    thinking_signature,
                    text,
                    usage,
                },
            );
        } else {
            self.session.append_assistant_tool_calls(
                api,
                &self.provider_id,
                &self.provider.model,
                AssistantToolResponse {
                    thinking,
                    thinking_signature,
                    text,
                    calls,
                    usage,
                },
            );
        }
        self.save_session()
    }
}
