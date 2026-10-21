//! Pi-shaped agent loop types (`packages/agent/src/types.ts`).
//!
//! Messages keep Pi's JSON shape so they can be persisted without translation.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

/// Reasoning level requested for a provider call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    #[default]
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

/// How tool calls from one assistant message are scheduled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ToolExecutionMode {
    Sequential,
    #[default]
    Parallel,
}

/// A content block shared by user, assistant, and tool-result messages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    #[serde(rename_all = "camelCase")]
    Image {
        data: String,
        mime_type: String,
    },
    Thinking {
        thinking: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
}

/// A tool as declared to the model (executable and display-only fields stripped).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constrained_sampling: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolReference {
    pub name: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemMessage {
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sections: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools_added: Vec<ToolDeclaration>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools_removed: Vec<ToolReference>,
    pub timestamp: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    pub content: Vec<ContentBlock>,
    pub timestamp: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ThinkingLevel>,
    pub timestamp: i64,
}

impl AssistantMessage {
    pub fn tool_calls(&self) -> Vec<ToolCall> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                } => Some(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                }),
                _ => None,
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ContentBlock>,
    pub details: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    pub timestamp: i64,
}

/// A message in an agent transcript.
///
/// `Custom` messages are application-defined and never reach the provider unless
/// `AgentHooks::convert_to_llm` converts them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum AgentMessage {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
    #[serde(skip)]
    Custom {
        role: String,
        content: Value,
    },
}

impl AgentMessage {
    pub fn user_text(text: impl Into<String>, timestamp: i64) -> Self {
        Self::User(UserMessage {
            content: vec![ContentBlock::text(text)],
            timestamp,
        })
    }

    pub fn role(&self) -> &str {
        match self {
            Self::System(_) => "system",
            Self::User(_) => "user",
            Self::Assistant(_) => "assistant",
            Self::ToolResult(_) => "toolResult",
            Self::Custom { role, .. } => role,
        }
    }
}

/// One tool call content block.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Final or partial result produced by a tool.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AgentToolResult {
    pub content: Vec<ContentBlock>,
    pub details: Value,
    pub structured_content: Option<Value>,
    /// Usage of the tool execution itself; not used for main context accounting.
    pub usage: Option<Value>,
    pub is_error: bool,
    /// Hint to stop after the current tool batch; only honored when every result in the batch sets it.
    pub terminate: bool,
}

impl AgentToolResult {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(text)],
            details: Value::Object(Map::new()),
            ..Self::default()
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self::text(text)
    }
}

/// Final outcome of a tool call after hooks ran.
#[derive(Clone, Debug, PartialEq)]
pub struct AgentToolCallOutcome {
    pub tool_call: ToolCall,
    pub result: AgentToolResult,
    pub is_error: bool,
    pub duration_ms: Option<u64>,
}

pub type ToolUpdateCallback<'a> = &'a (dyn Fn(AgentToolResult) + Send + Sync);

/// Executable tool used by the agent runtime.
#[async_trait]
pub trait AgentTool: Send + Sync {
    fn declaration(&self) -> ToolDeclaration;

    fn label(&self) -> String {
        self.declaration().name
    }

    /// Per-tool override of the configured execution mode.
    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        None
    }

    /// Compatibility shim for raw arguments before schema validation.
    fn prepare_arguments(&self, arguments: Value) -> Value {
        arguments
    }

    /// Run the call. `Err` is reported to the model as an error result.
    async fn execute(
        &self,
        tool_call_id: &str,
        params: Value,
        signal: &CancellationToken,
        on_update: ToolUpdateCallback<'_>,
    ) -> Result<AgentToolResult, String>;
}

pub type SharedTool = Arc<dyn AgentTool>;

/// Context snapshot passed into the loop.
#[derive(Clone, Default)]
pub struct AgentContext {
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<SharedTool>,
}

/// Model identity used for provider requests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Model {
    pub id: String,
    pub provider: String,
}

/// Events emitted for UI updates (`AgentEvent` in Pi).
#[derive(Clone, Debug, PartialEq)]
pub enum AgentEvent {
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
    },
    TurnStart,
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    MessageStart {
        message: AgentMessage,
    },
    MessageUpdate {
        message: AgentMessage,
        event: AssistantMessageEventKind,
    },
    MessageEnd {
        message: AgentMessage,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        partial_result: AgentToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: AgentToolResult,
        is_error: bool,
        duration_ms: Option<u64>,
    },
}

pub type EventSink = Arc<dyn Fn(AgentEvent) + Send + Sync>;

/// Streaming sub-events of an assistant message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssistantMessageEventKind {
    TextStart,
    TextDelta,
    TextEnd,
    ThinkingStart,
    ThinkingDelta,
    ThinkingEnd,
    ToolcallStart,
    ToolcallDelta,
    ToolcallEnd,
}

/// Provider stream protocol: `Start` carries the first partial; updates replace it.
#[derive(Clone, Debug, PartialEq)]
pub enum AssistantMessageEvent {
    Start {
        partial: AssistantMessage,
    },
    Update {
        kind: AssistantMessageEventKind,
        partial: AssistantMessage,
    },
}

/// Request handed to the stream function.
#[derive(Clone, Debug)]
pub struct StreamRequest {
    pub model: Model,
    pub messages: Vec<AgentMessage>,
    pub reasoning: Option<ThinkingLevel>,
    pub api_key: Option<String>,
}

/// Provider stream function.
///
/// Contract (as in Pi): never fail by panicking or returning `Err`; encode failures in the
/// returned message with `StopReason::Error`/`Aborted` and `error_message`.
#[async_trait]
pub trait StreamFn: Send + Sync {
    async fn stream(
        &self,
        request: StreamRequest,
        signal: &CancellationToken,
        on_event: &mut (dyn FnMut(AssistantMessageEvent) + Send),
    ) -> AssistantMessage;
}

/// Decision returned by [`AgentHooks::finish_turn`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentTurnDecision {
    Continue,
    End,
}

/// Context passed to completed-turn callbacks.
#[derive(Clone)]
pub struct AgentTurnContext {
    pub message: AssistantMessage,
    pub tool_results: Vec<ToolResultMessage>,
    pub context: AgentContext,
    /// Messages this loop invocation returns if it exits now.
    pub new_messages: Vec<AgentMessage>,
}

/// Replacement runtime state used before another provider request.
#[derive(Clone, Default)]
pub struct AgentLoopTurnUpdate {
    pub context: Option<AgentContext>,
    pub messages: Vec<AgentMessage>,
    pub model: Option<Model>,
    pub thinking_level: Option<ThinkingLevel>,
}

/// Runtime state available immediately before a provider request.
pub struct PrepareRequestContext<'a> {
    pub context: &'a AgentContext,
    pub model: &'a Model,
    pub thinking_level: ThinkingLevel,
}

/// Replacement runtime state for the request being prepared.
#[derive(Clone, Default)]
pub struct AgentRequestUpdate {
    pub context: Option<AgentContext>,
    pub model: Option<Model>,
    pub thinking_level: Option<ThinkingLevel>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BeforeToolCallResult {
    pub block: bool,
    pub reason: Option<String>,
    pub terminate: bool,
}

/// Partial override returned from `after_tool_call` (field-by-field replacement).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AfterToolCallResult {
    pub content: Option<Vec<ContentBlock>>,
    pub details: Option<Value>,
    pub structured_content: Option<Value>,
    pub usage: Option<Value>,
    pub is_error: Option<bool>,
    pub terminate: Option<bool>,
}

pub struct BeforeToolCallContext<'a> {
    pub assistant_message: &'a AssistantMessage,
    pub tool_call: &'a ToolCall,
    /// Validated arguments. Mutations are executed without revalidation, as in Pi.
    pub args: &'a mut Value,
    pub context: &'a AgentContext,
}

pub struct AfterToolCallContext<'a> {
    pub assistant_message: &'a AssistantMessage,
    pub tool_call: &'a ToolCall,
    pub args: &'a Value,
    pub result: &'a AgentToolResult,
    pub is_error: bool,
    pub context: &'a AgentContext,
}

/// Loop callbacks. Every method has Pi's "no hook configured" default.
#[async_trait]
pub trait AgentHooks: Send + Sync {
    /// Converts transcript messages to provider messages; the default drops custom messages.
    async fn convert_to_llm(&self, messages: &[AgentMessage]) -> Vec<AgentMessage> {
        messages
            .iter()
            .filter(|message| !matches!(message, AgentMessage::Custom { .. }))
            .cloned()
            .collect()
    }

    async fn transform_context(
        &self,
        messages: Vec<AgentMessage>,
        _signal: &CancellationToken,
    ) -> Vec<AgentMessage> {
        messages
    }

    async fn get_api_key(&self, _provider: &str) -> Option<String> {
        None
    }

    async fn finish_turn(
        &self,
        _turn: &AgentTurnContext,
        _signal: &CancellationToken,
    ) -> Option<AgentTurnDecision> {
        None
    }

    async fn prepare_request(
        &self,
        _request: PrepareRequestContext<'_>,
        _signal: &CancellationToken,
    ) -> Option<AgentRequestUpdate> {
        None
    }

    async fn prepare_next_turn(&self, _turn: &AgentTurnContext) -> Option<AgentLoopTurnUpdate> {
        None
    }

    async fn get_steering_messages(&self) -> Vec<AgentMessage> {
        Vec::new()
    }

    async fn get_follow_up_messages(&self) -> Vec<AgentMessage> {
        Vec::new()
    }

    async fn before_tool_call(
        &self,
        _context: BeforeToolCallContext<'_>,
        _signal: &CancellationToken,
    ) -> Option<BeforeToolCallResult> {
        None
    }

    async fn after_tool_call(
        &self,
        _context: AfterToolCallContext<'_>,
        _signal: &CancellationToken,
    ) -> Option<AfterToolCallResult> {
        None
    }
}

/// Hooks with no behavior; every callback keeps its default.
pub struct NoHooks;

#[async_trait]
impl AgentHooks for NoHooks {}

#[derive(Clone)]
pub struct AgentLoopConfig {
    pub model: Model,
    pub reasoning: Option<ThinkingLevel>,
    pub api_key: Option<String>,
    pub tool_execution: ToolExecutionMode,
    pub hooks: Arc<dyn AgentHooks>,
}

impl AgentLoopConfig {
    pub fn new(model: Model, hooks: Arc<dyn AgentHooks>) -> Self {
        Self {
            model,
            reasoning: None,
            api_key: None,
            tool_execution: ToolExecutionMode::Parallel,
            hooks,
        }
    }
}
