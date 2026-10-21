//! The agent loop (`packages/agent/src/agent-loop.ts`).
//!
//! Works with [`AgentMessage`] throughout and converts to provider messages only at the
//! request boundary.

use std::time::Instant;

use futures_util::future::join_all;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::{
    transcript::{declare_tool_changes, now_ms},
    types::{
        AfterToolCallContext, AgentContext, AgentEvent, AgentLoopConfig, AgentMessage,
        AgentToolCallOutcome, AgentToolResult, AgentTurnContext, AgentTurnDecision,
        AssistantMessage, AssistantMessageEvent, BeforeToolCallContext, EventSink,
        PrepareRequestContext, SharedTool, StopReason, StreamFn, StreamRequest, ThinkingLevel,
        ToolCall, ToolExecutionMode, ToolResultMessage,
    },
    validation::validate_tool_arguments,
};

/// Start a loop with new prompt messages appended to `context`.
pub async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EventSink,
    signal: &CancellationToken,
    stream_fn: &dyn StreamFn,
) -> Vec<AgentMessage> {
    let initial_messages = declare_tool_changes(&context, prompts);
    let mut new_messages = initial_messages.clone();
    let mut current_context = context;
    current_context.messages.extend(initial_messages.clone());

    emit(AgentEvent::AgentStart);
    emit(AgentEvent::TurnStart);
    for message in initial_messages {
        emit(AgentEvent::MessageStart {
            message: message.clone(),
        });
        emit(AgentEvent::MessageEnd { message });
    }

    run_loop(
        current_context,
        &mut new_messages,
        config,
        signal,
        &emit,
        stream_fn,
    )
    .await;
    new_messages
}

/// Why a loop cannot continue from the current context.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum ContinueError {
    #[error("Cannot continue: no messages in context")]
    NoMessages,
    #[error("Cannot continue from message role: assistant")]
    LastMessageIsAssistant,
}

/// Continue from the current context without adding a message (used for retries).
///
/// The last message must convert to a `user` or `toolResult` message via `convert_to_llm`;
/// the provider rejects the request otherwise.
pub async fn run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EventSink,
    signal: &CancellationToken,
    stream_fn: &dyn StreamFn,
) -> Result<Vec<AgentMessage>, ContinueError> {
    match context.messages.last() {
        None => return Err(ContinueError::NoMessages),
        Some(AgentMessage::Assistant(_)) => return Err(ContinueError::LastMessageIsAssistant),
        Some(_) => {}
    }
    let mut new_messages = Vec::new();
    emit(AgentEvent::AgentStart);
    emit(AgentEvent::TurnStart);
    run_loop(context, &mut new_messages, config, signal, &emit, stream_fn).await;
    Ok(new_messages)
}

fn resolve_reasoning(
    current: Option<ThinkingLevel>,
    update: Option<ThinkingLevel>,
) -> Option<ThinkingLevel> {
    match update {
        None => current,
        Some(ThinkingLevel::Off) => None,
        Some(level) => Some(level),
    }
}

fn turn_context(
    message: &AssistantMessage,
    tool_results: Vec<ToolResultMessage>,
    context: &AgentContext,
    new_messages: &[AgentMessage],
) -> AgentTurnContext {
    AgentTurnContext {
        message: message.clone(),
        tool_results,
        context: context.clone(),
        new_messages: new_messages.to_vec(),
    }
}

async fn run_loop(
    initial_context: AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    initial_config: AgentLoopConfig,
    signal: &CancellationToken,
    emit: &EventSink,
    stream_fn: &dyn StreamFn,
) {
    let mut current_context = initial_context;
    let mut config = initial_config;
    let mut last_completed_turn: Option<AgentTurnContext> = None;
    let mut explicit_continuation = false;
    // Steering queued while the user waited is delivered before the first request.
    let mut pending_messages = config.hooks.get_steering_messages().await;

    // Outer loop: continues when queued follow-ups arrive after the agent would stop.
    loop {
        let mut has_more_tool_calls = true;

        // Inner loop: process tool calls and steering messages.
        while has_more_tool_calls || !pending_messages.is_empty() {
            let mut prepared_messages = Vec::new();
            if let Some(turn) = &last_completed_turn {
                if let Some(update) = config.hooks.prepare_next_turn(turn).await {
                    if let Some(context) = update.context {
                        current_context = context;
                    }
                    prepared_messages = update.messages;
                    if let Some(model) = update.model {
                        config.model = model;
                    }
                    config.reasoning = resolve_reasoning(config.reasoning, update.thinking_level);
                }
                // Preparation can be long-running (compaction). Pick up steering queued meanwhile,
                // but only when the earlier poll returned nothing: otherwise one-at-a-time mode
                // would deliver two messages this turn.
                if pending_messages.is_empty() {
                    pending_messages = config.hooks.get_steering_messages().await;
                }
                emit(AgentEvent::TurnStart);
            }

            // Prepared and queued messages come before the next assistant response.
            let mut to_process = prepared_messages;
            to_process.append(&mut pending_messages);
            for message in declare_tool_changes(&current_context, to_process) {
                emit(AgentEvent::MessageStart {
                    message: message.clone(),
                });
                emit(AgentEvent::MessageEnd {
                    message: message.clone(),
                });
                current_context.messages.push(message.clone());
                new_messages.push(message);
            }

            let update = config
                .hooks
                .prepare_request(
                    PrepareRequestContext {
                        context: &current_context,
                        model: &config.model,
                        thinking_level: config.reasoning.unwrap_or_default(),
                    },
                    signal,
                )
                .await;
            if let Some(update) = update {
                if let Some(context) = update.context {
                    current_context = context;
                }
                if let Some(model) = update.model {
                    config.model = model;
                }
                config.reasoning = resolve_reasoning(config.reasoning, update.thinking_level);
            }

            let message =
                stream_assistant_response(&mut current_context, &config, signal, emit, stream_fn)
                    .await;
            new_messages.push(AgentMessage::Assistant(message.clone()));

            if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
                let turn = turn_context(&message, Vec::new(), &current_context, new_messages);
                config.hooks.finish_turn(&turn, signal).await;
                emit(AgentEvent::TurnEnd {
                    message: AgentMessage::Assistant(message),
                    tool_results: Vec::new(),
                });
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                });
                return;
            }

            let tool_calls = message.tool_calls();
            let mut tool_results = Vec::new();
            has_more_tool_calls = false;
            if !tool_calls.is_empty() {
                // A "length" stop means output was cut off by the token limit, so every call in
                // the message may carry truncated arguments. Fail them all instead of executing
                // potentially broken calls.
                let batch = if message.stop_reason == StopReason::Length {
                    fail_tool_calls_from_truncated_message(&tool_calls, emit)
                } else {
                    execute_tool_calls(&current_context, &message, &config, signal, emit).await
                };
                has_more_tool_calls = !batch.terminate;
                for result in batch.messages {
                    let message = AgentMessage::ToolResult(result.clone());
                    current_context.messages.push(message.clone());
                    new_messages.push(message);
                    tool_results.push(result);
                }
            }

            let turn = turn_context(
                &message,
                tool_results.clone(),
                &current_context,
                new_messages,
            );
            last_completed_turn = Some(turn.clone());
            let decision = config.hooks.finish_turn(&turn, signal).await;
            emit(AgentEvent::TurnEnd {
                message: AgentMessage::Assistant(message),
                tool_results,
            });

            if decision == Some(AgentTurnDecision::End) {
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                });
                return;
            }

            explicit_continuation = decision == Some(AgentTurnDecision::Continue);
            pending_messages = config.hooks.get_steering_messages().await;
            if has_more_tool_calls || !pending_messages.is_empty() {
                explicit_continuation = false;
            }
        }

        // The agent would stop here. Check for follow-up messages.
        let follow_ups = config.hooks.get_follow_up_messages().await;
        if !follow_ups.is_empty() {
            explicit_continuation = false;
            pending_messages = follow_ups;
            continue;
        }

        // No natural request was selected, so fulfil the continuation with one context-only turn.
        if explicit_continuation {
            explicit_continuation = false;
            continue;
        }

        break;
    }

    emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    });
}

/// Stream an assistant response; this is where the transcript becomes provider messages.
async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    signal: &CancellationToken,
    emit: &EventSink,
    stream_fn: &dyn StreamFn,
) -> AssistantMessage {
    let transformed = config
        .hooks
        .transform_context(context.messages.clone(), signal)
        .await;
    let messages = config.hooks.convert_to_llm(&transformed).await;
    let api_key = match config.hooks.get_api_key(&config.model.provider).await {
        Some(key) if !key.is_empty() => Some(key),
        _ => config.api_key.clone(),
    };
    let request = StreamRequest {
        model: config.model.clone(),
        messages,
        reasoning: config.reasoning,
        api_key,
    };

    let mut added_partial = false;
    let mut final_message = {
        let transcript = &mut context.messages;
        let added_partial = &mut added_partial;
        stream_fn
            .stream(request, signal, &mut |event| match event {
                AssistantMessageEvent::Start { partial } => {
                    transcript.push(AgentMessage::Assistant(partial.clone()));
                    *added_partial = true;
                    emit(AgentEvent::MessageStart {
                        message: AgentMessage::Assistant(partial),
                    });
                }
                AssistantMessageEvent::Update { kind, partial } => {
                    if *added_partial {
                        if let Some(last) = transcript.last_mut() {
                            *last = AgentMessage::Assistant(partial.clone());
                        }
                        emit(AgentEvent::MessageUpdate {
                            message: AgentMessage::Assistant(partial),
                            event: kind,
                        });
                    }
                }
            })
            .await
    };
    // Record the requested level, whichever stream function answered.
    final_message.thinking_level = Some(config.reasoning.unwrap_or_default());

    let message = AgentMessage::Assistant(final_message.clone());
    if added_partial {
        if let Some(last) = context.messages.last_mut() {
            *last = message.clone();
        }
    } else {
        context.messages.push(message.clone());
        emit(AgentEvent::MessageStart {
            message: message.clone(),
        });
    }
    emit(AgentEvent::MessageEnd { message });
    final_message
}

struct ExecutedToolCallBatch {
    messages: Vec<ToolResultMessage>,
    terminate: bool,
}

fn emit_tool_start(emit: &EventSink, tool_call: &ToolCall) {
    emit(AgentEvent::ToolExecutionStart {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        args: tool_call.arguments.clone(),
    });
}

/// Fail all tool calls of a message truncated by the output token limit.
///
/// Streamed arguments may parse and validate yet be silently incomplete, so none are safe to
/// execute; each is reported as an error so the model can re-issue it.
fn fail_tool_calls_from_truncated_message(
    tool_calls: &[ToolCall],
    emit: &EventSink,
) -> ExecutedToolCallBatch {
    let mut messages = Vec::new();
    for tool_call in tool_calls {
        emit_tool_start(emit, tool_call);
        let finalized = AgentToolCallOutcome {
            tool_call: tool_call.clone(),
            result: error_tool_result(format!(
                "Tool call \"{}\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
                tool_call.name
            )),
            is_error: true,
            duration_ms: None,
        };
        emit_tool_execution_end(&finalized, emit);
        let message = tool_result_message(&finalized);
        emit_tool_result_message(&message, emit);
        messages.push(message);
    }
    ExecutedToolCallBatch {
        messages,
        terminate: false,
    }
}

async fn execute_tool_calls(
    context: &AgentContext,
    assistant: &AssistantMessage,
    config: &AgentLoopConfig,
    signal: &CancellationToken,
    emit: &EventSink,
) -> ExecutedToolCallBatch {
    let tool_calls = assistant.tool_calls();
    let has_sequential_tool = tool_calls.iter().any(|call| {
        find_tool(&context.tools, &call.name)
            .is_some_and(|tool| tool.execution_mode() == Some(ToolExecutionMode::Sequential))
    });
    if config.tool_execution == ToolExecutionMode::Sequential || has_sequential_tool {
        execute_tool_calls_sequential(context, assistant, &tool_calls, config, signal, emit).await
    } else {
        execute_tool_calls_parallel(context, assistant, &tool_calls, config, signal, emit).await
    }
}

fn find_tool<'a>(tools: &'a [SharedTool], name: &str) -> Option<&'a SharedTool> {
    tools.iter().find(|tool| tool.declaration().name == name)
}

async fn execute_tool_calls_sequential(
    context: &AgentContext,
    assistant: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    signal: &CancellationToken,
    emit: &EventSink,
) -> ExecutedToolCallBatch {
    let mut finalized_calls = Vec::new();
    let mut messages = Vec::new();

    for tool_call in tool_calls {
        emit_tool_start(emit, tool_call);
        let preparation = prepare_tool_call(
            context,
            assistant,
            tool_call,
            config,
            signal,
            &context.tools,
        )
        .await;
        let finalized = match preparation {
            Preparation::Immediate { result, is_error } => AgentToolCallOutcome {
                tool_call: tool_call.clone(),
                result,
                is_error,
                duration_ms: None,
            },
            Preparation::Prepared(prepared) => {
                let on_update = tool_update_sink(tool_call, emit);
                let executed = execute_prepared_tool_call(&prepared, signal, &*on_update).await;
                finalize_executed_tool_call(context, assistant, &prepared, executed, config, signal)
                    .await
            }
        };
        emit_tool_execution_end(&finalized, emit);
        let message = tool_result_message(&finalized);
        emit_tool_result_message(&message, emit);
        finalized_calls.push(finalized);
        messages.push(message);
        if signal.is_cancelled() {
            break;
        }
    }

    ExecutedToolCallBatch {
        messages,
        terminate: should_terminate_batch(&finalized_calls),
    }
}

enum ParallelEntry {
    Done(AgentToolCallOutcome),
    Pending(Prepared),
}

async fn execute_tool_calls_parallel(
    context: &AgentContext,
    assistant: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    signal: &CancellationToken,
    emit: &EventSink,
) -> ExecutedToolCallBatch {
    // Calls are prepared sequentially; allowed tools then execute concurrently.
    let mut entries = Vec::new();
    for tool_call in tool_calls {
        emit_tool_start(emit, tool_call);
        match prepare_tool_call(
            context,
            assistant,
            tool_call,
            config,
            signal,
            &context.tools,
        )
        .await
        {
            Preparation::Immediate { result, is_error } => {
                let finalized = AgentToolCallOutcome {
                    tool_call: tool_call.clone(),
                    result,
                    is_error,
                    duration_ms: None,
                };
                emit_tool_execution_end(&finalized, emit);
                entries.push(ParallelEntry::Done(finalized));
            }
            Preparation::Prepared(prepared) => entries.push(ParallelEntry::Pending(prepared)),
        }
        if signal.is_cancelled() {
            break;
        }
    }

    // `tool_execution_end` fires in completion order; result messages follow in source order.
    let finalized_calls = join_all(entries.into_iter().map(|entry| async move {
        match entry {
            ParallelEntry::Done(finalized) => finalized,
            ParallelEntry::Pending(prepared) => {
                let finalized = if signal.is_cancelled() {
                    AgentToolCallOutcome {
                        tool_call: prepared.tool_call.clone(),
                        result: error_tool_result("Operation aborted"),
                        is_error: true,
                        duration_ms: None,
                    }
                } else {
                    let on_update = tool_update_sink(&prepared.tool_call, emit);
                    let executed = execute_prepared_tool_call(&prepared, signal, &*on_update).await;
                    finalize_executed_tool_call(
                        context, assistant, &prepared, executed, config, signal,
                    )
                    .await
                };
                emit_tool_execution_end(&finalized, emit);
                finalized
            }
        }
    }))
    .await;

    let mut messages = Vec::new();
    for finalized in &finalized_calls {
        let message = tool_result_message(finalized);
        emit_tool_result_message(&message, emit);
        messages.push(message);
    }
    ExecutedToolCallBatch {
        messages,
        terminate: should_terminate_batch(&finalized_calls),
    }
}

struct Prepared {
    tool_call: ToolCall,
    tool: SharedTool,
    args: Value,
}

enum Preparation {
    Prepared(Prepared),
    Immediate {
        result: AgentToolResult,
        is_error: bool,
    },
}

struct ExecutedToolCallOutcome {
    result: AgentToolResult,
    is_error: bool,
    duration_ms: u64,
}

/// Early termination only happens when every finalized result in the batch sets `terminate`.
fn should_terminate_batch(finalized_calls: &[AgentToolCallOutcome]) -> bool {
    !finalized_calls.is_empty()
        && finalized_calls
            .iter()
            .all(|finalized| finalized.result.terminate)
}

fn immediate_error(message: impl Into<String>) -> Preparation {
    Preparation::Immediate {
        result: error_tool_result(message),
        is_error: true,
    }
}

async fn prepare_tool_call(
    context: &AgentContext,
    assistant: &AssistantMessage,
    tool_call: &ToolCall,
    config: &AgentLoopConfig,
    signal: &CancellationToken,
    tools: &[SharedTool],
) -> Preparation {
    let Some(tool) = find_tool(tools, &tool_call.name) else {
        return immediate_error(format!("Tool {} not found", tool_call.name));
    };
    let prepared_arguments = tool.prepare_arguments(tool_call.arguments.clone());
    let declaration = tool.declaration();
    let mut args = match validate_tool_arguments(
        &tool_call.name,
        &declaration.parameters,
        &prepared_arguments,
    ) {
        Ok(args) => args,
        Err(message) => return immediate_error(message),
    };
    let before = config
        .hooks
        .before_tool_call(
            BeforeToolCallContext {
                assistant_message: assistant,
                tool_call,
                args: &mut args,
                context,
            },
            signal,
        )
        .await;
    if signal.is_cancelled() {
        return immediate_error("Operation aborted");
    }
    if let Some(before) = before.filter(|before| before.block) {
        let mut result = error_tool_result(
            before
                .reason
                .filter(|reason| !reason.is_empty())
                .unwrap_or_else(|| "Tool execution was blocked".into()),
        );
        result.terminate = before.terminate;
        return Preparation::Immediate {
            result,
            is_error: true,
        };
    }
    Preparation::Prepared(Prepared {
        tool_call: tool_call.clone(),
        tool: tool.clone(),
        args,
    })
}

fn tool_update_sink(
    tool_call: &ToolCall,
    emit: &EventSink,
) -> Box<dyn Fn(AgentToolResult) + Send + Sync> {
    let tool_call = tool_call.clone();
    let emit = emit.clone();
    Box::new(move |partial_result| {
        emit(AgentEvent::ToolExecutionUpdate {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: tool_call.arguments.clone(),
            partial_result,
        });
    })
}

async fn execute_prepared_tool_call(
    prepared: &Prepared,
    signal: &CancellationToken,
    on_update: &(dyn Fn(AgentToolResult) + Send + Sync),
) -> ExecutedToolCallOutcome {
    let started_at = Instant::now();
    let outcome = prepared
        .tool
        .execute(
            &prepared.tool_call.id,
            prepared.args.clone(),
            signal,
            on_update,
        )
        .await;
    let duration_ms = started_at.elapsed().as_millis() as u64;
    match outcome {
        Ok(result) => ExecutedToolCallOutcome {
            is_error: result.is_error,
            result,
            duration_ms,
        },
        Err(message) => ExecutedToolCallOutcome {
            result: error_tool_result(message),
            is_error: true,
            duration_ms,
        },
    }
}

async fn finalize_executed_tool_call(
    context: &AgentContext,
    assistant: &AssistantMessage,
    prepared: &Prepared,
    executed: ExecutedToolCallOutcome,
    config: &AgentLoopConfig,
    signal: &CancellationToken,
) -> AgentToolCallOutcome {
    let mut result = executed.result;
    let mut is_error = executed.is_error;

    let after = config
        .hooks
        .after_tool_call(
            AfterToolCallContext {
                assistant_message: assistant,
                tool_call: &prepared.tool_call,
                args: &prepared.args,
                result: &result,
                is_error,
                context,
            },
            signal,
        )
        .await;
    if let Some(after) = after {
        // Structured content not replaced along with the content may no longer match it.
        let structured_content = after.structured_content.or_else(|| {
            if after.content.is_some() {
                None
            } else {
                result.structured_content.clone()
            }
        });
        result = AgentToolResult {
            content: after.content.unwrap_or(result.content),
            details: after.details.unwrap_or(result.details),
            structured_content,
            usage: after.usage.or(result.usage),
            is_error: result.is_error,
            terminate: after.terminate.unwrap_or(result.terminate),
        };
        is_error = after.is_error.unwrap_or(is_error);
    }

    AgentToolCallOutcome {
        tool_call: prepared.tool_call.clone(),
        result,
        is_error,
        duration_ms: Some(executed.duration_ms),
    }
}

/// Options for [`run_tool_call`].
pub struct RunToolCallOptions<'a> {
    pub tools: &'a [SharedTool],
    pub assistant_message: &'a AssistantMessage,
    pub context: &'a AgentContext,
    pub config: &'a AgentLoopConfig,
    pub signal: &'a CancellationToken,
    pub on_update: Option<&'a (dyn Fn(AgentToolResult) + Send + Sync)>,
}

/// Run one tool call through the same steps as a model-issued call: argument preparation,
/// schema validation, `before_tool_call`, execution, and `after_tool_call`.
///
/// Emits no events and adds no messages. Never fails for tool problems: unknown tools,
/// validation errors, blocked calls, and thrown errors come back as `is_error` outcomes.
pub async fn run_tool_call(
    tool_call: &ToolCall,
    options: RunToolCallOptions<'_>,
) -> AgentToolCallOutcome {
    let preparation = prepare_tool_call(
        options.context,
        options.assistant_message,
        tool_call,
        options.config,
        options.signal,
        options.tools,
    )
    .await;
    match preparation {
        Preparation::Immediate { result, is_error } => AgentToolCallOutcome {
            tool_call: tool_call.clone(),
            result,
            is_error,
            duration_ms: None,
        },
        Preparation::Prepared(prepared) => {
            let ignore = |_: AgentToolResult| {};
            let executed = execute_prepared_tool_call(
                &prepared,
                options.signal,
                options.on_update.unwrap_or(&ignore),
            )
            .await;
            finalize_executed_tool_call(
                options.context,
                options.assistant_message,
                &prepared,
                executed,
                options.config,
                options.signal,
            )
            .await
        }
    }
}

fn error_tool_result(message: impl Into<String>) -> AgentToolResult {
    AgentToolResult::error(message)
}

fn emit_tool_execution_end(finalized: &AgentToolCallOutcome, emit: &EventSink) {
    emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        result: finalized.result.clone(),
        is_error: finalized.is_error,
        duration_ms: finalized.duration_ms,
    });
}

fn tool_result_message(finalized: &AgentToolCallOutcome) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        content: finalized.result.content.clone(),
        details: finalized.result.details.clone(),
        usage: finalized.result.usage.clone(),
        is_error: finalized.is_error,
        duration_ms: finalized.duration_ms,
        timestamp: now_ms(),
    }
}

fn emit_tool_result_message(message: &ToolResultMessage, emit: &EventSink) {
    let message = AgentMessage::ToolResult(message.clone());
    emit(AgentEvent::MessageStart {
        message: message.clone(),
    });
    emit(AgentEvent::MessageEnd { message });
}
