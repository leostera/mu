//! Ported from `packages/agent/test/agent-loop.test.ts` (pinned Pi revision).

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::*;

type Responder = dyn Fn(usize, &StreamRequest) -> AssistantMessage + Send + Sync;

/// Scripted provider: `respond(call_index, request)` decides each reply.
struct ScriptedStream {
    calls: AtomicUsize,
    requests: Mutex<Vec<StreamRequest>>,
    respond: Box<Responder>,
}

impl ScriptedStream {
    fn new(
        respond: impl Fn(usize, &StreamRequest) -> AssistantMessage + Send + Sync + 'static,
    ) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            requests: Mutex::new(Vec::new()),
            respond: Box::new(respond),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn requests(&self) -> Vec<StreamRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl StreamFn for ScriptedStream {
    async fn stream(
        &self,
        request: StreamRequest,
        _signal: &CancellationToken,
        _on_event: &mut (dyn FnMut(AssistantMessageEvent) + Send),
    ) -> AssistantMessage {
        let index = self.calls.fetch_add(1, Ordering::SeqCst);
        let reply = (self.respond)(index, &request);
        self.requests.lock().unwrap().push(request);
        reply
    }
}

fn assistant(content: Vec<ContentBlock>, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage::new(content, stop_reason)
}

fn text_reply(text: &str) -> AssistantMessage {
    assistant(vec![ContentBlock::text(text)], StopReason::Stop)
}

fn call(id: &str, name: &str, arguments: Value) -> ContentBlock {
    ContentBlock::ToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
    }
}

fn tool_use(calls: Vec<ContentBlock>) -> AssistantMessage {
    assistant(calls, StopReason::ToolUse)
}

fn user(text: &str) -> AgentMessage {
    AgentMessage::user_text(text, 0)
}

fn model() -> Model {
    Model {
        id: "mock".into(),
        provider: "openai".into(),
    }
}

type RunFn = dyn Fn(String, Value) -> futures_util::future::BoxFuture<'static, Result<AgentToolResult, String>>
    + Send
    + Sync;

struct FnTool {
    name: String,
    parameters: Value,
    mode: Option<ToolExecutionMode>,
    prepare: Option<fn(Value) -> Value>,
    run: Box<RunFn>,
}

#[async_trait]
impl AgentTool for FnTool {
    fn declaration(&self) -> ToolDeclaration {
        ToolDeclaration {
            name: self.name.clone(),
            description: format!("{} tool", self.name),
            parameters: self.parameters.clone(),
            constrained_sampling: None,
        }
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        self.mode
    }

    fn prepare_arguments(&self, arguments: Value) -> Value {
        self.prepare
            .map_or(arguments.clone(), |prepare| prepare(arguments))
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        params: Value,
        _signal: &CancellationToken,
        _on_update: ToolUpdateCallback<'_>,
    ) -> Result<AgentToolResult, String> {
        (self.run)(tool_call_id.to_owned(), params).await
    }
}

fn value_schema() -> Value {
    json!({"type":"object","properties":{"value":{"type":"string"}},"required":["value"]})
}

fn fn_tool<F, Fut>(name: &str, parameters: Value, run: F) -> FnTool
where
    F: Fn(String, Value) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<AgentToolResult, String>> + Send + 'static,
{
    FnTool {
        name: name.into(),
        parameters,
        mode: None,
        prepare: None,
        run: Box::new(move |id, params| Box::pin(run(id, params))),
    }
}

/// Echo tool that records executed values and returns `echoed: <value>`.
fn echo_tool(executed: Arc<Mutex<Vec<String>>>) -> FnTool {
    fn_tool("echo", value_schema(), move |_, params| {
        let executed = executed.clone();
        async move {
            let value = params["value"].as_str().unwrap_or_default().to_owned();
            executed.lock().unwrap().push(value.clone());
            Ok(AgentToolResult::text(format!("echoed: {value}")))
        }
    })
}

fn noop_tool() -> FnTool {
    fn_tool(
        "noop",
        json!({"type":"object","properties":{}}),
        |_, _| async { Ok(AgentToolResult::text("done")) },
    )
}

type Hook<A, R> = Option<Box<dyn Fn(A) -> R + Send + Sync>>;

/// Closure-driven hooks; unset hooks keep Pi's defaults.
#[derive(Default)]
struct TestHooks {
    convert: Hook<Vec<AgentMessage>, Vec<AgentMessage>>,
    transform: Hook<Vec<AgentMessage>, Vec<AgentMessage>>,
    before: Hook<(ToolCall, Value), (Option<BeforeToolCallResult>, Value)>,
    after: Hook<AgentToolResult, Option<AfterToolCallResult>>,
    steering: Hook<(), Vec<AgentMessage>>,
    follow_up: Hook<(), Vec<AgentMessage>>,
    finish_turn: Hook<AgentTurnContext, Option<AgentTurnDecision>>,
    prepare_request: Hook<(AgentContext, Model, ThinkingLevel), Option<AgentRequestUpdate>>,
    prepare_next_turn: Hook<AgentTurnContext, Option<AgentLoopTurnUpdate>>,
    before_delay: Option<Duration>,
}

#[async_trait]
impl AgentHooks for TestHooks {
    async fn convert_to_llm(&self, messages: &[AgentMessage]) -> Vec<AgentMessage> {
        match &self.convert {
            Some(convert) => convert(messages.to_vec()),
            None => NoHooks.convert_to_llm(messages).await,
        }
    }

    async fn transform_context(
        &self,
        messages: Vec<AgentMessage>,
        _signal: &CancellationToken,
    ) -> Vec<AgentMessage> {
        match &self.transform {
            Some(transform) => transform(messages),
            None => messages,
        }
    }

    async fn finish_turn(
        &self,
        turn: &AgentTurnContext,
        _signal: &CancellationToken,
    ) -> Option<AgentTurnDecision> {
        self.finish_turn
            .as_ref()
            .and_then(|hook| hook(turn.clone()))
    }

    async fn prepare_request(
        &self,
        request: PrepareRequestContext<'_>,
        _signal: &CancellationToken,
    ) -> Option<AgentRequestUpdate> {
        self.prepare_request.as_ref().and_then(|hook| {
            hook((
                request.context.clone(),
                request.model.clone(),
                request.thinking_level,
            ))
        })
    }

    async fn prepare_next_turn(&self, turn: &AgentTurnContext) -> Option<AgentLoopTurnUpdate> {
        self.prepare_next_turn
            .as_ref()
            .and_then(|hook| hook(turn.clone()))
    }

    async fn get_steering_messages(&self) -> Vec<AgentMessage> {
        self.steering
            .as_ref()
            .map_or_else(Vec::new, |hook| hook(()))
    }

    async fn get_follow_up_messages(&self) -> Vec<AgentMessage> {
        self.follow_up
            .as_ref()
            .map_or_else(Vec::new, |hook| hook(()))
    }

    async fn before_tool_call(
        &self,
        context: BeforeToolCallContext<'_>,
        _signal: &CancellationToken,
    ) -> Option<BeforeToolCallResult> {
        if let Some(delay) = self.before_delay {
            tokio::time::sleep(delay).await;
        }
        let hook = self.before.as_ref()?;
        let (result, args) = hook((context.tool_call.clone(), context.args.clone()));
        *context.args = args;
        result
    }

    async fn after_tool_call(
        &self,
        context: AfterToolCallContext<'_>,
        _signal: &CancellationToken,
    ) -> Option<AfterToolCallResult> {
        self.after
            .as_ref()
            .and_then(|hook| hook(context.result.clone()))
    }
}

struct Run {
    events: Vec<AgentEvent>,
    messages: Vec<AgentMessage>,
}

fn collecting_sink() -> (EventSink, Arc<Mutex<Vec<AgentEvent>>>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let sink_events = events.clone();
    (
        Arc::new(move |event| sink_events.lock().unwrap().push(event)),
        events,
    )
}

async fn run(
    prompts: Vec<AgentMessage>,
    tools: Vec<FnTool>,
    hooks: TestHooks,
    tool_execution: ToolExecutionMode,
    stream: &ScriptedStream,
) -> Run {
    let mut config = AgentLoopConfig::new(model(), Arc::new(hooks));
    config.tool_execution = tool_execution;
    let context = AgentContext {
        messages: Vec::new(),
        tools: tools
            .into_iter()
            .map(|tool| Arc::new(tool) as SharedTool)
            .collect(),
    };
    let (sink, events) = collecting_sink();
    let messages = run_agent_loop(
        prompts,
        context,
        config,
        sink,
        &CancellationToken::new(),
        stream,
    )
    .await;
    let events = events.lock().unwrap().clone();
    Run { events, messages }
}

fn roles(messages: &[AgentMessage]) -> Vec<&str> {
    messages.iter().map(AgentMessage::role).collect()
}

fn tool_results(messages: &[AgentMessage]) -> Vec<&ToolResultMessage> {
    messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect()
}

fn event_names(events: &[AgentEvent]) -> Vec<&'static str> {
    events
        .iter()
        .map(|event| match event {
            AgentEvent::AgentStart => "agent_start",
            AgentEvent::AgentEnd { .. } => "agent_end",
            AgentEvent::TurnStart => "turn_start",
            AgentEvent::TurnEnd { .. } => "turn_end",
            AgentEvent::MessageStart { .. } => "message_start",
            AgentEvent::MessageUpdate { .. } => "message_update",
            AgentEvent::MessageEnd { .. } => "message_end",
            AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
            AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
            AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
        })
        .collect()
}

fn user_texts(messages: &[AgentMessage]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::User(UserMessage { content, .. }) => match content.first() {
                Some(ContentBlock::Text { text }) => Some(text.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// First call answers with `first`, every later call with "done".
fn then_done(first: AssistantMessage) -> ScriptedStream {
    let first = Mutex::new(Some(first));
    ScriptedStream::new(move |_, _| {
        first
            .lock()
            .unwrap()
            .take()
            .unwrap_or_else(|| text_reply("done"))
    })
}

fn echo_call(id: &str, value: &str) -> ContentBlock {
    call(id, "echo", json!({"value": value}))
}

#[tokio::test]
async fn emits_agent_events_for_a_plain_prompt() {
    let stream = ScriptedStream::new(|_, _| text_reply("Hi there!"));
    let run = run(
        vec![user("Hello")],
        vec![],
        TestHooks::default(),
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    assert_eq!(
        event_names(&run.events),
        [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end"
        ]
    );
    assert_eq!(roles(&run.messages), ["user", "assistant"]);
}

#[tokio::test]
async fn builds_provider_context_from_transcript_messages_only() {
    let stream = ScriptedStream::new(|_, _| text_reply("ok"));
    let tools = vec![noop_tool()];
    run(
        vec![user("hi")],
        tools,
        TestHooks::default(),
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    let request = &stream.requests()[0];
    let AgentMessage::System(system) = &request.messages[0] else {
        panic!("the loop declares tools in a leading system message");
    };
    assert_eq!(system.tools_added[0].name, "noop");
    assert_eq!(roles(&request.messages), ["system", "user"]);
}

#[tokio::test]
async fn custom_messages_convert_through_convert_to_llm() {
    let stream = ScriptedStream::new(|_, _| text_reply("ok"));
    let hooks = TestHooks {
        convert: Some(Box::new(|messages| {
            messages
                .into_iter()
                .filter_map(|message| match message {
                    AgentMessage::Custom { content, .. } => Some(user(content.as_str().unwrap())),
                    AgentMessage::System(_) => None,
                    other => Some(other),
                })
                .collect()
        })),
        ..TestHooks::default()
    };
    let mut config = AgentLoopConfig::new(model(), Arc::new(hooks));
    config.reasoning = Some(ThinkingLevel::Low);
    let context = AgentContext {
        messages: vec![AgentMessage::Custom {
            role: "notification".into(),
            content: json!("Hook content"),
        }],
        tools: vec![],
    };
    let (sink, _) = collecting_sink();
    let messages =
        run_agent_loop_continue(context, config, sink, &CancellationToken::new(), &stream)
            .await
            .unwrap();

    assert_eq!(roles(&messages), ["assistant"]);
    assert_eq!(user_texts(&stream.requests()[0].messages), ["Hook content"]);
    let AgentMessage::Assistant(reply) = &messages[0] else {
        unreachable!()
    };
    assert_eq!(reply.thinking_level, Some(ThinkingLevel::Low));
}

#[tokio::test]
async fn applies_transform_context_before_convert_to_llm() {
    let stream = ScriptedStream::new(|_, _| text_reply("ok"));
    let hooks = TestHooks {
        transform: Some(Box::new(|messages| {
            messages.into_iter().rev().take(1).collect()
        })),
        convert: Some(Box::new(|messages| {
            // transform ran first, so a single message remains
            assert_eq!(messages.len(), 1);
            messages
        })),
        ..TestHooks::default()
    };
    let mut context_messages = vec![user("old one"), user("old two")];
    context_messages.push(user("newest"));
    let config = AgentLoopConfig::new(model(), Arc::new(hooks));
    let (sink, _) = collecting_sink();
    run_agent_loop(
        vec![user("prompt")],
        AgentContext {
            messages: context_messages,
            tools: vec![],
        },
        config,
        sink,
        &CancellationToken::new(),
        &stream,
    )
    .await;
    assert_eq!(user_texts(&stream.requests()[0].messages), ["prompt"]);
}

#[tokio::test]
async fn handles_tool_calls_and_results_with_usage_overrides() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = {
        let executed = executed.clone();
        fn_tool("echo", value_schema(), move |_, params| {
            let executed = executed.clone();
            async move {
                executed
                    .lock()
                    .unwrap()
                    .push(params["value"].as_str().unwrap().to_owned());
                Ok(AgentToolResult {
                    usage: Some(json!({"input": 1})),
                    ..AgentToolResult::text("echoed")
                })
            }
        })
    };
    let observed = Arc::new(Mutex::new(None));
    let hooks = TestHooks {
        after: Some({
            let observed = observed.clone();
            Box::new(move |result| {
                *observed.lock().unwrap() = result.usage.clone();
                Some(AfterToolCallResult {
                    usage: Some(json!({"input": 5})),
                    ..AfterToolCallResult::default()
                })
            })
        }),
        ..TestHooks::default()
    };
    let stream = then_done(tool_use(vec![echo_call("tool-1", "hello")]));
    let run = run(
        vec![user("echo something")],
        vec![tool],
        hooks,
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    assert_eq!(*executed.lock().unwrap(), ["hello"]);
    assert_eq!(*observed.lock().unwrap(), Some(json!({"input": 1})));
    let results = tool_results(&run.messages);
    assert_eq!(results[0].usage, Some(json!({"input": 5})));
    assert!(run.events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolExecutionEnd {
            is_error: false,
            ..
        }
    )));
}

#[tokio::test]
async fn records_execute_duration_excluding_hooks() {
    let tool = fn_tool("echo", value_schema(), |_, params| async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        Ok(AgentToolResult::text(
            params["value"].as_str().unwrap().to_owned(),
        ))
    });
    let hooks = TestHooks {
        before_delay: Some(Duration::from_millis(100)),
        before: Some(Box::new(|(call, args)| {
            let block = (call.id == "blocked").then(|| BeforeToolCallResult {
                block: true,
                reason: Some("no".into()),
                terminate: false,
            });
            (block, args)
        })),
        ..TestHooks::default()
    };
    let stream = then_done(tool_use(vec![
        echo_call("ran", "a"),
        echo_call("blocked", "b"),
    ]));
    let run = run(
        vec![user("go")],
        vec![tool],
        hooks,
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    let results = tool_results(&run.messages);
    let ran = results[0].duration_ms.expect("ran has a duration");
    assert!(
        (25..100).contains(&ran),
        "duration {ran}ms should exclude the 100ms hook"
    );
    assert!(results[1].is_error);
    assert_eq!(results[1].duration_ms, None);
}

#[tokio::test]
async fn does_not_execute_tool_calls_from_a_length_truncated_message() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let stream = then_done(assistant(
        vec![echo_call("tool-1", "hello")],
        StopReason::Length,
    ));
    let run = run(
        vec![user("go")],
        vec![echo_tool(executed.clone())],
        TestHooks::default(),
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    assert!(executed.lock().unwrap().is_empty());
    let result = tool_results(&run.messages)[0];
    assert!(result.is_error);
    assert!(matches!(
        &result.content[0],
        ContentBlock::Text { text } if text.contains("hit the output token limit")
    ));
    assert_eq!(stream.calls(), 2);
}

#[tokio::test]
async fn executes_mutated_before_tool_call_args_without_revalidation() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let tool = {
        let executed = executed.clone();
        fn_tool("echo", value_schema(), move |_, params| {
            let executed = executed.clone();
            async move {
                executed.lock().unwrap().push(params["value"].clone());
                Ok(AgentToolResult::text("ok"))
            }
        })
    };
    let hooks = TestHooks {
        before: Some(Box::new(|(_, mut args)| {
            args["value"] = json!(123);
            (None, args)
        })),
        ..TestHooks::default()
    };
    let stream = then_done(tool_use(vec![echo_call("tool-1", "hello")]));
    run(
        vec![user("go")],
        vec![tool],
        hooks,
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;
    assert_eq!(*executed.lock().unwrap(), [json!(123)]);
}

#[tokio::test]
async fn prepares_tool_arguments_before_validation() {
    fn prepare(args: Value) -> Value {
        match (args.get("oldText"), args.get("newText")) {
            (Some(old), Some(new)) => json!({"edits":[{"oldText": old, "newText": new}]}),
            _ => args,
        }
    }
    let executed = Arc::new(Mutex::new(Vec::new()));
    let mut tool = {
        let executed = executed.clone();
        fn_tool(
            "edit",
            json!({"type":"object","properties":{"edits":{"type":"array","items":{
                "type":"object",
                "properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},
                "required":["oldText","newText"]}}},"required":["edits"]}),
            move |_, params| {
                let executed = executed.clone();
                async move {
                    executed.lock().unwrap().push(params["edits"].clone());
                    Ok(AgentToolResult::text("edited"))
                }
            },
        )
    };
    tool.prepare = Some(prepare);
    let stream = then_done(tool_use(vec![call(
        "tool-1",
        "edit",
        json!({"oldText":"before","newText":"after"}),
    )]));
    run(
        vec![user("edit")],
        vec![tool],
        TestHooks::default(),
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;
    assert_eq!(
        *executed.lock().unwrap(),
        [json!([{"oldText":"before","newText":"after"}])]
    );
}

#[tokio::test]
async fn parallel_tools_end_in_completion_order_but_results_keep_source_order() {
    let release_first = Arc::new(Notify::new());
    let first_resolved = Arc::new(AtomicUsize::new(0));
    let parallel_observed = Arc::new(AtomicUsize::new(0));
    let tool = {
        let (release, resolved, observed) = (
            release_first.clone(),
            first_resolved.clone(),
            parallel_observed.clone(),
        );
        fn_tool("echo", value_schema(), move |_, params| {
            let (release, resolved, observed) =
                (release.clone(), resolved.clone(), observed.clone());
            async move {
                let value = params["value"].as_str().unwrap().to_owned();
                if value == "first" {
                    release.notified().await;
                    resolved.store(1, Ordering::SeqCst);
                } else if resolved.load(Ordering::SeqCst) == 0 {
                    observed.store(1, Ordering::SeqCst);
                    release.notify_one();
                }
                Ok(AgentToolResult::text(format!("echoed: {value}")))
            }
        })
    };
    let stream = then_done(tool_use(vec![
        echo_call("tool-1", "first"),
        echo_call("tool-2", "second"),
    ]));
    let run = run(
        vec![user("echo both")],
        vec![tool],
        TestHooks::default(),
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    let end_ids = run
        .events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let result_ids = run
        .events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd {
                message: AgentMessage::ToolResult(result),
            } => Some(result.tool_call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let turn_ids = run
        .events
        .iter()
        .find_map(|event| match event {
            AgentEvent::TurnEnd { tool_results, .. } if !tool_results.is_empty() => Some(
                tool_results
                    .iter()
                    .map(|r| r.tool_call_id.as_str())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .unwrap();

    assert_eq!(parallel_observed.load(Ordering::SeqCst), 1);
    assert_eq!(end_ids, ["tool-2", "tool-1"]);
    assert_eq!(result_ids, ["tool-1", "tool-2"]);
    assert_eq!(turn_ids, ["tool-1", "tool-2"]);
}

#[tokio::test]
async fn injects_steering_after_all_tool_calls_complete() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let delivered = Arc::new(Mutex::new(false));
    let hooks = TestHooks {
        steering: Some({
            let (executed, delivered) = (executed.clone(), delivered.clone());
            Box::new(move |()| {
                let mut delivered = delivered.lock().unwrap();
                if !executed.lock().unwrap().is_empty() && !*delivered {
                    *delivered = true;
                    return vec![user("interrupt")];
                }
                Vec::new()
            })
        }),
        ..TestHooks::default()
    };
    let stream = then_done(tool_use(vec![
        echo_call("tool-1", "first"),
        echo_call("tool-2", "second"),
    ]));
    let run = run(
        vec![user("start")],
        vec![echo_tool(executed.clone())],
        hooks,
        ToolExecutionMode::Sequential,
        &stream,
    )
    .await;

    assert_eq!(*executed.lock().unwrap(), ["first", "second"]);
    let sequence = run
        .events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageStart {
                message: AgentMessage::ToolResult(r),
            } => Some(format!("tool:{}", r.tool_call_id)),
            AgentEvent::MessageStart {
                message: AgentMessage::User(_),
            } => None,
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(sequence, ["tool:tool-1", "tool:tool-2"]);
    // The interrupt reached the second request after both tool results.
    assert!(user_texts(&stream.requests()[1].messages).contains(&"interrupt".to_string()));
    let requests = stream.requests();
    let order = roles(&requests[1].messages);
    assert_eq!(
        order,
        [
            "system",
            "user",
            "assistant",
            "toolResult",
            "toolResult",
            "user"
        ]
    );
}

/// Two echo calls; the tool records whether the calls overlapped.
fn overlap_probe(mode: Option<ToolExecutionMode>) -> (FnTool, Arc<AtomicUsize>) {
    let overlapped = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let (probe, flight) = (overlapped.clone(), in_flight);
    let mut tool = fn_tool("echo", value_schema(), move |_, params| {
        let (probe, flight) = (probe.clone(), flight.clone());
        async move {
            if flight.fetch_add(1, Ordering::SeqCst) > 0 {
                probe.store(1, Ordering::SeqCst);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
            flight.fetch_sub(1, Ordering::SeqCst);
            Ok(AgentToolResult::text(
                params["value"].as_str().unwrap().to_owned(),
            ))
        }
    });
    tool.mode = mode;
    (tool, overlapped)
}

#[tokio::test]
async fn sequential_tool_mode_overrides_default_parallel_config() {
    let (tool, overlapped) = overlap_probe(Some(ToolExecutionMode::Sequential));
    let stream = then_done(tool_use(vec![
        echo_call("tool-1", "first"),
        echo_call("tool-2", "second"),
    ]));
    run(
        vec![user("go")],
        vec![tool],
        TestHooks::default(),
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;
    assert_eq!(overlapped.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn one_sequential_tool_forces_the_whole_batch_sequential() {
    let (sequential, overlapped) = overlap_probe(Some(ToolExecutionMode::Sequential));
    let mut parallel = fn_tool("other", value_schema(), |_, _| async {
        Ok(AgentToolResult::text("x"))
    });
    parallel.mode = Some(ToolExecutionMode::Parallel);
    let stream = then_done(tool_use(vec![
        echo_call("tool-1", "first"),
        call("tool-2", "other", json!({"value":"x"})),
    ]));
    run(
        vec![user("go")],
        vec![sequential, parallel],
        TestHooks::default(),
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;
    assert_eq!(overlapped.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn parallel_mode_tools_overlap() {
    let (tool, overlapped) = overlap_probe(Some(ToolExecutionMode::Parallel));
    let stream = then_done(tool_use(vec![
        echo_call("tool-1", "first"),
        echo_call("tool-2", "second"),
    ]));
    run(
        vec![user("go")],
        vec![tool],
        TestHooks::default(),
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;
    assert_eq!(overlapped.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn finish_turn_runs_after_tool_results_and_before_turn_end() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let tool = fn_tool("echo", value_schema(), |_, _| async {
        Ok(AgentToolResult {
            terminate: true,
            ..AgentToolResult::text("hello")
        })
    });
    let hooks = TestHooks {
        finish_turn: Some({
            let order = order.clone();
            Box::new(move |turn| {
                order.lock().unwrap().push("finishTurn".to_string());
                assert_eq!(turn.tool_results.len(), 1);
                assert_eq!(turn.context.messages.last().unwrap().role(), "toolResult");
                None
            })
        }),
        ..TestHooks::default()
    };
    let stream = then_done(tool_use(vec![echo_call("tool-1", "hello")]));
    let run = run(
        vec![user("echo")],
        vec![tool],
        hooks,
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;
    let mut sequence = Vec::new();
    for event in &run.events {
        match event {
            AgentEvent::MessageEnd { message } => {
                sequence.push(format!("message_end:{}", message.role()))
            }
            AgentEvent::TurnEnd { .. } => sequence.push("turn_end".into()),
            _ => {}
        }
        if sequence
            .last()
            .is_some_and(|last| last == "message_end:toolResult")
        {
            sequence.push(order.lock().unwrap().last().cloned().unwrap_or_default());
        }
    }
    let tail = &sequence[sequence.len() - 3..];
    assert_eq!(tail, ["message_end:toolResult", "finishTurn", "turn_end"]);
}

#[tokio::test]
async fn finish_turn_runs_for_error_and_aborted_without_changing_the_hard_exit() {
    for reason in [StopReason::Error, StopReason::Aborted] {
        let finish_calls = Arc::new(AtomicUsize::new(0));
        let steering_polls = Arc::new(AtomicUsize::new(0));
        let follow_up_polls = Arc::new(AtomicUsize::new(0));
        let hooks = TestHooks {
            finish_turn: Some({
                let finish_calls = finish_calls.clone();
                Box::new(move |turn| {
                    assert_eq!(turn.message.stop_reason, reason);
                    finish_calls.fetch_add(1, Ordering::SeqCst);
                    Some(AgentTurnDecision::Continue)
                })
            }),
            steering: Some({
                let polls = steering_polls.clone();
                Box::new(move |()| {
                    polls.fetch_add(1, Ordering::SeqCst);
                    Vec::new()
                })
            }),
            follow_up: Some({
                let polls = follow_up_polls.clone();
                Box::new(move |()| {
                    polls.fetch_add(1, Ordering::SeqCst);
                    vec![user("queued")]
                })
            }),
            ..TestHooks::default()
        };
        let stream = ScriptedStream::new(move |_, _| AssistantMessage {
            error_message: Some("failed".into()),
            ..assistant(vec![], reason)
        });
        let run = run(
            vec![user("run")],
            vec![],
            hooks,
            ToolExecutionMode::Parallel,
            &stream,
        )
        .await;

        assert_eq!(finish_calls.load(Ordering::SeqCst), 1);
        assert_eq!(stream.calls(), 1);
        assert_eq!(steering_polls.load(Ordering::SeqCst), 1);
        assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
        assert!(matches!(
            run.events.last(),
            Some(AgentEvent::AgentEnd { .. })
        ));
    }
}

#[tokio::test]
async fn finish_turn_end_skips_queue_polling_and_next_turn_preparation() {
    let steering_polls = Arc::new(AtomicUsize::new(0));
    let follow_up_polls = Arc::new(AtomicUsize::new(0));
    let prepare_calls = Arc::new(AtomicUsize::new(0));
    let hooks = TestHooks {
        finish_turn: Some(Box::new(|_| Some(AgentTurnDecision::End))),
        prepare_next_turn: Some({
            let calls = prepare_calls.clone();
            Box::new(move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                None
            })
        }),
        steering: Some({
            let polls = steering_polls.clone();
            Box::new(move |()| {
                polls.fetch_add(1, Ordering::SeqCst);
                Vec::new()
            })
        }),
        follow_up: Some({
            let polls = follow_up_polls.clone();
            Box::new(move |()| {
                polls.fetch_add(1, Ordering::SeqCst);
                vec![user("queued")]
            })
        }),
        ..TestHooks::default()
    };
    let stream = ScriptedStream::new(|_, _| tool_use(vec![call("tool-1", "noop", json!({}))]));
    let run = run(
        vec![user("run")],
        vec![noop_tool()],
        hooks,
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    assert_eq!(stream.calls(), 1);
    assert_eq!(steering_polls.load(Ordering::SeqCst), 1);
    assert_eq!(follow_up_polls.load(Ordering::SeqCst), 0);
    assert_eq!(prepare_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        roles(&run.messages),
        ["system", "user", "assistant", "toolResult"]
    );
    assert_eq!(
        event_names(&run.events),
        [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "tool_execution_start",
            "tool_execution_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end"
        ]
    );
}

fn continue_once() -> TestHooks {
    let finish_calls = AtomicUsize::new(0);
    TestHooks {
        finish_turn: Some(Box::new(move |_| {
            (finish_calls.fetch_add(1, Ordering::SeqCst) == 0)
                .then_some(AgentTurnDecision::Continue)
        })),
        ..TestHooks::default()
    }
}

#[tokio::test]
async fn continue_decision_makes_exactly_one_context_only_request() {
    let stream = ScriptedStream::new(|_, _| text_reply("response"));
    run(
        vec![user("run")],
        vec![],
        continue_once(),
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;
    assert_eq!(stream.calls(), 2);
}

#[tokio::test]
async fn natural_tool_result_request_satisfies_continuation() {
    let stream = then_done(tool_use(vec![call("tool-1", "noop", json!({}))]));
    run(
        vec![user("run")],
        vec![noop_tool()],
        continue_once(),
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;
    assert_eq!(stream.calls(), 2);
}

#[tokio::test]
async fn natural_steering_or_follow_up_satisfies_continuation() {
    for kind in ["steering", "follow-up"] {
        let steering_polls = Arc::new(AtomicUsize::new(0));
        let follow_up_delivered = Arc::new(Mutex::new(false));
        let mut hooks = continue_once();
        hooks.steering = Some({
            let polls = steering_polls.clone();
            Box::new(move |()| {
                let poll = polls.fetch_add(1, Ordering::SeqCst) + 1;
                if kind == "steering" && poll == 2 {
                    vec![user(kind)]
                } else {
                    Vec::new()
                }
            })
        });
        hooks.follow_up = Some({
            let delivered = follow_up_delivered.clone();
            Box::new(move |()| {
                let mut delivered = delivered.lock().unwrap();
                if kind != "follow-up" || *delivered {
                    return Vec::new();
                }
                *delivered = true;
                vec![user(kind)]
            })
        });
        let stream = ScriptedStream::new(|_, _| text_reply("done"));
        run(
            vec![user("run")],
            vec![],
            hooks,
            ToolExecutionMode::Parallel,
            &stream,
        )
        .await;

        assert_eq!(stream.calls(), 2, "{kind}");
        assert!(user_texts(&stream.requests()[1].messages).contains(&kind.to_string()));
    }
}

#[tokio::test]
async fn prepare_request_runs_after_pending_messages_and_replaces_state() {
    let replacement = Model {
        id: "replacement".into(),
        provider: "openai".into(),
    };
    let delivered = Mutex::new(false);
    let prepare_calls = Arc::new(AtomicUsize::new(0));
    let hooks = TestHooks {
        steering: Some(Box::new(move |()| {
            let mut delivered = delivered.lock().unwrap();
            if std::mem::replace(&mut *delivered, true) {
                Vec::new()
            } else {
                vec![user("steering")]
            }
        })),
        prepare_request: Some({
            let (calls, replacement) = (prepare_calls.clone(), replacement.clone());
            Box::new(move |(context, _, _)| {
                calls.fetch_add(1, Ordering::SeqCst);
                assert!(user_texts(&context.messages).contains(&"steering".to_string()));
                Some(AgentRequestUpdate {
                    context: Some(AgentContext {
                        messages: vec![user("canonical projection")],
                        tools: context.tools.clone(),
                    }),
                    model: Some(replacement.clone()),
                    thinking_level: Some(ThinkingLevel::High),
                })
            })
        }),
        ..TestHooks::default()
    };
    let stream = ScriptedStream::new(|_, _| text_reply("done"));
    run(
        vec![user("prompt")],
        vec![],
        hooks,
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    let request = &stream.requests()[0];
    assert_eq!(request.model, replacement);
    assert_eq!(user_texts(&request.messages), ["canonical projection"]);
    assert_eq!(request.reasoning, Some(ThinkingLevel::High));
    assert_eq!(prepare_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn does_not_poll_steering_after_prepare_request() {
    let queued = Arc::new(Mutex::new(Vec::<AgentMessage>::new()));
    let preparations = Arc::new(AtomicUsize::new(0));
    let polls = Arc::new(AtomicUsize::new(0));
    let hooks = TestHooks {
        steering: Some({
            let (queued, polls) = (queued.clone(), polls.clone());
            Box::new(move |()| {
                polls.fetch_add(1, Ordering::SeqCst);
                std::mem::take(&mut *queued.lock().unwrap())
            })
        }),
        prepare_request: Some({
            let (queued, preparations) = (queued.clone(), preparations.clone());
            Box::new(move |_| {
                if preparations.fetch_add(1, Ordering::SeqCst) == 0 {
                    queued.lock().unwrap().push(user("late steering"));
                }
                None
            })
        }),
        ..TestHooks::default()
    };
    let stream = ScriptedStream::new(|_, _| text_reply("done"));
    run(
        vec![user("run")],
        vec![],
        hooks,
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    let included = stream
        .requests()
        .iter()
        .map(|request| user_texts(&request.messages).contains(&"late steering".to_string()))
        .collect::<Vec<_>>();
    assert_eq!(included, [false, true]);
    assert_eq!(preparations.load(Ordering::SeqCst), 2);
    // Startup, post-turn delivery, then the final natural-stop check.
    assert_eq!(polls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn prepare_next_turn_snapshot_applies_before_continuing() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let prepare_calls = Arc::new(AtomicUsize::new(0));
    let hooks = TestHooks {
        prepare_next_turn: Some({
            let calls = prepare_calls.clone();
            Box::new(move |turn| {
                if calls.fetch_add(1, Ordering::SeqCst) > 0 {
                    return None;
                }
                Some(AgentLoopTurnUpdate {
                    context: Some(turn.context.clone()),
                    messages: vec![AgentMessage::System(SystemMessage {
                        content: "updated guidance".into(),
                        timestamp: 1,
                        ..SystemMessage::default()
                    })],
                    ..AgentLoopTurnUpdate::default()
                })
            })
        }),
        ..TestHooks::default()
    };
    let stream = then_done(tool_use(vec![echo_call("tool-1", "hello")]));
    run(
        vec![user("echo something")],
        vec![echo_tool(executed)],
        hooks,
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    assert_eq!(stream.calls(), 2);
    assert_eq!(prepare_calls.load(Ordering::SeqCst), 1);
    assert!(stream.requests()[1].messages.iter().any(|message| matches!(
        message,
        AgentMessage::System(system) if system.content == "updated guidance"
    )));
}

#[tokio::test]
async fn steering_queued_during_prepare_next_turn_joins_the_next_request() {
    let queued = Arc::new(Mutex::new(Vec::<AgentMessage>::new()));
    let hooks = TestHooks {
        prepare_next_turn: Some({
            let queued = queued.clone();
            Box::new(move |_| {
                queued.lock().unwrap().push(user("late steering"));
                None
            })
        }),
        steering: Some({
            let queued = queued.clone();
            Box::new(move |()| std::mem::take(&mut *queued.lock().unwrap()))
        }),
        ..TestHooks::default()
    };
    let stream = then_done(tool_use(vec![call("tool-1", "noop", json!({}))]));
    run(
        vec![user("run")],
        vec![noop_tool()],
        hooks,
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    assert_eq!(stream.calls(), 2);
    assert!(user_texts(&stream.requests()[1].messages).contains(&"late steering".to_string()));
}

#[tokio::test]
async fn stops_after_a_batch_when_every_result_terminates() {
    let tool = fn_tool("echo", value_schema(), |_, _| async {
        Ok(AgentToolResult {
            terminate: true,
            ..AgentToolResult::text("echoed")
        })
    });
    let stream = ScriptedStream::new(|_, _| tool_use(vec![echo_call("tool-1", "hello")]));
    let run = run(
        vec![user("go")],
        vec![tool],
        TestHooks::default(),
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    assert_eq!(stream.calls(), 1);
    assert_eq!(
        roles(&run.messages),
        ["system", "user", "assistant", "toolResult"]
    );
    assert_eq!(
        event_names(&run.events)
            .iter()
            .filter(|name| **name == "turn_end")
            .count(),
        1
    );
}

#[tokio::test]
async fn blocked_call_with_terminate_stops_the_loop_without_executing() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let hooks = TestHooks {
        before: Some(Box::new(|(_, args)| {
            (
                Some(BeforeToolCallResult {
                    block: true,
                    reason: Some("Blocked by policy".into()),
                    terminate: true,
                }),
                args,
            )
        })),
        ..TestHooks::default()
    };
    let stream = ScriptedStream::new(|_, _| tool_use(vec![echo_call("tool-1", "hello")]));
    let run = run(
        vec![user("go")],
        vec![echo_tool(executed.clone())],
        hooks,
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;

    assert!(executed.lock().unwrap().is_empty());
    assert_eq!(stream.calls(), 1);
    let result = tool_results(&run.messages)[0];
    assert!(result.is_error);
    assert!(
        result
            .content
            .contains(&ContentBlock::text("Blocked by policy"))
    );
}

#[tokio::test]
async fn mixed_batch_with_one_terminating_blocked_call_continues() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let hooks = TestHooks {
        before: Some(Box::new(|(_, args)| {
            let blocked = (args["value"] == "first").then(|| BeforeToolCallResult {
                block: true,
                reason: Some("Blocked first".into()),
                terminate: true,
            });
            (blocked, args)
        })),
        ..TestHooks::default()
    };
    let stream = then_done(tool_use(vec![
        echo_call("tool-1", "first"),
        echo_call("tool-2", "second"),
    ]));
    run(
        vec![user("both")],
        vec![echo_tool(executed.clone())],
        hooks,
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;
    assert_eq!(*executed.lock().unwrap(), ["second"]);
    assert_eq!(stream.calls(), 2);
}

#[tokio::test]
async fn parallel_batch_continues_when_not_all_results_terminate() {
    let tool = fn_tool("echo", value_schema(), |_, params| async move {
        Ok(AgentToolResult {
            terminate: params["value"] == "first",
            ..AgentToolResult::text("echoed")
        })
    });
    let stream = then_done(tool_use(vec![
        echo_call("tool-1", "first"),
        echo_call("tool-2", "second"),
    ]));
    let run = run(
        vec![user("both")],
        vec![tool],
        TestHooks::default(),
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;
    assert_eq!(stream.calls(), 2);
    assert_eq!(
        roles(&run.messages),
        [
            "system",
            "user",
            "assistant",
            "toolResult",
            "toolResult",
            "assistant"
        ]
    );
}

#[tokio::test]
async fn after_tool_call_can_mark_a_batch_as_terminating() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let hooks = TestHooks {
        after: Some(Box::new(|_| {
            Some(AfterToolCallResult {
                terminate: Some(true),
                ..AfterToolCallResult::default()
            })
        })),
        ..TestHooks::default()
    };
    let stream = ScriptedStream::new(|_, _| tool_use(vec![echo_call("tool-1", "hello")]));
    run(
        vec![user("go")],
        vec![echo_tool(executed)],
        hooks,
        ToolExecutionMode::Parallel,
        &stream,
    )
    .await;
    assert_eq!(stream.calls(), 1);
}

#[tokio::test]
async fn continue_requires_a_non_assistant_last_message() {
    let stream = ScriptedStream::new(|_, _| text_reply("unused"));
    let (sink, _) = collecting_sink();
    let config = AgentLoopConfig::new(model(), Arc::new(NoHooks));
    let empty = run_agent_loop_continue(
        AgentContext::default(),
        config.clone(),
        sink.clone(),
        &CancellationToken::new(),
        &stream,
    )
    .await;
    assert_eq!(empty.unwrap_err(), ContinueError::NoMessages);
    assert_eq!(empty_message(), "Cannot continue: no messages in context");

    let after_assistant = AgentContext {
        messages: vec![AgentMessage::Assistant(text_reply("hi"))],
        tools: vec![],
    };
    let error = run_agent_loop_continue(
        after_assistant,
        config,
        sink,
        &CancellationToken::new(),
        &stream,
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Cannot continue from message role: assistant"
    );
    assert_eq!(stream.calls(), 0);
}

fn empty_message() -> String {
    ContinueError::NoMessages.to_string()
}

#[tokio::test]
async fn continue_does_not_emit_events_for_existing_messages() {
    let stream = ScriptedStream::new(|_, _| text_reply("Response"));
    let (sink, events) = collecting_sink();
    let messages = run_agent_loop_continue(
        AgentContext {
            messages: vec![user("Hello")],
            tools: vec![],
        },
        AgentLoopConfig::new(model(), Arc::new(NoHooks)),
        sink,
        &CancellationToken::new(),
        &stream,
    )
    .await
    .unwrap();

    assert_eq!(roles(&messages), ["assistant"]);
    let message_ends = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| matches!(event, AgentEvent::MessageEnd { .. }))
        .count();
    assert_eq!(message_ends, 1);
}

fn run_tool_options<'a>(
    tools: &'a [SharedTool],
    assistant: &'a AssistantMessage,
    context: &'a AgentContext,
    config: &'a AgentLoopConfig,
    signal: &'a CancellationToken,
    on_update: Option<&'a (dyn Fn(AgentToolResult) + Send + Sync)>,
) -> RunToolCallOptions<'a> {
    RunToolCallOptions {
        tools,
        assistant_message: assistant,
        context,
        config,
        signal,
        on_update,
    }
}

fn structured_echo() -> FnTool {
    fn_tool("echo", value_schema(), |_, params| async move {
        Ok(AgentToolResult {
            structured_content: Some(json!({"value": params["value"]})),
            ..AgentToolResult::text(params["value"].as_str().unwrap().to_owned())
        })
    })
}

#[tokio::test]
async fn run_tool_call_validates_runs_hooks_and_reports_failures() {
    let hook_calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let hooks = TestHooks {
        before: Some({
            let hook_calls = hook_calls.clone();
            Box::new(move |(call, args)| {
                hook_calls
                    .lock()
                    .unwrap()
                    .push(format!("before {}", call.id));
                let blocked = (args["value"] == "blocked").then(|| BeforeToolCallResult {
                    block: true,
                    reason: Some("nope".into()),
                    terminate: false,
                });
                (blocked, args)
            })
        }),
        after: Some({
            let hook_calls = hook_calls.clone();
            Box::new(move |_| {
                hook_calls.lock().unwrap().push("after".into());
                None
            })
        }),
        ..TestHooks::default()
    };
    let failing = fn_tool(
        "failing",
        json!({"type":"object","properties":{}}),
        |_, _| async {
            Ok(AgentToolResult {
                details: json!({"partial": true}),
                is_error: true,
                ..AgentToolResult::text("bad")
            })
        },
    );
    let tools: Vec<SharedTool> = vec![Arc::new(structured_echo()), Arc::new(failing)];
    let config = AgentLoopConfig::new(model(), Arc::new(hooks));
    let assistant = assistant(vec![], StopReason::Stop);
    let context = AgentContext::default();
    let signal = CancellationToken::new();
    let updates = Mutex::new(0);
    let on_update = |_: AgentToolResult| *updates.lock().unwrap() += 1;
    let call_tool = |id: &str, name: &str, arguments: Value| {
        let call = ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        };
        let (tools, assistant, context, config, signal, on_update) =
            (&tools, &assistant, &context, &config, &signal, &on_update);
        async move {
            run_tool_call(
                &call,
                run_tool_options(tools, assistant, context, config, signal, Some(on_update)),
            )
            .await
        }
    };

    let ok = call_tool("a", "echo", json!({"value":"a"})).await;
    assert!(!ok.is_error);
    assert_eq!(ok.result.structured_content, Some(json!({"value":"a"})));
    assert!(
        call_tool("b", "echo", json!({"value":{"nested":true}}))
            .await
            .is_error
    );
    let blocked = call_tool("c", "echo", json!({"value":"blocked"})).await;
    assert!(blocked.is_error);
    assert_eq!(blocked.result.content, [ContentBlock::text("nope")]);
    let missing = call_tool("d", "missing", json!({})).await;
    assert_eq!(
        missing.result.content,
        [ContentBlock::text("Tool missing not found")]
    );
    let failed = call_tool("e", "failing", json!({})).await;
    assert!(failed.is_error);
    assert_eq!(failed.result.details, json!({"partial": true}));
    // Validation failures and unknown tools never reach the hooks.
    assert_eq!(
        *hook_calls.lock().unwrap(),
        ["before a", "after", "before c", "before e", "after"]
    );
}

#[tokio::test]
async fn after_tool_call_replaces_structured_content_or_drops_it_with_new_content() {
    let redacted = vec![ContentBlock::text("redacted")];
    let overrides = vec![
        AfterToolCallResult {
            content: Some(redacted.clone()),
            ..Default::default()
        },
        AfterToolCallResult {
            structured_content: Some(json!({"value":"replaced"})),
            ..Default::default()
        },
        AfterToolCallResult {
            content: Some(redacted),
            structured_content: Some(json!({"value":"both"})),
            ..Default::default()
        },
        AfterToolCallResult {
            details: Some(json!({"note":"kept"})),
            ..Default::default()
        },
    ];
    let mut seen = Vec::new();
    for after in overrides {
        let hooks = TestHooks {
            after: Some(Box::new(move |_| Some(after.clone()))),
            ..TestHooks::default()
        };
        let tools: Vec<SharedTool> = vec![Arc::new(structured_echo())];
        let config = AgentLoopConfig::new(model(), Arc::new(hooks));
        let assistant = assistant(vec![], StopReason::Stop);
        let context = AgentContext::default();
        let signal = CancellationToken::new();
        let call = ToolCall {
            id: "x".into(),
            name: "echo".into(),
            arguments: json!({"value":"original"}),
        };
        let outcome = run_tool_call(
            &call,
            run_tool_options(&tools, &assistant, &context, &config, &signal, None),
        )
        .await;
        seen.push(outcome.result.structured_content);
    }
    assert_eq!(
        seen,
        [
            None,
            Some(json!({"value":"replaced"})),
            Some(json!({"value":"both"})),
            Some(json!({"value":"original"}))
        ]
    );
}

#[test]
fn tool_declaration_replays_adds_changes_and_removals() {
    let declaration = |name: &str, description: &str| ToolDeclaration {
        name: name.into(),
        description: description.into(),
        parameters: json!({"type":"object"}),
        constrained_sampling: None,
    };
    let previous = vec![declaration("a", "one"), declaration("b", "two")];
    let current = vec![declaration("a", "changed"), declaration("c", "three")];
    let changes = tool_state_changes(&previous, &current);
    assert_eq!(
        changes
            .tools_added
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["a", "c"]
    );
    assert_eq!(
        changes
            .tools_removed
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    let transcript = vec![
        AgentMessage::System(SystemMessage {
            tools_added: previous,
            ..SystemMessage::default()
        }),
        AgentMessage::System(SystemMessage {
            tools_added: changes.tools_added,
            tools_removed: changes.tools_removed,
            ..SystemMessage::default()
        }),
    ];
    assert_eq!(current_tools(&transcript), current);
}

#[test]
fn validation_coerces_primitives_and_formats_pi_errors() {
    let schema = json!({
        "type":"object",
        "properties":{"count":{"type":"integer"},"flag":{"type":"boolean"},"note":{"type":"string"}},
        "required":["count"]
    });
    let coerced = validate_tool_arguments(
        "t",
        &schema,
        &json!({"count":"3","flag":"true","note":null}),
    )
    .unwrap();
    assert_eq!(coerced, json!({"count":3,"flag":true}));

    let error = validate_tool_arguments("t", &schema, &json!({})).unwrap_err();
    assert!(
        error.starts_with("Validation failed for tool \"t\":\n  - count: "),
        "{error}"
    );
    assert!(error.ends_with("Received arguments:\n{}"), "{error}");
}
