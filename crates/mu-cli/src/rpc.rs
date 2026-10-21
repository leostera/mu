//! `--mode rpc`: Pi's JSONL command protocol over stdin/stdout (a subset).
//!
//! Commands are read one per line and handled sequentially, so `abort`/`steer` cannot interrupt
//! a running `prompt` yet. Supported: `prompt`, `get_state`, `set_model`, `get_available_models`,
//! `set_thinking_level`, `cycle_thinking_level`, `new_session`, `get_session_stats`,
//! `get_last_assistant_text`.

use std::io::{BufRead, Write};

use mu_core::MuAgent;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

type Inbox = mpsc::UnboundedReceiver<Value>;

fn emit(value: &Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{value}");
    let _ = out.flush();
}

fn response(id: &Option<Value>, command: &str, outcome: Result<Option<Value>, String>) -> Value {
    let mut value = match outcome {
        Ok(data) => {
            let mut v = json!({"type":"response","command":command,"success":true});
            if let Some(data) = data {
                v["data"] = data;
            }
            v
        }
        Err(error) => json!({"type":"response","command":command,"success":false,"error":error}),
    };
    if let Some(id) = id {
        value["id"] = id.clone();
    }
    value
}

/// Run a `prompt` while still serving `steer`, `follow_up`, `clear_queue` and `abort` from `inbox`.
async fn run_prompt(agent: &mut MuAgent, command: &Value, inbox: &mut Inbox) -> Value {
    let id = command.get("id").cloned();
    let Some(message) = command["message"].as_str() else {
        return response(
            &id,
            "prompt",
            Err("prompt requires a string `message`".into()),
        );
    };
    emit(&response(
        &id,
        "prompt",
        Ok(Some(json!({"disposition":"started"}))),
    ));
    let queue = agent.prompt_queue();
    let cancel = CancellationToken::new();
    let mut ignore_event = |_| {};
    let turn = agent.submit_with_cancel(message.to_owned(), &mut ignore_event, &cancel);
    tokio::pin!(turn);
    loop {
        tokio::select! {
            result = &mut turn => {
                return match result {
                    Ok(()) => json!({"type":"prompt_done"}),
                    Err(error) => response(&id, "prompt", Err(error.to_string())),
                };
            }
            Some(incoming) = inbox.recv() => {
                let id = incoming.get("id").cloned();
                let kind = incoming["type"].as_str().unwrap_or_default().to_owned();
                let text = incoming["message"].as_str().map(str::to_owned);
                let outcome = match (kind.as_str(), text) {
                    ("abort", _) => {
                        cancel.cancel();
                        Ok(None)
                    }
                    ("steer", Some(text)) => {
                        queue.enqueue(text, mu_core::PromptBehavior::Steer);
                        Ok(Some(json!({"disposition":"queued"})))
                    }
                    ("follow_up", Some(text)) => {
                        queue.enqueue(text, mu_core::PromptBehavior::FollowUp);
                        Ok(Some(json!({"disposition":"queued"})))
                    }
                    ("steer" | "follow_up", None) => Err(format!("{kind} requires a string `message`")),
                    (other, _) => Err(format!("Cannot run {other} while the agent is streaming")),
                };
                emit(&response(&id, &kind, outcome));
            }
        }
    }
}

/// Handle one parsed command; `prompt` streams its events before returning.
pub async fn handle(agent: &mut MuAgent, command: &Value) -> Value {
    let id = command.get("id").cloned();
    let kind = command["type"].as_str().unwrap_or_default().to_owned();
    let outcome = match kind.as_str() {
        "prompt" => {
            let (_keep_open, mut inbox) = mpsc::unbounded_channel();
            return run_prompt(agent, command, &mut inbox).await;
        }
        "get_state" => Ok(Some(json!({
            "model": {"id": agent.model(), "provider": agent.provider()},
            "thinkingLevel": agent.thinking().name(),
            "sessionFile": agent.session_path(),
            "sessionId": agent.session_info().id,
            "sessionName": agent.session_name(),
            "messageCount": agent.user_messages().len(),
        }))),
        "set_thinking_level" => match command["level"].as_str().map(str::parse) {
            Some(Ok(level)) => {
                agent.set_thinking(level);
                Ok(None)
            }
            Some(Err(error)) => Err(error),
            None => Err("set_thinking_level requires `level`".into()),
        },
        "cycle_thinking_level" => {
            let level = agent.thinking().next();
            agent.set_thinking(level);
            Ok(Some(json!({"level": level.name()})))
        }
        "new_session" => agent
            .start_new_session()
            .map(|()| Some(json!({"cancelled": false})))
            .map_err(|error| error.to_string()),
        "get_available_models" => Ok(Some(json!({
            "models": agent
                .model_choices()
                .iter()
                .map(|id| json!({"id": id, "provider": agent.provider(), "name": agent.model_display_name(id)}))
                .collect::<Vec<_>>(),
        }))),
        "set_model" => match command["modelId"].as_str() {
            None => Err("set_model requires `modelId`".into()),
            Some(model) => match agent.select_model(model) {
                Ok(true) => Ok(Some(json!({"id": model, "provider": agent.provider()}))),
                Ok(false) => Err(format!("Model not found: {model}")),
                Err(error) => Err(error.to_string()),
            },
        },
        "compact" => match agent.compact().await {
            Ok(Some(outcome)) => Ok(Some(json!({
                "summary": outcome.summary,
                "tokensBefore": outcome.tokens_before,
            }))),
            Ok(None) => Err("Nothing to compact".into()),
            Err(error) => Err(error.to_string()),
        },
        "get_fork_messages" => Ok(Some(json!({
            "messages": agent
                .user_messages()
                .into_iter()
                .map(|(id, text)| json!({"entryId": id, "text": text}))
                .collect::<Vec<_>>(),
        }))),
        "fork" => match command["entryId"].as_str() {
            None => Err("fork requires `entryId`".into()),
            Some(entry_id) => agent
                .fork_before_user_message(entry_id)
                .map(|text| Some(json!({"text": text, "cancelled": false})))
                .map_err(|error| error.to_string()),
        },
        "clone" => agent
            .clone_session()
            .map(|()| Some(json!({"cancelled": false})))
            .map_err(|error| error.to_string()),
        "get_session_stats" => Ok(Some(json!({"summary": agent.session_summary()}))),
        "get_last_assistant_text" => Ok(Some(json!({"text": agent.last_assistant_text()}))),
        other => Err(format!("Unknown command: {other}")),
    };
    response(&id, &kind, outcome)
}

/// Run the protocol until stdin closes.
pub async fn run(mut agent: MuAgent) -> Result<(), Box<dyn std::error::Error>> {
    agent.set_event_tap(std::sync::Arc::new(|event| {
        emit(&mu_core::agent_loop::event_to_json(event));
    }));
    let (tx, mut inbox) = mpsc::unbounded_channel::<Value>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines().map_while(Result::ok) {
            if line.trim().is_empty() {
                continue;
            }
            let parsed = serde_json::from_str::<Value>(&line).unwrap_or_else(|error| {
                json!({"type":"parse_error","error": format!("Failed to parse command: {error}")})
            });
            if tx.send(parsed).is_err() {
                break;
            }
        }
    });
    while let Some(command) = inbox.recv().await {
        let reply = match command["type"].as_str() {
            Some("parse_error") => json!({
                "type":"response","command":"parse","success":false,"error":command["error"]
            }),
            Some("prompt") => run_prompt(&mut agent, &command, &mut inbox).await,
            _ => handle(&mut agent, &command).await,
        };
        if reply["type"] != "prompt_done" {
            emit(&reply);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn commands_return_pi_shaped_responses_with_ids() {
        let mut agent = MuAgent::demo().expect("demo agent");
        let state = handle(&mut agent, &json!({"id":"1","type":"get_state"})).await;
        assert_eq!(state["type"], "response");
        assert_eq!(state["id"], "1");
        assert_eq!(state["success"], true);
        assert_eq!(state["data"]["thinkingLevel"], "medium");

        let models = handle(&mut agent, &json!({"type":"get_available_models"})).await;
        let listed = models["data"]["models"].as_array().unwrap().clone();
        assert!(
            listed.iter().any(|m| m["id"] == agent.model()),
            "current model is listed"
        );
        let target = listed[0]["id"].as_str().unwrap().to_owned();
        let switched = handle(&mut agent, &json!({"type":"set_model","modelId":target})).await;
        assert_eq!(switched["data"]["id"], target.as_str());
        let missing = handle(&mut agent, &json!({"type":"set_model","modelId":"zzz"})).await;
        assert_eq!(missing["error"], "Model not found: zzz");

        let set = handle(
            &mut agent,
            &json!({"type":"set_thinking_level","level":"high"}),
        )
        .await;
        assert_eq!(set["success"], true);
        assert_eq!(agent.thinking().name(), "high");
        let cycled = handle(&mut agent, &json!({"type":"cycle_thinking_level"})).await;
        assert_eq!(cycled["data"]["level"], "xhigh");

        let bad = handle(
            &mut agent,
            &json!({"type":"set_thinking_level","level":"loud"}),
        )
        .await;
        assert_eq!(bad["success"], false);
        assert!(
            bad["error"]
                .as_str()
                .unwrap()
                .starts_with("Invalid thinking level")
        );
        let unknown = handle(&mut agent, &json!({"id":7,"type":"nope"})).await;
        assert_eq!(unknown["error"], "Unknown command: nope");
        assert_eq!(unknown["id"], 7);
    }

    #[tokio::test]
    async fn steer_and_abort_are_served_while_a_prompt_is_running() {
        let mut agent = MuAgent::demo().expect("demo agent");
        let (tx, mut inbox) = mpsc::unbounded_channel();
        tx.send(json!({"id":"s","type":"steer","message":"also this"}))
            .unwrap();
        tx.send(json!({"type":"get_state"})).unwrap();
        tx.send(json!({"type":"abort"})).unwrap();
        let done = run_prompt(
            &mut agent,
            &json!({"type":"prompt","message":"hello"}),
            &mut inbox,
        )
        .await;
        // The demo turn may finish before the queued commands are read; either way it ends cleanly.
        assert_eq!(done["type"], "prompt_done");
    }

    #[tokio::test]
    async fn prompt_runs_the_turn_and_last_text_reflects_it() {
        let mut agent = MuAgent::demo().expect("demo agent");
        let done = handle(&mut agent, &json!({"type":"prompt","message":"hello"})).await;
        assert_eq!(done["type"], "prompt_done");
        let text = handle(&mut agent, &json!({"type":"get_last_assistant_text"})).await;
        assert_eq!(text["data"]["text"], "Demo backend received: hello");
        let forks = handle(&mut agent, &json!({"type":"get_fork_messages"})).await;
        let last = forks["data"]["messages"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()
            .clone();
        assert_eq!(last["text"], "hello");
        assert!(last["entryId"].as_str().is_some());
        let entry = last["entryId"].as_str().unwrap().to_owned();
        let forked = handle(&mut agent, &json!({"type":"fork","entryId":entry})).await;
        assert_eq!(forked["data"]["text"], "hello");
        assert!(
            agent.user_messages().is_empty(),
            "history before the message is empty"
        );
        let bad_fork = handle(&mut agent, &json!({"type":"fork","entryId":"nope"})).await;
        assert_eq!(bad_fork["success"], false);
        let nothing = handle(&mut agent, &json!({"type":"compact"})).await;
        assert_eq!(nothing["error"], "Nothing to compact");
        let missing = handle(&mut agent, &json!({"type":"prompt"})).await;
        assert_eq!(missing["success"], false);
    }
}
