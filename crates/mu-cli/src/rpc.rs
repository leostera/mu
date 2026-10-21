//! `--mode rpc`: Pi's JSONL command protocol over stdin/stdout (a subset).
//!
//! Commands are read one per line and handled sequentially, so `abort`/`steer` cannot interrupt
//! a running `prompt` yet. Supported: `prompt`, `get_state`, `set_model`, `get_available_models`,
//! `set_thinking_level`, `cycle_thinking_level`, `new_session`, `get_session_stats`,
//! `get_last_assistant_text`.

use std::io::{BufRead, Write};

use mu_core::MuAgent;
use serde_json::{Value, json};

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

/// Handle one parsed command; `prompt` streams its events before returning.
pub async fn handle(agent: &mut MuAgent, command: &Value) -> Value {
    let id = command.get("id").cloned();
    let kind = command["type"].as_str().unwrap_or_default().to_owned();
    let outcome = match kind.as_str() {
        "prompt" => match command["message"].as_str() {
            Some(message) => {
                emit(&response(
                    &id,
                    "prompt",
                    Ok(Some(json!({"disposition":"started"}))),
                ));
                match agent.submit(message.to_owned(), &mut |_| {}).await {
                    Ok(()) => return json!({"type":"prompt_done"}),
                    Err(error) => Err(error.to_string()),
                }
            }
            None => Err("prompt requires a string `message`".into()),
        },
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
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(&line) {
            Ok(command) => {
                let reply = handle(&mut agent, &command).await;
                if reply["type"] != "prompt_done" {
                    emit(&reply);
                }
            }
            Err(error) => emit(&json!({
                "type":"response","command":"parse","success":false,
                "error": format!("Failed to parse command: {error}")
            })),
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
        let nothing = handle(&mut agent, &json!({"type":"compact"})).await;
        assert_eq!(nothing["error"], "Nothing to compact");
        let missing = handle(&mut agent, &json!({"type":"prompt"})).await;
        assert_eq!(missing["success"], false);
    }
}
