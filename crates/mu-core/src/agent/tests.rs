use std::collections::HashSet;

use super::runtime::{responses_input, session_messages};
use super::{
    AgentPromptQueue, FunctionCall, PromptBehavior, QueueMode, Reply, TerminalResponse,
    configured_models, find_session_id_path, latest_session_path, load_session,
    merge_final_response_text, resolve_session_options, response_message_text,
    session_active_mcp_tools, session_codemode_store, session_preview, session_transcript,
    trusted_url, validate_session_id,
};
use super::{Config, SessionOptions, ThinkingSetting};
use mu_session::{
    AssistantToolResponse, AssistantTurnResponse, MessageUsage, SessionDocument, ToolCallRecord,
};
use serde_json::Value;
use std::path::Path;

#[test]
fn session_transcript_reconstructs_text_thinking_and_completed_tools() {
    let mut session = SessionDocument::new("/tmp/project");
    session.append_user_text("Run the command");
    session.append_assistant_tool_calls(
        "openai-responses",
        "provider",
        "model",
        AssistantToolResponse {
            thinking: "Checking first".into(),
            thinking_signature: None,
            text: "I will inspect this.".into(),
            calls: vec![ToolCallRecord {
                id: "call-1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command":"pwd"}),
            }],
            usage: MessageUsage::default(),
        },
    );
    session.append_tool_result("call-1", "bash", "success: /tmp/project".into(), true);
    session.append_assistant_response(
        "openai-responses",
        "provider",
        "model",
        AssistantTurnResponse {
            thinking: String::new(),
            thinking_signature: None,
            text: "Done.".into(),
            usage: MessageUsage::default(),
        },
    );

    let transcript = session_transcript(&session);
    assert_eq!(transcript.len(), 3);
    assert_eq!(transcript[0].role, super::SessionTranscriptRole::User);
    assert_eq!(transcript[0].text, "Run the command");
    assert_eq!(transcript[1].role, super::SessionTranscriptRole::Assistant);
    assert_eq!(transcript[1].thinking, "Checking first");
    assert_eq!(transcript[1].text, "I will inspect this.");
    assert_eq!(transcript[1].tools[0].name, "bash");
    assert_eq!(
        transcript[1].tools[0].output.as_deref(),
        Some("success: /tmp/project")
    );
    assert!(transcript[1].tools[0].is_error);
    assert_eq!(transcript[2].text, "Done.");
}

#[test]
fn session_preview_uses_the_latest_user_text_and_compacts_whitespace() {
    let mut session = SessionDocument::new("/tmp/project");
    session.append_user_text("earlier prompt");
    session.append_user_text("  latest\n prompt   text  ");
    assert_eq!(
        session_preview(&session).as_deref(),
        Some("latest prompt text")
    );

    let long_prompt = "x".repeat(200);
    session.append_user_text(&long_prompt);
    assert_eq!(
        session_preview(&session).unwrap(),
        format!("{}…", "x".repeat(160))
    );
}

#[test]
fn project_shell_command_prefix_overrides_agent_settings() {
    let root = std::env::temp_dir().join(format!("mu-shell-prefix-{}", uuid::Uuid::new_v4()));
    let agent_dir = root.join("agent");
    let project_dir = root.join("project/.pi");
    std::fs::create_dir_all(&agent_dir).unwrap();
    std::fs::create_dir_all(&project_dir).unwrap();
    crate::mcp::set_project_trust_override(None);
    std::fs::write(
        agent_dir.join("settings.json"),
        r#"{"shellCommandPrefix":"global setup","enableSkillCommands":true}"#,
    )
    .unwrap();
    std::fs::write(
            project_dir.join("settings.json"),
            r#"{"shellCommandPrefix":"project setup","shellPath":"bin/bash","enableSkillCommands":false}"#,
        )
        .unwrap();
    crate::mcp::save_project_trust(&agent_dir, &root.join("project"), true).unwrap();
    assert_eq!(
        super::shell_setting(&agent_dir, &root.join("project"), "shellCommandPrefix").as_deref(),
        Some("project setup")
    );
    assert_eq!(
        super::configured_bash_shell_path(&agent_dir, &root.join("project")).as_deref(),
        Some(root.join("project/bin/bash").to_string_lossy().as_ref())
    );
    assert!(!super::skill_commands_enabled(
        &agent_dir,
        &root.join("project")
    ));
    assert!(super::skill_commands_enabled(
        &root.join("missing-agent"),
        &root
    ));
    std::fs::remove_dir_all(root).unwrap();
    crate::mcp::set_project_trust_override(None);
}

#[test]
fn settings_can_be_written_to_the_trusted_project_scope() {
    let root =
        std::env::temp_dir().join(format!("mu-project-setting-write-{}", uuid::Uuid::new_v4()));
    let agent = scripted_agent(&root, vec![]);
    crate::mcp::set_project_trust_override(None);
    assert!(
        agent
            .toggle_setting_in_scope(crate::settings::Scope::Global, "quietStartup")
            .unwrap()
    );

    let denied =
        agent.set_setting_text_in_scope(crate::settings::Scope::Project, "tuiMode", "regular");
    assert!(denied.is_err(), "untrusted project writes must be refused");

    crate::mcp::save_project_trust(agent.agent_dir(), agent.cwd(), true).unwrap();
    assert_eq!(
        agent
            .set_setting_text_in_scope(crate::settings::Scope::Project, "tuiMode", "regular",)
            .unwrap(),
        serde_json::json!("regular")
    );
    assert!(
        !agent
            .toggle_setting_in_scope(crate::settings::Scope::Project, "quietStartup")
            .unwrap()
    );
    let settings = agent.settings();
    assert_eq!(
        settings.get_for_scope(crate::settings::Scope::Global, "quietStartup"),
        Some(serde_json::json!(true))
    );
    assert_eq!(
        settings.get_for_scope(crate::settings::Scope::Project, "quietStartup"),
        Some(serde_json::json!(false))
    );
    let project_settings = std::fs::read_to_string(root.join(".pi/settings.json")).unwrap();
    assert!(project_settings.contains("\"tuiMode\": \"regular\""));
    assert!(project_settings.contains("\"quietStartup\": false"));
    assert!(
        !std::fs::read_to_string(root.join("agent/settings.json"))
            .unwrap()
            .contains("tuiMode")
    );

    std::fs::remove_dir_all(root).unwrap();
    crate::mcp::set_project_trust_override(None);
}

#[test]
fn project_settings_are_ignored_until_the_project_is_trusted() {
    let root = std::env::temp_dir().join(format!("mu-settings-trust-{}", uuid::Uuid::new_v4()));
    let agent_dir = root.join("agent");
    let cwd = root.join("project");
    std::fs::create_dir_all(&agent_dir).unwrap();
    std::fs::create_dir_all(cwd.join(".pi")).unwrap();
    std::fs::write(
        agent_dir.join("settings.json"),
        r#"{"shellCommandPrefix":"global setup"}"#,
    )
    .unwrap();
    std::fs::write(
        cwd.join(".pi/settings.json"),
        r#"{"shellCommandPrefix":"project setup","autocompleteMaxVisible":11}"#,
    )
    .unwrap();
    crate::mcp::set_project_trust_override(None);
    crate::mcp::save_project_trust(&agent_dir, &cwd, false).unwrap();

    assert_eq!(
        super::shell_setting(&agent_dir, &cwd, "shellCommandPrefix").as_deref(),
        Some("global setup")
    );

    crate::mcp::save_project_trust(&agent_dir, &cwd, true).unwrap();
    assert_eq!(
        super::shell_setting(&agent_dir, &cwd, "shellCommandPrefix").as_deref(),
        Some("project setup")
    );
    assert_eq!(
        crate::settings::Settings::load_for_project(&agent_dir, &cwd).get("autocompleteMaxVisible"),
        Some(serde_json::json!(11))
    );

    std::fs::write(agent_dir.join("trust.json"), "not json").unwrap();
    assert_eq!(
        super::shell_setting(&agent_dir, &cwd, "shellCommandPrefix").as_deref(),
        Some("global setup"),
        "malformed trust data fails closed"
    );
    std::fs::remove_dir_all(root).unwrap();
    crate::mcp::set_project_trust_override(None);
}

#[test]
fn cloning_session_copies_the_active_branch_to_a_new_file() {
    let root = std::env::temp_dir().join(format!("mu-session-clone-{}", uuid::Uuid::new_v4()));
    let cwd = root.join("project");
    let agent_dir = root.join("agent");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&agent_dir).unwrap();
    let config = super::Config {
        agent_dir,
        cwd: cwd.clone(),
        provider: "demo".into(),
        model: "demo".into(),
        model_choices: vec!["demo".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: super::resolve_default_tools(None),
    };
    let mut agent = super::MuAgent::demo_from_config(config).unwrap();
    agent
        .session
        .append_user_text("keep <script>alert('x')</script>");
    agent.session.append_assistant_turn(
        "responses",
        "demo",
        "demo",
        "thinking <private>",
        "assistant says <hello>",
        mu_session::MessageUsage::default(),
    );
    agent.session.append_tool_call(
        "call-1",
        "read",
        serde_json::json!({"path":"a&b.txt"}),
        "responses",
        "demo",
        "demo",
    );
    agent.session.append_tool_result(
        "call-1",
        "read",
        "<script>tool output</script>".into(),
        true,
    );
    agent.save_session().unwrap();
    let original_path = agent.session_path().to_owned();
    let original_id = agent.session.header().unwrap().id.as_str().to_owned();
    let entry_count = agent.session.active_branch_entries().len();

    agent.clone_session().unwrap();

    let cloned_header = agent.session.header().unwrap();
    assert_ne!(cloned_header.id.as_str(), original_id);
    assert_eq!(
        cloned_header.parent_session.as_deref(),
        Some(original_path.to_string_lossy().as_ref())
    );
    assert_eq!(agent.session.active_branch_entries().len(), entry_count);
    assert!(agent.session_path().is_file());
    assert_ne!(agent.session_path(), original_path);

    let clone_id = cloned_header.id.as_str().to_owned();
    let html_path = root.join("export.html");
    agent.export_session(&html_path).unwrap();
    let html = std::fs::read_to_string(html_path).unwrap();
    assert!(html.contains("keep &lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;"));
    assert!(html.contains("thinking &lt;private&gt;"));
    assert!(html.contains("assistant says &lt;hello&gt;"));
    assert!(html.contains("<details class=\"error\"><summary>Tool: read"));
    assert!(html.contains("a&amp;b.txt"));
    assert!(html.contains("&lt;script&gt;tool output&lt;/script&gt;"));
    assert!(!html.contains("<script>alert('x')</script>"));
    assert!(!html.contains("<script>tool output</script>"));
    assert!(agent.export_session(&root.join("export.txt")).is_err());

    let export_path = root.join("export.jsonl");
    agent.export_session(&export_path).unwrap();
    agent.start_new_session().unwrap();
    agent.import_session(&export_path).unwrap();
    assert_eq!(agent.session.header().unwrap().id.as_str(), clone_id);
    assert_eq!(agent.session.active_branch_entries().len(), entry_count);
    assert_eq!(agent.session_path(), export_path);
    agent.name_session("  Refactor session  ").unwrap();
    assert_eq!(agent.session.session_name(), Some("Refactor session"));
    let (saved_session, _) = load_session(&export_path).unwrap();
    assert_eq!(saved_session.session_name(), Some("Refactor session"));
    assert!(agent.name_session("  ").is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn skill_commands_expand_frontmatter_body_and_arguments() {
    let root = std::env::temp_dir().join(format!("mu-skill-command-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let file_path = root.join("review/SKILL.md");
    std::fs::create_dir_all(file_path.parent().unwrap()).unwrap();
    std::fs::write(
        &file_path,
        "---\nname: review\ndescription: review code\n---\nReview this file carefully.\n",
    )
    .unwrap();
    let skills = crate::skills::SkillCatalog {
        skills: vec![crate::skills::Skill {
            name: "review".into(),
            description: "review code".into(),
            file_path: file_path.clone(),
            disable_model_invocation: true,
        }],
        diagnostics: Vec::new(),
    };

    let expanded =
        super::expand_skill_command("/skill:review  check edge cases ".into(), &skills).unwrap();
    assert_eq!(
        expanded,
        format!(
            "<skill name=\"review\" location=\"{}\">\nReferences are relative to {}.\n\nReview this file carefully.\n</skill>\n\ncheck edge cases",
            file_path.display(),
            file_path.parent().unwrap().display(),
        )
    );
    assert_eq!(
        super::expand_skill_command("/skill:missing argument".into(), &skills).unwrap(),
        "/skill:missing argument"
    );
    assert_eq!(
        super::expand_skill_command("/skill:review ".into(), &skills).unwrap(),
        expanded.strip_suffix("\n\ncheck edge cases").unwrap(),
        "a space with no remaining argument still expands the skill"
    );
    for prompt in ["/skill:review\targument", "/skill:review\u{00a0}argument"] {
        assert_eq!(
            super::expand_skill_command(prompt.into(), &skills).unwrap(),
            prompt,
            "Pi uses a literal ASCII space as the skill-argument delimiter"
        );
    }
    let mut warning = None;
    let unchanged =
        super::expand_skill_prompt("/skill:missing argument".into(), &skills, &mut |event| {
            warning = Some(event)
        });
    assert_eq!(unchanged, "/skill:missing argument");
    assert!(warning.is_none());
    std::fs::remove_file(&file_path).unwrap();
    let mut warning = None;
    assert_eq!(
        super::expand_skill_prompt("/skill:review".into(), &skills, &mut |event| {
            warning = Some(event)
        }),
        "/skill:review"
    );
    assert!(matches!(
        warning,
        Some(super::AgentEvent::SkillCommandWarning(message))
            if message.contains("Could not read skill 'review'")
    ));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn incomplete_responses_preserve_only_the_output_token_limit_as_a_retryable_result() {
    assert_eq!(TerminalResponse::Completed.error_message(), None);
    assert_eq!(
        TerminalResponse::Incomplete {
            reason: Some("max_output_tokens".into())
        }
        .error_message(),
        None
    );
    assert_eq!(
        TerminalResponse::Incomplete {
            reason: Some("content_filter".into())
        }
        .error_message()
        .as_deref(),
        Some("Response incomplete: content_filter")
    );
    assert_eq!(
        TerminalResponse::Incomplete { reason: None }
            .error_message()
            .as_deref(),
        Some("Response incomplete without a provider reason")
    );
}

#[test]
fn mcp_tool_summaries_show_first_description_line() {
    assert_eq!(
        super::format_mcp_tool_summary(&serde_json::json!({
            "name":"mcp__fs__read",
            "description":"Read a file.\nReturns the content."
        }))
        .as_deref(),
        Some("mcp__fs__read: Read a file.")
    );
    assert_eq!(
        super::format_mcp_tool_summary(&serde_json::json!({"name":"mcp__fs__empty"})).as_deref(),
        Some("mcp__fs__empty")
    );
    assert_eq!(
        super::format_mcp_tool_summary(&serde_json::json!({"description":"missing name"})),
        None
    );
}

#[test]
fn responses_failure_events_preserve_provider_error_details() {
    assert_eq!(
        super::response_failed_error(&serde_json::json!({
            "response":{"error":{"code":"server_error","message":"unavailable"}}
        }))
        .to_string(),
        "provider: server_error: unavailable"
    );
    assert_eq!(
        super::response_failed_error(&serde_json::json!({
            "response":{"incomplete_details":{"reason":"content_filter"}}
        }))
        .to_string(),
        "provider: incomplete: content_filter"
    );
    assert_eq!(
        super::response_failed_error(&serde_json::json!({"response":{}})).to_string(),
        "provider: Unknown error (no error details in response)"
    );
    assert_eq!(
        super::response_error_event(&serde_json::json!({
            "code":"rate_limit_exceeded","message":"slow down"
        }))
        .to_string(),
        "provider: Error Code rate_limit_exceeded: slow down"
    );
    assert_eq!(
        super::response_error_event(&serde_json::json!({})).to_string(),
        "provider: Error Code undefined: undefined"
    );
}

#[test]
fn responses_final_message_recovers_text_and_refusals_missing_from_deltas() {
    let output = vec![serde_json::json!({
        "type":"message",
        "content":[
            {"type":"output_text","text":"partial "},
            {"type":"refusal","refusal":"answer"}
        ]
    })];
    assert_eq!(response_message_text(&output), "partial answer");

    let mut streamed = String::new();
    assert_eq!(
        merge_final_response_text(&mut streamed, &output).as_deref(),
        Some("partial answer")
    );
    assert_eq!(streamed, "partial answer");
    assert_eq!(merge_final_response_text(&mut streamed, &output), None);
}

#[test]
fn responses_final_message_appends_only_the_unstreamed_suffix() {
    let output = vec![serde_json::json!({
        "type":"message",
        "content":[{"type":"refusal","refusal":"Refused."}]
    })];
    let mut streamed = "Ref".to_owned();
    assert_eq!(
        merge_final_response_text(&mut streamed, &output).as_deref(),
        Some("used.")
    );
    assert_eq!(streamed, "Refused.");
}

#[test]
fn latest_session_for_a_working_directory_uses_mtime_and_validates_session_cwd() {
    let root = std::env::temp_dir().join(format!("mu-session-list-{}", uuid::Uuid::new_v4()));
    let session_dir = root.join("sessions/--tmp-project--");
    std::fs::create_dir_all(&session_dir).unwrap();
    let header = |cwd: &str| {
        format!(
            "{}\n",
            serde_json::json!({
                "type":"session",
                "id":"session-id",
                "timestamp":"2025-01-01T00:00:00.000Z",
                "cwd":cwd
            })
        )
    };
    let old_name = session_dir.join("2024-01-01_old.jsonl");
    let lexically_newer_name = session_dir.join("2025-01-01_new.jsonl");
    let other_cwd_name = session_dir.join("2026-01-01-other.jsonl");
    std::fs::write(&old_name, header("/tmp/project")).unwrap();
    std::fs::write(&lexically_newer_name, header("/tmp/project")).unwrap();
    std::fs::write(&other_cwd_name, header("/tmp/other-project")).unwrap();
    std::fs::write(session_dir.join("invalid.jsonl"), "not json").unwrap();
    std::fs::create_dir(session_dir.join("2027-01-01-directory.jsonl")).unwrap();
    let set_mtime = |path: &Path, seconds| {
        std::fs::File::open(path)
            .unwrap()
            .set_times(
                std::fs::FileTimes::new()
                    .set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(seconds)),
            )
            .unwrap();
    };
    set_mtime(&old_name, 300);
    set_mtime(&lexically_newer_name, 200);
    set_mtime(&other_cwd_name, 400);

    assert_eq!(
        latest_session_path(&root, Path::new("/tmp/project"), None),
        Some(old_name)
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn exact_session_id_lookup_honors_custom_directory_and_project_cwd() {
    let root = std::env::temp_dir().join(format!("mu-session-id-{}", uuid::Uuid::new_v4()));
    let session_dir = root.join("custom-sessions");
    std::fs::create_dir_all(&session_dir).unwrap();
    for (id, cwd, file_name) in [
        ("wanted-id", "/tmp/project", "project.jsonl"),
        ("wanted-id-prefix", "/tmp/project", "prefix.jsonl"),
        ("foreign-id", "/tmp/other-project", "foreign.jsonl"),
    ] {
        SessionDocument::new_with_id(cwd, id)
            .write_atomic(&session_dir.join(file_name))
            .unwrap();
    }

    assert_eq!(
        find_session_id_path(
            &root,
            Path::new("/tmp/project"),
            Some(&session_dir),
            "wanted-id",
            false
        ),
        Some(session_dir.join("project.jsonl"))
    );
    assert!(
        find_session_id_path(
            &root,
            Path::new("/tmp/project"),
            Some(&session_dir),
            "wanted",
            false
        )
        .is_none()
    );
    assert!(validate_session_id("Session_1.test-2").is_ok());
    assert!(validate_session_id("-invalid").is_err());
    assert!(validate_session_id("invalid id").is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn session_id_options_open_prefixes_or_create_the_requested_id_in_custom_directory() {
    let root = std::env::temp_dir().join(format!("mu-session-options-{}", uuid::Uuid::new_v4()));
    let cwd = root.join("project");
    let session_dir = root.join("session-files");
    std::fs::create_dir_all(&cwd).unwrap();
    let config = Config {
        agent_dir: root.join("agent"),
        cwd: cwd.clone(),
        provider: "test-provider".into(),
        model: "test-model".into(),
        model_choices: vec!["test-model".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: super::resolve_default_tools(None),
    };

    let (new_path, new_session, _) = resolve_session_options(
        &config,
        SessionOptions {
            directory: Some(session_dir.clone()),
            id: Some("requested-id".into()),
            name: Some("  Feature\nname  ".into()),
            ..SessionOptions::default()
        },
    )
    .unwrap();
    assert_eq!(new_path.parent(), Some(session_dir.as_path()));
    assert_eq!(new_session.header().unwrap().id.as_str(), "requested-id");
    assert_eq!(new_session.session_name(), Some("Feature name"));
    new_session.write_atomic(&new_path).unwrap();

    let (opened_path, opened_session, _) = resolve_session_options(
        &config,
        SessionOptions {
            directory: Some(session_dir.clone()),
            id_prefix: Some("requested".into()),
            ..SessionOptions::default()
        },
    )
    .unwrap();
    assert_eq!(opened_path, new_path);
    assert_eq!(opened_session.header().unwrap().id.as_str(), "requested-id");
    let (continued_path, _, _) = resolve_session_options(
        &config,
        SessionOptions {
            directory: Some(session_dir.clone()),
            continue_latest: true,
            ..SessionOptions::default()
        },
    )
    .unwrap();
    assert_eq!(continued_path, new_path);
    assert!(
        resolve_session_options(
            &config,
            SessionOptions {
                directory: Some(session_dir),
                id_prefix: Some("missing".into()),
                ..SessionOptions::default()
            },
        )
        .is_err()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn forking_a_session_creates_a_new_header_and_preserves_source_history() {
    let root = std::env::temp_dir().join(format!("mu-session-fork-{}", uuid::Uuid::new_v4()));
    let source_path = root.join("source.jsonl");
    let target_cwd = root.join("target");
    let session_dir = root.join("forks");
    std::fs::create_dir_all(&target_cwd).unwrap();
    let mut source = SessionDocument::new_with_id("/source", "source-id");
    source.append_user_text("forked context");
    source.append_custom_entry("custom-state", serde_json::json!({"value":7}));
    source.write_atomic(&source_path).unwrap();
    let config = Config {
        agent_dir: root.join("agent"),
        cwd: target_cwd.clone(),
        provider: "test-provider".into(),
        model: "test-model".into(),
        model_choices: vec!["test-model".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: super::resolve_default_tools(None),
    };

    let (fork_path, fork, history) = resolve_session_options(
        &config,
        SessionOptions {
            directory: Some(session_dir.clone()),
            id: Some("fork-id".into()),
            fork_from: Some(source_path.clone()),
            ..SessionOptions::default()
        },
    )
    .unwrap();
    let header = fork.header().unwrap();
    assert_eq!(header.id.as_str(), "fork-id");
    assert_eq!(header.cwd, target_cwd.to_string_lossy());
    assert_eq!(
        header.parent_session.as_deref(),
        Some(source_path.to_string_lossy().as_ref())
    );
    let history = responses_input(&history);
    assert_eq!(history[0]["content"][0]["text"], "forked context");
    assert!(fork_path.exists());
    assert_eq!(source.header().unwrap().id.as_str(), "source-id");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn loading_a_session_file_restores_its_document_and_model_history() {
    let root = std::env::temp_dir().join(format!("mu-session-load-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("session.jsonl");
    let mut session = SessionDocument::new("/tmp/project");
    session.append_user_text("remember this");
    session.append_assistant_turn(
        "openai-responses",
        "test-provider",
        "test-model",
        "",
        "I will.",
        MessageUsage::default(),
    );
    session.append_custom_entry(
        "mcp-tool-loadout",
        serde_json::json!({"active":["mcp__test__echo"]}),
    );
    session.append_custom_entry("codemode-store", serde_json::json!({"set":{"answer":42}}));
    session.write_atomic(&path).unwrap();

    let (loaded, history) = load_session(&path).unwrap();
    assert_eq!(loaded.header().unwrap().cwd, "/tmp/project");
    let history = responses_input(&history);
    assert_eq!(history.len(), 2);
    assert_eq!(history[0]["content"][0]["text"], "remember this");
    assert_eq!(history[1]["content"][0]["text"], "I will.");
    assert!(session_active_mcp_tools(&loaded).contains("mcp__test__echo"));
    assert_eq!(session_codemode_store(&loaded)["answer"], 42);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn resumed_history_reconstructs_user_assistant_tool_call_and_tool_result() {
    let mut session = SessionDocument::new("/tmp/project");
    session.append_user_text("hello");
    session.append_assistant_tool_calls(
        "openai-responses",
        "test-provider",
        "test-model",
        AssistantToolResponse {
            thinking: String::new(),
            thinking_signature: None,
            text: String::new(),
            calls: vec![ToolCallRecord {
                id: "call-1".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path":"file.txt"}),
            }],
            usage: MessageUsage::default(),
        },
    );
    session.append_tool_result("call-1", "read", "file contents".into(), false);
    session.append_assistant_turn(
        "openai-responses",
        "test-provider",
        "test-model",
        "",
        "done",
        MessageUsage::default(),
    );

    let history = responses_input(&session_messages(&session));
    assert_eq!(history.len(), 4);
    assert_eq!(history[0]["role"], "user");
    assert_eq!(history[0]["content"][0]["text"], "hello");
    assert_eq!(history[1]["type"], "function_call");
    assert_eq!(history[1]["call_id"], "call-1");
    assert_eq!(history[1]["arguments"], r#"{"path":"file.txt"}"#);
    assert_eq!(history[2]["type"], "function_call_output");
    assert_eq!(history[2]["output"], "file contents");
    assert_eq!(history[3]["content"][0]["text"], "done");
}

#[tokio::test]
async fn ephemeral_sessions_update_memory_without_creating_a_session_file() {
    let root = std::env::temp_dir().join(format!("mu-ephemeral-session-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let config = super::Config {
        agent_dir: root.join("agent"),
        cwd: root.clone(),
        provider: "mu-demo".into(),
        model: "mu-demo".into(),
        model_choices: vec!["mu-demo".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: super::resolve_default_tools(None),
    };
    let mut agent = super::MuAgent::demo_from_config(config).unwrap();
    agent.persist_session = false;
    let path = agent.session_path().to_owned();

    agent.submit("hello".into(), &mut |_| {}).await.unwrap();
    assert!(!path.exists());
    assert_eq!(agent.session.active_branch_entries().len(), 2);
    agent.start_new_session().unwrap();
    assert!(!agent.session_path().exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn default_tools_merge_and_resolve_like_pi_settings() {
    let mut entries = None;
    assert_eq!(
        super::resolve_default_tools(entries.clone()),
        ["read", "bash", "edit", "write"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    );
    super::merge_default_tools(
        &mut entries,
        &serde_json::json!({"defaultTools":["read", "bash", "+grep"]}),
    )
    .unwrap();
    super::merge_default_tools(
        &mut entries,
        &serde_json::json!({"defaultTools":["-bash", "+find"]}),
    )
    .unwrap();
    assert_eq!(
        super::resolve_default_tools(entries),
        ["read", "grep", "find"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    );

    let mut empty = None;
    super::merge_default_tools(&mut empty, &serde_json::json!({"defaultTools":[]})).unwrap();
    assert!(super::resolve_default_tools(empty).is_empty());
    assert!(
        super::merge_default_tools(&mut None, &serde_json::json!({"defaultTools":["read", 4]}))
            .is_err()
    );

    let root = std::env::temp_dir().join(format!("mu-tool-loadout-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let mut agent = super::MuAgent::demo_from_config(super::Config {
        agent_dir: root.join("agent"),
        cwd: root.clone(),
        provider: "mu-demo".into(),
        model: "mu-demo".into(),
        model_choices: vec!["mu-demo".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: vec!["grep".to_owned()],
    })
    .unwrap();
    let shell_environment = agent.shell_environment();
    assert_eq!(
        shell_environment["PI_SESSION_ID"],
        agent.session.header().unwrap().id.as_str()
    );
    assert_eq!(shell_environment["PI_PROVIDER"], "mu-demo");
    assert_eq!(shell_environment["PI_MODEL"], "mu-demo");
    assert_eq!(
        shell_environment["PI_SESSION_FILE"],
        agent.session_path().to_string_lossy()
    );
    let exposed = agent
        .tool_definitions(&HashSet::new())
        .into_iter()
        .filter_map(|definition| {
            definition
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect::<HashSet<_>>();
    assert_eq!(exposed, ["grep".to_owned()].into_iter().collect());
    agent.enabled_tools = super::resolve_default_tools(None);
    let strict_definitions = agent.tool_definitions(&HashSet::new());
    for definition in &strict_definitions {
        if matches!(
            definition["name"].as_str(),
            Some("read" | "bash" | "edit" | "write")
        ) {
            assert_eq!(definition["strict"], true);
        }
    }
    let read = strict_definitions
        .iter()
        .find(|definition| definition["name"] == "read")
        .unwrap();
    let required = read["parameters"]["required"].as_array().unwrap();
    assert_eq!(required.len(), 3);
    assert!(
        read["parameters"]["properties"]["offset"]["anyOf"]
            .as_array()
            .unwrap()
            .iter()
            .any(|variant| variant["type"] == "null")
    );
    agent.provider.supports_strict_sampling = false;
    let unsupported_definitions = agent.tool_definitions(&HashSet::new());
    let unsupported_read = unsupported_definitions
        .iter()
        .find(|definition| definition["name"] == "read")
        .unwrap();
    assert!(unsupported_read.get("strict").is_none());
    assert_eq!(
        unsupported_read["parameters"]["required"],
        serde_json::json!(["path"])
    );
    agent.no_tools = true;
    assert!(agent.tool_definitions(&HashSet::new()).is_empty());
    std::fs::remove_dir_all(root).unwrap();

    let defaults = super::resolve_default_tools(None);
    assert_eq!(
        super::apply_tool_selection(
            &defaults,
            &super::ToolSelection::parse(vec!["-bash".into(), "+grep".into()]).unwrap(),
        ),
        ["read", "edit", "write", "grep"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        super::apply_tool_selection(
            &defaults,
            &super::ToolSelection::parse(vec!["read".into(), "gr*".into()]).unwrap(),
        ),
        ["read", "grep"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    );
    assert!(super::ToolSelection::parse(vec!["read".into(), "+grep".into()]).is_err());
    assert_eq!(
        super::apply_tool_selection(
            &defaults,
            &super::ToolSelection::parse(vec!["coding".into()]).unwrap(),
        ),
        ["read", "bash", "edit", "write"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        super::apply_tool_selection(
            &defaults,
            &super::ToolSelection::parse(vec!["read-only".into()]).unwrap(),
        ),
        ["read", "grep", "find", "ls"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        super::apply_tool_selection(
            &defaults,
            &super::ToolSelection::parse(vec!["all".into()]).unwrap(),
        )
        .len(),
        8
    );
    let definitions = vec![
        serde_json::json!({"name":"read"}),
        serde_json::json!({"name":"mcp__docs__search"}),
        serde_json::json!({"name":"tool_search"}),
    ];
    let allowlist =
        super::ToolSelection::parse(vec!["read".into(), "mcp__docs__*".into()]).unwrap();
    let allowlisted = super::filter_tool_definitions(definitions.clone(), &allowlist);
    assert_eq!(allowlisted.len(), 2);
    let all = super::ToolSelection::parse(vec!["all".into()]).unwrap();
    assert_eq!(
        super::filter_tool_definitions(definitions.clone(), &all).len(),
        definitions.len()
    );
    let modifiers = super::ToolSelection::parse(vec!["-tool_search".into()]).unwrap();
    let modified = super::filter_tool_definitions(definitions, &modifiers);
    assert_eq!(modified.len(), 2);
}

#[test]
fn builtin_prompt_contributions_follow_the_selected_tools() {
    let read_and_edit = vec![
        serde_json::json!({"name":"read"}),
        serde_json::json!({"name":"edit"}),
    ];
    let instructions = super::instructions_for_tools("base instructions", &read_and_edit);
    assert!(instructions.contains("- read: Read file contents"));
    assert!(instructions.contains("- edit: Make precise file edits"));
    assert!(instructions.contains("Use read to examine files instead of cat or sed."));
    assert!(
        instructions.contains("Use edit for precise changes (edits[].oldText must match exactly)")
    );
    assert!(instructions.contains("Show file paths clearly when working with files"));

    let shell_only =
        super::instructions_for_tools("base instructions", &[serde_json::json!({"name":"bash"})]);
    assert!(shell_only.contains("Use bash for file operations like ls, rg, find"));
    assert!(shell_only.contains("PI_* environment variables"));
    let shell_with_search = super::instructions_for_tools(
        "base instructions",
        &[
            serde_json::json!({"name":"bash"}),
            serde_json::json!({"name":"grep"}),
        ],
    );
    assert!(!shell_with_search.contains("Use bash for file operations like ls, rg, find"));
}

#[test]
fn model_strict_sampling_capability_defaults_to_unsupported_and_honors_opt_in() {
    assert!(!super::model_supports_strict_sampling(&serde_json::json!(
        {}
    )));
    assert!(super::model_supports_strict_sampling(&serde_json::json!({
        "compat":{"supportsStrictMode":true}
    })));
    assert!(!super::model_supports_strict_sampling(&serde_json::json!({
        "compat":{"supportsStrictMode":false}
    })));
}

#[test]
fn strict_sampling_builtins_use_required_nullable_optional_parameters() {
    let definitions = crate::tools::definitions()
        .into_iter()
        .map(super::mark_strict_builtin_tool)
        .collect::<Vec<_>>();
    let read = definitions
        .iter()
        .find(|definition| definition["name"] == "read")
        .unwrap();
    assert_eq!(read["strict"], true);
    assert_eq!(read["parameters"]["required"].as_array().unwrap().len(), 3);
    assert!(
        read["parameters"]["properties"]["offset"]["anyOf"]
            .as_array()
            .unwrap()
            .iter()
            .any(|schema| schema["type"] == "null")
    );

    let edit = definitions
        .iter()
        .find(|definition| definition["name"] == "edit")
        .unwrap();
    let item_schema = &edit["parameters"]["properties"]["edits"]["items"];
    assert_eq!(item_schema["required"].as_array().unwrap().len(), 2);
    assert_eq!(item_schema["additionalProperties"], false);
    let grep = definitions
        .iter()
        .find(|definition| definition["name"] == "grep")
        .unwrap();
    assert!(grep.get("strict").is_none());
    assert!(super::strict_schema::schema_allows_null(
        &serde_json::json!({"type":["string","null"]})
    ));
    assert!(super::strict_schema::schema_allows_null(
        &serde_json::json!({"const":null})
    ));
    assert!(super::strict_schema::schema_allows_null(
        &serde_json::json!({"enum":["value",null]})
    ));
    assert!(super::strict_schema::schema_allows_null(
        &serde_json::json!({"anyOf":[{"type":"null"}]})
    ));
    assert_eq!(
        super::ToolSelection::parse(vec!["+gr*".into()])
            .unwrap_err()
            .to_string(),
        "configuration: +name and -name entries take exact tool names, not patterns"
    );
}

#[tokio::test]
async fn scripted_provider_runs_a_complete_streaming_tool_turn_offline() {
    let root = std::env::temp_dir().join(format!("mu-scripted-agent-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).expect("create temporary project");
    let config = super::Config {
        agent_dir: root.join("agent"),
        cwd: root.clone(),
        provider: "test-provider".into(),
        model: "test-model".into(),
        model_choices: vec!["test-model".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: super::resolve_default_tools(None),
    };
    let mut agent = super::MuAgent::demo_from_config(config).expect("demo agent");
    let replies = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from([
        Ok(Reply {
            text: String::new(),
            thinking: String::new(),
            thinking_signature: None,
            output_items: vec![serde_json::json!({
                "type":"function_call",
                "call_id":"call-write",
                "name":"write",
                "arguments":"{\"path\":\"result.txt\",\"content\":\"saved\"}"
            })],
            tool_calls: vec![FunctionCall {
                call_id: "call-write".into(),
                name: "write".into(),
                arguments: serde_json::json!({"path":"result.txt","content":"saved"}),
            }],
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            reasoning: None,
            total_tokens: 0,
            incomplete_reason: None,
        }),
        Ok(Reply {
            text: "saved file".into(),
            thinking: "verify result".into(),
            thinking_signature: None,
            output_items: Vec::new(),
            tool_calls: Vec::new(),
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            reasoning: None,
            total_tokens: 0,
            incomplete_reason: None,
        }),
    ])));
    agent.provider.transport = super::ProviderTransport::Scripted(replies);
    let mut events = Vec::new();

    agent
        .submit("create the result file".into(), &mut |event| {
            events.push(event)
        })
        .await
        .expect("complete scripted agent turn");

    assert_eq!(
        std::fs::read_to_string(root.join("result.txt")).expect("tool output file"),
        "saved"
    );
    assert!(events.contains(&super::AgentEvent::ToolStarted {
        call_id: "call-write".into(),
        name: "write".into(),
        arguments: serde_json::json!({"path":"result.txt","content":"saved"}),
    }));
    assert!(
        events.contains(&super::AgentEvent::ToolFinished {
            call_id: "call-write".into(),
            output: "Successfully wrote to result.txt".into(),
            is_error: false,
            details: events
                .iter()
                .find_map(|event| match event {
                    super::AgentEvent::ToolFinished { details, .. } => Some(details.clone()),
                    _ => None,
                })
                .flatten(),
            duration_ms: events.iter().find_map(|event| match event {
                super::AgentEvent::ToolFinished { duration_ms, .. } => *duration_ms,
                _ => None,
            }),
        })
    );
    assert!(events.contains(&super::AgentEvent::ThinkingDelta("v".into())));
    assert!(events.contains(&super::AgentEvent::TextDelta("s".into())));
    assert_eq!(
        responses_input(&agent.transcript)[1]["type"],
        "function_call"
    );
    assert_eq!(
        responses_input(&agent.transcript)[2]["type"],
        "function_call_output"
    );
    assert_eq!(
        responses_input(&agent.transcript)[2]["output"],
        "Successfully wrote to result.txt"
    );
    let session = std::fs::read_to_string(agent.session_path()).expect("read session");
    let parsed = SessionDocument::parse_jsonl(&session).expect("parse session");
    assert!(parsed.entries().any(|entry| {
        entry
            .extra_fields()
            .get("message")
            .and_then(Value::as_object)
            .and_then(|message| message.get("role"))
            .and_then(Value::as_str)
            == Some("toolResult")
    }));
    std::fs::remove_dir_all(root).expect("remove temporary project");
}

#[tokio::test]
async fn output_token_limited_tool_arguments_are_reported_but_never_executed() {
    let root = std::env::temp_dir().join(format!("mu-truncated-tool-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let config = super::Config {
        agent_dir: root.join("agent"),
        cwd: root.clone(),
        provider: "test-provider".into(),
        model: "test-model".into(),
        model_choices: vec!["test-model".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: super::resolve_default_tools(None),
    };
    let mut agent = super::MuAgent::demo_from_config(config).unwrap();
    let arguments = serde_json::json!({"path":"truncated.txt","content":"must not be written"});
    let replies = std::collections::VecDeque::from([
        Ok(Reply {
            text: String::new(),
            thinking: String::new(),
            thinking_signature: None,
            output_items: vec![serde_json::json!({
                "type":"function_call",
                "call_id":"truncated-call",
                "name":"write",
                "arguments":arguments.to_string()
            })],
            tool_calls: vec![FunctionCall {
                call_id: "truncated-call".into(),
                name: "write".into(),
                arguments,
            }],
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            reasoning: None,
            total_tokens: 0,
            incomplete_reason: Some("max_output_tokens".into()),
        }),
        Ok(Reply {
            text: "I will retry with a complete tool call".into(),
            thinking: String::new(),
            thinking_signature: None,
            output_items: Vec::new(),
            tool_calls: Vec::new(),
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            reasoning: None,
            total_tokens: 0,
            incomplete_reason: None,
        }),
    ]);
    agent.provider.transport =
        super::ProviderTransport::Scripted(std::sync::Arc::new(std::sync::Mutex::new(replies)));
    let mut events = Vec::new();

    agent
        .submit("write a file".into(), &mut |event| events.push(event))
        .await
        .unwrap();

    assert!(!root.join("truncated.txt").exists());
    assert!(events.contains(&super::AgentEvent::ToolFinished {
            call_id: "truncated-call".into(),
            output: "Tool call \"write\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.".into(),
            is_error: true,
            details: None,
            duration_ms: None,
        }));
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn scripted_provider_can_complete_more_than_eight_tool_rounds() {
    let root = std::env::temp_dir().join(format!("mu-many-tool-rounds-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).expect("create temporary project");
    let config = super::Config {
        agent_dir: root.join("agent"),
        cwd: root.clone(),
        provider: "test-provider".into(),
        model: "test-model".into(),
        model_choices: vec!["test-model".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: super::resolve_default_tools(None),
    };
    let mut agent = super::MuAgent::demo_from_config(config).expect("demo agent");
    let mut scripted = std::collections::VecDeque::new();
    for round in 0..9 {
        let call_id = format!("call-{round}");
        let arguments = serde_json::json!({"command":"true"});
        scripted.push_back(Ok(Reply {
            text: String::new(),
            thinking: String::new(),
            thinking_signature: None,
            output_items: vec![serde_json::json!({
                "type":"function_call",
                "call_id":call_id,
                "name":"bash",
                "arguments":arguments.to_string()
            })],
            tool_calls: vec![FunctionCall {
                call_id,
                name: "bash".into(),
                arguments,
            }],
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            reasoning: None,
            total_tokens: 0,
            incomplete_reason: None,
        }));
    }
    scripted.push_back(Ok(Reply {
        text: "all nine rounds completed".into(),
        thinking: String::new(),
        thinking_signature: None,
        output_items: Vec::new(),
        tool_calls: Vec::new(),
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        reasoning: None,
        total_tokens: 0,
        incomplete_reason: None,
    }));
    agent.provider.transport =
        super::ProviderTransport::Scripted(std::sync::Arc::new(std::sync::Mutex::new(scripted)));

    agent
        .submit("run nine independent commands".into(), &mut |_| {})
        .await
        .expect("agent loop should have no arbitrary tool-round limit");

    assert_eq!(responses_input(&agent.transcript).len(), 20);
    assert_eq!(
        responses_input(&agent.transcript).last().unwrap()["content"][0]["text"],
        "all nine rounds completed"
    );
    std::fs::remove_dir_all(root).expect("remove temporary project");
}

#[tokio::test]
async fn scripted_provider_retries_transient_failures_with_bounded_attempt_events() {
    let root = std::env::temp_dir().join(format!("mu-provider-retry-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).expect("create temporary project");
    let config = super::Config {
        agent_dir: root.join("agent"),
        cwd: root.clone(),
        provider: "test-provider".into(),
        model: "test-model".into(),
        model_choices: vec!["test-model".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy {
            enabled: true,
            max_retries: 2,
            base_delay_ms: 0,
            max_delay_ms: 0,
        },
        enabled_tools: super::resolve_default_tools(None),
    };
    let mut agent = super::MuAgent::demo_from_config(config).expect("demo agent");
    agent.provider.transport = super::ProviderTransport::Scripted(std::sync::Arc::new(
        std::sync::Mutex::new(std::collections::VecDeque::from([
            Err("model request returned HTTP 503".into()),
            Err("upstream connection timed out".into()),
            Ok(Reply {
                text: "recovered".into(),
                thinking: String::new(),
                thinking_signature: None,
                output_items: Vec::new(),
                tool_calls: Vec::new(),
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                reasoning: None,
                total_tokens: 0,
                incomplete_reason: None,
            }),
        ])),
    ));
    let mut events = Vec::new();

    agent
        .submit("retry this request".into(), &mut |event| events.push(event))
        .await
        .expect("retry eventually succeeds");

    let scheduled = events
        .iter()
        .filter_map(|event| match event {
            super::AgentEvent::RetryScheduled {
                attempt,
                max_attempts,
                delay_ms,
                error,
            } => Some((*attempt, *max_attempts, *delay_ms, error.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        scheduled,
        [
            (1, 2, 0, "provider: model request returned HTTP 503"),
            (2, 2, 0, "provider: upstream connection timed out")
        ]
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, super::AgentEvent::RetryAttemptStarted { .. }))
            .count(),
        2
    );
    assert!(events.contains(&super::AgentEvent::TextDelta("r".into())));
    assert_eq!(responses_input(&agent.transcript)[0]["role"], "user");
    assert_eq!(responses_input(&agent.transcript)[1]["role"], "assistant");
    std::fs::remove_dir_all(root).expect("remove temporary project");
}

#[tokio::test]
async fn cancelling_during_retry_backoff_does_not_start_another_attempt() {
    let root = std::env::temp_dir().join(format!("mu-retry-cancel-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).expect("create temporary project");
    let config = super::Config {
        agent_dir: root.join("agent"),
        cwd: root.clone(),
        provider: "test-provider".into(),
        model: "test-model".into(),
        model_choices: vec!["test-model".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy {
            enabled: true,
            max_retries: 1,
            base_delay_ms: 60_000,
            max_delay_ms: 60_000,
        },
        enabled_tools: super::resolve_default_tools(None),
    };
    let mut agent = super::MuAgent::demo_from_config(config).expect("demo agent");
    agent.provider.transport = super::ProviderTransport::Scripted(std::sync::Arc::new(
        std::sync::Mutex::new(std::collections::VecDeque::from([
            Err("HTTP 503 temporary failure".into()),
            Ok(Reply {
                text: "must not run".into(),
                thinking: String::new(),
                thinking_signature: None,
                output_items: Vec::new(),
                tool_calls: Vec::new(),
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                reasoning: None,
                total_tokens: 0,
                incomplete_reason: None,
            }),
        ])),
    ));
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        agent
            .submit("cancel during backoff".into(), &mut |event| {
                let _ = event_tx.send(event);
            })
            .await
    });
    let mut scheduled = false;
    while let Some(event) = event_rx.recv().await {
        if matches!(event, super::AgentEvent::RetryScheduled { .. }) {
            scheduled = true;
            break;
        }
    }
    assert!(scheduled);

    task.abort();

    assert!(task.await.unwrap_err().is_cancelled());
    std::fs::remove_dir_all(root).expect("remove temporary project");
}

#[test]
fn retry_policy_excludes_quota_errors_and_caps_exponential_backoff() {
    let mut configured = super::RetryPolicy::default();
    super::update_retry_policy(
        &mut configured,
        &serde_json::json!({"retry":{"enabled":false,"maxRetries":1,"baseDelayMs":50,"maxAgentDelayMs":200}}),
    );
    assert_eq!(
        configured,
        super::RetryPolicy {
            enabled: false,
            max_retries: 1,
            base_delay_ms: 50,
            max_delay_ms: 200,
        }
    );
    super::update_retry_policy(
        &mut configured,
        &serde_json::json!({"retry":{"maxDelayMs":300}}),
    );
    assert_eq!(configured.max_delay_ms, 300);
    assert!(super::retryable_provider_error(
        &super::AgentError::Provider("HTTP 503 service unavailable".into())
    ));
    assert!(!super::retryable_provider_error(
        &super::AgentError::Provider("insufficient_quota: billing limit".into())
    ));
    let policy = super::RetryPolicy {
        enabled: true,
        max_retries: 4,
        base_delay_ms: 2_000,
        max_delay_ms: 5_000,
    };
    assert_eq!(super::retry_delay_ms(policy, 1), 2_000);
    assert_eq!(super::retry_delay_ms(policy, 2), 4_000);
    assert_eq!(super::retry_delay_ms(policy, 3), 5_000);
}

#[tokio::test]
async fn scripted_provider_failure_persists_the_user_turn_without_an_assistant_turn() {
    let root = std::env::temp_dir().join(format!("mu-provider-error-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).expect("create temporary project");
    let config = super::Config {
        agent_dir: root.join("agent"),
        cwd: root.clone(),
        provider: "test-provider".into(),
        model: "test-model".into(),
        model_choices: vec!["test-model".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: super::resolve_default_tools(None),
    };
    let mut agent = super::MuAgent::demo_from_config(config).expect("demo agent");
    agent.provider.transport =
        super::ProviderTransport::Scripted(std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::VecDeque::from([Err("invalid request: malformed payload".into())]),
        )));
    let mut events = Vec::new();

    let error = agent
        .submit("request that fails".into(), &mut |event| events.push(event))
        .await
        .expect_err("scripted provider failure should propagate");

    assert_eq!(
        error.to_string(),
        "provider: invalid request: malformed payload"
    );
    assert!(events.is_empty());
    assert_eq!(responses_input(&agent.transcript).len(), 1);
    assert_eq!(responses_input(&agent.transcript)[0]["role"], "user");
    let session = std::fs::read_to_string(agent.session_path()).expect("read session");
    let parsed = SessionDocument::parse_jsonl(&session).expect("parse session");
    let messages = parsed
        .active_branch_entries()
        .into_iter()
        .filter(|entry| entry.entry_type.as_str() == "message")
        .filter_map(|entry| entry.extra_fields().get("message"))
        .collect::<Vec<_>>();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    std::fs::remove_dir_all(root).expect("remove temporary project");
}

fn scripted_reply(text: &str, calls: Vec<(&str, &str, Value)>) -> Reply {
    Reply {
        text: text.into(),
        thinking: String::new(),
        thinking_signature: None,
        output_items: Vec::new(),
        tool_calls: calls
            .into_iter()
            .map(|(call_id, name, arguments)| FunctionCall {
                call_id: call_id.into(),
                name: name.into(),
                arguments,
            })
            .collect(),
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        reasoning: None,
        total_tokens: 0,
        incomplete_reason: None,
    }
}

fn scripted_agent(root: &Path, replies: Vec<Result<Reply, String>>) -> super::MuAgent {
    std::fs::create_dir_all(root).expect("create temporary project");
    let mut agent = super::MuAgent::demo_from_config(super::Config {
        agent_dir: root.join("agent"),
        cwd: root.to_owned(),
        provider: "test-provider".into(),
        model: "test-model".into(),
        model_choices: vec!["test-model".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: super::resolve_default_tools(None),
    })
    .expect("demo agent");
    agent.provider.transport = super::ProviderTransport::Scripted(std::sync::Arc::new(
        std::sync::Mutex::new(replies.into()),
    ));
    agent
}

#[tokio::test]
async fn continue_session_runs_from_a_pending_user_message_without_adding_another() {
    let root = std::env::temp_dir().join(format!("mu-continue-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(
        &root,
        vec![
            Ok(scripted_reply("continued", vec![])),
            Ok(scripted_reply("processed", vec![])),
        ],
    );
    agent
        .prompt_queue
        .enqueue("keep queued".to_owned(), super::PromptBehavior::Steer);
    assert_eq!(
        agent
            .continue_session(&mut |_| {})
            .await
            .unwrap_err()
            .to_string(),
        "session: No messages to continue from"
    );
    assert_eq!(
        agent.prompt_queue.take_next(),
        Some(("keep queued".to_owned(), super::PromptBehavior::Steer))
    );
    agent
        .transcript
        .push(crate::agent_loop::AgentMessage::System(
            crate::agent_loop::SystemMessage {
                content: "system only".into(),
                sections: None,
                tools_added: Vec::new(),
                tools_removed: Vec::new(),
                timestamp: 0,
            },
        ));
    agent
        .prompt_queue
        .enqueue("still queued".to_owned(), super::PromptBehavior::FollowUp);
    assert_eq!(
        agent
            .continue_session(&mut |_| {})
            .await
            .unwrap_err()
            .to_string(),
        "session: No messages to continue from"
    );
    assert_eq!(
        agent.prompt_queue.take_next(),
        Some(("still queued".to_owned(), super::PromptBehavior::FollowUp))
    );
    agent.transcript.clear();

    agent.session.append_user_text("pending prompt");
    agent
        .transcript
        .push(crate::agent_loop::AgentMessage::user_text(
            "pending prompt",
            0,
        ));
    agent.save_session().unwrap();
    agent.continue_session(&mut |_| {}).await.unwrap();

    assert_eq!(responses_input(&agent.transcript).len(), 2);
    assert_eq!(responses_input(&agent.transcript)[0]["role"], "user");
    assert_eq!(
        responses_input(&agent.transcript)[0]["content"][0]["text"],
        "pending prompt"
    );
    assert_eq!(responses_input(&agent.transcript)[1]["role"], "assistant");
    assert!(matches!(
        agent.transcript.last(),
        Some(crate::agent_loop::AgentMessage::Assistant(message))
            if message.content.iter().any(
                |block| matches!(block, crate::agent_loop::ContentBlock::Text { text } if text == "continued")
            )
    ));
    assert_eq!(
        agent
            .continue_session(&mut |_| {})
            .await
            .unwrap_err()
            .to_string(),
        "session: Cannot continue from message role: assistant"
    );
    agent.prompt_queue.enqueue(
        "queued follow-up".to_owned(),
        super::PromptBehavior::FollowUp,
    );
    agent.continue_session(&mut |_| {}).await.unwrap();
    assert_eq!(responses_input(&agent.transcript).len(), 4);
    assert_eq!(responses_input(&agent.transcript)[2]["role"], "user");
    assert_eq!(
        responses_input(&agent.transcript)[2]["content"][0]["text"],
        "queued follow-up"
    );
    let session = std::fs::read_to_string(agent.session_path()).unwrap();
    let parsed = SessionDocument::parse_jsonl(&session).unwrap();
    assert_eq!(
        parsed
            .active_branch_entries()
            .into_iter()
            .filter(|entry| entry.entry_type.as_str() == "message")
            .count(),
        4
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn continue_session_accepts_a_pending_tool_result_tail() {
    let root = std::env::temp_dir().join(format!("mu-continue-tool-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(&root, vec![Ok(scripted_reply("continued", vec![]))]);
    agent.session.append_user_text("run a tool");
    agent.session.append_assistant_tool_calls(
        "openai-responses",
        "test-provider",
        "test-model",
        mu_session::AssistantToolResponse {
            thinking: String::new(),
            thinking_signature: None,
            text: String::new(),
            calls: vec![mu_session::ToolCallRecord {
                id: "call-1".into(),
                name: "noop".into(),
                arguments: serde_json::json!({}),
            }],
            usage: mu_session::MessageUsage::default(),
        },
    );
    agent.session.append_tool_result_with_details(
        "call-1",
        "noop",
        "tool finished".to_owned(),
        false,
        None,
    );
    agent.transcript = vec![
        crate::agent_loop::AgentMessage::user_text("run a tool", 0),
        crate::agent_loop::AgentMessage::Assistant(crate::agent_loop::AssistantMessage::new(
            vec![crate::agent_loop::ContentBlock::ToolCall {
                id: "call-1".into(),
                name: "noop".into(),
                arguments: serde_json::json!({}),
            }],
            crate::agent_loop::StopReason::ToolUse,
        )),
        crate::agent_loop::AgentMessage::ToolResult(crate::agent_loop::ToolResultMessage {
            tool_call_id: "call-1".into(),
            tool_name: "noop".into(),
            content: vec![crate::agent_loop::ContentBlock::text("tool finished")],
            details: serde_json::json!({}),
            usage: None,
            is_error: false,
            duration_ms: None,
            timestamp: 0,
        }),
    ];
    agent.save_session().unwrap();

    agent.continue_session(&mut |_| {}).await.unwrap();
    assert_eq!(
        responses_input(&agent.transcript).last().unwrap()["role"],
        "assistant"
    );
    assert!(matches!(
        agent.transcript.last(),
        Some(crate::agent_loop::AgentMessage::Assistant(message))
            if message.content.iter().any(
                |block| matches!(block, crate::agent_loop::ContentBlock::Text { text } if text == "continued")
            )
    ));
    let session = std::fs::read_to_string(agent.session_path()).unwrap();
    let parsed = SessionDocument::parse_jsonl(&session).unwrap();
    assert_eq!(
        parsed
            .active_branch_entries()
            .into_iter()
            .filter(|entry| entry.entry_type.as_str() == "message")
            .count(),
        4
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn prompt_inputs_read_files_or_fall_back_to_literal_text() {
    let dir = std::env::temp_dir().join(format!("mu-prompt-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("extra.md");
    std::fs::write(&file, "from file").unwrap();
    let options = super::PromptOptions::from_session_options(&SessionOptions {
        system_prompt: Some("literal prompt".into()),
        append_system_prompt: vec![file.to_string_lossy().into_owned(), "inline".into()],
        ..SessionOptions::default()
    });
    assert_eq!(options.system_prompt.as_deref(), Some("literal prompt"));
    assert_eq!(options.appendix(), "from file\n\ninline");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn excluded_tools_remove_exact_and_wildcard_matches_after_selection() {
    let root = std::env::temp_dir().join(format!("mu-exclude-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(&root, vec![]);
    agent.enabled_tools = ["read", "bash", "grep", "find"].map(String::from).to_vec();
    agent.excluded_tools = vec!["bash".into(), "gr*".into()];
    let names = agent
        .tool_definitions(&HashSet::new())
        .iter()
        .filter_map(|d| d["name"].as_str().map(str::to_owned))
        .collect::<Vec<_>>();
    assert_eq!(names, ["read", "find"]);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn thinking_levels_parse_like_pi_and_map_to_effort() {
    assert_eq!(ThinkingSetting::Max.next(), ThinkingSetting::Off);
    assert_eq!(ThinkingSetting::Medium.next().name(), "high");
    assert_eq!("high".parse::<ThinkingSetting>(), Ok(ThinkingSetting::High));
    assert_eq!(ThinkingSetting::Off.effort(), None);
    assert_eq!(ThinkingSetting::Max.effort(), Some("xhigh"));
    assert_eq!(ThinkingSetting::default().effort(), Some("medium"));
    assert_eq!(
        "loud".parse::<ThinkingSetting>().unwrap_err(),
        "Invalid thinking level \"loud\". Valid values: off, minimal, low, medium, high, xhigh, max"
    );
}

#[test]
fn model_flags_override_settings_and_split_provider_prefixes() {
    let mk = || super::Config {
        agent_dir: std::path::PathBuf::from("/nonexistent"),
        cwd: std::path::PathBuf::from("/"),
        provider: "p".into(),
        model: "m".into(),
        model_choices: vec!["m".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: vec![],
    };
    let mut config = mk();
    super::apply_model_override(&mut config, None, Some("opencode-cf/gpt-x")).unwrap();
    assert_eq!(
        (config.provider.as_str(), config.model.as_str()),
        ("opencode-cf", "gpt-x")
    );
    let mut config = mk();
    super::apply_model_override(&mut config, Some("q"), Some("org/model")).unwrap();
    assert_eq!(
        (config.provider.as_str(), config.model.as_str()),
        ("q", "org/model")
    );
    let mut config = mk();
    super::apply_model_override(&mut config, None, None).unwrap();
    assert_eq!(config.model, "m");
}

#[test]
fn export_session_file_renders_html_without_a_provider() {
    let dir = std::env::temp_dir().join(format!("mu-export-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("s.jsonl");
    let mut session = SessionDocument::new("/tmp/p");
    session.append_user_text("hello <b>there</b>");
    session.write_atomic(&input).unwrap();
    let out = super::export_session_file(&input, Some(&dir.join("o.html"))).unwrap();
    let html = std::fs::read_to_string(out).unwrap();
    assert!(html.contains("hello &lt;b&gt;there&lt;/b&gt;"));
    assert!(super::export_session_file(&dir.join("missing.jsonl"), None).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn missing_api_keys_name_the_provider_and_how_to_fix_it() {
    let message = super::no_api_key_message("acme");
    assert!(message.starts_with("No API key found for acme."));
    assert!(message.contains("/login") && message.contains("models.json"));
}

#[test]
fn missing_model_settings_explain_how_to_log_in_and_pick_a_model() {
    let message = super::no_model_selected_message("Pi settings have no defaultModel");
    assert!(message.starts_with("Pi settings have no defaultModel. No model selected."));
    assert!(message.contains("/login") && message.ends_with("select a model."));
}

#[test]
fn http_failures_include_a_bounded_error_body_so_quota_errors_are_not_retried() {
    let status = reqwest::StatusCode::TOO_MANY_REQUESTS;
    assert_eq!(
        super::retry::http_failure_message(status, ""),
        "model request returned HTTP 429 Too Many Requests"
    );
    let quota = super::retry::http_failure_message(status, r#"{"error":"insufficient_quota"}"#);
    assert!(quota.ends_with(r#"{"error":"insufficient_quota"}"#));
    assert!(!super::retryable_provider_error(
        &crate::AgentError::Provider(quota)
    ));
    let long = super::retry::http_failure_message(status, &"x".repeat(900));
    assert!(long.ends_with('…') && long.len() < 600);
}

/// Ported from Pi's `tool-system-prompt-contributions.test.ts`: the snippets and guidelines
/// are exactly the ones in each tool's `*ToolSystemPromptContribution`.
#[test]
fn builtin_tool_prompt_contributions_match_pi() {
    let snippets = [
        ("read", "Read file contents"),
        ("bash", "Execute bash commands (ls, grep, find, etc.)"),
        ("powershell", "Execute PowerShell commands"),
        (
            "edit",
            "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
        ),
        ("write", "Create or overwrite files"),
        (
            "grep",
            "Search file contents for patterns (respects .gitignore)",
        ),
        ("find", "Find files by glob pattern (respects .gitignore)"),
        ("ls", "List directory contents"),
    ];
    for (name, snippet) in snippets {
        assert_eq!(
            super::tool_prompts::tool_prompt_snippet(name),
            Some(snippet),
            "{name}"
        );
    }
    let env = ["You can inspect PI_* environment variables for current model and session details."];
    assert_eq!(super::tool_prompts::tool_prompt_guidelines("bash"), env);
    assert_eq!(
        super::tool_prompts::tool_prompt_guidelines("powershell"),
        env
    );
    assert_eq!(
        super::tool_prompts::tool_prompt_guidelines("read"),
        ["Use read to examine files instead of cat or sed."]
    );
    assert_eq!(
        super::tool_prompts::tool_prompt_guidelines("write"),
        ["Use write only for new files or complete rewrites."]
    );
    assert_eq!(super::tool_prompts::tool_prompt_guidelines("edit").len(), 4);
    for name in ["grep", "find", "ls"] {
        assert!(
            super::tool_prompts::tool_prompt_guidelines(name).is_empty(),
            "{name}"
        );
    }
}

#[test]
fn sessions_with_a_vanished_cwd_are_refused_with_pis_message() {
    let dir = std::env::temp_dir().join(format!("mu-cwd-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("s.jsonl");
    let gone = dir.join("deleted-project");
    let existing = SessionDocument::new(dir.to_string_lossy().into_owned());
    assert_eq!(
        super::missing_session_cwd_error(&existing, &file, &dir),
        None
    );
    let missing = SessionDocument::new(gone.to_string_lossy().into_owned());
    let message = super::missing_session_cwd_error(&missing, &file, &dir).unwrap();
    assert_eq!(
        message,
        format!(
            "Stored session working directory does not exist: {}\nSession file: {}\nCurrent working directory: {}",
            gone.display(),
            file.display(),
            dir.display()
        )
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn importing_a_session_with_a_vanished_cwd_leaves_the_current_session_untouched() {
    let dir = std::env::temp_dir().join(format!("mu-import-cwd-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("s.jsonl");
    let doc = SessionDocument::new(dir.join("gone").to_string_lossy().into_owned());
    std::fs::write(&file, doc.to_jsonl().unwrap()).unwrap();
    let mut agent = super::MuAgent::demo().unwrap();
    let before = agent.session_path().to_owned();
    let error = agent.import_session(&file).unwrap_err().to_string();
    assert!(
        error.contains("Stored session working directory does not exist"),
        "{error}"
    );
    assert_eq!(agent.session_path(), before);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn session_dir_precedence_is_flag_then_env_then_setting() {
    let flag = Path::new("/flag");
    assert_eq!(
        super::sessions::configured_session_dir(Some(flag), Some("/env"), Some("/set")),
        Some(std::path::PathBuf::from("/flag"))
    );
    assert_eq!(
        super::sessions::configured_session_dir(None, Some("/env"), Some("/set")),
        Some(std::path::PathBuf::from("/env"))
    );
    assert_eq!(
        super::sessions::configured_session_dir(None, Some(""), Some("/set")),
        Some(std::path::PathBuf::from("/set"))
    );
    assert_eq!(
        super::sessions::configured_session_dir(None, None, None),
        None
    );
    if let Some(home) = std::env::var_os("HOME") {
        assert_eq!(
            super::sessions::expand_tilde("~/x"),
            std::path::PathBuf::from(home).join("x")
        );
    }
    assert_eq!(
        super::sessions::expand_tilde("/abs"),
        std::path::PathBuf::from("/abs")
    );
}

#[test]
fn branch_summaries_project_into_context_as_a_framed_user_message() {
    let mut session = SessionDocument::new("/w".to_owned());
    session.append_user_text("first");
    session.append_branch_summary("explored X", "abandoned-leaf");
    let messages = super::runtime::session_messages(&session);
    let inputs = super::runtime::responses_input(&messages);
    let text = inputs.last().unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(
        text,
        "The following is a summary of a branch that this conversation came back from:\n\n<summary>\nexplored X</summary>"
    );
    assert_eq!(inputs.len(), 2);
}

#[test]
fn vertex_base_url_is_built_from_project_and_location() {
    let url =
        super::vertex_base_url(&serde_json::json!({"project":"p1","location":"europe-west4"}))
            .unwrap();
    assert_eq!(
        url,
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/p1/locations/europe-west4/publishers/google"
    );
    let global =
        super::vertex_base_url(&serde_json::json!({"project":"p1","location":"global"})).unwrap();
    assert!(
        global.starts_with("https://aiplatform.googleapis.com/v1/projects/p1/locations/global/")
    );
    let default = super::vertex_base_url(&serde_json::json!({"project":"p1"})).unwrap();
    assert!(default.contains("us-central1-aiplatform"));
}

/// Pi's `transformMessages` orphan handling: unanswered calls get a synthetic error result
/// before the next turn begins, and answered calls are left alone.
#[test]
fn orphaned_tool_calls_get_synthetic_results_before_the_next_turn() {
    let call = |id: &str| serde_json::json!({"type":"function_call","call_id":id,"name":"read","arguments":"{}"});
    let output =
        |id: &str| serde_json::json!({"type":"function_call_output","call_id":id,"output":"ok"});
    let user = serde_json::json!({"role":"user","content":[{"type":"input_text","text":"next"}]});
    let items = super::runtime::close_orphaned_calls(vec![
        call("a"),
        call("b"),
        output("a"),
        user.clone(),
        call("c"),
    ]);
    let shape = items
        .iter()
        .map(|i| {
            format!(
                "{}:{}",
                i["type"].as_str().unwrap_or("message"),
                i["call_id"].as_str().unwrap_or("")
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        shape,
        [
            "function_call:a",
            "function_call:b",
            "function_call_output:a",
            "function_call_output:b",
            "message:",
            "function_call:c",
            "function_call_output:c"
        ]
    );
    assert_eq!(items[3]["output"], "No result provided");
    // A complete history is unchanged.
    let complete = vec![call("x"), output("x"), user];
    assert_eq!(
        super::runtime::close_orphaned_calls(complete.clone()),
        complete
    );
}

#[test]
fn image_tool_outputs_are_replaced_for_text_only_models() {
    let item = serde_json::json!({"type":"function_call_output","call_id":"c","output":[
            {"type":"input_text","text":"Read image file [image/png]"},
            {"type":"input_image","image_url":"data:a"},
            {"type":"input_image","image_url":"data:b"},
            {"type":"input_text","text":"after"}]});
    let out = super::runtime::downgrade_images(&[
        item,
        serde_json::json!({"type":"function_call_output","call_id":"d","output":"plain"}),
    ]);
    let texts = out[0]["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["text"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    assert_eq!(
        texts,
        [
            "Read image file [image/png]",
            "(tool image omitted: model does not support images)",
            "after"
        ]
    );
    assert_eq!(out[1]["output"], "plain");
}

#[test]
fn compaction_and_branch_summaries_appear_in_the_transcript() {
    let mut session = SessionDocument::new("/w".to_owned());
    session.append_user_text("one");
    let kept = session.entries().last().unwrap().id.clone().unwrap();
    session.append_compaction("COMPACTED", kept.as_str(), 77);
    session.append_branch_summary("BRANCHED", "x");
    let transcript = super::session_transcript(&session);
    let roles = transcript.iter().map(|m| m.role).collect::<Vec<_>>();
    assert_eq!(
        roles,
        [
            super::SessionTranscriptRole::User,
            super::SessionTranscriptRole::CompactionSummary(77),
            super::SessionTranscriptRole::BranchSummary
        ]
    );
    assert_eq!(transcript[1].text, "COMPACTED");
}

#[test]
fn user_images_round_trip_through_the_session_into_responses_input() {
    let mut session = SessionDocument::new("/w".to_owned());
    session.append_user_content("see", &[("image/png".to_owned(), "QUJD".to_owned())]);
    let reloaded = SessionDocument::parse_jsonl(&session.to_jsonl().unwrap()).unwrap();
    let input = super::runtime::responses_input(&super::runtime::session_messages(&reloaded));
    assert_eq!(input[0]["content"][0]["text"], "see");
    assert_eq!(input[0]["content"][1]["type"], "input_image");
    assert_eq!(
        input[0]["content"][1]["image_url"],
        "data:image/png;base64,QUJD"
    );
}

#[test]
fn codemode_only_hides_every_declaration_except_tool_search() {
    let tools = || {
        vec![
            serde_json::json!({"name":"read"}),
            serde_json::json!({"name":"mcp__x__y"}),
            serde_json::json!({"name":"tool_search"}),
        ]
    };
    let mut on = tools();
    super::runtime::hide_declarations_for_codemode_only(&mut on, false);
    assert_eq!(on.len(), 3);
    let mut only = tools();
    super::runtime::hide_declarations_for_codemode_only(&mut only, true);
    assert_eq!(only, vec![serde_json::json!({"name":"tool_search"})]);
}

#[test]
fn user_images_reach_every_adapter_in_its_own_format() {
    let user = serde_json::json!({"role":"user","content":[
            {"type":"input_text","text":"look"},
            {"type":"input_image","image_url":"data:image/png;base64,QUJD"}]});
    let anthropic = super::anthropic::messages(std::slice::from_ref(&user));
    assert_eq!(anthropic[0]["content"][1]["source"]["data"], "QUJD");
    let completions = super::completions::messages("sys", std::slice::from_ref(&user));
    assert_eq!(
        completions[1]["content"][1]["image_url"]["url"],
        "data:image/png;base64,QUJD"
    );
    let google = super::google::contents(std::slice::from_ref(&user));
    assert_eq!(google[0]["parts"][1]["inlineData"]["mimeType"], "image/png");
    let bedrock = super::bedrock::messages(std::slice::from_ref(&user));
    assert_eq!(bedrock[0]["content"][1]["image"]["format"], "png");
    let down = super::runtime::downgrade_images(&[user]);
    assert_eq!(
        down[0]["content"][1]["text"],
        "(image omitted: model does not support images)"
    );
}

#[test]
fn models_accept_images_unless_input_excludes_them() {
    assert!(super::model_accepts_images(&serde_json::json!({})));
    assert!(super::model_accepts_images(
        &serde_json::json!({"input":["text","image"]})
    ));
    assert!(!super::model_accepts_images(
        &serde_json::json!({"input":["text"]})
    ));
}

#[test]
fn bedrock_region_comes_from_the_runtime_host() {
    assert_eq!(
        super::bedrock_region_from_host("bedrock-runtime.eu-west-1.amazonaws.com").as_deref(),
        Some("eu-west-1")
    );
    assert_eq!(super::bedrock_region_from_host("example.com"), None);
}

#[tokio::test]
async fn provider_headers_are_resolved_and_sent_on_every_request() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buffer = vec![0; 4096];
        let read = socket.read(&mut buffer).await.unwrap();
        socket
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
            .await
            .unwrap();
        String::from_utf8_lossy(&buffer[..read]).to_lowercase()
    });
    let headers = serde_json::json!({"X-Static":"one","X-Command":"!echo two","X-Unset":"$MU_SURELY_UNSET_VAR"});
    let client = super::provider_http_client(Some(&headers), "acme").unwrap();
    client
        .get(format!("http://127.0.0.1:{port}/"))
        .send()
        .await
        .unwrap();
    let request = server.await.unwrap();
    assert!(request.contains("x-static: one"), "{request}");
    assert!(request.contains("x-command: two"), "{request}");
    assert!(!request.contains("x-unset"), "{request}");
    assert!(
        super::provider_http_client(Some(&serde_json::json!({"bad name":"x"})), "acme").is_err()
    );
}

#[tokio::test]
async fn github_copilot_requests_carry_pis_static_headers_and_user_headers_win() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buffer = vec![0; 4096];
        let read = socket.read(&mut buffer).await.unwrap();
        socket
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
            .await
            .unwrap();
        String::from_utf8_lossy(&buffer[..read]).to_lowercase()
    });
    let user = serde_json::json!({"editor-version":"custom/1"});
    let client = super::provider_http_client(Some(&user), "github-copilot").unwrap();
    client
        .get(format!("http://127.0.0.1:{port}/"))
        .send()
        .await
        .unwrap();
    let request = server.await.unwrap();
    assert!(
        request.contains("user-agent: githubcopilotchat/0.35.0"),
        "{request}"
    );
    assert!(
        request.contains("copilot-integration-id: vscode-chat"),
        "{request}"
    );
    assert!(
        request.contains("openai-intent: conversation-edits"),
        "{request}"
    );
    assert!(request.contains("editor-version: custom/1"), "{request}");
    assert!(
        !request.contains("vscode/1.107.0"),
        "the user header replaced the default"
    );
}

#[test]
fn github_copilot_base_url_comes_from_the_stored_token_when_models_json_has_none() {
    let dir = std::env::temp_dir().join(format!("mu-copilot-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
            dir.join("models.json"),
            r#"{"providers":{"github-copilot":{"api":"openai-completions","models":[{"id":"gpt-x"}]}}}"#,
        )
        .unwrap();
    let config = super::Config {
        agent_dir: dir.clone(),
        cwd: dir.clone(),
        provider: "github-copilot".into(),
        model: "gpt-x".into(),
        model_choices: vec![],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: vec![],
    };
    let prompt = super::PromptOptions::default();
    assert!(
        super::connect_custom_provider(&config, String::new(), &prompt).is_err(),
        "not logged in"
    );
    crate::auth::set_credential(
        &dir.join("auth.json"),
        "github-copilot",
        crate::auth::Credential::Oauth {
            access: "tid=1;proxy-ep=proxy.business.githubcopilot.com".into(),
            refresh: "gho".into(),
            expires: f64::MAX,
            extra: Default::default(),
        },
    )
    .unwrap();
    let provider = super::connect_custom_provider(&config, String::new(), &prompt).unwrap();
    let super::ProviderTransport::Completions { endpoint, key, .. } = provider.transport else {
        panic!("expected a completions transport");
    };
    assert_eq!(
        endpoint.as_str(),
        "https://api.business.githubcopilot.com/chat/completions"
    );
    assert_eq!(key, "tid=1;proxy-ep=proxy.business.githubcopilot.com");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn custom_providers_come_from_models_json_without_gateway_headers() {
    let dir = std::env::temp_dir().join(format!("mu-custom-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
            dir.join("models.json"),
            r#"{"providers":{"acme":{"api":"openai-responses","baseUrl":"https://api.acme.test/v1","apiKey":"literal-key","models":[{"id":"m1","reasoning":true}]},"bad":{"api":"google-genai","baseUrl":"x","apiKey":"k"}}}"#,
        )
        .unwrap();
    let config = |provider: &str| super::Config {
        agent_dir: dir.clone(),
        cwd: dir.clone(),
        provider: provider.into(),
        model: "m1".into(),
        model_choices: vec![],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: vec![],
    };
    let prompt = super::PromptOptions::default();
    let provider = super::connect_custom_provider(&config("acme"), String::new(), &prompt).unwrap();
    let super::ProviderTransport::OpenCode {
        endpoint,
        token,
        gateway_headers,
        ..
    } = provider.transport
    else {
        panic!("expected an HTTP transport");
    };
    assert_eq!(endpoint.as_str(), "https://api.acme.test/v1/responses");
    assert_eq!(token, "literal-key");
    assert!(!gateway_headers && provider.supports_reasoning);
    let error = super::connect_custom_provider(&config("bad"), String::new(), &prompt)
        .err()
        .unwrap();
    assert!(error.to_string().contains("unsupported api `google-genai`"));
    assert!(super::connect_custom_provider(&config("nope"), String::new(), &prompt).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn api_key_flag_overrides_the_models_json_key() {
    let dir = std::env::temp_dir().join(format!("mu-apikey-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
            dir.join("models.json"),
            r#"{"providers":{"acme":{"api":"openai-responses","baseUrl":"https://a.test/v1","apiKey":"from-file","models":[{"id":"m"}]}}}"#,
        )
        .unwrap();
    let config = super::Config {
        agent_dir: dir.clone(),
        cwd: dir.clone(),
        provider: "acme".into(),
        model: "m".into(),
        model_choices: vec![],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: vec![],
    };
    let prompt = super::PromptOptions::from_session_options(&SessionOptions {
        api_key: Some("runtime".into()),
        ..SessionOptions::default()
    });
    let provider = super::connect_custom_provider(&config, String::new(), &prompt).unwrap();
    let super::ProviderTransport::OpenCode { token, .. } = provider.transport else {
        panic!()
    };
    assert_eq!(token, "runtime");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn logout_removes_the_stored_credential() {
    let root = std::env::temp_dir().join(format!("mu-logout-{}", uuid::Uuid::new_v4()));
    let agent = scripted_agent(&root, vec![]);
    let auth = root.join("agent/auth.json");
    crate::auth::set_api_key(&auth, "acme", "k").unwrap();
    assert!(agent.logout("acme").unwrap());
    assert!(!agent.logout("acme").unwrap());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn fork_before_a_user_message_truncates_history_and_returns_its_text() {
    let root = std::env::temp_dir().join(format!("mu-fork-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(&root, vec![]);
    agent.session.append_user_text("first");
    agent.session.append_user_text("second");
    agent.session.append_user_text("third");
    let messages = agent.user_messages();
    assert_eq!(messages.len(), 3);
    let old_path = agent.session_path().to_owned();
    let text = agent.fork_before_user_message(&messages[1].0).unwrap();
    assert_eq!(text, "second");
    assert_eq!(agent.user_messages().len(), 1);
    assert_ne!(agent.session_path(), old_path);
    let saved = std::fs::read_to_string(agent.session_path()).unwrap();
    assert!(saved.contains("parentSession") && !saved.contains("second"));
    assert!(agent.fork_before_user_message("nope").is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn strict_schemas_reject_unsupported_shapes_like_pi() {
    let strict = |schema: Value| super::strict_schema::strict_json_schema(&schema);
    let ok = strict(serde_json::json!({"type":"object","properties":{"a":{"type":"string"},"b":{"type":"number"}},"required":["a"]})).unwrap();
    assert_eq!(ok["required"], serde_json::json!(["a", "b"]));
    assert_eq!(ok["additionalProperties"], false);
    assert!(ok["properties"]["b"]["anyOf"].is_array());
    for (schema, message) in [
        (
            serde_json::json!({"type":"string"}),
            "root schema must have type object",
        ),
        (
            serde_json::json!({"type":"object","oneOf":[{}]}),
            "oneOf schemas are unsupported",
        ),
        (
            serde_json::json!({"type":"object","additionalProperties":true}),
            "schema-valued or true additionalProperties is unsupported",
        ),
        (
            serde_json::json!({"type":"object","properties":{"t":{"type":"array","items":[{}]}}}),
            "tuple schemas are unsupported",
        ),
        (
            serde_json::json!({"type":"object","properties":{"u":{"anyOf":[{"type":"object"},{"type":"null"}]}}}),
            "object and array unions are unsupported",
        ),
        (
            serde_json::json!({"type":"object","required":["zz"]}),
            "required contains an unknown property",
        ),
        (
            serde_json::json!({"type":"object","properties":{"p":true}}),
            "boolean schemas are unsupported",
        ),
    ] {
        assert_eq!(strict(schema).unwrap_err(), message);
    }
    // An unsupported schema leaves the tool non-strict instead of failing the request.
    let tool = serde_json::json!({"name":"read","parameters":{"type":"object","oneOf":[{}]}});
    assert!(
        super::mark_strict_builtin_tool(tool)
            .get("strict")
            .is_none()
    );
}

#[test]
fn tree_navigation_branches_new_messages_from_the_chosen_entry() {
    let root = std::env::temp_dir().join(format!("mu-tree-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(&root, vec![]);
    agent.session.append_user_text("one");
    agent.session.append_user_text("two");
    let rows = agent.tree_rows();
    assert_eq!(rows.len(), 2);
    assert!(rows[1].is_leaf && rows[1].on_branch && rows[1].depth == 1);
    agent.navigate_to(&rows[0].id).unwrap();
    assert_eq!(agent.user_messages().len(), 1);
    agent.session.append_user_text("alt");
    let rows = agent.tree_rows();
    assert_eq!(rows.len(), 3);
    assert!(
        !rows[1].on_branch,
        "the abandoned branch is shown but off-branch"
    );
    assert_eq!(rows[2].depth, 1);
    assert!(agent.navigate_to("nope").is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn session_summary_counts_messages_and_tokens() {
    let root = std::env::temp_dir().join(format!("mu-summary-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(&root, vec![]);
    agent.session.append_user_text("hi");
    agent.session.append_assistant_response(
        "a",
        "p",
        "m",
        mu_session::AssistantTurnResponse {
            thinking: String::new(),
            thinking_signature: None,
            text: "yo".into(),
            usage: mu_session::MessageUsage {
                input: 7,
                output: 3,
                ..Default::default()
            },
        },
    );
    let summary = agent.session_summary();
    assert!(
        summary.contains("Messages: 1 user, 1 assistant, 0 tool calls, 0 tool results"),
        "{summary}"
    );
    assert!(summary.contains("Tokens: 7 input, 3 output"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn enabled_models_scope_the_cycle_list() {
    let choices = ["gpt-a", "gpt-b", "claude-x"].map(String::from).to_vec();
    let pats = |p: &[&str]| p.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(super::scope_models(&[], &choices, "p"), choices);
    assert_eq!(
        super::scope_models(&pats(&["claude-*"]), &choices, "p"),
        ["claude-x"]
    );
    assert_eq!(
        super::scope_models(&pats(&["P/GPT-B:high", "gpt-a"]), &choices, "p"),
        ["gpt-b", "gpt-a"]
    );
    assert_eq!(
        super::scope_models(&pats(&["nomatch"]), &choices, "p"),
        choices
    );
}

#[test]
fn reload_picks_up_new_prompt_templates() {
    let root = std::env::temp_dir().join(format!("mu-reload-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(&root, vec![]);
    assert!(agent.reload_resources().contains("0 prompt templates"));
    let prompts = root.join(".pi/prompts");
    std::fs::create_dir_all(&prompts).unwrap();
    std::fs::write(prompts.join("hi.md"), "Hello $1").unwrap();
    assert!(agent.reload_resources().contains("1 prompt templates"));
    assert_eq!(
        crate::prompt_templates::expand_prompt_template("/hi bob", &agent.templates),
        "Hello bob"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn startup_summary_lists_the_resolved_configuration() {
    let root = std::env::temp_dir().join(format!("mu-verbose-{}", uuid::Uuid::new_v4()));
    let agent = scripted_agent(&root, vec![]);
    let summary = agent.startup_summary();
    assert!(summary.starts_with("provider: test-provider\nmodel: test-model\nthinking: medium"));
    assert!(summary.contains("tools: read, bash, edit, write"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn queue_modes_drain_one_or_all_queued_messages() {
    let queue = AgentPromptQueue::default();
    for prompt in ["one", "two", "three"] {
        queue.enqueue(prompt.into(), PromptBehavior::Steer);
        queue.enqueue(prompt.into(), PromptBehavior::FollowUp);
    }
    assert_eq!(
        queue.take_steering_batch(super::QueueMode::OneAtATime),
        ["one"]
    );
    assert_eq!(
        queue.take_steering_batch(super::QueueMode::All),
        ["two", "three"]
    );
    assert!(queue.take_steering_batch(super::QueueMode::All).is_empty());
    assert_eq!(queue.take_follow_ups(super::QueueMode::OneAtATime), ["one"]);
    assert_eq!(
        queue.take_follow_ups(super::QueueMode::All),
        ["two", "three"]
    );
    assert_eq!(
        super::QueueMode::from_setting("all"),
        Some(super::QueueMode::All)
    );
    assert_eq!(super::QueueMode::from_setting("bogus"), None);
    assert_eq!(super::QueueMode::default(), super::QueueMode::OneAtATime);
}

#[tokio::test]
async fn compact_summarizes_old_history_and_rebuilds_the_transcript() {
    let root = std::env::temp_dir().join(format!("mu-compact-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(&root, vec![Ok(scripted_reply("## Goal\nship it", vec![]))]);
    for index in 0..6 {
        agent
            .session
            .append_user_text(format!("question {index} {}", "x".repeat(20_000)));
        agent.session.append_assistant_response(
            "openai-responses",
            "p",
            "m",
            mu_session::AssistantTurnResponse {
                thinking: String::new(),
                thinking_signature: None,
                text: format!("answer {index} {}", "y".repeat(20_000)),
                usage: mu_session::MessageUsage::default(),
            },
        );
    }
    let outcome = agent
        .compact()
        .await
        .unwrap()
        .expect("something to compact");
    assert_eq!(outcome.summary, "## Goal\nship it");
    assert!(outcome.tokens_before > 20_000);

    let input = responses_input(&agent.transcript);
    let first = input[0]["content"][0]["text"].as_str().unwrap();
    assert!(first.starts_with("The conversation history before this point was compacted"));
    assert!(first.contains("ship it"));
    assert!(
        input.len() < 12,
        "old messages were dropped: {}",
        input.len()
    );
    assert!(
        input.last().unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .starts_with("answer 5")
    );
    let saved = std::fs::read_to_string(agent.session_path()).unwrap();
    assert!(saved.contains("\"type\":\"compaction\""));
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn auto_compaction_runs_after_a_turn_that_fills_the_context_window() {
    let root = std::env::temp_dir().join(format!("mu-auto-compact-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(&root, vec![Ok(scripted_reply("summary text", vec![]))]);
    std::fs::create_dir_all(root.join("agent")).unwrap();
    std::fs::write(
        root.join("agent/models.json"),
        r#"{"providers":{"test-provider":{"models":[{"id":"test-model","contextWindow":30000}]}}}"#,
    )
    .unwrap();
    for index in 0..4 {
        agent
            .session
            .append_user_text(format!("q{index} {}", "x".repeat(20_000)));
        agent.session.append_assistant_response(
            "openai-responses",
            "p",
            "m",
            mu_session::AssistantTurnResponse {
                thinking: String::new(),
                thinking_signature: None,
                text: format!("a{index} {}", "y".repeat(20_000)),
                usage: mu_session::MessageUsage {
                    total_tokens: 29_000,
                    ..mu_session::MessageUsage::default()
                },
            },
        );
    }
    assert!(agent.needs_auto_compaction());
    agent.compact().await.unwrap();
    // The compaction entry carries no usage, but the last assistant usage is still full;
    // a disabled setting must suppress the trigger.
    std::fs::write(
        root.join("agent/settings.json"),
        r#"{"compaction":{"enabled":false}}"#,
    )
    .unwrap();
    assert!(!agent.needs_auto_compaction());
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn compact_does_nothing_for_short_sessions_and_keeps_failed_summaries_out() {
    let root = std::env::temp_dir().join(format!("mu-compact-short-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(&root, vec![]);
    agent.session.append_user_text("hi");
    assert_eq!(agent.compact().await.unwrap(), None);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn unknown_tools_become_error_results_the_model_can_see() {
    let root = std::env::temp_dir().join(format!("mu-unknown-tool-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(
        &root,
        vec![
            Ok(scripted_reply(
                "",
                vec![("call-1", "missing_tool", serde_json::json!({}))],
            )),
            Ok(scripted_reply("recovered", vec![])),
        ],
    );
    let mut finished = Vec::new();
    agent
        .submit("use a tool".into(), &mut |event| {
            if let super::AgentEvent::ToolFinished {
                output, is_error, ..
            } = event
            {
                finished.push((output, is_error));
            }
        })
        .await
        .expect("the loop continues after a failed tool call");

    assert_eq!(finished, [("Tool missing_tool not found".to_owned(), true)]);
    let input = responses_input(&agent.transcript);
    assert_eq!(input[2]["type"], "function_call_output");
    assert_eq!(input[3]["content"][0]["text"], "recovered");
    std::fs::remove_dir_all(root).expect("remove temporary project");
}

#[tokio::test]
async fn invalid_tool_arguments_report_pi_validation_errors_without_executing() {
    let root = std::env::temp_dir().join(format!("mu-invalid-args-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(
        &root,
        vec![
            Ok(scripted_reply(
                "",
                vec![("call-1", "write", serde_json::json!({"path": 7}))],
            )),
            Ok(scripted_reply("done", vec![])),
        ],
    );
    let mut outputs = Vec::new();
    agent
        .submit("write".into(), &mut |event| {
            if let super::AgentEvent::ToolFinished { output, .. } = event {
                outputs.push(output);
            }
        })
        .await
        .unwrap();

    assert!(
        outputs[0].starts_with("Validation failed for tool \"write\":"),
        "{}",
        outputs[0]
    );
    assert!(outputs[0].contains("Received arguments:"));
    std::fs::remove_dir_all(root).expect("remove temporary project");
}

#[tokio::test]
async fn a_cancelled_turn_ends_cleanly_without_persisting_an_assistant_message() {
    let root = std::env::temp_dir().join(format!("mu-cancelled-turn-{}", uuid::Uuid::new_v4()));
    let mut agent = scripted_agent(&root, vec![Ok(scripted_reply("never sent", vec![]))]);
    let cancel = tokio_util::sync::CancellationToken::new();
    cancel.cancel();

    agent
        .submit_with_cancel("hello".into(), &mut |_| {}, &cancel)
        .await
        .expect("an aborted turn is not an error");

    let session = std::fs::read_to_string(agent.session_path()).expect("read session");
    let parsed = SessionDocument::parse_jsonl(&session).expect("parse session");
    let roles = parsed
        .active_branch_entries()
        .into_iter()
        .filter_map(|entry| entry.extra_fields().get("message"))
        .map(|message| message["role"].as_str().unwrap_or_default().to_owned())
        .collect::<Vec<_>>();
    assert_eq!(roles, ["user"]);
    std::fs::remove_dir_all(root).expect("remove temporary project");
}

#[tokio::test]
async fn starting_a_new_session_clears_context_and_preserves_the_old_file() {
    let root = std::env::temp_dir().join(format!("mu-new-session-{}", uuid::Uuid::new_v4()));
    let agent_dir = root.join("agent");
    std::fs::create_dir_all(&root).expect("create temporary project");
    let config = super::Config {
        agent_dir,
        cwd: root.clone(),
        provider: "mu-demo".into(),
        model: "mu-demo".into(),
        model_choices: vec!["mu-demo".into()],
        hide_thinking_block: false,
        retry: super::RetryPolicy::default(),
        enabled_tools: super::resolve_default_tools(None),
    };
    let mut agent = super::MuAgent::demo_from_config(config).expect("demo agent");
    agent
        .submit("old conversation".into(), &mut |_| {})
        .await
        .expect("write old session");
    let old_path = agent.session_path().to_owned();
    agent.active_mcp_tools.insert("mcp__test__echo".into());
    agent
        .prompt_queue
        .enqueue("pending prompt".into(), PromptBehavior::FollowUp);

    agent.start_new_session().expect("start new session");

    assert_ne!(agent.session_path(), old_path);
    assert!(responses_input(&agent.transcript).is_empty());
    assert!(agent.active_mcp_tools.is_empty());
    assert_eq!(agent.prompt_queue.take_next(), None);
    let old_session = std::fs::read_to_string(&old_path).expect("preserve old session file");
    assert!(
        super::SessionDocument::parse_jsonl(&old_session)
            .expect("parse old session")
            .entries()
            .count()
            > 0
    );
    let new_session = std::fs::read_to_string(agent.session_path()).expect("new session file");
    assert_eq!(
        super::SessionDocument::parse_jsonl(&new_session)
            .expect("parse new session")
            .entries()
            .count(),
        0
    );
    std::fs::remove_dir_all(root).expect("remove temporary project");
}

#[test]
fn prompt_queue_prioritizes_fifo_steering_before_follow_ups() {
    let queue = AgentPromptQueue::default();
    queue.enqueue("later".into(), PromptBehavior::FollowUp);
    queue.enqueue("steer one".into(), PromptBehavior::Steer);
    queue.enqueue("steer two".into(), PromptBehavior::Steer);

    assert_eq!(
        queue.take_steering_batch(QueueMode::OneAtATime),
        ["steer one"]
    );
    assert_eq!(
        queue.take_steering_batch(QueueMode::OneAtATime),
        ["steer two"]
    );
    assert!(queue.take_steering_batch(QueueMode::OneAtATime).is_empty());
    assert_eq!(
        queue.take_next(),
        Some(("later".into(), PromptBehavior::FollowUp))
    );
    assert_eq!(queue.take_next(), None);
}

#[tokio::test]
async fn submit_consumes_steering_then_follow_up_prompts_as_new_user_turns() {
    let mut agent = super::MuAgent::demo().expect("demo agent");
    let queue = agent.prompt_queue();
    let mut inserted_steering = false;
    let mut inserted_follow_up = false;
    let mut queued_started = Vec::new();

    agent
        .submit("initial request".into(), &mut |event| match event {
            super::AgentEvent::TextDelta(_) if !inserted_steering => {
                inserted_steering = true;
                queue.enqueue("steer the current work".into(), PromptBehavior::Steer);
            }
            super::AgentEvent::QueuedPromptStarted {
                behavior: PromptBehavior::Steer,
                ..
            } if !inserted_follow_up => {
                inserted_follow_up = true;
                queued_started.push(PromptBehavior::Steer);
                queue.enqueue("after that, summarize".into(), PromptBehavior::FollowUp);
            }
            super::AgentEvent::QueuedPromptStarted { behavior, .. } => {
                queued_started.push(behavior);
            }
            _ => {}
        })
        .await
        .expect("complete queued turns");

    assert_eq!(
        queued_started,
        [PromptBehavior::Steer, PromptBehavior::FollowUp]
    );
    let input = responses_input(&agent.transcript);
    let user_text = input
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .filter_map(|message| message.pointer("/content/0/text").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert_eq!(
        user_text,
        [
            "initial request",
            "steer the current work",
            "after that, summarize",
        ]
    );
}

#[tokio::test]
async fn queue_mode_settings_control_scripted_agent_request_batches() {
    let cases = [
        ("steeringMode", PromptBehavior::Steer, "all", 1),
        ("steeringMode", PromptBehavior::Steer, "one-at-a-time", 2),
        ("followUpMode", PromptBehavior::FollowUp, "all", 2),
        ("followUpMode", PromptBehavior::FollowUp, "one-at-a-time", 3),
    ];
    for (key, behavior, mode, expected_assistant_turns) in cases {
        let root = std::env::temp_dir().join(format!("mu-queue-mode-{}", uuid::Uuid::new_v4()));
        let mut agent = scripted_agent(
            &root,
            (0..4)
                .map(|index| Ok(scripted_reply(&format!("reply {index}"), vec![])))
                .collect(),
        );
        let settings_path = root.join("agent/settings.json");
        std::fs::create_dir_all(settings_path.parent().unwrap()).unwrap();
        std::fs::write(&settings_path, format!(r#"{{"{key}":"{mode}"}}"#)).unwrap();
        agent
            .prompt_queue
            .enqueue("queued one".to_owned(), behavior);
        agent
            .prompt_queue
            .enqueue("queued two".to_owned(), behavior);

        agent
            .submit("initial prompt".to_owned(), &mut |_| {})
            .await
            .unwrap();
        let assistant_turns = agent
            .transcript
            .iter()
            .filter(|message| matches!(message, crate::agent_loop::AgentMessage::Assistant(_)))
            .count();
        assert_eq!(
            assistant_turns, expected_assistant_turns,
            "{key}={mode} should batch queued {behavior:?} messages as Pi does"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn restores_codemode_store_and_tool_loadout_from_the_active_session_branch() {
    let input = [
            r#"{"type":"session","version":3,"id":"session","timestamp":"t","cwd":"/tmp"}"#,
            r#"{"type":"custom","id":"root","parentId":null,"timestamp":"t","customType":"codemode-store","data":{"set":{"count":1,"old":true},"delete":[]}}"#,
            r#"{"type":"custom","id":"sibling","parentId":"root","timestamp":"t","customType":"codemode-store","data":{"set":{"sibling":true},"delete":[]}}"#,
            r#"{"type":"custom","id":"leaf","parentId":"root","timestamp":"t","customType":"codemode-store","data":{"set":{"count":2},"delete":["old"]}}"#,
            r#"{"type":"custom","id":"loadout","parentId":"leaf","timestamp":"t","customType":"mcp-tool-loadout","data":{"active":["mcp__test__echo"]}}"#,
        ]
        .join("\n");
    let session = SessionDocument::parse_jsonl(&input).expect("parse session branch");
    assert_eq!(
        session_codemode_store(&session),
        serde_json::json!({"count":2})
    );
    assert_eq!(
        session_active_mcp_tools(&session),
        ["mcp__test__echo".to_owned()].into()
    );
}

#[test]
fn reads_final_openai_responses_reasoning_when_no_reasoning_delta_arrived() {
    let item = serde_json::json!({
        "type":"reasoning",
        "summary":[{"type":"summary_text","text":"Plan the change."},{"type":"summary_text","text":"Then implement it."}],
        "content":[{"type":"reasoning_text","text":"private detail"}]
    });
    assert_eq!(
        super::response_reasoning(&item),
        "Plan the change.\n\nThen implement it."
    );
}

#[test]
fn allows_access_gateway_and_rejects_untrusted_hosts() {
    assert!(trusted_url("https://gateway.opencode.cloudflare.dev/openai").is_ok());
    assert!(trusted_url("https://attacker.example/openai").is_err());
}

#[test]
fn reads_custom_model_choices_and_keeps_the_default_model_available() {
    let dir = std::env::temp_dir().join(format!("mu-model-config-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create model config directory");
    std::fs::write(
        dir.join("models.json"),
        r#"{"providers":{"oracle":{"models":[{"id":"alpha"},{"id":"beta"}]}}}"#,
    )
    .expect("write model config");

    let models = configured_models(&dir, "oracle", "beta").expect("read model choices");
    assert_eq!(models, ["alpha", "beta"]);
    let includes_unlisted_default =
        configured_models(&dir, "oracle", "gamma").expect("include configured default");
    assert_eq!(includes_unlisted_default, ["gamma", "alpha", "beta"]);
    std::fs::remove_dir_all(dir).expect("remove model config directory");

    let absent = configured_models(Path::new("/no/such/mu/config"), "oracle", "default")
        .expect("fall back to default model");
    assert_eq!(absent, ["default"]);
}
