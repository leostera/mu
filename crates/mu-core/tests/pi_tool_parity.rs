//! Executable specifications ported from pinned Pi tool tests.
//!
//! The ignored tests are intentionally kept compiling as a backlog: each names a
//! concrete Pi behavior that Mu does not yet support. Remove its `ignore` marker
//! only when the matching implementation is ready.

use std::{
    fs,
    path::{Path, PathBuf},
};

use mu_core::tools::{definitions, execute};
use serde_json::{Value, json};

fn temp_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mu-pi-{label}-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&path).expect("create temporary fixture directory");
    path
}

fn cleanup(path: &Path) {
    fs::remove_dir_all(path).expect("remove temporary fixture directory");
}

#[test]
fn current_builtin_definitions_are_valid_json_schemas() {
    let definitions = definitions();
    let mut names = definitions
        .iter()
        .filter_map(|definition| definition.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "bash",
            "edit",
            "find",
            "grep",
            "ls",
            "powershell",
            "read",
            "write",
        ]
    );

    for definition in definitions {
        let schema = &definition["parameters"];
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        assert!(schema.get("required").is_none() || schema["required"].as_array().is_some());
    }
}

// Pi source: packages/coding-agent/test/tools.test.ts and test/default-tools-setting.test.ts.
#[tokio::test]
async fn pi_default_builtin_tool_set_contains_all_eight_tools() {
    let definitions = definitions();
    let mut names = definitions
        .iter()
        .filter_map(|definition| definition.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "bash",
            "edit",
            "find",
            "grep",
            "ls",
            "powershell",
            "read",
            "write"
        ]
    );
}

// Pi source: packages/coding-agent/test/tools.test.ts and edit-tool-legacy-input.test.ts.
#[tokio::test]
async fn pi_edit_applies_disjoint_edits_in_one_atomic_call() {
    let dir = temp_dir("edit-multi");
    let path = dir.join("file.txt");
    fs::write(&path, "alpha\nbeta\ngamma\ndelta\n").unwrap();

    let result = execute(
        &dir,
        "edit",
        &json!({
            "path":"file.txt",
            "edits":[
                {"oldText":"alpha\n","newText":"ALPHA\n"},
                {"oldText":"gamma\n","newText":"GAMMA\n"}
            ]
        }),
    )
    .await
    .unwrap();

    assert!(result.content.contains("Successfully replaced 2 block(s)"));
    let details = result.details.as_ref().unwrap();
    assert!(details["diff"].as_str().unwrap().contains("ALPHA"));
    assert!(details["diff"].as_str().unwrap().contains("GAMMA"));
    assert!(details["patch"].as_str().unwrap().contains("--- "));
    assert!(details["patch"].as_str().unwrap().contains("+++ "));
    assert!(details["patch"].as_str().unwrap().contains("@@"));
    assert_eq!(details["firstChangedLine"], 1);
    assert_eq!(
        fs::read_to_string(path).unwrap(),
        "ALPHA\nbeta\nGAMMA\ndelta\n"
    );
    cleanup(&dir);
}

#[tokio::test]
async fn pi_edit_reports_enoent_with_the_requested_path() {
    let dir = temp_dir("edit-enoent");
    let error = execute(
        &dir,
        "edit",
        &json!({"path":"missing.txt","edits":[{"oldText":"a","newText":"b"}]}),
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid arguments: Could not edit file: missing.txt. Error code: ENOENT."
    );
    assert!(!dir.join("missing.txt").exists());
    cleanup(&dir);
}

// Pi source: packages/coding-agent/test/edit-tool-legacy-input.test.ts.
#[tokio::test]
async fn pi_edit_accepts_legacy_mixed_object_and_stringified_edits() {
    let dir = temp_dir("edit-legacy-input");
    let path = dir.join("input.txt");
    fs::write(&path, "one two three").unwrap();

    let encoded_edits = serde_json::to_string(&json!([{"oldText":"one","newText":"ONE"}])).unwrap();
    let stringified = execute(
        &dir,
        "edit",
        &json!({"path":"input.txt","edits":encoded_edits}),
    )
    .await
    .unwrap();
    assert!(
        stringified
            .content
            .contains("Successfully replaced 1 block(s)")
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "ONE two three");

    execute(
        &dir,
        "edit",
        &json!({"path":"input.txt","edits":{"oldText":"two","newText":"TWO"}}),
    )
    .await
    .unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap(), "ONE TWO three");

    let mixed = execute(
        &dir,
        "edit",
        &json!({
            "path":"input.txt",
            "edits":[{"oldText":"three","newText":"THREE"}],
            "oldText":"ONE",
            "newText":"one"
        }),
    )
    .await
    .unwrap();
    assert!(mixed.content.contains("Successfully replaced 2 block(s)"));
    assert_eq!(fs::read_to_string(&path).unwrap(), "one TWO THREE");

    let malformed = execute(
        &dir,
        "edit",
        &json!({"path":"input.txt","edits":"not json"}),
    )
    .await
    .unwrap_err();
    assert!(
        malformed
            .to_string()
            .contains("Edit tool input is invalid. edits must contain at least one replacement.")
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "one TWO THREE");
    cleanup(&dir);
}

// Pi source: packages/coding-agent/test/tools.test.ts, edit diff cases.
#[tokio::test]
async fn pi_edit_diff_collapses_large_unchanged_gaps() {
    let dir = temp_dir("edit-diff-gaps");
    let path = dir.join("many-lines.txt");
    let lines = (1..=600)
        .map(|line| format!("line {line:03}"))
        .collect::<Vec<_>>();
    fs::write(&path, lines.join("\n") + "\n").unwrap();

    let result = execute(
        &dir,
        "edit",
        &json!({"path":"many-lines.txt","edits":[
            {"oldText":"line 100\n","newText":"LINE 100\n"},
            {"oldText":"line 300\n","newText":"LINE 300\n"},
            {"oldText":"line 500\n","newText":"LINE 500\n"}
        ]}),
    )
    .await
    .unwrap();
    let diff = result.details.unwrap()["diff"].as_str().unwrap().to_owned();
    assert!(diff.contains("LINE 100"));
    assert!(diff.contains("LINE 300"));
    assert!(diff.contains("LINE 500"));
    assert!(diff.contains("..."));
    assert!(diff.lines().count() < 50);
    cleanup(&dir);
}

// Pi source: packages/coding-agent/test/tools.test.ts, grep tool cases.
#[tokio::test]
async fn pi_grep_obeys_global_limit_and_context_without_returning_later_matches() {
    let dir = temp_dir("grep-context");
    fs::write(
        dir.join("context.txt"),
        "before\nmatch one\nafter\nmiddle\nmatch two\nafter two",
    )
    .unwrap();

    let result = execute(
        &dir,
        "grep",
        &json!({"pattern":"match","path":"context.txt","limit":1,"context":1}),
    )
    .await
    .unwrap();

    assert!(result.content.contains("context.txt-1- before"));
    assert!(result.content.contains("context.txt:2: match one"));
    assert!(result.content.contains("context.txt-3- after"));
    assert!(
        result
            .content
            .contains("[1 matches limit reached. Use limit=2 for more, or refine pattern]")
    );
    assert_eq!(result.details.as_ref().unwrap()["matchLimitReached"], 1);
    assert!(!result.content.contains("match two"));
    cleanup(&dir);
}

#[tokio::test]
async fn pi_grep_treats_flag_like_patterns_as_search_text_and_supports_literal_mode() {
    let dir = temp_dir("grep-patterns");
    fs::write(dir.join("sample.txt"), "-flag value\nNeedle\nliteral.dot\n").unwrap();

    let flag = execute(
        &dir,
        "grep",
        &json!({"pattern":"-flag","path":"sample.txt"}),
    )
    .await
    .unwrap();
    assert!(flag.content.contains("sample.txt:1: -flag value"));

    let case_insensitive = execute(
        &dir,
        "grep",
        &json!({"pattern":"needle","path":"sample.txt","ignoreCase":true}),
    )
    .await
    .unwrap();
    assert!(case_insensitive.content.contains("sample.txt:2: Needle"));

    let literal = execute(
        &dir,
        "grep",
        &json!({"pattern":".","path":"sample.txt","literal":true}),
    )
    .await
    .unwrap();
    assert_eq!(literal.content, "sample.txt:3: literal.dot");
    cleanup(&dir);
}

#[tokio::test]
async fn search_tools_resolve_normalized_paths_against_the_execution_cwd() {
    let dir = temp_dir("normalized-search-path");
    let folder = dir.join("spaced folder");
    fs::create_dir_all(&folder).unwrap();
    fs::write(folder.join("source.txt"), "needle\n").unwrap();
    let path_with_nonbreaking_space = "@spaced\u{00a0}folder";

    let grep = execute(
        &dir,
        "grep",
        &json!({"pattern":"needle","path":path_with_nonbreaking_space}),
    )
    .await
    .unwrap();
    assert_eq!(grep.content, "source.txt:1: needle");

    let find = execute(
        &dir,
        "find",
        &json!({"pattern":"*.txt","path":path_with_nonbreaking_space}),
    )
    .await
    .unwrap();
    assert_eq!(find.content, "source.txt");

    let ls = execute(&dir, "ls", &json!({"path":path_with_nonbreaking_space}))
        .await
        .unwrap();
    assert_eq!(ls.content, "source.txt");
    cleanup(&dir);
}

// Pi source: packages/coding-agent/test/tools.test.ts, find result-limit cases.
#[tokio::test]
async fn pi_find_preserves_result_limit_details() {
    let dir = temp_dir("find-limit");
    fs::write(dir.join("one.txt"), "one").unwrap();
    fs::write(dir.join("two.txt"), "two").unwrap();
    let result = execute(
        &dir,
        "find",
        &json!({"pattern":"*.txt","path":".","limit":1}),
    )
    .await
    .unwrap();
    assert!(result.content.contains("[1 results limit reached]"));
    assert_eq!(result.details.as_ref().unwrap()["resultLimitReached"], 1);
    cleanup(&dir);
}

// Pi source: packages/coding-agent/test/tools.test.ts, grep limit cases.
#[tokio::test]
async fn pi_grep_reports_a_limit_only_when_an_additional_match_exists() {
    let dir = temp_dir("grep-limit-boundary");
    fs::write(dir.join("one.txt"), "needle\n").unwrap();
    fs::write(dir.join("two.txt"), "needle\nneedle\n").unwrap();

    let exactly_one = execute(
        &dir,
        "grep",
        &json!({"pattern":"needle","path":"one.txt","limit":1}),
    )
    .await
    .unwrap();
    assert!(!exactly_one.content.contains("limit reached"));

    let exactly_two = execute(
        &dir,
        "grep",
        &json!({"pattern":"needle","path":"two.txt","limit":2}),
    )
    .await
    .unwrap();
    assert!(!exactly_two.content.contains("limit reached"));

    let limited = execute(
        &dir,
        "grep",
        &json!({"pattern":"needle","path":"two.txt","limit":1}),
    )
    .await
    .unwrap();
    assert!(limited.content.contains("1 matches limit reached"));
    cleanup(&dir);
}

// Pi source: packages/coding-agent/test/tools.test.ts, find and ls cases.
#[tokio::test]
async fn pi_find_includes_hidden_files_and_respects_gitignore() {
    let dir = temp_dir("find");
    fs::create_dir_all(dir.join(".secret")).unwrap();
    fs::write(dir.join(".secret/hidden.txt"), "hidden").unwrap();
    fs::write(dir.join("visible.txt"), "visible").unwrap();
    fs::write(dir.join(".gitignore"), "ignored.txt\n").unwrap();
    fs::write(dir.join("ignored.txt"), "ignored").unwrap();

    let result = execute(&dir, "find", &json!({"pattern":"**/*.txt","path":"."}))
        .await
        .unwrap();

    assert!(result.content.contains("visible.txt"));
    assert!(result.content.contains(".secret/hidden.txt"));
    assert!(!result.content.contains("ignored.txt"));
    cleanup(&dir);
}

#[tokio::test]
async fn pi_find_treats_flag_like_patterns_as_globs_and_reports_invalid_globs() {
    let dir = temp_dir("find-flags");
    fs::write(dir.join("-report.txt"), "report").unwrap();

    let found = execute(&dir, "find", &json!({"pattern":"-report.txt","path":"."}))
        .await
        .unwrap();
    assert_eq!(found.content, "-report.txt");

    let error = execute(&dir, "find", &json!({"pattern":"{invalid","path":"."}))
        .await
        .unwrap_err();
    assert!(!error.to_string().is_empty());
    cleanup(&dir);
}

#[tokio::test]
async fn pi_ls_lists_dotfiles_and_marks_directories() {
    let dir = temp_dir("ls");
    fs::write(dir.join(".hidden-file"), "secret").unwrap();
    fs::create_dir(dir.join(".hidden-dir")).unwrap();

    let result = execute(&dir, "ls", &json!({"path":"."})).await.unwrap();

    assert!(result.content.contains(".hidden-file"));
    assert!(result.content.contains(".hidden-dir/"));
    cleanup(&dir);
}

// Pi source: packages/coding-agent/test/tools.test.ts, bash nonzero-exit case.
#[tokio::test]
async fn pi_bash_nonzero_exit_is_returned_as_an_error_tool_result() {
    let dir = temp_dir("bash-exit");
    let result = execute(&dir, "bash", &json!({"command":"echo out; exit 3"}))
        .await
        .unwrap();
    assert_eq!(result.content, "out\n\n\nCommand exited with code 3");
    assert!(result.is_error);
    assert_eq!(
        result.structured_content.as_ref().unwrap()["output"],
        "out\n"
    );
    assert_eq!(
        result.structured_content.as_ref().unwrap()["truncated"],
        false
    );
    assert_eq!(result.structured_content.as_ref().unwrap()["exit_code"], 3);
    assert!(result.structured_content.as_ref().unwrap()["wall_time_seconds"].is_number());
    cleanup(&dir);
}

#[cfg(not(windows))]
#[tokio::test]
async fn powershell_reports_its_windows_only_platform_requirement() {
    let dir = temp_dir("powershell-platform");
    let error = execute(&dir, "powershell", &json!({"command":"Get-Location"}))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("only available on Windows"));
    cleanup(&dir);
}

#[cfg(windows)]
#[tokio::test]
async fn pi_powershell_runs_with_process_local_utf8_configuration() {
    let dir = temp_dir("powershell");
    let result = execute(
        &dir,
        "powershell",
        &json!({"command":"[Console]::Write('héllo')"}),
    )
    .await
    .unwrap();
    assert!(result.content.contains("héllo"));
    cleanup(&dir);
}
