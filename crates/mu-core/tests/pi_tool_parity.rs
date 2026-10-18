//! Executable specifications ported from pinned Pi tool tests.
//!
//! The ignored tests are intentionally kept compiling as a backlog: each names a
//! concrete Pi behavior that Mu does not yet support. Remove its `ignore` marker
//! only when the matching implementation is ready.

use std::{
    fs,
    path::{Path, PathBuf},
};

use mu_core::tools::{TextEdit, compute_edit_diff, definitions, execute};
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
fn builtin_definition_order_matches_pi_tool_registry() {
    let tool_definitions = definitions();
    let names = tool_definitions
        .iter()
        .filter_map(|definition| definition.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            "read",
            "bash",
            "powershell",
            "edit",
            "write",
            "grep",
            "find",
            "ls"
        ]
    );
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

#[test]
fn builtin_definitions_match_pi_descriptions_and_numeric_parameter_shapes() {
    let definitions = definitions();
    let find = |name: &str| {
        definitions
            .iter()
            .find(|definition| definition["name"] == name)
            .unwrap()
    };

    assert_eq!(
        find("read")["parameters"]["properties"]["offset"]["type"],
        "number"
    );
    assert_eq!(
        find("read")["parameters"]["properties"]["offset"]["description"],
        "Line number to start reading from (1-indexed)"
    );
    assert!(find("read")["parameters"]["properties"]["offset"]["minimum"].is_null());
    assert!(find("read")["parameters"]["properties"]["limit"]["minimum"].is_null());
    assert_eq!(
        find("grep")["parameters"]["properties"]["literal"]["description"],
        "Treat pattern as literal string instead of regex (default: false)"
    );
    assert!(
        find("read")["description"]
            .as_str()
            .unwrap()
            .contains("Images are sent as attachments")
    );
    assert!(
        find("bash")["description"]
            .as_str()
            .unwrap()
            .contains("full output is saved to a temp file")
    );
    assert!(find("edit")["parameters"]["properties"]["edits"]["minItems"].is_null());
    assert_eq!(
        find("edit")["parameters"]["properties"]["edits"]["items"]["properties"]["oldText"]["description"],
        "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."
    );
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
async fn pi_write_formats_directory_target_errors_like_node_fs_open() {
    let dir = temp_dir("write-directory-error");
    let error = execute(&dir, "write", &json!({"path":".","content":"content"}))
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        format!(
            "EISDIR: illegal operation on a directory, open '{}'",
            dir.display()
        )
    );
    cleanup(&dir);
}

#[tokio::test]
async fn pi_write_formats_parent_creation_errors_like_node_fs_mkdir() {
    let dir = temp_dir("write-mkdir-error");
    fs::write(dir.join("blocker"), "file").unwrap();
    let parent = dir.join("blocker/child");

    let error = execute(
        &dir,
        "write",
        &json!({"path":"blocker/child/file.txt","content":"content"}),
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        format!("ENOTDIR: not a directory, mkdir '{}'", parent.display())
    );
    cleanup(&dir);
}

#[tokio::test]
async fn pi_edit_formats_directory_read_errors_like_node_fs_read() {
    let dir = temp_dir("edit-directory-error");
    let error = execute(
        &dir,
        "edit",
        &json!({"path":".","edits":[{"oldText":"old","newText":"new"}]}),
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        format!(
            "EISDIR: illegal operation on a directory, read '{}'",
            dir.display()
        )
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
        "Could not edit file: missing.txt. Error code: ENOENT."
    );
    assert!(!dir.join("missing.txt").exists());
    cleanup(&dir);
}

// Pi source: packages/coding-agent/test/edit-tool-legacy-input.test.ts.
#[tokio::test]
async fn pi_edit_application_errors_are_returned_without_mu_prefixes() {
    let dir = temp_dir("edit-apply-error");
    let path = dir.join("file.txt");
    fs::write(&path, "original content").unwrap();
    let error = execute(
        &dir,
        "edit",
        &json!({"path":"file.txt","edits":[{"oldText":"missing","newText":"replacement"}]}),
    )
    .await
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "Could not find the exact text in file.txt. The old text must match exactly including all whitespace and newlines."
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "original content");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_edit_diff_line_number_width_counts_trailing_empty_line() {
    let dir = temp_dir("edit-diff-line-width");
    let path = dir.join("nine-lines.txt");
    let original = (1..=9)
        .map(|line| format!("line{line}"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(&path, &original).unwrap();

    let result = execute(
        &dir,
        "edit",
        &json!({"path":"nine-lines.txt","edits":[{"oldText":"line9","newText":"changed"}]}),
    )
    .await
    .unwrap();
    let diff = result.details.unwrap()["diff"].as_str().unwrap().to_owned();
    assert!(diff.contains("- 9 line9"), "{diff}");
    assert!(diff.contains("+ 9 changed"), "{diff}");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_edit_rejects_noop_replacements_with_single_and_multi_edit_errors() {
    let dir = temp_dir("edit-noop");
    let path = dir.join("same.txt");
    fs::write(&path, "alpha beta").unwrap();

    let single = execute(
        &dir,
        "edit",
        &json!({"path":"same.txt","edits":[{"oldText":"alpha","newText":"alpha"}]}),
    )
    .await
    .unwrap_err();
    assert_eq!(
        single.to_string(),
        "No changes made to same.txt. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
    );

    let multiple = execute(
        &dir,
        "edit",
        &json!({"path":"same.txt","edits":[
            {"oldText":"alpha","newText":"alpha"},
            {"oldText":"beta","newText":"beta"}
        ]}),
    )
    .await
    .unwrap_err();
    assert_eq!(
        multiple.to_string(),
        "No changes made to same.txt. The replacements produced identical content."
    );
    assert_eq!(fs::read_to_string(path).unwrap(), "alpha beta");
    cleanup(&dir);
}

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

    let legacy_over_invalid_edits = execute(
        &dir,
        "edit",
        &json!({"path":"input.txt","edits":"not an array","oldText":"one","newText":"ONE"}),
    )
    .await
    .unwrap();
    assert!(
        legacy_over_invalid_edits
            .content
            .contains("Successfully replaced 1 block(s)")
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "ONE TWO THREE");

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
    let malformed_object = execute(
        &dir,
        "edit",
        &json!({"path":"input.txt","edits":{"unexpected":"shape"}}),
    )
    .await
    .unwrap_err();
    assert_eq!(
        malformed_object.to_string(),
        "Edit tool input is invalid. edits must contain at least one replacement."
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "ONE TWO THREE");
    cleanup(&dir);
}

// Pi source: packages/coding-agent/test/tools.test.ts, edit diff cases.
#[test]
fn pi_edit_preview_is_non_mutating_and_returns_the_renderer_result_shape() {
    let dir = temp_dir("edit-preview");
    let original = "\u{feff}before\r\nafter\r\n";
    fs::write(dir.join("preview.txt"), original).unwrap();
    let preview = compute_edit_diff(
        &dir,
        "preview.txt",
        &[TextEdit {
            old_text: "before".into(),
            new_text: "changed".into(),
        }],
    );

    assert_eq!(preview["firstChangedLine"], 1);
    assert!(preview["diff"].as_str().unwrap().contains("changed"));
    assert_eq!(
        fs::read_to_string(dir.join("preview.txt")).unwrap(),
        original
    );

    let missing = compute_edit_diff(
        &dir,
        "missing.txt",
        &[TextEdit {
            old_text: "old".into(),
            new_text: "new".into(),
        }],
    );
    assert_eq!(
        missing["error"],
        "Could not edit file: missing.txt. Error code: ENOENT."
    );
    cleanup(&dir);
}

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
async fn pi_grep_limit_preserves_javascript_number_semantics() {
    let dir = temp_dir("grep-number-limit");
    fs::write(
        dir.join("matches.txt"),
        "match one\nmatch two\nmatch three\n",
    )
    .unwrap();

    let fractional = execute(
        &dir,
        "grep",
        &json!({"pattern":"match","path":"matches.txt","limit":1.5}),
    )
    .await
    .unwrap();
    assert!(fractional.content.contains("matches.txt:1: match one"));
    assert!(fractional.content.contains("matches.txt:2: match two"));
    assert!(!fractional.content.contains("matches.txt:3: match three"));
    assert!(
        fractional
            .content
            .contains("1.5 matches limit reached. Use limit=3 for more")
    );
    assert_eq!(fractional.details.unwrap()["matchLimitReached"], 1.5);

    let negative = execute(
        &dir,
        "grep",
        &json!({"pattern":"match","path":"matches.txt","limit":-2}),
    )
    .await
    .unwrap();
    assert!(negative.content.contains("matches.txt:1: match one"));
    assert!(!negative.content.contains("matches.txt:2: match two"));
    assert!(negative.content.contains("1 matches limit reached"));
    assert_eq!(negative.details.unwrap()["matchLimitReached"], 1);
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

// Pi source: packages/coding-agent/test/tools.test.ts, read boundary and empty-file cases.
#[tokio::test]
async fn pi_grep_fractional_context_preserves_pi_number_and_array_index_semantics() {
    let dir = temp_dir("grep-fractional-context");
    fs::write(dir.join("input.txt"), "zero\nmatch\nafter\n").unwrap();

    let result = execute(&dir, "grep", &json!({"pattern":"match","context":0.5}))
        .await
        .unwrap();
    assert_eq!(result.content, "input.txt-1.5- \ninput.txt-2.5- ");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_grep_binary_context_uses_original_nul_containing_file_lines() {
    let dir = temp_dir("grep-nul-context");
    fs::write(dir.join("binary.txt"), b"left\0right\n").unwrap();

    let result = execute(&dir, "grep", &json!({"pattern":"right","context":1}))
        .await
        .unwrap();
    assert_eq!(result.content, "binary.txt-1- left\0right\nbinary.txt:2: ");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_grep_context_uses_pi_carriage_return_line_normalization() {
    let dir = temp_dir("grep-cr-context");
    fs::write(dir.join("input.txt"), b"target\rcontext\rend\n").unwrap();

    let result = execute(&dir, "grep", &json!({"pattern":"target","context":1}))
        .await
        .unwrap();
    assert_eq!(result.content, "input.txt:1: target\ninput.txt-2- context");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_grep_removes_embedded_carriage_returns_from_match_lines() {
    let dir = temp_dir("grep-carriage-return");
    fs::write(dir.join("input.txt"), b"a\rb\rc\n").unwrap();

    let result = execute(&dir, "grep", &json!({"pattern":"b"}))
        .await
        .unwrap();
    assert_eq!(result.content, "input.txt:1: abc");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_grep_decodes_invalid_utf8_and_treats_nul_as_a_line_boundary() {
    let dir = temp_dir("grep-binary-decoding");
    fs::write(dir.join("invalid.txt"), b"left\xffright\n").unwrap();
    fs::write(dir.join("binary.txt"), b"left\0right\n").unwrap();

    let result = execute(&dir, "grep", &json!({"pattern":"right"}))
        .await
        .unwrap();
    assert_eq!(
        result.content,
        "binary.txt:2: right\ninvalid.txt:1: left\u{fffd}right"
    );
    cleanup(&dir);
}

#[tokio::test]
async fn pi_read_formats_directory_errors_like_node_fs_read() {
    let dir = temp_dir("read-directory-error");
    let error = execute(&dir, "read", &json!({"path":"."}))
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        format!(
            "EISDIR: illegal operation on a directory, read '{}'",
            dir.display()
        )
    );
    cleanup(&dir);
}

#[tokio::test]
async fn pi_read_formats_missing_file_errors_like_node_fs_access() {
    let dir = temp_dir("read-missing-error");
    let path = dir.join("missing.txt");
    let error = execute(&dir, "read", &json!({"path":"missing.txt"}))
        .await
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        format!(
            "ENOENT: no such file or directory, access '{}'",
            path.display()
        )
    );
    cleanup(&dir);
}

#[tokio::test]
async fn pi_text_tools_replace_malformed_utf8_like_node_buffer_decoding() {
    let dir = temp_dir("lossy-utf8-tools");
    let path = dir.join("invalid.txt");
    fs::write(&path, b"left\xffright").unwrap();

    let read = execute(&dir, "read", &json!({"path":"invalid.txt"}))
        .await
        .unwrap();
    assert_eq!(read.content, "left\u{fffd}right");

    let edit = execute(
        &dir,
        "edit",
        &json!({"path":"invalid.txt","edits":[{"oldText":"right","newText":"RIGHT"}]}),
    )
    .await
    .unwrap();
    assert!(edit.content.contains("Successfully replaced 1 block(s)"));
    assert_eq!(fs::read_to_string(path).unwrap(), "left\u{fffd}RIGHT");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_read_matches_empty_file_and_trailing_newline_boundaries() {
    let dir = temp_dir("read-empty-boundaries");
    fs::write(dir.join("empty.txt"), "").unwrap();
    fs::write(dir.join("trailing-newline.txt"), "one\n").unwrap();

    assert_eq!(
        execute(&dir, "read", &json!({"path":"empty.txt"}))
            .await
            .unwrap()
            .content,
        ""
    );
    assert_eq!(
        execute(&dir, "read", &json!({"path":"empty.txt","offset":1}))
            .await
            .unwrap()
            .content,
        ""
    );
    assert_eq!(
        execute(
            &dir,
            "read",
            &json!({"path":"trailing-newline.txt","offset":2})
        )
        .await
        .unwrap()
        .content,
        ""
    );
    let beyond_empty = execute(&dir, "read", &json!({"path":"empty.txt","offset":2}))
        .await
        .unwrap_err();
    assert!(
        beyond_empty
            .to_string()
            .contains("Offset 2 is beyond end of file (1 lines total)")
    );

    let zero_limit = execute(&dir, "read", &json!({"path":"empty.txt","limit":0}))
        .await
        .unwrap();
    assert_eq!(
        zero_limit.content,
        "\n\n[1 more lines in file. Use offset=1 to continue.]"
    );
    cleanup(&dir);
}

#[tokio::test]
async fn pi_read_preserves_numeric_slice_semantics_for_fractional_ranges() {
    let dir = temp_dir("read-fractional-ranges");
    fs::write(dir.join("file.txt"), "one\ntwo\nthree").unwrap();

    let result = execute(
        &dir,
        "read",
        &json!({"path":"file.txt","offset":1.5,"limit":1.5}),
    )
    .await
    .unwrap();

    assert_eq!(
        result.content,
        "one\ntwo\n\n[1 more lines in file. Use offset=3 to continue.]"
    );
    cleanup(&dir);
}

// Pi source: packages/coding-agent/test/tools.test.ts, find result-limit cases.
#[tokio::test]
async fn pi_find_uses_fd_smart_case_glob_matching() {
    let dir = temp_dir("find-smart-case");
    fs::write(dir.join("UPPER.TXT"), "").unwrap();
    fs::write(dir.join("lower.txt"), "").unwrap();

    let insensitive = execute(&dir, "find", &json!({"pattern":"*.txt"}))
        .await
        .unwrap();
    assert_eq!(insensitive.content, "UPPER.TXT\nlower.txt");

    let sensitive = execute(&dir, "find", &json!({"pattern":"*.TXT"}))
        .await
        .unwrap();
    assert_eq!(sensitive.content, "UPPER.TXT");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_find_matches_path_globs_from_nested_roots_and_absolute_patterns() {
    let dir = temp_dir("find-path-glob");
    fs::create_dir_all(dir.join("outer/src")).unwrap();
    fs::write(dir.join("outer/src/target.rs"), "").unwrap();
    fs::write(dir.join("outer/other.rs"), "").unwrap();

    let relative = execute(&dir, "find", &json!({"pattern":"src/*.rs"}))
        .await
        .unwrap();
    assert_eq!(relative.content, "outer/src/target.rs");

    let absolute_pattern = format!("{}/*.rs", dir.join("outer/src").display());
    let absolute = execute(&dir, "find", &json!({"pattern":absolute_pattern}))
        .await
        .unwrap();
    assert_eq!(absolute.content, "outer/src/target.rs");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_grep_ignores_an_empty_glob_filter() {
    let dir = temp_dir("grep-empty-glob");
    fs::write(dir.join("match.txt"), "needle\n").unwrap();
    fs::write(dir.join("other.md"), "needle\n").unwrap();

    let result = execute(&dir, "grep", &json!({"pattern":"needle","glob":""}))
        .await
        .unwrap();
    assert_eq!(result.content, "match.txt:1: needle\nother.md:1: needle");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_optional_search_arguments_treat_null_like_pi_defaults() {
    let dir = temp_dir("search-null-defaults");
    fs::write(dir.join("match.txt"), "needle\n").unwrap();

    let ls = execute(&dir, "ls", &json!({"path":null,"limit":null}))
        .await
        .unwrap();
    assert_eq!(ls.content, "match.txt");
    let find = execute(
        &dir,
        "find",
        &json!({"path":null,"pattern":"*.txt","limit":null}),
    )
    .await
    .unwrap();
    assert_eq!(find.content, "match.txt");
    let grep = execute(
        &dir,
        "grep",
        &json!({
            "path":null,
            "pattern":"needle",
            "glob":null,
            "ignoreCase":null,
            "literal":null,
            "context":null,
            "limit":null
        }),
    )
    .await
    .unwrap();
    assert_eq!(grep.content, "match.txt:1: needle");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_find_preserves_fd_numeric_limit_parse_errors() {
    let dir = temp_dir("find-invalid-limit");
    fs::write(dir.join("one.txt"), "").unwrap();

    let negative = execute(&dir, "find", &json!({"pattern":"*.txt","limit":-1}))
        .await
        .unwrap_err();
    assert_eq!(
        negative.to_string(),
        "error: a value is required for '--max-results <count>' but none was supplied\n\nFor more information, try '--help'."
    );

    let fractional = execute(&dir, "find", &json!({"pattern":"*.txt","limit":1.5}))
        .await
        .unwrap_err();
    assert_eq!(
        fractional.to_string(),
        "error: invalid value '1.5' for '--max-results <count>': invalid digit found in string\n\nFor more information, try '--help'."
    );
    cleanup(&dir);
}

#[tokio::test]
async fn pi_find_treats_zero_limit_as_unbounded_fd_results() {
    let dir = temp_dir("find-zero-limit");
    fs::write(dir.join("one.txt"), "1").unwrap();
    fs::write(dir.join("two.txt"), "2").unwrap();

    let result = execute(&dir, "find", &json!({"pattern":"*.txt","limit":0}))
        .await
        .unwrap();
    assert_eq!(
        result.content,
        "one.txt\ntwo.txt\n\n[0 results limit reached]"
    );
    assert_eq!(result.details.unwrap()["resultLimitReached"], 0);
    cleanup(&dir);
}

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

#[tokio::test]
async fn pi_grep_stops_parent_gitignore_rules_at_nested_repository_boundaries() {
    let dir = temp_dir("grep-nested-repo");
    fs::create_dir_all(dir.join(".git")).unwrap();
    fs::create_dir_all(dir.join("nested/.git")).unwrap();
    fs::write(dir.join(".gitignore"), "nested/secret.txt\n").unwrap();
    fs::write(dir.join("nested/secret.txt"), "needle\n").unwrap();

    let result = execute(&dir, "grep", &json!({"pattern":"needle","path":"."}))
        .await
        .unwrap();

    assert_eq!(result.content, "nested/secret.txt:1: needle");
    cleanup(&dir);
}

// Pi source: packages/coding-agent/test/tools.test.ts, grep limit cases.
#[tokio::test]
async fn pi_grep_reports_a_limit_when_match_count_reaches_the_limit() {
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
    assert!(exactly_one.content.contains("1 matches limit reached"));
    assert_eq!(
        exactly_one.details.as_ref().unwrap()["matchLimitReached"],
        1
    );

    let exactly_two = execute(
        &dir,
        "grep",
        &json!({"pattern":"needle","path":"two.txt","limit":2}),
    )
    .await
    .unwrap();
    assert!(exactly_two.content.contains("2 matches limit reached"));
    assert_eq!(
        exactly_two.details.as_ref().unwrap()["matchLimitReached"],
        2
    );

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
async fn pi_find_orders_results_like_fd_default_sorting() {
    let dir = temp_dir("find-order");
    fs::create_dir_all(dir.join("nested")).unwrap();
    for path in ["z.txt", "a.txt", "nested/b.txt", "10.txt", "2.txt"] {
        fs::write(dir.join(path), "file").unwrap();
    }

    let result = execute(&dir, "find", &json!({"pattern":"**/*.txt","path":"."}))
        .await
        .unwrap();

    assert_eq!(result.content, "10.txt\n2.txt\na.txt\nnested/b.txt\nz.txt");
    cleanup(&dir);
}

#[cfg(unix)]
#[tokio::test]
async fn pi_find_does_not_mark_directory_symlinks_as_directories() {
    let dir = temp_dir("find-symlink");
    fs::create_dir(dir.join("actual-dir")).unwrap();
    std::os::unix::fs::symlink(dir.join("actual-dir"), dir.join("linked-dir")).unwrap();

    let result = execute(&dir, "find", &json!({"pattern":"linked-dir","path":"."}))
        .await
        .unwrap();

    assert_eq!(result.content, "linked-dir");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_find_stops_parent_gitignore_rules_at_nested_repository_boundaries() {
    let dir = temp_dir("find-nested-repo");
    fs::create_dir_all(dir.join(".git")).unwrap();
    fs::create_dir_all(dir.join("nested/.git")).unwrap();
    fs::write(dir.join(".gitignore"), "nested/secret.txt\n").unwrap();
    fs::write(
        dir.join("nested/secret.txt"),
        "visible in nested repository",
    )
    .unwrap();

    let result = execute(&dir, "find", &json!({"pattern":"**/*.txt","path":"."}))
        .await
        .unwrap();

    assert_eq!(result.content, "nested/secret.txt");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_find_keeps_fd_root_marker_when_a_name_trims_to_empty() {
    let dir = temp_dir("find-whitespace-only-name");
    fs::write(dir.join("   "), "content").unwrap();

    let result = execute(&dir, "find", &json!({"pattern":"*"}))
        .await
        .unwrap();
    assert_eq!(result.content, "/");
    cleanup(&dir);
}

#[tokio::test]
async fn pi_find_trims_whitespace_from_fd_output_paths() {
    let dir = temp_dir("find-trim-output");
    fs::write(dir.join(" leading.txt "), "content").unwrap();

    let result = execute(&dir, "find", &json!({"pattern":"*leading.txt*"}))
        .await
        .unwrap();
    assert_eq!(result.content, " leading.txt");
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
    assert!(
        error
            .to_string()
            .starts_with("[fd error]: error parsing glob")
    );
    cleanup(&dir);
}

#[tokio::test]
async fn pi_search_tools_return_raw_regex_and_glob_errors() {
    let dir = temp_dir("search-parse-errors");

    let regex_error = execute(&dir, "grep", &json!({"pattern":"("}))
        .await
        .unwrap_err();
    assert!(
        regex_error
            .to_string()
            .starts_with("rg: regex parse error:")
    );

    let glob_error = execute(&dir, "grep", &json!({"pattern":"needle","glob":"{"}))
        .await
        .unwrap_err();
    assert!(glob_error.to_string().starts_with("rg: error parsing glob"));
    cleanup(&dir);
}

#[tokio::test]
async fn pi_search_tools_return_runtime_path_errors_without_mu_argument_prefixes() {
    let dir = temp_dir("search-path-errors");
    let missing = dir.join("missing");
    let expected_missing = format!("Path not found: {}", missing.display());
    for (tool, arguments) in [
        ("grep", json!({"path":"missing","pattern":"needle"})),
        ("ls", json!({"path":"missing"})),
    ] {
        let error = execute(&dir, tool, &arguments).await.unwrap_err();
        assert_eq!(error.to_string(), expected_missing, "{tool}");
    }
    let missing_find = execute(&dir, "find", &json!({"path":"missing","pattern":"*.txt"}))
        .await
        .unwrap_err();
    assert_eq!(
        missing_find.to_string(),
        format!(
            "[fd error]: Search path '{}' is not a directory.\n[fd error]: No valid search paths given.",
            missing.display()
        )
    );

    let file = dir.join("file.txt");
    fs::write(&file, "content").unwrap();
    let not_directory = execute(&dir, "ls", &json!({"path":"file.txt"}))
        .await
        .unwrap_err();
    assert_eq!(
        not_directory.to_string(),
        format!("Not a directory: {}", file.display())
    );

    let find_file = execute(&dir, "find", &json!({"path":"file.txt","pattern":"*.txt"}))
        .await
        .unwrap_err();
    assert_eq!(
        find_file.to_string(),
        format!(
            "[fd error]: Search path '{}' is not a directory.\n[fd error]: No valid search paths given.",
            file.display()
        )
    );
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

#[tokio::test]
async fn pi_bash_reports_a_missing_working_directory_before_spawn() {
    let missing = temp_dir("bash-missing-cwd").join("missing");
    let error = execute(&missing, "bash", &json!({"command":"true"}))
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        format!(
            "Working directory does not exist: {}\nCannot execute bash commands.",
            missing.display()
        )
    );
    cleanup(missing.parent().unwrap());
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
