use std::{
    io::{Read, Write},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use serde_json::{Value, json};
use vt100::Parser;

const ORACLE_VERSION: &str = "1.3.0";
const COLUMNS: u16 = 100;
const ROWS: u16 = 30;

#[test]
fn recorded_pi_ui_oracle_contains_100_independent_sessions() {
    let fixture_dir = fixture_dir();
    let index: Value = serde_json::from_slice(
        &std::fs::read(fixture_dir.join("index.json")).expect("read Pi UI oracle index"),
    )
    .expect("parse Pi UI oracle index");
    assert_eq!(index["oracleVersion"], ORACLE_VERSION);
    let sessions = index["sessions"].as_array().expect("session records");
    assert_eq!(sessions.len(), 100);
    let mut categories = std::collections::HashMap::new();
    for session in sessions {
        let filename = session["file"].as_str().expect("session fixture filename");
        assert!(fixture_dir.join(filename).is_file(), "missing {filename}");
        *categories
            .entry(session["category"].as_str().unwrap())
            .or_insert(0) += 1;
    }
    for category in [
        "model-cycle",
        "slash-command-menu",
        "composer-sizing-and-editing",
        "terminal-resize",
        "conversation-scrolling",
    ] {
        assert_eq!(
            categories.get(category),
            Some(&20),
            "coverage for {category}"
        );
    }

    let first_cycle = read_record(&fixture_dir, "001-model-cycle-01.json");
    assert!(
        first_cycle["checkpoints"][1]["screen"]
            .as_str()
            .unwrap()
            .contains("Switched to Beta Model")
    );
    let slash_menu = read_record(&fixture_dir, "021-slash-menu-01-empty.json");
    assert!(
        slash_menu["checkpoints"][1]["screen"]
            .as_str()
            .unwrap()
            .contains("settings")
    );
    let composer = read_record(&fixture_dir, "044-composer-04.json");
    assert!(
        composer["checkpoints"][1]["screen"]
            .as_str()
            .unwrap()
            .contains("first\nsecond")
    );
    assert!(
        !composer["checkpoints"][3]["screen"]
            .as_str()
            .unwrap()
            .contains("second")
    );
    let resize = read_record(&fixture_dir, "061-resize-01-80x24.json");
    assert_eq!(resize["checkpoints"][2]["terminalSize"]["columns"], 80);
    assert_eq!(resize["checkpoints"][2]["terminalSize"]["rows"], 24);
    let scroll = read_record(&fixture_dir, "081-scroll-01.json");
    assert!(
        scroll["checkpoints"][1]["screen"]
            .as_str()
            .unwrap()
            .contains("ASSISTANT-MARKER-25")
    );
}

fn read_record(fixture_dir: &std::path::Path, filename: &str) -> Value {
    serde_json::from_slice(&std::fs::read(fixture_dir.join(filename)).expect("read UI session"))
        .expect("parse UI session")
}

#[test]
#[ignore = "opt-in: set PI_ORACLE_BIN to run 100 interactive Pi sessions and record their screens"]
fn record_100_live_pi_ui_sessions() -> Result<(), Box<dyn std::error::Error>> {
    let binary = std::env::var("PI_ORACLE_BIN").expect("set PI_ORACLE_BIN to Pi 1.3.0");
    let output_dir = fixture_dir();
    std::fs::create_dir_all(&output_dir)?;
    let scenarios = scenarios();
    assert_eq!(scenarios.len(), 100);
    let start = std::env::var("PI_UI_ORACLE_START")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
        .min(scenarios.len());
    let limit = std::env::var("PI_UI_ORACLE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(scenarios.len())
        .min(scenarios.len() - start);

    let mut index_sessions = Vec::with_capacity(limit);
    for (index, scenario) in scenarios.iter().skip(start).take(limit).enumerate() {
        let sequence = start + index + 1;
        eprintln!("Recording Pi UI session {sequence}/100: {}", scenario.name);
        let record = capture_session(&binary, scenario)?;
        let filename = format!("{sequence:03}-{}.json", scenario.name);
        std::fs::write(
            output_dir.join(&filename),
            serde_json::to_vec_pretty(&record)?,
        )?;
        index_sessions.push(json!({
            "id": format!("pi-ui-{sequence:03}"),
            "category": scenario.category,
            "intent": scenario.name,
            "file": filename,
        }));
    }

    let manifest = json!({
        "oracle": "Pi interactive TUI",
        "oracleVersion": ORACLE_VERSION,
        "pinnedReferenceRevision": "698a2d23b86a5045dd54554354a5e0cb7e466795",
        "binarySource": "selected by PI_ORACLE_BIN; version verified, source revision not independently established",
        "environment": "isolated HOME/settings, PI_OFFLINE=1, --offline, no extensions or skills, no model requests",
        "sessionCount": index_sessions.len(),
        "sessions": index_sessions,
    });
    std::fs::write(
        output_dir.join("index.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}

struct Scenario {
    category: &'static str,
    name: String,
    initial_size: (u16, u16),
    scroll_history: bool,
    steps: Vec<Step>,
}

struct Step {
    label: String,
    input: StepInput,
}

enum StepInput {
    Keys(Vec<u8>),
    Resize { columns: u16, rows: u16 },
}

fn scenarios() -> Vec<Scenario> {
    let mut cases = Vec::with_capacity(100);

    for cycles in 1..=20 {
        cases.push(Scenario {
            category: "model-cycle",
            name: format!("model-cycle-{cycles:02}"),
            initial_size: (COLUMNS, ROWS),
            scroll_history: false,
            steps: vec![Step {
                label: format!("press Ctrl+P {cycles} times"),
                input: StepInput::Keys(vec![0x10; cycles]),
            }],
        });
    }

    let command_prefixes = [
        "", "m", "mo", "model", "s", "se", "set", "t", "th", "think", "h", "he", "help", "n",
        "new", "r", "res", "tools", "scoped", "unknown",
    ];
    for (index, prefix) in command_prefixes.into_iter().enumerate() {
        let mut steps = vec![Step {
            label: "open slash command menu".into(),
            input: StepInput::Keys(vec![b'/']),
        }];
        if !prefix.is_empty() {
            steps.push(Step {
                label: format!("filter slash commands by {prefix:?}"),
                input: StepInput::Keys(prefix.as_bytes().to_vec()),
            });
        }
        steps.push(Step {
            label: "cancel slash command menu".into(),
            input: StepInput::Keys(vec![0x1b]),
        });
        cases.push(Scenario {
            category: "slash-command-menu",
            name: format!("slash-menu-{:02}-{}", index + 1, prefix_key(prefix)),
            initial_size: (COLUMNS, ROWS),
            scroll_history: false,
            steps,
        });
    }

    let compositions = vec![
        "a".to_owned(),
        "a moderately sized single-line prompt".to_owned(),
        "x".repeat(140),
        "first\nsecond".to_owned(),
        "one\ntwo\nthree".to_owned(),
        "line\n".repeat(8),
        "line\n".repeat(20),
        "Unicode: café naïve résumé".to_owned(),
        "CJK: 你好世界こんにちは".to_owned(),
        "emoji: 🧑‍💻 🚀 🌊".to_owned(),
        "combining: e\u{301} a\u{308}".to_owned(),
        "punctuation: !@#$%^&*()[]{}<>?".to_owned(),
        "leading and trailing spaces   ".to_owned(),
        "tab\tseparated\ttext".to_owned(),
        "long-word-".repeat(12),
        "first\n\nthird".to_owned(),
        "αβγδεζηθικλμνξοπρστυφχψω".to_owned(),
        "مرحبا بالعالم".to_owned(),
        "שלום עולם".to_owned(),
        "line one\nline two\nline three\nline four\nline five".to_owned(),
    ];
    for (index, text) in compositions.iter().enumerate() {
        cases.push(Scenario {
            category: "composer-sizing-and-editing",
            name: format!("composer-{:02}", index + 1),
            initial_size: (COLUMNS, ROWS),
            scroll_history: false,
            steps: vec![
                Step {
                    label: format!("type composer content: {text:?}"),
                    input: StepInput::Keys(text.as_bytes().to_vec()),
                },
                Step {
                    label: "shrink composer by clearing the active line with Ctrl+U".into(),
                    input: StepInput::Keys(vec![0x15]),
                },
                Step {
                    label: "remove the empty line break with Backspace".into(),
                    input: StepInput::Keys(vec![0x7f]),
                },
            ],
        });
    }

    let resize_targets = [
        (80, 24),
        (120, 40),
        (60, 24),
        (140, 40),
        (90, 20),
        (110, 32),
        (72, 28),
        (160, 45),
        (50, 18),
        (100, 24),
        (132, 36),
        (84, 30),
        (118, 26),
        (96, 42),
        (64, 22),
        (150, 34),
        (105, 28),
        (78, 36),
        (128, 20),
        (88, 44),
    ];
    for (index, (columns, rows)) in resize_targets.into_iter().enumerate() {
        cases.push(Scenario {
            category: "terminal-resize",
            name: format!("resize-{:02}-{}x{}", index + 1, columns, rows),
            initial_size: (COLUMNS, ROWS),
            scroll_history: false,
            steps: vec![
                Step {
                    label: "type multiline content before resizing".into(),
                    input: StepInput::Keys(b"first line\nsecond line\nthird line".to_vec()),
                },
                Step {
                    label: format!("resize terminal to {columns} columns by {rows} rows"),
                    input: StepInput::Resize { columns, rows },
                },
            ],
        });
    }

    let scroll_inputs: [&[u8]; 20] = [
        b"\x1b[<64;20;10M",
        b"\x1b[<64;20;10M\x1b[<64;20;10M",
        b"\x1b[<64;20;10M\x1b[<64;64;10M",
        b"\x1b[<65;20;10M",
        b"\x1b[<65;20;10M\x1b[<65;20;10M",
        b"\x1b[5~",
        b"\x1b[6~",
        b"\x1b[5~\x1b[5~",
        b"\x1b[6~\x1b[6~",
        b"\x1b[<64;40;12M",
        b"\x1b[<64;40;12M\x1b[<64;40;12M\x1b[<64;40;12M",
        b"\x1b[<65;40;12M",
        b"\x1b[<64;10;8M\x1b[<65;10;8M",
        b"\x1b[5~\x1b[6~",
        b"\x1b[<64;50;18M",
        b"\x1b[<65;50;18M\x1b[<64;50;18M",
        b"\x1b[5~\x1b[5~\x1b[6~",
        b"\x1b[<64;25;15M\x1b[<64;25;15M\x1b[<64;25;15M\x1b[<64;25;15M",
        b"\x1b[<65;25;15M\x1b[<65;25;15M",
        b"\x1b[6~\x1b[5~",
    ];
    for (index, keys) in scroll_inputs.into_iter().enumerate() {
        cases.push(Scenario {
            category: "conversation-scrolling",
            name: format!("scroll-{:02}", index + 1),
            initial_size: (COLUMNS, ROWS),
            scroll_history: true,
            steps: vec![Step {
                label: "scroll the populated conversation viewport".into(),
                input: StepInput::Keys(keys.to_vec()),
            }],
        });
    }

    cases
}

fn prefix_key(prefix: &str) -> String {
    if prefix.is_empty() {
        "empty".into()
    } else {
        prefix.to_owned()
    }
}

fn capture_session(binary: &str, scenario: &Scenario) -> Result<Value, Box<dyn std::error::Error>> {
    let sandbox = tempfile::tempdir()?;
    let home = sandbox.path().join("home");
    let agent = sandbox.path().join("agent");
    let sessions = sandbox.path().join("sessions");
    let project = sandbox.path().join("project");
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&agent)?;
    std::fs::create_dir_all(&sessions)?;
    std::fs::create_dir_all(&project)?;
    let project = std::fs::canonicalize(project)?;
    std::fs::write(
        agent.join("settings.json"),
        r#"{"defaultProvider":"oracle","defaultModel":"alpha"}"#,
    )?;
    std::fs::write(
        agent.join("models.json"),
        serde_json::to_vec(&json!({
            "providers": {
                "oracle": {
                    "baseUrl": "http://127.0.0.1:9/v1",
                    "api": "openai-completions",
                    "apiKey": "offline-test-key",
                    "models": [
                        {"id":"alpha","name":"Alpha Model","contextWindow":32768,"maxTokens":4096},
                        {"id":"beta","name":"Beta Model","contextWindow":32768,"maxTokens":4096}
                    ]
                }
            }
        }))?,
    )?;

    let mut args = vec![
        "--offline".to_owned(),
        "--no-extensions".to_owned(),
        "--no-skills".to_owned(),
    ];
    if scenario.scroll_history {
        let session_file = sandbox.path().join("scroll-session.jsonl");
        std::fs::write(&session_file, scroll_session(&project)?)?;
        args.extend([
            "--session".to_owned(),
            session_file.to_string_lossy().into_owned(),
        ]);
    }

    let (columns, rows) = scenario.initial_size;
    let pair = native_pty_system().openpty(PtySize {
        rows,
        cols: columns,
        pixel_width: 0,
        pixel_height: 0,
    })?;
    let mut command = CommandBuilder::new(binary);
    command.args(&args);
    command.cwd(&project);
    command.env_clear();
    command.env("PATH", std::env::var_os("PATH").unwrap_or_default());
    command.env("HOME", &home);
    command.env("TERM", "xterm-256color");
    command.env("PI_CODING_AGENT_DIR", &agent);
    command.env("PI_CODING_AGENT_SESSION_DIR", &sessions);
    command.env("PI_OFFLINE", "1");
    command.env("PI_TELEMETRY", "0");
    let mut child = pair.slave.spawn_command(command)?;
    drop(pair.slave);

    let screen = Arc::new(Mutex::new(Parser::new(rows, columns, 1000)));
    let mut reader = pair.master.try_clone_reader()?;
    let reader_screen = Arc::clone(&screen);
    let reader_thread = thread::spawn(move || {
        let mut buffer = [0; 8192];
        while let Ok(count) = reader.read(&mut buffer) {
            if count == 0 {
                break;
            }
            reader_screen
                .lock()
                .expect("screen mutex")
                .process(&buffer[..count]);
        }
    });
    let mut writer = pair.master.take_writer()?;

    wait_for(&screen, Duration::from_secs(20), |text| {
        text.contains("v1.3.0") || text.contains("commands") || text.contains("USER-MARKER-29")
    })?;
    let mut checkpoints =
        vec![json!({"label":"startup","screen":normalized_screen(&screen, &project, &home)})];
    let mut current_size = scenario.initial_size;
    for step in &scenario.steps {
        match &step.input {
            StepInput::Keys(keys) => {
                writer.write_all(keys)?;
                writer.flush()?;
            }
            StepInput::Resize { columns, rows } => {
                *screen.lock().expect("screen mutex") = Parser::new(*rows, *columns, 1000);
                pair.master.resize(PtySize {
                    rows: *rows,
                    cols: *columns,
                    pixel_width: 0,
                    pixel_height: 0,
                })?;
                current_size = (*columns, *rows);
            }
        }
        thread::sleep(Duration::from_millis(350));
        checkpoints.push(json!({
            "label":step.label,
            "terminalSize":{"columns":current_size.0,"rows":current_size.1},
            "screen":normalized_screen(&screen, &project, &home)
        }));
    }

    child.kill()?;
    let exit = child.wait()?;
    drop(writer);
    drop(pair.master);
    reader_thread.join().expect("Pi PTY reader thread");

    Ok(json!({
        "oracleVersion": ORACLE_VERSION,
        "category": scenario.category,
        "name": scenario.name,
        "terminalSize":{"columns":scenario.initial_size.0,"rows":scenario.initial_size.1},
        "actions":scenario.steps.iter().map(|step| match &step.input {
            StepInput::Keys(keys) => json!({"label":step.label,"kind":"keys","bytes":keys}),
            StepInput::Resize{columns,rows} => json!({"label":step.label,"kind":"resize","columns":columns,"rows":rows}),
        }).collect::<Vec<_>>(),
        "checkpoints":checkpoints,
        "oracleProcessExit":{"success":exit.success(),"code":exit.exit_code()},
        "cleanup":"PTY closed after checkpoint capture; no prompt was submitted",
    }))
}

fn scroll_session(project: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut lines = vec![
        json!({
            "type":"session",
            "version":3,
            "id":"00000000-0000-4000-8000-000000000001",
            "timestamp":"2025-01-01T00:00:00.000Z",
            "cwd":project,
            "provider":"oracle",
            "modelId":"alpha",
            "thinkingLevel":"off"
        })
        .to_string(),
    ];
    let mut parent: Option<String> = None;
    let mut timestamp = 1_735_689_600_000i64;
    for index in 0..30 {
        for role in ["user", "assistant"] {
            let id = format!("{index:07x}{}", if role == "user" { "a" } else { "b" });
            let text = if role == "user" {
                format!("USER-MARKER-{index:02}: scroll oracle prompt")
            } else {
                format!(
                    "ASSISTANT-MARKER-{index:02}: a long transcript line for viewport scrolling"
                )
            };
            let mut entry = json!({
                "type":"message",
                "id":id,
                "parentId":parent.clone(),
                "timestamp":"2025-01-01T00:00:00.000Z",
                "message":{
                    "role":role,
                    "content": if role == "user" { json!(text) } else { json!([{"type":"text","text":text}]) },
                    "timestamp":timestamp
                }
            });
            if role == "assistant" {
                entry["message"]["api"] = json!("openai-completions");
                entry["message"]["provider"] = json!("oracle");
                entry["message"]["model"] = json!("alpha");
                entry["message"]["usage"] = json!({
                    "input":1,"output":1,"cacheRead":0,"cacheWrite":0,"totalTokens":2,
                    "cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}
                });
                entry["message"]["stopReason"] = json!("stop");
            }
            lines.push(entry.to_string());
            parent = Some(id);
            timestamp += 1000;
        }
    }
    Ok(format!("{}\n", lines.join("\n")))
}

fn normalized_screen(
    screen: &Arc<Mutex<Parser>>,
    project: &std::path::Path,
    home: &std::path::Path,
) -> String {
    screen
        .lock()
        .expect("screen mutex")
        .screen()
        .contents()
        .replace(&project.to_string_lossy().to_string(), "<PROJECT>")
        .replace(&home.to_string_lossy().to_string(), "<HOME>")
        .lines()
        .map(|line| {
            if line.contains("/private/var/folders/")
                || line.contains("/var/folders/")
                || line.contains("/tmp/")
            {
                "<PROJECT>"
            } else {
                line.trim_end()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .to_owned()
}

fn wait_for(
    screen: &Arc<Mutex<Parser>>,
    timeout: Duration,
    predicate: impl Fn(&str) -> bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let deadline = Instant::now() + timeout;
    loop {
        let contents = screen.lock().expect("screen mutex").screen().contents();
        if predicate(&contents) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!("timed out starting Pi UI; screen was:\n{contents}").into());
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../mu-e2e/fixtures")
        .join(format!("pi-ui-v{ORACLE_VERSION}"))
}
