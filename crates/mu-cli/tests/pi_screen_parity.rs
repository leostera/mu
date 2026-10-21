//! Replays the 100 recorded Pi UI sessions against the real `mu` binary and compares screens.
//!
//! Every wait is bounded; a hung scenario is killed and reported as a failure. Set
//! `PI_PARITY_ONLY=<substring>` to run matching fixtures and `PI_PARITY_VERBOSE=1` to print diffs.

use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use serde_json::{Value, json};
use vt100::Parser;

const SCENARIO_DEADLINE: Duration = Duration::from_secs(25);
const STABLE_FOR: Duration = Duration::from_millis(250);
const SETTLE_LIMIT: Duration = Duration::from_secs(4);

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../mu-e2e/fixtures/pi-ui-v1.3.0")
}

struct Session {
    screen: Arc<Mutex<Parser>>,
}

impl Session {
    fn contents(&self) -> String {
        self.screen.lock().unwrap().screen().contents()
    }

    /// Wait until the screen stops changing (bounded).
    fn settle(&self) {
        let start = Instant::now();
        let mut last = self.contents();
        let mut stable_since = Instant::now();
        while start.elapsed() < SETTLE_LIMIT {
            thread::sleep(Duration::from_millis(30));
            let now = self.contents();
            if now != last {
                last = now;
                stable_since = Instant::now();
            } else if stable_since.elapsed() >= STABLE_FOR && !last.trim().is_empty() {
                return;
            }
        }
    }
}

fn normalize(contents: &str, project: &Path, home: &Path) -> String {
    contents
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

fn scroll_session(project: &Path) -> String {
    let mut lines = vec![
        json!({
            "type":"session","version":3,"id":"00000000-0000-4000-8000-000000000001",
            "timestamp":"2025-01-01T00:00:00.000Z","cwd":project,
            "provider":"oracle","modelId":"alpha","thinkingLevel":"off"
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
                "type":"message","id":id,"parentId":parent.clone(),
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
    format!("{}\n", lines.join("\n"))
}

/// Replay one fixture; returns a description of the first mismatch.
fn replay(fixture: &Value) -> Result<(), String> {
    let sandbox = tempfile::tempdir().map_err(|e| e.to_string())?;
    let home = sandbox.path().join("home");
    let agent = sandbox.path().join("agent");
    let project = sandbox.path().join("project");
    for dir in [&home, &agent, &project] {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let project = std::fs::canonicalize(project).map_err(|e| e.to_string())?;
    let home = std::fs::canonicalize(home).map_err(|e| e.to_string())?;
    std::fs::write(
        agent.join("settings.json"),
        r#"{"defaultProvider":"oracle","defaultModel":"alpha"}"#,
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(
        agent.join("models.json"),
        serde_json::to_vec(&json!({"providers":{"oracle":{
        "baseUrl":"http://127.0.0.1:9/v1","api":"openai-completions","apiKey":"offline-test-key",
        "models":[
            {"id":"alpha","name":"Alpha Model","contextWindow":32768,"maxTokens":4096},
            {"id":"beta","name":"Beta Model","contextWindow":32768,"maxTokens":4096}
        ]}}}))
        .unwrap(),
    )
    .map_err(|e| e.to_string())?;

    let mut args = vec![
        "--demo".to_owned(),
        "--offline".into(),
        "--no-extensions".into(),
        "--no-skills".into(),
    ];
    if fixture["category"] == "conversation-scrolling" {
        let file = sandbox.path().join("scroll-session.jsonl");
        std::fs::write(&file, scroll_session(&project)).map_err(|e| e.to_string())?;
        args.extend(["--session".into(), file.to_string_lossy().into_owned()]);
    }
    let columns = fixture["terminalSize"]["columns"].as_u64().unwrap() as u16;
    let rows = fixture["terminalSize"]["rows"].as_u64().unwrap() as u16;
    let pair = native_pty_system()
        .openpty(PtySize {
            rows,
            cols: columns,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| e.to_string())?;
    let mut command = CommandBuilder::new(env!("CARGO_BIN_EXE_mu"));
    command.args(&args);
    command.cwd(&project);
    command.env_clear();
    command.env("PATH", std::env::var_os("PATH").unwrap_or_default());
    command.env("HOME", &home);
    command.env("TERM", "xterm-256color");
    command.env("PI_CODING_AGENT_DIR", &agent);
    command.env("PI_OFFLINE", "1");
    let mut child = pair
        .slave
        .spawn_command(command)
        .map_err(|e| e.to_string())?;
    drop(pair.slave);

    let screen = Arc::new(Mutex::new(Parser::new(rows, columns, 1000)));
    let mut reader = pair.master.try_clone_reader().map_err(|e| e.to_string())?;
    let reader_screen = Arc::clone(&screen);
    thread::spawn(move || {
        let mut buffer = [0; 8192];
        while let Ok(count) = reader.read(&mut buffer) {
            if count == 0 {
                break;
            }
            reader_screen.lock().unwrap().process(&buffer[..count]);
        }
    });
    let mut writer = pair.master.take_writer().map_err(|e| e.to_string())?;
    let session = Session {
        screen: Arc::clone(&screen),
    };
    let started = Instant::now();

    let result = (|| {
        session.settle();
        let checkpoints = fixture["checkpoints"].as_array().unwrap();
        compare(
            0,
            "startup",
            &normalize(&session.contents(), &project, &home),
            checkpoints[0]["screen"].as_str().unwrap(),
        )?;
        for (index, action) in fixture["actions"].as_array().unwrap().iter().enumerate() {
            if started.elapsed() > SCENARIO_DEADLINE {
                return Err("scenario deadline exceeded".to_owned());
            }
            match action["kind"].as_str().unwrap() {
                "keys" => {
                    let bytes = action["bytes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|b| b.as_u64().unwrap() as u8)
                        .collect::<Vec<_>>();
                    writer
                        .write_all(&bytes)
                        .and_then(|()| writer.flush())
                        .map_err(|e| e.to_string())?;
                }
                _ => {
                    let (c, r) = (
                        action["columns"].as_u64().unwrap() as u16,
                        action["rows"].as_u64().unwrap() as u16,
                    );
                    *screen.lock().unwrap() = Parser::new(r, c, 1000);
                    pair.master
                        .resize(PtySize {
                            rows: r,
                            cols: c,
                            pixel_width: 0,
                            pixel_height: 0,
                        })
                        .map_err(|e| e.to_string())?;
                }
            }
            session.settle();
            let expected = checkpoints[index + 1]["screen"].as_str().unwrap();
            compare(
                index + 1,
                action["label"].as_str().unwrap(),
                &normalize(&session.contents(), &project, &home),
                expected,
            )?;
        }
        Ok(())
    })();
    let _ = child.kill();
    let _ = child.wait();
    result
}

/// Mu's own product text (logo, version, tagline) replaces Pi's in the recorded screens.
fn brand(expected: &str) -> String {
    expected
        .replace(" ▀▀█  v1.3.0", " ▛▜▌  v0.1.0")
        .replace(" █▀ █ escape", " ▙▟ ▐ escape")
        .replace(
            "Pi can explain its own features and look up its docs. Ask it how to use or extend Pi.",
            "Mu can explain its own features and look up its docs. Ask it how to use or extend Mu.",
        )
        .replace(" extend Pi.", " extend Mu.")
        .replace(" Pi can explain", " Mu can explain")
        .replace(
            "Report a bug to the Pi developers",
            "Report a bug to the Pi developers",
        )
}

fn compare(index: usize, label: &str, actual: &str, expected: &str) -> Result<(), String> {
    let expected = brand(expected);
    let expected = expected.as_str();
    if actual == expected {
        return Ok(());
    }
    let (a, e): (Vec<_>, Vec<_>) = (actual.lines().collect(), expected.lines().collect());
    let mut report = format!("checkpoint {index} ({label}): first differing rows\n");
    if std::env::var("PI_PARITY_DUMP").is_ok() {
        report.push_str(&format!(
            "--- mu screen\n{actual}\n--- pi screen\n{expected}\n---\n"
        ));
    }
    let mut shown = 0;
    for row in 0..a.len().max(e.len()) {
        let (left, right) = (
            a.get(row).copied().unwrap_or("<missing>"),
            e.get(row).copied().unwrap_or("<missing>"),
        );
        if left != right && shown < 6 {
            report.push_str(&format!(
                "  row {row}:\n    mu: {left:?}\n    pi: {right:?}\n"
            ));
            shown += 1;
        }
    }
    Err(report)
}

#[test]
fn mu_matches_recorded_pi_screens() {
    let only = std::env::var("PI_PARITY_ONLY").ok();
    let verbose = std::env::var("PI_PARITY_VERBOSE").is_ok();
    let index: Value =
        serde_json::from_slice(&std::fs::read(fixture_dir().join("index.json")).unwrap()).unwrap();
    let files = index["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["file"].as_str().unwrap().to_owned())
        .filter(|f| only.as_deref().is_none_or(|needle| f.contains(needle)))
        .collect::<Vec<_>>();
    let queue = Arc::new(Mutex::new(files));
    let failures = Arc::new(Mutex::new(Vec::<(String, String)>::new()));
    let workers = (0..6)
        .map(|_| {
            let (queue, failures) = (Arc::clone(&queue), Arc::clone(&failures));
            thread::spawn(move || {
                loop {
                    let Some(file) = queue.lock().unwrap().pop() else {
                        return;
                    };
                    let fixture: Value =
                        serde_json::from_slice(&std::fs::read(fixture_dir().join(&file)).unwrap())
                            .unwrap();
                    if let Err(error) = replay(&fixture) {
                        failures.lock().unwrap().push((file, error));
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }
    let mut failures = failures.lock().unwrap().clone();
    failures.sort();
    if verbose {
        for (file, error) in &failures {
            eprintln!("--- {file}\n{error}");
        }
    }
    assert!(
        failures.is_empty(),
        "{} of the recorded Pi screens differ: {}\n{}",
        failures.len(),
        failures
            .iter()
            .map(|(f, _)| f.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        failures
            .first()
            .map(|(f, e)| format!("--- {f}\n{e}"))
            .unwrap_or_default()
    );
}
