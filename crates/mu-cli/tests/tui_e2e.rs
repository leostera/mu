use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use vt100::Parser;

#[test]
fn demo_backend_drives_real_tui_without_gateway_or_credentials()
-> Result<(), Box<dyn std::error::Error>> {
    let agent_dir = tempfile::tempdir()?;
    // Keep process-level key interactions at a fixed size; resize geometry is covered by TestBackend.
    let pair = native_pty_system().openpty(PtySize {
        rows: 30,
        cols: 100,
        pixel_width: 0,
        pixel_height: 0,
    })?;
    let mut command = CommandBuilder::new(env!("CARGO_BIN_EXE_mu"));
    command.arg("--demo");
    command.env_clear();
    command.env("TERM", "xterm-256color");
    command.env("HOME", agent_dir.path());
    command.env("PI_CODING_AGENT_DIR", agent_dir.path());
    let mut child = pair.slave.spawn_command(command)?;
    drop(pair.slave);

    let screen = Arc::new(Mutex::new(Parser::new(30, 100, 1000)));
    let mut reader = pair.master.try_clone_reader()?;
    let reader_screen = Arc::clone(&screen);
    let reader_thread = thread::spawn(move || {
        let mut buffer = [0; 4096];
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

    wait_for(&screen, |text| {
        text.contains("Mu can explain its own features")
    })?;
    assert!(
        screen
            .lock()
            .expect("screen mutex")
            .screen()
            .alternate_screen()
    );
    // Keys sent while the agent is still connecting are ignored, so resend Ctrl+P until it lands.
    let mut cycled = false;
    for _ in 0..5 {
        writer.write_all(&[0x10])?;
        writer.flush()?;
        if wait_for_within(&screen, Duration::from_secs(2), |text| {
            text.contains("mu-demo-alt")
        })
        .is_ok()
        {
            cycled = true;
            break;
        }
    }
    assert!(cycled, "Ctrl+P never switched to mu-demo-alt");
    writer.write_all(&[0x10])?;
    writer.flush()?;
    wait_for(&screen, |text| !text.contains("mu-demo-alt"))?;

    writer.write_all(b"/")?;
    writer.flush()?;
    wait_for(&screen, |text| {
        text.contains("model") && text.contains("(1/24)")
    })?;
    writer.write_all(b"m")?;
    writer.flush()?;
    wait_for(&screen, |text| text.contains("→ model"))?;
    writer.write_all(&[0x1b])?;
    writer.flush()?;
    wait_for(&screen, |text| {
        text.contains("/m") && !text.contains("→ model")
    })?;
    writer.write_all(&[0x1b])?;
    writer.flush()?;
    wait_for(&screen, |text| !text.contains("/m"))?;

    writer.write_all(b"/model")?;
    writer.flush()?;
    wait_for(&screen, |text| text.contains("→ model"))?;
    writer.write_all(b"\r")?;
    writer.flush()?;
    wait_for(&screen, |text| text.contains("mu-demo-alt"))?;
    writer.write_all(b"\x1b[B\r")?;
    writer.flush()?;
    wait_for(&screen, |text| text.contains("Switched to mu-demo-alt"))?;

    writer.write_all(b"temporary editor text")?;
    writer.flush()?;
    wait_for(&screen, |text| text.contains("temporary editor text"))?;
    writer.write_all(&[0x1b])?;
    writer.flush()?;
    wait_for(&screen, |text| !text.contains("temporary editor text"))?;

    writer.write_all(b"\x1b[200~pasted first line\r\npasted second line\x1b[201~")?;
    writer.flush()?;
    wait_for(&screen, |text| {
        text.contains("pasted first line") && text.contains("pasted second line")
    })?;
    writer.write_all(&[0x1b])?;
    writer.flush()?;
    wait_for(&screen, |text| !text.contains("pasted first line"))?;

    writer.write_all(b"hello from PTY\r")?;
    writer.flush()?;
    wait_for(&screen, |text| {
        text.contains("Demo backend received: hello from PTY")
    })?;
    writer.write_all(b"\x1b[A")?;
    writer.flush()?;
    wait_for(&screen, |text| text.matches("hello from PTY").count() >= 3)?;
    writer.write_all(&[0x1b])?;
    writer.flush()?;
    wait_for(&screen, |text| text.matches("hello from PTY").count() == 2)?;

    for index in 0..8 {
        let prompt = format!("scroll marker {index}");
        writer.write_all(format!("{prompt}\r").as_bytes())?;
        writer.flush()?;
        wait_for(&screen, |text| {
            text.contains(&format!("Demo backend received: {prompt}"))
        })?;
    }
    for _ in 0..6 {
        writer.write_all(b"\x1b[5~")?;
        writer.flush()?;
        thread::sleep(Duration::from_millis(30));
    }
    wait_for(&screen, |text| text.contains("scroll marker 0"))?;

    writer.write_all(&[0x03])?;
    writer.flush()?;
    thread::sleep(Duration::from_millis(50));
    writer.write_all(&[0x03])?;
    writer.flush()?;
    assert!(child.wait()?.success());
    wait_for(&screen, |_| {
        !screen
            .lock()
            .expect("screen mutex")
            .screen()
            .alternate_screen()
    })?;
    drop(writer);
    reader_thread.join().expect("PTY reader thread");

    let session_file = find_session_file(agent_dir.path())?;
    let records = std::fs::read_to_string(session_file)?
        .lines()
        .map(serde_json::from_str::<serde_json::Value>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(records[0]["type"], "session");
    assert_eq!(records[0]["version"], 3);
    let model_changes = records
        .iter()
        .filter(|record| record["type"] == "model_change")
        .collect::<Vec<_>>();
    assert!(!model_changes.is_empty());
    assert_eq!(model_changes.last().unwrap()["modelId"], "mu-demo-alt");
    let messages = records
        .iter()
        .filter(|record| record["type"] == "message")
        .collect::<Vec<_>>();
    assert_eq!(messages[0]["message"]["role"], "user");
    assert_eq!(messages[1]["message"]["role"], "assistant");
    assert_eq!(messages[1]["message"]["api"], "mu-demo");
    assert_eq!(messages[1]["message"]["model"], "mu-demo-alt");
    Ok(())
}

fn wait_for(
    screen: &Arc<Mutex<Parser>>,
    predicate: impl Fn(&str) -> bool,
) -> Result<(), Box<dyn std::error::Error>> {
    wait_for_within(screen, Duration::from_secs(5), predicate)
}

fn wait_for_within(
    screen: &Arc<Mutex<Parser>>,
    timeout: Duration,
    predicate: impl Fn(&str) -> bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let deadline = Instant::now() + timeout;
    loop {
        let text = screen.lock().expect("screen mutex").screen().contents();
        if predicate(&text) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!("timed out waiting for TUI state; screen was:\n{text}").into());
        }
        thread::sleep(Duration::from_millis(15));
    }
}

fn find_session_file(agent_dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    for cwd_dir in std::fs::read_dir(agent_dir.join("sessions"))? {
        for entry in std::fs::read_dir(cwd_dir?.path())? {
            let path = entry?.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "jsonl")
            {
                return Ok(path);
            }
        }
    }
    Err("demo backend did not persist a Pi session".into())
}
