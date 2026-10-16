use std::path::PathBuf;

use serde_json::Value;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../mu-e2e/fixtures/pi-vhs-v1.3.0")
}

#[test]
fn recorded_vhs_fixture_contains_pi_screens_and_full_terminal_video() {
    let dir = fixture_dir();
    let metadata: Value = serde_json::from_slice(
        &std::fs::read(dir.join("metadata.json")).expect("read VHS metadata"),
    )
    .expect("parse VHS metadata");
    assert_eq!(metadata["oracle"], "Pi interactive TUI captured by VHS");
    assert_eq!(metadata["piVersion"], "1.3.0");
    assert!(
        metadata["vhsVersion"]
            .as_str()
            .is_some_and(|version| version.starts_with("vhs version "))
    );
    assert!(
        metadata["environment"]
            .as_str()
            .unwrap()
            .contains("PI_OFFLINE=1")
    );

    let screen = std::fs::read_to_string(dir.join("pi-session.ascii"))
        .expect("read recorded Pi screen snapshots");
    assert!(screen.contains("Pi can explain its own features"));
    assert!(
        screen.contains("→ model"),
        "slash-command menu should be captured"
    );
    assert!(
        screen.contains("composer oracle prompt"),
        "composer should be captured"
    );

    let video = std::fs::read(dir.join("pi-session.gif")).expect("read full-terminal recording");
    assert!(video.starts_with(b"GIF89a"), "VHS video must be a GIF");
    assert!(
        video.len() > 10_000,
        "recording should contain multiple rendered frames"
    );
}
