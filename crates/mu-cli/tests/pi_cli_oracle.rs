use std::{
    ffi::OsStr,
    process::{Command, Output, Stdio},
};

const PI_VERSION: &str = include_str!("../../mu-e2e/fixtures/pi-v1.3.0/version.stdout");
const PI_SHORT_VERSION: &str = include_str!("../../mu-e2e/fixtures/pi-v1.3.0/short-version.stdout");
const PI_HELP: &str = include_str!("../../mu-e2e/fixtures/pi-v1.3.0/help.stdout");
const PI_UNKNOWN_STDERR: &str = include_str!("../../mu-e2e/fixtures/pi-v1.3.0/unknown-flag.stderr");
const PI_INVALID_MODE_STDERR: &str =
    include_str!("../../mu-e2e/fixtures/pi-v1.3.0/invalid-mode.stderr");

fn run_mu(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mu"))
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("run Mu CLI")
}

fn run_pi(binary: &OsStr, args: &[&str]) -> Output {
    let home = tempfile::tempdir().expect("create isolated Pi home");
    let path = std::env::var_os("PATH").unwrap_or_default();
    Command::new(binary)
        .args(args)
        .env_clear()
        .env("PATH", path)
        .env("HOME", home.path())
        .env("PI_CODING_AGENT_DIR", home.path().join("agent"))
        .env("PI_OFFLINE", "1")
        .stdin(Stdio::null())
        .output()
        .expect("run Pi CLI oracle")
}

#[test]
fn recorded_pi_cli_baseline_covers_version_help_and_argument_modes() {
    assert_eq!(PI_VERSION.trim(), "1.3.0");
    assert_eq!(PI_SHORT_VERSION, PI_VERSION);
    assert!(PI_HELP.contains("Usage:"));
    assert!(PI_HELP.contains("--print, -p"));
    assert!(PI_HELP.contains("--mode <mode>"));
    assert!(PI_HELP.contains("--session <path|id>"));
    assert_eq!(
        PI_UNKNOWN_STDERR,
        "Error: Unknown option: --definitely-not-a-pi-flag\n"
    );
    assert!(PI_INVALID_MODE_STDERR.contains("Valid values: text, json, rpc"));
    assert_eq!(
        include_str!("../../mu-e2e/fixtures/pi-v1.3.0/print-empty.stdout"),
        ""
    );
    assert_eq!(
        include_str!("../../mu-e2e/fixtures/pi-v1.3.0/print-empty.stderr"),
        ""
    );
    for (case, status_text, expected_status) in [
        (
            "version",
            include_str!("../../mu-e2e/fixtures/pi-v1.3.0/version.exit"),
            '0',
        ),
        (
            "short-version",
            include_str!("../../mu-e2e/fixtures/pi-v1.3.0/short-version.exit"),
            '0',
        ),
        (
            "help",
            include_str!("../../mu-e2e/fixtures/pi-v1.3.0/help.exit"),
            '0',
        ),
        (
            "unknown-flag",
            include_str!("../../mu-e2e/fixtures/pi-v1.3.0/unknown-flag.exit"),
            '1',
        ),
        (
            "invalid-mode",
            include_str!("../../mu-e2e/fixtures/pi-v1.3.0/invalid-mode.exit"),
            '1',
        ),
        (
            "print-empty",
            include_str!("../../mu-e2e/fixtures/pi-v1.3.0/print-empty.exit"),
            '0',
        ),
    ] {
        assert!(
            status_text.starts_with(expected_status),
            "unexpected Pi status for {case}"
        );
    }
}

#[test]
fn mu_matches_the_shared_version_help_error_and_empty_print_contracts() {
    let version = run_mu(&["--version"]);
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8_lossy(&version.stdout).trim(),
        format!("mu {}", env!("CARGO_PKG_VERSION"))
    );

    let short_version = run_mu(&["-v"]);
    assert!(short_version.status.success());
    assert_eq!(short_version.stdout, version.stdout);

    let help = run_mu(&["--help"]);
    assert!(help.status.success());
    let help = String::from_utf8_lossy(&help.stdout);
    for shared_flag in ["--help", "--version", "--print", "--mode"] {
        assert!(
            PI_HELP.contains(shared_flag),
            "Pi baseline lacks {shared_flag}"
        );
        assert!(help.contains(shared_flag), "Mu help lacks {shared_flag}");
    }

    let invalid = run_mu(&["--definitely-not-a-pi-flag"]);
    assert_eq!(invalid.status.code(), Some(1));
    assert!(!invalid.stderr.is_empty());

    let invalid_mode = run_mu(&["--mode", "unsupported"]);
    assert_eq!(invalid_mode.status.code(), Some(1));
    let invalid_mode_stderr = String::from_utf8_lossy(&invalid_mode.stderr);
    assert!(invalid_mode_stderr.contains("text"));
    assert!(invalid_mode_stderr.contains("json"));
    assert!(invalid_mode_stderr.contains("rpc"));

    for args in [&["--print"][..], &[][..]] {
        let empty = run_mu(args);
        assert_eq!(empty.status.code(), Some(0));
        assert!(empty.stdout.is_empty());
        assert!(empty.stderr.is_empty());
    }
}

#[test]
#[ignore = "opt-in: set PI_ORACLE_BIN to run the installed Pi executable"]
fn live_pi_executable_matches_the_recorded_baseline() {
    let binary = std::env::var_os("PI_ORACLE_BIN").expect("set PI_ORACLE_BIN to Pi executable");
    let version = run_pi(&binary, &["--version"]);
    assert_eq!(version.status.code(), Some(0));
    assert_eq!(version.stdout, PI_VERSION.as_bytes());
    assert!(version.stderr.is_empty());

    let short_version = run_pi(&binary, &["-v"]);
    assert_eq!(short_version.status.code(), Some(0));
    assert_eq!(short_version.stdout, PI_SHORT_VERSION.as_bytes());
    assert!(short_version.stderr.is_empty());

    let help = run_pi(&binary, &["--help"]);
    assert_eq!(help.status.code(), Some(0));
    assert_eq!(help.stdout, PI_HELP.as_bytes());
    assert!(help.stderr.is_empty());

    let invalid = run_pi(&binary, &["--definitely-not-a-pi-flag"]);
    assert_eq!(invalid.status.code(), Some(1));
    assert!(invalid.stdout.is_empty());
    assert_eq!(invalid.stderr, PI_UNKNOWN_STDERR.as_bytes());

    let invalid_mode = run_pi(&binary, &["--mode", "unsupported"]);
    assert_eq!(invalid_mode.status.code(), Some(1));
    assert!(invalid_mode.stdout.is_empty());
    assert_eq!(invalid_mode.stderr, PI_INVALID_MODE_STDERR.as_bytes());

    let empty_print = run_pi(&binary, &["--print"]);
    assert_eq!(empty_print.status.code(), Some(0));
    assert!(empty_print.stdout.is_empty());
    assert!(empty_print.stderr.is_empty());
}
