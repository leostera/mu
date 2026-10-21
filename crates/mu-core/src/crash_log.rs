//! Local crash log (Pi's `core/crash-log.ts`): `crashes.json` in the agent directory keeps the
//! last few crashes so the next start can mention one. Nothing is uploaded.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const MAX_CRASH_RECORDS: usize = 5;
const MAX_AGE_SECONDS: i64 = 7 * 24 * 60 * 60;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CrashRecord {
    pub timestamp: String,
    pub version: String,
    /// `uncaught_exception` or `fatal_error`.
    pub kind: String,
    pub message: String,
    pub stack: Option<String>,
    pub session_file: Option<String>,
    pub cwd: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notified: Option<bool>,
}

pub fn crash_log_path(agent_dir: &Path) -> PathBuf {
    agent_dir.join("crashes.json")
}

/// Records are skipped when malformed, as Pi does; a missing or invalid file is empty.
pub fn read(path: &Path) -> Vec<CrashRecord> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(serde_json::Value::Array(items)) = serde_json::from_str(&text) else {
        return Vec::new();
    };
    items
        .into_iter()
        .filter_map(|item| serde_json::from_value(item).ok())
        .collect()
}

fn write(path: &Path, records: &[CrashRecord]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = serde_json::to_string_pretty(records).expect("crash records serialize");
    text.push('\n');
    std::fs::write(path, text)
}

/// Best-effort append for callers that are already crashing; keeps the newest five.
pub fn record(
    path: &Path,
    kind: &str,
    message: &str,
    stack: Option<String>,
    session_file: Option<String>,
    cwd: &Path,
) -> Option<CrashRecord> {
    let entry = CrashRecord {
        timestamp: Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        version: env!("CARGO_PKG_VERSION").to_owned(),
        kind: kind.to_owned(),
        message: message.to_owned(),
        stack,
        session_file,
        cwd: cwd.display().to_string(),
        notified: None,
    };
    let mut records = read(path);
    records.push(entry.clone());
    let keep_from = records.len().saturating_sub(MAX_CRASH_RECORDS);
    write(path, &records[keep_from..]).ok()?;
    Some(entry)
}

/// The newest recent crash not yet announced; marks pending records as announced.
pub fn take_unnotified(path: &Path, now: DateTime<Utc>) -> Option<CrashRecord> {
    let records = read(path);
    let crash = records
        .iter()
        .rev()
        .find(|r| {
            r.notified != Some(true)
                && DateTime::parse_from_rfc3339(&r.timestamp)
                    .is_ok_and(|t| (now - t.with_timezone(&Utc)).num_seconds() <= MAX_AGE_SECONDS)
        })?
        .clone();
    let marked = records
        .into_iter()
        .map(|r| {
            if r.notified == Some(true) {
                r
            } else {
                CrashRecord {
                    notified: Some(true),
                    ..r
                }
            }
        })
        .collect::<Vec<_>>();
    let _ = write(path, &marked);
    Some(crash)
}

pub fn clear(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// Record panics to the crash log before the default panic output runs.
pub fn install_panic_hook(agent_dir: PathBuf, cwd: PathBuf) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let message = info.to_string();
        let stack = Some(std::backtrace::Backtrace::force_capture().to_string());
        record(
            &crash_log_path(&agent_dir),
            "uncaught_exception",
            &message,
            stack,
            None,
            &cwd,
        );
        previous(info);
    }));
}

#[cfg(test)]
mod tests {
    use chrono::Duration;

    use super::*;

    fn path() -> PathBuf {
        std::env::temp_dir().join(format!("mu-crash-{}/crashes.json", uuid::Uuid::new_v4()))
    }

    #[test]
    fn keeps_only_the_newest_five_records() {
        let path = path();
        for i in 0..7 {
            record(
                &path,
                "fatal_error",
                &format!("boom {i}"),
                None,
                Some("s.jsonl".into()),
                Path::new("/w"),
            );
        }
        let records = read(&path);
        assert_eq!(records.len(), 5);
        assert_eq!(records[0].message, "boom 2");
        assert_eq!(records[4].session_file.as_deref(), Some("s.jsonl"));
        clear(&path);
        assert!(read(&path).is_empty());
    }

    #[test]
    fn malformed_files_and_records_read_as_empty_or_are_skipped() {
        let path = path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "not json").unwrap();
        assert!(read(&path).is_empty());
        std::fs::write(&path, r#"[{"timestamp":"t"},{"timestamp":"t","version":"v","kind":"k","message":"m","stack":null,"sessionFile":null,"cwd":"/"}]"#).unwrap();
        assert_eq!(read(&path).len(), 1);
    }

    #[test]
    fn unnotified_crashes_are_announced_once_and_expire_after_a_week() {
        let path = path();
        record(
            &path,
            "uncaught_exception",
            "boom",
            Some("trace".into()),
            None,
            Path::new("/w"),
        )
        .unwrap();
        let now = Utc::now();
        let crash = take_unnotified(&path, now).unwrap();
        assert_eq!(crash.message, "boom");
        assert_eq!(take_unnotified(&path, now), None, "marked as announced");
        assert_eq!(read(&path)[0].notified, Some(true));
        record(&path, "fatal_error", "later", None, None, Path::new("/w")).unwrap();
        assert_eq!(
            take_unnotified(&path, now + Duration::days(8)),
            None,
            "too old"
        );
    }
}
