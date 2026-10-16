//! Pi-compatible session JSONL records.
//!
//! The parser validates the fields Mu needs to navigate sessions while keeping
//! every unrecognized field as JSON. This lets Mu inspect an entry without
//! erasing fields it does not yet understand when the document is written back.

use std::collections::{BTreeMap, HashSet};

use serde_json::{Map, Value};
use thiserror::Error;

pub const CURRENT_SESSION_VERSION: u64 = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionId(String);

impl SessionId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn from_json(value: Value, line: usize) -> Result<Self, SessionParseError> {
        value
            .as_str()
            .map(|value| Self(value.to_owned()))
            .ok_or(SessionParseError::InvalidField {
                line,
                field: "id",
                expected: "a string",
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryId(String);

impl EntryId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn from_json(value: Value, line: usize) -> Result<Self, SessionParseError> {
        value
            .as_str()
            .map(|value| Self(value.to_owned()))
            .ok_or(SessionParseError::InvalidField {
                line,
                field: "id",
                expected: "a string",
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryType(String);

impl EntryType {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn from_json(value: Value, line: usize) -> Result<Self, SessionParseError> {
        value
            .as_str()
            .map(|value| Self(value.to_owned()))
            .ok_or(SessionParseError::InvalidField {
                line,
                field: "type",
                expected: "a string",
            })
    }
}

/// A timestamp string as stored by Pi. Session-entry and nested-message
/// timestamps intentionally remain distinct representations in Pi; this type
/// preserves the entry value without interpreting or normalizing it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionTimestamp(String);

impl SessionTimestamp {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn from_json(value: Value, line: usize) -> Result<Self, SessionParseError> {
        value
            .as_str()
            .map(|value| Self(value.to_owned()))
            .ok_or(SessionParseError::InvalidField {
                line,
                field: "timestamp",
                expected: "a string",
            })
    }
}

/// Pi's session header. Fields unknown to this version of Mu are retained.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionHeader {
    pub version: Option<u64>,
    pub id: SessionId,
    pub timestamp: SessionTimestamp,
    pub cwd: String,
    pub parent_session: Option<String>,
    extra: BTreeMap<String, Value>,
}

impl SessionHeader {
    pub fn extra_fields(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }

    fn from_value(mut fields: Map<String, Value>, line: usize) -> Result<Self, SessionParseError> {
        let entry_type = take_string(&mut fields, "type", line)?;
        if entry_type != "session" {
            return Err(SessionParseError::ExpectedSessionHeader { line });
        }

        let version = match fields.remove("version") {
            Some(Value::Number(value)) => {
                Some(value.as_u64().ok_or(SessionParseError::InvalidField {
                    line,
                    field: "version",
                    expected: "a non-negative integer",
                })?)
            }
            Some(_) => {
                return Err(SessionParseError::InvalidField {
                    line,
                    field: "version",
                    expected: "a non-negative integer",
                });
            }
            None => None,
        };

        let id = SessionId::from_json(take_value(&mut fields, "id", line)?, line)?;
        let timestamp =
            SessionTimestamp::from_json(take_value(&mut fields, "timestamp", line)?, line)?;
        let cwd = take_string(&mut fields, "cwd", line)?;
        let parent_session = match fields.remove("parentSession") {
            Some(Value::String(path)) => Some(path),
            Some(_) => {
                return Err(SessionParseError::InvalidField {
                    line,
                    field: "parentSession",
                    expected: "a string when present",
                });
            }
            None => None,
        };

        Ok(Self {
            version,
            id,
            timestamp,
            cwd,
            parent_session,
            extra: fields.into_iter().collect(),
        })
    }

    fn into_json(self) -> Value {
        let mut fields = self.extra.into_iter().collect::<Map<_, _>>();
        fields.insert("type".into(), Value::String("session".into()));
        if let Some(version) = self.version {
            fields.insert("version".into(), Value::from(version));
        }
        fields.insert("id".into(), Value::String(self.id.0));
        fields.insert("timestamp".into(), Value::String(self.timestamp.0));
        fields.insert("cwd".into(), Value::String(self.cwd));
        if let Some(parent_session) = self.parent_session {
            fields.insert("parentSession".into(), Value::String(parent_session));
        }
        Value::Object(fields)
    }
}

/// A parent link preserves the distinction between legacy v1 fields that are
/// absent and v2+ roots that explicitly serialize `parentId: null`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParentReference {
    LegacyMissing,
    Root,
    Entry(EntryId),
}

/// A Pi session entry with typed tree-navigation fields and opaque payload.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionEntry {
    pub entry_type: EntryType,
    pub id: Option<EntryId>,
    pub parent_id: ParentReference,
    pub timestamp: SessionTimestamp,
    extra: BTreeMap<String, Value>,
}

impl SessionEntry {
    pub fn extra_fields(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }

    fn from_value(mut fields: Map<String, Value>, line: usize) -> Result<Self, SessionParseError> {
        let entry_type = EntryType::from_json(take_value(&mut fields, "type", line)?, line)?;
        if entry_type.as_str() == "session" {
            return Err(SessionParseError::UnexpectedSessionHeader { line });
        }

        let id = fields
            .remove("id")
            .map(|value| EntryId::from_json(value, line))
            .transpose()?;
        let parent_id = match fields.remove("parentId") {
            Some(Value::Null) => ParentReference::Root,
            Some(Value::String(id)) => ParentReference::Entry(EntryId(id)),
            Some(_) => {
                return Err(SessionParseError::InvalidField {
                    line,
                    field: "parentId",
                    expected: "a string or null",
                });
            }
            None => ParentReference::LegacyMissing,
        };
        let timestamp =
            SessionTimestamp::from_json(take_value(&mut fields, "timestamp", line)?, line)?;

        Ok(Self {
            entry_type,
            id,
            parent_id,
            timestamp,
            extra: fields.into_iter().collect(),
        })
    }

    fn into_json(self) -> Value {
        let mut fields = self.extra.into_iter().collect::<Map<_, _>>();
        fields.insert("type".into(), Value::String(self.entry_type.0));
        if let Some(id) = self.id {
            fields.insert("id".into(), Value::String(id.0));
        }
        match self.parent_id {
            ParentReference::LegacyMissing => {}
            ParentReference::Root => {
                fields.insert("parentId".into(), Value::Null);
            }
            ParentReference::Entry(id) => {
                fields.insert("parentId".into(), Value::String(id.0));
            }
        }
        fields.insert("timestamp".into(), Value::String(self.timestamp.0));
        Value::Object(fields)
    }
}

/// One record in Pi's JSONL session file.
#[derive(Clone, Debug, PartialEq)]
pub enum SessionRecord {
    Header(SessionHeader),
    Entry(SessionEntry),
}

/// A parsed Pi JSONL session. Original record order is retained.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionDocument {
    records: Vec<SessionRecord>,
}

impl SessionDocument {
    pub fn new(cwd: impl Into<String>) -> Self {
        let timestamp = now_timestamp();
        let header = SessionHeader {
            version: Some(CURRENT_SESSION_VERSION),
            id: SessionId(uuid::Uuid::new_v4().to_string()),
            timestamp,
            cwd: cwd.into(),
            parent_session: None,
            extra: BTreeMap::new(),
        };
        Self {
            records: vec![SessionRecord::Header(header)],
        }
    }

    pub fn append_user_text(&mut self, text: impl Into<String>) {
        let message_timestamp = chrono::Utc::now().timestamp_millis();
        self.append_message(serde_json::json!({
            "role": "user",
            "content": [{ "type": "text", "text": text.into() }],
            "timestamp": message_timestamp,
        }));
    }

    pub fn append_assistant_text(
        &mut self,
        api: &str,
        provider: &str,
        model: &str,
        text: impl Into<String>,
        usage: MessageUsage,
    ) {
        let message_timestamp = chrono::Utc::now().timestamp_millis();
        self.append_message(serde_json::json!({
            "role": "assistant",
            "content": [{ "type": "text", "text": text.into() }],
            "api": api,
            "provider": provider,
            "model": model,
            "usage": usage,
            "stopReason": "stop",
            "timestamp": message_timestamp,
        }));
    }

    fn append_message(&mut self, message: Value) {
        let parent_id = self.entries().last().and_then(|entry| entry.id.clone());
        let id = next_entry_id(
            &mut self
                .entries()
                .filter_map(|entry| entry.id.as_ref().map(|id| id.as_str().to_owned()))
                .collect(),
        );
        let mut extra = BTreeMap::new();
        extra.insert("message".into(), message);
        self.records.push(SessionRecord::Entry(SessionEntry {
            entry_type: EntryType("message".into()),
            id: Some(id),
            parent_id: parent_id
                .map(ParentReference::Entry)
                .unwrap_or(ParentReference::Root),
            timestamp: now_timestamp(),
            extra,
        }));
    }

    pub fn parse_jsonl(input: &str) -> Result<Self, SessionParseError> {
        let mut records = Vec::new();

        for (line_index, line) in input.lines().enumerate() {
            let line_number = line_index + 1;
            if line.trim().is_empty() {
                continue;
            }

            let value: Value =
                serde_json::from_str(line).map_err(|error| SessionParseError::InvalidJson {
                    line: line_number,
                    message: error.to_string(),
                })?;
            let Value::Object(fields) = value else {
                return Err(SessionParseError::RecordIsNotObject { line: line_number });
            };
            let entry_type = fields.get("type").and_then(Value::as_str).ok_or(
                SessionParseError::InvalidField {
                    line: line_number,
                    field: "type",
                    expected: "a string",
                },
            )?;

            let record = if entry_type == "session" {
                SessionRecord::Header(SessionHeader::from_value(fields, line_number)?)
            } else {
                SessionRecord::Entry(SessionEntry::from_value(fields, line_number)?)
            };
            records.push(record);
        }

        Ok(Self { records })
    }

    pub fn records(&self) -> &[SessionRecord] {
        &self.records
    }

    pub fn header(&self) -> Option<&SessionHeader> {
        self.records.iter().find_map(|record| match record {
            SessionRecord::Header(header) => Some(header),
            SessionRecord::Entry(_) => None,
        })
    }

    /// Upgrade legacy session data using Pi's v1 -> v2 -> v3 migrations.
    /// Returns whether the document changed. Sessions without a header are
    /// left alone; the session manager is responsible for supplying metadata.
    pub fn migrate_to_current_version(&mut self) -> bool {
        if self.header().is_none() {
            return false;
        }
        let version = self.header().and_then(|header| header.version).unwrap_or(1);
        if version >= CURRENT_SESSION_VERSION {
            return false;
        }

        if version < 2 {
            self.migrate_v1_to_v2();
        }
        if version < 3 {
            self.migrate_v2_to_v3();
        }
        true
    }

    fn migrate_v1_to_v2(&mut self) {
        let mut seen_ids = HashSet::new();
        let mut previous_id: Option<EntryId> = None;

        for record in &mut self.records {
            match record {
                SessionRecord::Header(header) => header.version = Some(2),
                SessionRecord::Entry(entry) => {
                    let id = next_entry_id(&mut seen_ids);
                    entry.id = Some(id.clone());
                    entry.parent_id = match previous_id.replace(id) {
                        Some(parent_id) => ParentReference::Entry(parent_id),
                        None => ParentReference::Root,
                    };
                }
            }
        }

        let ids_by_record_index = self
            .records
            .iter()
            .map(|record| match record {
                SessionRecord::Header(_) => None,
                SessionRecord::Entry(entry) => entry.id.clone(),
            })
            .collect::<Vec<_>>();
        for record in &mut self.records {
            if let SessionRecord::Entry(entry) = record
                && entry.entry_type.as_str() == "compaction"
            {
                migrate_compaction_index(entry, &ids_by_record_index);
            }
        }
    }

    fn migrate_v2_to_v3(&mut self) {
        for record in &mut self.records {
            match record {
                SessionRecord::Header(header) => header.version = Some(3),
                SessionRecord::Entry(entry) if entry.entry_type.as_str() == "message" => {
                    if let Some(Value::Object(message)) = entry.extra.get_mut("message") {
                        if message.get("role").and_then(Value::as_str) == Some("hookMessage") {
                            message.insert("role".into(), Value::String("custom".into()));
                        }
                    }
                }
                SessionRecord::Entry(_) => {}
            }
        }
    }

    pub fn entries(&self) -> impl Iterator<Item = &SessionEntry> {
        self.records.iter().filter_map(|record| match record {
            SessionRecord::Header(_) => None,
            SessionRecord::Entry(entry) => Some(entry),
        })
    }

    pub fn to_jsonl(&self) -> Result<String, SessionWriteError> {
        let mut output = String::new();
        for record in &self.records {
            let value = match record {
                SessionRecord::Header(header) => header.clone().into_json(),
                SessionRecord::Entry(entry) => entry.clone().into_json(),
            };
            output.push_str(
                &serde_json::to_string(&value)
                    .map_err(|error| SessionWriteError::Serialize(error.to_string()))?,
            );
            output.push('\n');
        }
        Ok(output)
    }

    /// Atomically replace a session file with this document's Pi JSONL form.
    pub fn write_atomic(&self, path: &std::path::Path) -> Result<(), SessionFileError> {
        use std::io::Write;

        let contents = self
            .to_jsonl()
            .map_err(|error| SessionFileError::Serialize(error.to_string()))?;
        let parent = path
            .parent()
            .ok_or_else(|| SessionFileError::Io("session path has no parent directory".into()))?;
        std::fs::create_dir_all(parent).map_err(|error| SessionFileError::Io(error.to_string()))?;
        let temp_path = path.with_extension(format!("jsonl.tmp-{}", uuid::Uuid::new_v4()));
        let result = (|| {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)
                .map_err(|error| SessionFileError::Io(error.to_string()))?;
            file.write_all(contents.as_bytes())
                .map_err(|error| SessionFileError::Io(error.to_string()))?;
            file.sync_all()
                .map_err(|error| SessionFileError::Io(error.to_string()))?;
            std::fs::rename(&temp_path, path)
                .map_err(|error| SessionFileError::Io(error.to_string()))
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp_path);
        }
        result
    }
}

#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageUsage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    pub total_tokens: u64,
    pub cost: MessageCost,
}

#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

fn now_timestamp() -> SessionTimestamp {
    SessionTimestamp(chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
}

fn next_entry_id(seen_ids: &mut HashSet<String>) -> EntryId {
    for _ in 0..100 {
        let candidate = uuid::Uuid::new_v4().simple().to_string()[..8].to_owned();
        if seen_ids.insert(candidate.clone()) {
            return EntryId(candidate);
        }
    }

    let candidate = uuid::Uuid::new_v4().to_string();
    seen_ids.insert(candidate.clone());
    EntryId(candidate)
}

fn migrate_compaction_index(entry: &mut SessionEntry, ids_by_record_index: &[Option<EntryId>]) {
    let Some(index_value) = entry.extra.get("firstKeptEntryIndex") else {
        return;
    };
    if !index_value.is_number() {
        return;
    }

    let index = entry
        .extra
        .remove("firstKeptEntryIndex")
        .and_then(|value| value.as_u64().and_then(|index| usize::try_from(index).ok()));
    if let Some(target_id) = index
        .and_then(|index| ids_by_record_index.get(index))
        .and_then(Option::as_ref)
    {
        entry.extra.insert(
            "firstKeptEntryId".into(),
            Value::String(target_id.as_str().to_owned()),
        );
    }
}

fn take_value(
    fields: &mut Map<String, Value>,
    field: &'static str,
    line: usize,
) -> Result<Value, SessionParseError> {
    fields
        .remove(field)
        .ok_or(SessionParseError::MissingField { line, field })
}

fn take_string(
    fields: &mut Map<String, Value>,
    field: &'static str,
    line: usize,
) -> Result<String, SessionParseError> {
    take_value(fields, field, line)?
        .as_str()
        .map(str::to_owned)
        .ok_or(SessionParseError::InvalidField {
            line,
            field,
            expected: "a string",
        })
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SessionParseError {
    #[error("invalid JSON on session line {line}: {message}")]
    InvalidJson { line: usize, message: String },
    #[error("session line {line} is not a JSON object")]
    RecordIsNotObject { line: usize },
    #[error("session line {line} is missing required field `{field}`")]
    MissingField { line: usize, field: &'static str },
    #[error("session line {line} field `{field}` must be {expected}")]
    InvalidField {
        line: usize,
        field: &'static str,
        expected: &'static str,
    },
    #[error("expected a session header on line {line}")]
    ExpectedSessionHeader { line: usize },
    #[error("unexpected session header after the first record on line {line}")]
    UnexpectedSessionHeader { line: usize },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SessionFileError {
    #[error("could not serialize session file: {0}")]
    Serialize(String),
    #[error("could not write session file: {0}")]
    Io(String),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SessionWriteError {
    #[error("could not serialize session JSONL: {0}")]
    Serialize(String),
}

#[cfg(test)]
mod tests {
    use super::{
        CURRENT_SESSION_VERSION, MessageCost, MessageUsage, ParentReference, SessionDocument,
        SessionRecord,
    };

    #[test]
    fn creates_pi_compatible_linked_message_records() {
        let mut session = SessionDocument::new("/repo");
        session.append_user_text("hello");
        session.append_assistant_text(
            "openai-responses",
            "opencode-cf",
            "gpt-6-luna",
            "hi there",
            MessageUsage {
                input: 3,
                output: 2,
                cache_read: 0,
                cache_write: 0,
                reasoning: None,
                total_tokens: 5,
                cost: MessageCost {
                    input: 0.0,
                    output: 0.0,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0,
                },
            },
        );

        let serialized = session.to_jsonl().expect("serialize new session");
        let parsed = SessionDocument::parse_jsonl(&serialized).expect("parse new session");
        assert_eq!(
            parsed.header().unwrap().version,
            Some(CURRENT_SESSION_VERSION)
        );
        let entries = parsed.entries().collect::<Vec<_>>();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].parent_id, ParentReference::Root);
        assert_eq!(
            entries[1].parent_id,
            ParentReference::Entry(entries[0].id.clone().unwrap())
        );
        assert_eq!(entries[0].extra_fields()["message"]["role"], "user");
        assert_eq!(
            entries[1].extra_fields()["message"]["provider"],
            "opencode-cf"
        );
        assert_eq!(
            entries[1].extra_fields()["message"]["usage"]["totalTokens"],
            5
        );
    }

    #[test]
    fn preserves_legacy_header_and_entries_without_tree_fields() {
        let input = concat!(
            r#"{"type":"session","id":"s1","timestamp":"2025-11-20T23:33:50.805Z","cwd":"/repo","provider":"anthropic","modelId":"model-x"}"#,
            "\n",
            r#"{"type":"message","timestamp":"2025-11-20T23:33:51.000Z","message":{"role":"user","content":"hello","timestamp":1763681631000}}"#,
            "\n"
        );
        let parsed = SessionDocument::parse_jsonl(input).expect("parse legacy session");

        let SessionRecord::Header(header) = &parsed.records()[0] else {
            panic!("first record should be session header");
        };
        assert_eq!(header.version, None);
        assert_eq!(header.extra_fields()["provider"], "anthropic");
        assert_eq!(
            parsed.entries().next().unwrap().parent_id,
            ParentReference::LegacyMissing
        );

        let serialized = parsed.to_jsonl().expect("serialize session");
        let reparsed = SessionDocument::parse_jsonl(&serialized).expect("parse serialized session");
        assert_eq!(reparsed, parsed);
        assert_eq!(CURRENT_SESSION_VERSION, 3);
    }

    #[test]
    fn migrates_v1_tree_fields_and_hook_message_role_to_v3() {
        let input = concat!(
            r#"{"type":"session","id":"s1","timestamp":"t","cwd":"/repo"}"#,
            "\n",
            r#"{"type":"message","timestamp":"t","message":{"role":"hookMessage","content":"legacy"}}"#,
            "\n",
            r#"{"type":"compaction","timestamp":"t","firstKeptEntryIndex":1,"summary":"summary"}"#,
            "\n"
        );
        let mut session = SessionDocument::parse_jsonl(input).expect("parse v1 session");
        assert!(session.migrate_to_current_version());
        assert_eq!(session.header().unwrap().version, Some(3));

        let entries = session.entries().collect::<Vec<_>>();
        assert_eq!(entries[0].parent_id, ParentReference::Root);
        assert_eq!(
            entries[1].parent_id,
            ParentReference::Entry(entries[0].id.clone().unwrap())
        );
        assert_eq!(
            entries[1].extra_fields()["firstKeptEntryId"],
            entries[0].id.as_ref().unwrap().as_str()
        );
        assert!(
            !entries[1]
                .extra_fields()
                .contains_key("firstKeptEntryIndex")
        );
        assert_eq!(entries[0].extra_fields()["message"]["role"], "custom");
        assert!(!session.migrate_to_current_version());
    }

    #[test]
    fn rejects_invalid_input_without_returning_a_partial_document() {
        let error = SessionDocument::parse_jsonl(
            "{\"type\":\"session\",\"id\":\"s1\",\"timestamp\":\"t\",\"cwd\":\"/repo\"}\nnot-json\n",
        )
        .expect_err("malformed record must fail parsing");

        assert!(error.to_string().contains("line 2"));
    }

    #[test]
    fn preserves_unknown_nested_entry_fields() {
        let input = concat!(
            r#"{"type":"session","version":3,"id":"s1","timestamp":"t","cwd":"/repo"}"#,
            "\n",
            r#"{"type":"custom","id":"e1","parentId":null,"timestamp":"t","customType":"plug","data":{"future":[1,true,null]}}"#,
            "\n"
        );
        let parsed = SessionDocument::parse_jsonl(input).expect("parse session");
        let serialized = parsed.to_jsonl().expect("serialize session");
        let reparsed = SessionDocument::parse_jsonl(&serialized).expect("parse serialized session");

        assert_eq!(reparsed, parsed);
        assert_eq!(
            reparsed.entries().next().unwrap().extra_fields()["data"]["future"][1],
            true
        );
    }
}
