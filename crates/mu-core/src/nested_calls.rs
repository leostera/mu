//! Record of tool calls a tool makes while it runs (Pi's `NestedCallRecorder` in
//! `core/nested-tool-calls.ts`), kept on the model-issued call's tool result.

use std::time::Instant;

use serde_json::Value;

/// Limits of the record: oversized arguments are omitted, calls beyond the count are dropped, and
/// the record is marked incomplete when either happens.
pub const MAX_CALLS: usize = 256;
pub const MAX_ARGUMENT_BYTES_PER_CALL: usize = 8 * 1024;
pub const MAX_ARGUMENT_BYTES_TOTAL: usize = 32 * 1024;
pub const MAX_ERROR_CHARS: usize = 500;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallStatus {
    Unfinished,
    Ok,
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NestedCall {
    pub id: String,
    pub name: String,
    pub status: CallStatus,
    pub arguments: Option<Value>,
    /// Size of the arguments when they were too large to keep.
    pub arguments_bytes: Option<usize>,
    pub duration_ms: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NestedCalls {
    pub calls: Vec<NestedCall>,
    pub complete: bool,
}

/// Handle to a started call, for [`NestedCallRecorder::finish`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallHandle(usize);

#[derive(Debug)]
pub struct NestedCallRecorder {
    calls: Vec<NestedCall>,
    started_at: Vec<Instant>,
    complete: bool,
    argument_bytes: usize,
}

impl Default for NestedCallRecorder {
    fn default() -> Self {
        Self {
            calls: Vec::new(),
            started_at: Vec::new(),
            complete: true,
            argument_bytes: 0,
        }
    }
}

impl NestedCallRecorder {
    /// Record a call as it starts; `None` when the call is dropped.
    pub fn start(&mut self, id: &str, name: &str, arguments: &Value) -> Option<CallHandle> {
        if self.calls.len() >= MAX_CALLS {
            self.complete = false;
            return None;
        }
        let json = serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_owned());
        let bytes = json.len();
        let mut call = NestedCall {
            id: id.to_owned(),
            name: name.to_owned(),
            status: CallStatus::Unfinished,
            arguments: None,
            arguments_bytes: None,
            duration_ms: None,
            error: None,
        };
        if bytes > MAX_ARGUMENT_BYTES_PER_CALL
            || self.argument_bytes + bytes > MAX_ARGUMENT_BYTES_TOTAL
        {
            call.arguments_bytes = Some(bytes);
            self.complete = false;
        } else {
            call.arguments = Some(arguments.clone());
            self.argument_bytes += bytes;
        }
        self.calls.push(call);
        self.started_at.push(Instant::now());
        Some(CallHandle(self.calls.len() - 1))
    }

    pub fn finish(&mut self, handle: Option<CallHandle>, is_error: bool, error_text: &str) {
        let Some(CallHandle(index)) = handle else {
            return;
        };
        let call = &mut self.calls[index];
        call.status = if is_error {
            CallStatus::Error
        } else {
            CallStatus::Ok
        };
        call.duration_ms = Some(self.started_at[index].elapsed().as_millis() as u64);
        if is_error && !error_text.is_empty() {
            call.error = Some(error_text.chars().take(MAX_ERROR_CHARS).collect());
        }
    }

    /// Copy of the record so far, or `None` when no nested call was made.
    pub fn snapshot(&self) -> Option<NestedCalls> {
        if self.calls.is_empty() && self.complete {
            return None;
        }
        let complete = self.complete
            && self
                .calls
                .iter()
                .all(|c| c.status != CallStatus::Unfinished);
        Some(NestedCalls {
            calls: self.calls.clone(),
            complete,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Ported from Pi's `nested-tool-calls.test.ts` (`NestedCallRecorder`).
    #[test]
    fn omits_oversized_arguments_and_drops_calls_beyond_the_limit() {
        let mut recorder = NestedCallRecorder::default();
        assert_eq!(recorder.snapshot(), None);
        let small = recorder.start("a", "t", &json!({"x": 1}));
        recorder.finish(small, false, "");
        let snapshot = recorder.snapshot().unwrap();
        assert!(snapshot.complete);
        assert_eq!(snapshot.calls[0].arguments, Some(json!({"x": 1})));
        assert_eq!(snapshot.calls[0].status, CallStatus::Ok);
        assert!(snapshot.calls[0].duration_ms.is_some());

        let big = recorder.start(
            "b",
            "t",
            &json!({"text": "x".repeat(MAX_ARGUMENT_BYTES_PER_CALL)}),
        );
        recorder.finish(big, true, &"e".repeat(1000));
        let snapshot = recorder.snapshot().unwrap();
        assert!(!snapshot.complete);
        assert_eq!(snapshot.calls[1].status, CallStatus::Error);
        assert_eq!(snapshot.calls[1].arguments, None);
        assert!(snapshot.calls[1].arguments_bytes.unwrap() > MAX_ARGUMENT_BYTES_PER_CALL);
        assert_eq!(
            snapshot.calls[1].error.as_ref().unwrap().chars().count(),
            MAX_ERROR_CHARS
        );

        for i in 0..MAX_CALLS {
            let handle = recorder.start(&format!("c{i}"), "t", &json!({}));
            recorder.finish(handle, false, "");
        }
        assert_eq!(recorder.snapshot().unwrap().calls.len(), MAX_CALLS);
    }

    #[test]
    fn caps_total_argument_size_and_marks_unfinished_calls_incomplete() {
        let mut recorder = NestedCallRecorder::default();
        let chunk = json!({"text": "x".repeat(7000)});
        let handles = (0..6)
            .map(|i| recorder.start(&format!("c{i}"), "t", &chunk))
            .collect::<Vec<_>>();
        let snapshot = recorder.snapshot().unwrap();
        // 32 KiB fits four 7000-byte argument objects.
        assert_eq!(
            snapshot
                .calls
                .iter()
                .filter(|c| c.arguments.is_some())
                .count(),
            4
        );
        assert!(
            snapshot
                .calls
                .iter()
                .all(|c| c.status == CallStatus::Unfinished)
        );
        assert!(!snapshot.complete);
        assert_eq!(handles.len(), 6);
    }
}
