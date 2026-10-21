//! Startup profiling (Pi's `core/timings.ts`): with `PI_TIMING=1`, labelled steps are timed and
//! printed to stderr as a table.

use std::{sync::Mutex, time::Instant};

struct State {
    last: Instant,
    steps: Vec<(String, u128)>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn enabled() -> bool {
    std::env::var("PI_TIMING").is_ok_and(|value| value == "1")
}

/// Record the time since the previous call (or the first call) under `label`.
pub fn time(label: &str) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = state.get_or_insert_with(|| State {
        last: now,
        steps: Vec::new(),
    });
    state
        .steps
        .push((label.to_owned(), now.duration_since(state.last).as_millis()));
    state.last = now;
}

/// The table Pi prints: each step, then the total.
pub fn format_steps(title: &str, steps: &[(String, u128)]) -> Option<String> {
    if steps.is_empty() {
        return None;
    }
    let mut out = format!("\n--- {title} ---\n");
    for (label, ms) in steps {
        out.push_str(&format!("  {label}: {ms}ms\n"));
    }
    out.push_str(&format!(
        "  TOTAL: {}ms\n",
        steps.iter().map(|(_, ms)| ms).sum::<u128>()
    ));
    out.push_str(&"-".repeat(title.len() + 8));
    out.push('\n');
    Some(out)
}

pub fn print() {
    if !enabled() {
        return;
    }
    let state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(table) = state
        .as_ref()
        .and_then(|s| format_steps("Startup Timings: main", &s.steps))
    {
        eprintln!("{table}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_lists_steps_and_totals_like_pi() {
        let steps = vec![("args".to_owned(), 3), ("session".to_owned(), 10)];
        assert_eq!(
            format_steps("Startup Timings: main", &steps).unwrap(),
            "\n--- Startup Timings: main ---\n  args: 3ms\n  session: 10ms\n  TOTAL: 13ms\n-----------------------------\n"
        );
        assert_eq!(format_steps("x", &[]), None);
    }
}
