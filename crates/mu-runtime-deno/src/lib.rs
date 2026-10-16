//! Embedded JavaScript runtime for Pi-compatible extensions.
//!
//! This first slice only proves that the project can host V8. Plugin API
//! bindings, module loading, permissions, and lifecycle management follow later.

use deno_core::{JsRuntime, RuntimeOptions};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("JavaScript evaluation failed: {0}")]
    Evaluation(String),
}

/// A per-agent-session JavaScript isolate.
pub struct PluginRuntime {
    runtime: JsRuntime,
}

impl PluginRuntime {
    pub fn new() -> Self {
        Self {
            runtime: JsRuntime::new(RuntimeOptions::default()),
        }
    }

    /// Evaluate a JavaScript plugin source string in this runtime.
    pub fn evaluate(&mut self, specifier: &str, source: &str) -> Result<(), RuntimeError> {
        self.runtime
            .execute_script(specifier.to_owned(), source.to_owned())
            .map_err(|error| RuntimeError::Evaluation(error.to_string()))?;
        Ok(())
    }
}

impl Default for PluginRuntime {
    fn default() -> Self {
        Self::new()
    }
}
