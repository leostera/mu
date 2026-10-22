//! Embedded JavaScript runtime for Pi-compatible extensions.
//!
//! This runtime evaluates JavaScript and transpiles TypeScript script sources. Plugin API bindings,
//! module loading, permissions, and lifecycle management remain future slices.

use deno_ast::{
    EmitOptions, MediaType, ModuleSpecifier, ParseParams, TranspileModuleOptions, TranspileOptions,
    parse_program,
};
use deno_core::{JsRuntime, RuntimeOptions};
use std::sync::Arc;
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

    /// Evaluate JavaScript or a TypeScript script source string in this runtime.
    ///
    /// TypeScript is selected from the module specifier's media type and transpiled without type
    /// checking. This executes script-shaped entries; module resolution and ES module loading are
    /// separate extension-runtime work.
    pub fn evaluate(&mut self, specifier: &str, source: &str) -> Result<(), RuntimeError> {
        let source = transpile_typescript(specifier, source)?;
        self.runtime
            .execute_script(specifier.to_owned(), source)
            .map_err(|error| RuntimeError::Evaluation(error.to_string()))?;
        Ok(())
    }
}

fn transpile_typescript(specifier: &str, source: &str) -> Result<String, RuntimeError> {
    let path = specifier.split(['?', '#']).next().unwrap_or(specifier);
    if ![".ts", ".mts", ".cts"]
        .iter()
        .any(|suffix| path.ends_with(suffix))
    {
        return Ok(source.to_owned());
    }

    let module_specifier = ModuleSpecifier::parse(specifier)
        .map_err(|error| RuntimeError::Evaluation(error.to_string()))?;
    let media_type = MediaType::from_specifier(&module_specifier);
    if !matches!(
        media_type,
        MediaType::TypeScript | MediaType::Mts | MediaType::Cts
    ) {
        return Ok(source.to_owned());
    }

    let parsed = parse_program(ParseParams {
        specifier: module_specifier,
        text: Arc::from(source),
        media_type,
        capture_tokens: false,
        scope_analysis: false,
        maybe_syntax: None,
    })
    .map_err(|error| RuntimeError::Evaluation(error.to_string()))?;
    let emitted = parsed
        .transpile(
            &TranspileOptions::default(),
            &TranspileModuleOptions::default(),
            &EmitOptions::default(),
        )
        .map_err(|error| RuntimeError::Evaluation(error.to_string()))?
        .into_source();
    Ok(emitted.text)
}

impl Default for PluginRuntime {
    fn default() -> Self {
        Self::new()
    }
}
