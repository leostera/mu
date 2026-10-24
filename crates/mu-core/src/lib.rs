//! Mu's Rust-native agent loop and orchestration layer.
//!
//! Pi-specific persistence is defined separately in [`session`], so the
//! runtime can evolve without changing the on-disk compatibility contract.

mod agent;
pub mod agent_loop;
pub mod auth;
mod codemode;
pub mod compaction;
pub mod config_value;
mod context_files;
pub mod crash_log;
pub mod image_process;
mod mcp;
pub(crate) mod mcp_oauth;
pub mod nested_calls;
pub mod oauth;
pub mod path_utils;
pub mod prompt_templates;
pub mod settings;
pub mod skills;
pub mod sse;
pub mod timings;
mod tool_search;
pub mod tools;

pub use agent::{
    AgentError, AgentEvent, AgentPromptQueue, CompactionOutcome, FooterStats, MuAgent,
    PromptBehavior, SessionInfo, SessionOptions, SessionTranscriptMessage, SessionTranscriptRole,
    SessionTranscriptTool, ThinkingSetting, TreeRow, export_session_file,
    latest_session_for_current_cwd, list_sessions_for_current_cwd, run_cli, run_prompt,
    run_prompt_json, run_prompt_with_options, run_prompt_with_session,
};
pub use mu_session as session;

/// Entry points into crate-private hot paths, for benchmarks only.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod bench_internals {
    use serde_json::Value;

    pub use crate::agent::{decode_anthropic_events, decode_completions_chunks};

    pub fn rank_tools(query: &str, tools: &[Value], limit: usize) -> Vec<Value> {
        crate::tool_search::rank(query, tools, limit)
    }

    /// Pi's MCP output limit, with the full-output save stubbed out.
    pub fn limit_mcp_text(text: &str) -> String {
        crate::mcp::limit_mcp_text(text, |_| Ok("/dev/null".into()))
    }
}
