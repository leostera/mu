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
pub mod image_process;
mod mcp;
pub mod nested_calls;
pub mod oauth;
pub mod path_utils;
pub mod prompt_templates;
pub mod settings;
pub mod skills;
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
