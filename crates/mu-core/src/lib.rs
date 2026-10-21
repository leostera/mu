//! Mu's Rust-native agent loop and orchestration layer.
//!
//! Pi-specific persistence is defined separately in [`session`], so the
//! runtime can evolve without changing the on-disk compatibility contract.

mod agent;
pub mod agent_loop;
mod codemode;
mod context_files;
mod mcp;
pub mod path_utils;
pub mod skills;
mod tool_search;
pub mod tools;

pub use agent::{
    AgentError, AgentEvent, AgentPromptQueue, FooterStats, MuAgent, PromptBehavior, SessionInfo,
    SessionOptions, SessionTranscriptMessage, SessionTranscriptRole, SessionTranscriptTool,
    latest_session_for_current_cwd, list_sessions_for_current_cwd, run_cli, run_prompt,
    run_prompt_json, run_prompt_with_options, run_prompt_with_session,
};
pub use mu_session as session;
