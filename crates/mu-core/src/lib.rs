//! Mu's Rust-native agent loop and orchestration layer.
//!
//! Pi-specific persistence is defined separately in [`session`], so the
//! runtime can evolve without changing the on-disk compatibility contract.

mod agent;
mod codemode;
mod mcp;
pub mod skills;
mod tool_search;
pub mod tools;

pub use agent::{
    AgentError, AgentEvent, AgentPromptQueue, MuAgent, PromptBehavior,
    latest_session_for_current_cwd, run_cli, run_prompt, run_prompt_with_session,
};
pub use mu_session as session;
