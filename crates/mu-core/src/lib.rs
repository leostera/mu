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
    AgentError, AgentEvent, AgentPromptQueue, MuAgent, PromptBehavior, run_cli, run_prompt,
};
pub use mu_session as session;
