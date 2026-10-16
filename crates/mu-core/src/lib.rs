//! Mu's Rust-native agent loop and orchestration layer.
//!
//! Pi-specific persistence is defined separately in [`session`], so the
//! runtime can evolve without changing the on-disk compatibility contract.

mod agent;

pub use agent::{AgentError, run_cli};
pub use mu_session as session;
