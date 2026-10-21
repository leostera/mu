//! Pi-compatible agent loop: events, hooks, tool scheduling, and transcript tool state.

mod run;
mod transcript;
mod types;
mod validation;

#[cfg(test)]
mod tests;

pub use run::{
    ContinueError, RunToolCallOptions, run_agent_loop, run_agent_loop_continue, run_tool_call,
};
pub use transcript::{ToolStateChanges, current_tools, declare_tool_changes, tool_state_changes};
pub use types::*;
pub use validation::validate_tool_arguments;
