//! Prompts queued while a run is active: steering and follow-up queues and their delivery modes.

use std::{
    collections::VecDeque,
    path::Path,
    sync::{Arc, Mutex as StdMutex},
};

use super::configured_setting;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromptBehavior {
    Steer,
    FollowUp,
}

#[derive(Clone)]
pub struct AgentPromptQueue {
    pending: Arc<StdMutex<PendingPrompts>>,
}

/// How many queued messages the loop injects at a queue drain point (Pi's `QueueMode`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum QueueMode {
    All,
    #[default]
    OneAtATime,
}

impl QueueMode {
    pub(super) fn from_setting(value: &str) -> Option<Self> {
        match value {
            "all" => Some(Self::All),
            "one-at-a-time" => Some(Self::OneAtATime),
            _ => None,
        }
    }

    fn drain(self, queue: &mut VecDeque<String>) -> Vec<String> {
        match self {
            Self::All => queue.drain(..).collect(),
            Self::OneAtATime => queue.pop_front().into_iter().collect(),
        }
    }
}

#[derive(Default)]
pub(super) struct PendingPrompts {
    steering: VecDeque<String>,
    follow_up: VecDeque<String>,
}

impl AgentPromptQueue {
    pub fn enqueue(&self, prompt: String, behavior: PromptBehavior) {
        if prompt.trim().is_empty() {
            return;
        }
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match behavior {
            PromptBehavior::Steer => pending.steering.push_back(prompt),
            PromptBehavior::FollowUp => pending.follow_up.push_back(prompt),
        }
    }

    pub fn take_next(&self) -> Option<(String, PromptBehavior)> {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        pending
            .steering
            .pop_front()
            .map(|prompt| (prompt, PromptBehavior::Steer))
            .or_else(|| {
                pending
                    .follow_up
                    .pop_front()
                    .map(|prompt| (prompt, PromptBehavior::FollowUp))
            })
    }

    /// Drain queued follow-ups: every one for `All`, only the oldest for `OneAtATime`.
    pub(super) fn take_follow_ups(&self, mode: QueueMode) -> Vec<String> {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        mode.drain(&mut pending.follow_up)
    }

    /// Drain queued steering messages according to `mode`.
    pub(super) fn take_steering_batch(&self, mode: QueueMode) -> Vec<String> {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        mode.drain(&mut pending.steering)
    }

    pub fn clear(&self) {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        pending.steering.clear();
        pending.follow_up.clear();
    }
}

impl Default for AgentPromptQueue {
    fn default() -> Self {
        Self {
            pending: Arc::new(StdMutex::new(PendingPrompts::default())),
        }
    }
}

pub(super) fn configured_queue_mode(agent_dir: &Path, cwd: &Path, key: &str) -> QueueMode {
    configured_setting(agent_dir, cwd, key, |value| {
        value.as_str().and_then(QueueMode::from_setting)
    })
    .unwrap_or_default()
}
