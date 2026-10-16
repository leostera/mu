use std::io;

use crossterm::{
    cursor::Show,
    event::{
        DisableMouseCapture, EnableMouseCapture, KeyboardEnhancementFlags,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TerminalMode {
    AlternateScreen,
    MouseCapture,
    KeyboardEnhancement,
}

trait TerminalControl {
    fn set_raw_mode(&mut self, enabled: bool) -> io::Result<()>;
    fn set_mode(&mut self, mode: TerminalMode, enabled: bool) -> io::Result<()>;
    fn show_cursor(&mut self) -> io::Result<()>;
}

pub(super) struct TerminalSession {
    _protocol: TerminalProtocol<CrosstermControl>,
}

impl TerminalSession {
    pub(super) fn enter() -> io::Result<Self> {
        Ok(Self {
            _protocol: TerminalProtocol::enter_with(CrosstermControl::default())?,
        })
    }
}

struct TerminalProtocol<C: TerminalControl> {
    control: C,
    acquired: Vec<TerminalMode>,
    raw_mode: bool,
}

impl<C: TerminalControl> TerminalProtocol<C> {
    fn enter_with(mut control: C) -> io::Result<Self> {
        control.set_raw_mode(true)?;
        let mut session = Self {
            control,
            acquired: Vec::new(),
            raw_mode: true,
        };
        for mode in [
            TerminalMode::AlternateScreen,
            TerminalMode::MouseCapture,
            TerminalMode::KeyboardEnhancement,
        ] {
            session.control.set_mode(mode, true)?;
            session.acquired.push(mode);
        }
        Ok(session)
    }
}

impl<C: TerminalControl> Drop for TerminalProtocol<C> {
    fn drop(&mut self) {
        while let Some(mode) = self.acquired.pop() {
            let _ = self.control.set_mode(mode, false);
        }
        if self.raw_mode {
            let _ = self.control.set_raw_mode(false);
            self.raw_mode = false;
        }
        let _ = self.control.show_cursor();
    }
}

struct CrosstermControl {
    output: io::Stdout,
}

impl Default for CrosstermControl {
    fn default() -> Self {
        Self {
            output: io::stdout(),
        }
    }
}

impl TerminalControl for CrosstermControl {
    fn set_raw_mode(&mut self, enabled: bool) -> io::Result<()> {
        if enabled {
            enable_raw_mode()
        } else {
            disable_raw_mode()
        }
    }

    fn set_mode(&mut self, mode: TerminalMode, enabled: bool) -> io::Result<()> {
        match (mode, enabled) {
            (TerminalMode::AlternateScreen, true) => execute!(self.output, EnterAlternateScreen),
            (TerminalMode::AlternateScreen, false) => execute!(self.output, LeaveAlternateScreen),
            (TerminalMode::MouseCapture, true) => execute!(self.output, EnableMouseCapture),
            (TerminalMode::MouseCapture, false) => execute!(self.output, DisableMouseCapture),
            (TerminalMode::KeyboardEnhancement, true) => execute!(
                self.output,
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
            ),
            (TerminalMode::KeyboardEnhancement, false) => {
                execute!(self.output, PopKeyboardEnhancementFlags)
            }
        }
    }

    fn show_cursor(&mut self) -> io::Result<()> {
        execute!(self.output, Show)
    }
}

#[cfg(test)]
mod tests {
    use super::{TerminalControl, TerminalMode, TerminalProtocol};
    use std::{
        io,
        sync::{Arc, Mutex},
    };

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum Operation {
        Raw(bool),
        Mode(TerminalMode, bool),
        ShowCursor,
    }

    #[derive(Clone, Default)]
    struct MockControl {
        operations: Arc<Mutex<Vec<Operation>>>,
        fail_on: Option<Operation>,
    }

    impl MockControl {
        fn record(&self, operation: Operation) -> io::Result<()> {
            let should_fail = self.fail_on.as_ref() == Some(&operation);
            self.operations.lock().expect("mock log").push(operation);
            if should_fail {
                Err(io::Error::other("injected terminal protocol failure"))
            } else {
                Ok(())
            }
        }
    }

    impl TerminalControl for MockControl {
        fn set_raw_mode(&mut self, enabled: bool) -> io::Result<()> {
            self.record(Operation::Raw(enabled))
        }

        fn set_mode(&mut self, mode: TerminalMode, enabled: bool) -> io::Result<()> {
            self.record(Operation::Mode(mode, enabled))
        }

        fn show_cursor(&mut self) -> io::Result<()> {
            self.record(Operation::ShowCursor)
        }
    }

    #[test]
    fn restores_terminal_modes_in_reverse_order() {
        let control = MockControl::default();
        let operations = Arc::clone(&control.operations);
        let session = TerminalProtocol::enter_with(control).expect("enter terminal modes");
        drop(session);

        assert_eq!(
            *operations.lock().expect("mock log"),
            [
                Operation::Raw(true),
                Operation::Mode(TerminalMode::AlternateScreen, true),
                Operation::Mode(TerminalMode::MouseCapture, true),
                Operation::Mode(TerminalMode::KeyboardEnhancement, true),
                Operation::Mode(TerminalMode::KeyboardEnhancement, false),
                Operation::Mode(TerminalMode::MouseCapture, false),
                Operation::Mode(TerminalMode::AlternateScreen, false),
                Operation::Raw(false),
                Operation::ShowCursor,
            ]
        );
    }

    #[test]
    fn partial_setup_failure_rolls_back_already_enabled_modes() {
        let control = MockControl {
            fail_on: Some(Operation::Mode(TerminalMode::MouseCapture, true)),
            ..MockControl::default()
        };
        let operations = Arc::clone(&control.operations);

        assert!(TerminalProtocol::enter_with(control).is_err());
        assert_eq!(
            *operations.lock().expect("mock log"),
            [
                Operation::Raw(true),
                Operation::Mode(TerminalMode::AlternateScreen, true),
                Operation::Mode(TerminalMode::MouseCapture, true),
                Operation::Mode(TerminalMode::AlternateScreen, false),
                Operation::Raw(false),
                Operation::ShowCursor,
            ]
        );
    }
}
