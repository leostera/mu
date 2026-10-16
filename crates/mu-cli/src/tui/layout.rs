use ratatui::layout::Rect;

/// Geometry for Pi's fullscreen chat: a scrollable document above a fixed dock.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ChatViewport {
    pub transcript: Rect,
    pub command_menu: Rect,
    pub editor: Rect,
    pub footer: Rect,
}

impl ChatViewport {
    pub fn new(area: Rect, command_menu_rows: u16, editor_content_rows: u16) -> Self {
        let footer_height = area.height.min(2);
        let remaining = area.height - footer_height;
        let editor_height = remaining.min(editor_content_rows.saturating_add(2));
        let remaining = remaining - editor_height;
        let menu_height = remaining.min(command_menu_rows);
        let transcript_height = remaining - menu_height;

        let transcript = Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: transcript_height,
        };
        let command_menu = Rect {
            y: transcript.y.saturating_add(transcript.height),
            height: menu_height,
            ..transcript
        };
        let editor = Rect {
            y: command_menu.y.saturating_add(command_menu.height),
            height: editor_height,
            ..transcript
        };
        let footer = Rect {
            y: editor.y.saturating_add(editor.height),
            height: footer_height,
            ..transcript
        };

        Self {
            transcript,
            command_menu,
            editor,
            footer,
        }
    }
}

#[cfg(test)]
mod tests {
    use ratatui::layout::Rect;

    use super::ChatViewport;

    #[test]
    fn keeps_editor_and_footer_fixed_below_the_transcript() {
        let layout = ChatViewport::new(Rect::new(3, 5, 80, 24), 4, 3);

        assert_eq!(layout.transcript, Rect::new(3, 5, 80, 13));
        assert_eq!(layout.command_menu, Rect::new(3, 18, 80, 4));
        assert_eq!(layout.editor, Rect::new(3, 22, 80, 5));
        assert_eq!(layout.footer, Rect::new(3, 27, 80, 2));
    }

    #[test]
    fn shrinks_dock_before_consuming_the_last_transcript_rows() {
        let layout = ChatViewport::new(Rect::new(0, 0, 40, 5), 3, 8);

        assert_eq!(layout.transcript.height, 0);
        assert_eq!(layout.command_menu.height, 0);
        assert_eq!(layout.editor.height, 3);
        assert_eq!(layout.footer.height, 2);
    }

    #[test]
    fn handles_zero_sized_terminal_area() {
        let layout = ChatViewport::new(Rect::new(0, 0, 0, 0), 3, 8);

        assert_eq!(layout.transcript, Rect::default());
        assert_eq!(layout.command_menu, Rect::default());
        assert_eq!(layout.editor, Rect::default());
        assert_eq!(layout.footer, Rect::default());
    }
}
