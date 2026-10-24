//! Server-sent-event framing shared by the streaming provider adapters.
//!
//! A frame ends at a blank line (`\n\n`). Mu reads the first `data:` line of each frame and
//! skips frames without one (comments, `event:`-only keep-alives), as the adapters did before
//! this module existed.

/// Accumulates stream chunks and yields the trimmed `data:` payload of each complete frame.
#[derive(Debug, Default)]
pub struct SseFrames {
    buffer: Vec<u8>,
    /// Start of the first unread frame in `buffer`.
    start: usize,
    /// Bytes before this offset hold no frame terminator, so the search resumes here.
    scanned: usize,
}

impl SseFrames {
    pub fn push(&mut self, chunk: &[u8]) {
        // Drop consumed frames once they dominate the buffer, so pushes stay amortized O(1).
        if self.start > 0 && self.start >= self.buffer.len() / 2 {
            self.buffer.drain(..self.start);
            self.scanned -= self.start;
            self.start = 0;
        }
        self.buffer.extend_from_slice(chunk);
    }

    /// The next complete frame's `data:` payload, or `None` until more bytes arrive.
    pub fn next_data(&mut self) -> Option<String> {
        loop {
            let frame_end = self.next_frame_end()?;
            let frame = String::from_utf8_lossy(&self.buffer[self.start..frame_end]);
            let data = frame
                .lines()
                .find_map(|line| line.strip_prefix("data:"))
                .map(|data| data.trim().to_owned());
            self.start = frame_end;
            self.scanned = frame_end;
            if data.is_some() {
                return data;
            }
        }
    }

    /// End (exclusive, after the blank line) of the frame starting at `start`.
    fn next_frame_end(&mut self) -> Option<usize> {
        // A terminator may straddle the previous scan boundary.
        let from = self.scanned.saturating_sub(1).max(self.start);
        match self.buffer[from..]
            .windows(2)
            .position(|pair| pair == b"\n\n")
        {
            Some(offset) => Some(from + offset + 2),
            None => {
                self.scanned = self.buffer.len();
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(frames: &mut SseFrames) -> Vec<String> {
        std::iter::from_fn(|| frames.next_data()).collect()
    }

    #[test]
    fn yields_data_of_complete_frames_and_skips_frames_without_data() {
        let mut frames = SseFrames::default();
        frames.push(b": keep-alive\n\nevent: x\ndata: {\"a\":1} \n\ndata: [DONE]\n\ndata: tail");
        assert_eq!(drain(&mut frames), ["{\"a\":1}", "[DONE]"]);
        frames.push(b"\n\n");
        assert_eq!(drain(&mut frames), ["tail"]);
    }

    #[test]
    fn terminators_split_across_chunks_are_found() {
        let mut frames = SseFrames::default();
        frames.push(b"data: one\n");
        assert_eq!(frames.next_data(), None);
        frames.push(b"\ndata: two\n");
        assert_eq!(drain(&mut frames), ["one"]);
        frames.push(b"\n");
        assert_eq!(drain(&mut frames), ["two"]);
    }

    #[test]
    fn only_the_first_data_line_of_a_frame_is_read() {
        let mut frames = SseFrames::default();
        frames.push(b"data: first\ndata: second\n\n");
        assert_eq!(drain(&mut frames), ["first"]);
    }

    #[test]
    fn invalid_utf8_is_replaced_rather_than_dropping_the_frame() {
        let mut frames = SseFrames::default();
        frames.push(b"data: a\xffb\n\n");
        assert_eq!(drain(&mut frames), ["a\u{fffd}b"]);
    }
}
