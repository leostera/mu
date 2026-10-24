//! Standalone HTML export of a session's active branch (`/export <path>.html`).

use std::path::{Path, PathBuf};

use mu_session::SessionDocument;

use super::{
    AgentError, SessionTranscriptMessage, SessionTranscriptRole, SessionTranscriptTool,
    load_session, session_transcript,
};

pub(super) fn render_session_html(session: &SessionDocument) -> String {
    let header = session.header();
    let id = header.map(|h| h.id.as_str().to_owned()).unwrap_or_default();
    let timestamp = header
        .map(|h| h.timestamp.as_str().to_owned())
        .unwrap_or_default();
    let title = session.session_name().unwrap_or("Mu session");
    let mut html = format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{}</title><style>{SESSION_HTML_STYLE}</style></head><body><main><header><h1>{}</h1><p class=\"meta\">Session {} · {}</p></header>",
        escape_html(title),
        escape_html(title),
        escape_html(&id),
        escape_html(&timestamp),
    );
    for message in session_transcript(session) {
        append_transcript_message_html(&mut html, message);
    }
    html.push_str("</main></body></html>");
    html
}

/// `--export <session.jsonl> [output.html]`: render a session file without connecting a provider.
///
/// The default output is `mu-session-<name>.html` in the working directory (Pi's naming scheme).
pub fn export_session_file(input: &Path, output: Option<&Path>) -> Result<PathBuf, AgentError> {
    let (session, _) = load_session(input)?;
    let output = match output {
        Some(path) => path.to_owned(),
        None => {
            let stem = input
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            std::env::current_dir()?.join(format!("mu-session-{stem}.html"))
        }
    };
    std::fs::write(&output, render_session_html(&session))
        .map_err(|error| AgentError::Session(error.to_string()))?;
    Ok(output)
}

pub(super) const SESSION_HTML_STYLE: &str = "body{margin:0;background:#17191f;color:#e5e7eb;font:16px/1.6 system-ui,sans-serif}main{max-width:900px;margin:3rem auto;padding:0 1.25rem}header{border-bottom:1px solid #454955;margin-bottom:2rem}article{padding:1rem 1.25rem;margin:1rem 0;border-radius:12px;background:#22252d}article.user{background:#292d38}h1,h2{line-height:1.2}h2{font-size:1rem;color:#aeb8ce}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:.5rem 0}details{margin:.75rem 0;padding:.75rem;background:#191b21;border-radius:8px}.error{color:#ff8a8a}.meta{color:#9ca3af;font-size:.9rem}";

pub(super) fn append_transcript_message_html(html: &mut String, message: SessionTranscriptMessage) {
    let (role, class) = match message.role {
        SessionTranscriptRole::User => ("User", "user"),
        SessionTranscriptRole::Assistant => ("Assistant", "assistant"),
        SessionTranscriptRole::CompactionSummary(_) => ("Compaction summary", "assistant"),
        SessionTranscriptRole::BranchSummary => ("Branch summary", "assistant"),
    };
    html.push_str(&format!("<article class=\"{class}\"><h2>{role}</h2>"));
    if !message.thinking.is_empty() {
        html.push_str(&format!(
            "<details><summary>Thinking</summary><pre>{}</pre></details>",
            escape_html(&message.thinking)
        ));
    }
    if !message.text.is_empty() {
        html.push_str(&format!("<pre>{}</pre>", escape_html(&message.text)));
    }
    for tool in message.tools {
        append_tool_html(html, tool);
    }
    html.push_str("</article>");
}

pub(super) fn append_tool_html(html: &mut String, tool: SessionTranscriptTool) {
    let arguments = serde_json::to_string_pretty(&tool.arguments)
        .unwrap_or_else(|_| tool.arguments.to_string());
    let error_class = if tool.is_error {
        " class=\"error\""
    } else {
        ""
    };
    html.push_str(&format!(
        "<details{error_class}><summary>Tool: {}</summary><pre>{}</pre>",
        escape_html(&tool.name),
        escape_html(&arguments)
    ));
    if let Some(output) = tool.output {
        html.push_str(&format!("<pre>{}</pre>", escape_html(&output)));
    }
    html.push_str("</details>");
}

pub(super) fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
