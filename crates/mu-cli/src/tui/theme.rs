//! Pi's JSON themes (`modes/interactive/theme/{dark,light}.json`) resolved to Ratatui colors.
//!
//! Colors may be hex, `okhsl(h s% l%)` (converted with the `palette` crate), `""` (terminal
//! default), or a reference to an entry in the theme's `vars` table.

use std::collections::HashMap;

use palette::{FromColor, Okhsl, Srgb};
use ratatui::style::{Color, Style};
use serde_json::Value;

const DARK: &str = include_str!("../../assets/themes/dark.json");
const LIGHT: &str = include_str!("../../assets/themes/light.json");

#[derive(Clone, Debug)]
pub struct Theme {
    colors: HashMap<String, Color>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ThemeError {
    #[error("invalid theme JSON: {0}")]
    Json(String),
    #[error("Variable reference not found: {0}")]
    UnknownVariable(String),
    #[error("Circular variable reference detected: {0}")]
    CircularVariable(String),
    #[error("invalid color `{0}`")]
    InvalidColor(String),
}

impl Theme {
    pub fn dark() -> Self {
        Self::parse(DARK).expect("the bundled dark theme is valid")
    }

    pub fn light() -> Self {
        Self::parse(LIGHT).expect("the bundled light theme is valid")
    }

    /// A bundled theme by name (`dark` or `light`).
    pub fn named(name: &str) -> Option<Self> {
        match name {
            "dark" => Some(Self::dark()),
            "light" => Some(Self::light()),
            _ => None,
        }
    }

    pub fn parse(json: &str) -> Result<Self, ThemeError> {
        let document: Value =
            serde_json::from_str(json).map_err(|error| ThemeError::Json(error.to_string()))?;
        let vars = document.get("vars").and_then(Value::as_object);
        let colors = document
            .get("colors")
            .and_then(Value::as_object)
            .ok_or_else(|| ThemeError::Json("missing `colors`".into()))?;
        let mut resolved = HashMap::new();
        for (token, value) in colors {
            resolved.insert(token.clone(), resolve(value, vars, &mut Vec::new())?);
        }
        Ok(Self { colors: resolved })
    }

    /// The color for `token`; unknown tokens use the terminal default.
    pub fn color(&self, token: &str) -> Color {
        self.colors.get(token).copied().unwrap_or(Color::Reset)
    }

    /// Foreground style for `token`.
    pub fn fg(&self, token: &str) -> Style {
        Style::default().fg(self.color(token))
    }
}

fn resolve(
    value: &Value,
    vars: Option<&serde_json::Map<String, Value>>,
    visiting: &mut Vec<String>,
) -> Result<Color, ThemeError> {
    match value {
        Value::Number(index) => index
            .as_u64()
            .and_then(|index| u8::try_from(index).ok())
            .map(Color::Indexed)
            .ok_or_else(|| ThemeError::InvalidColor(index.to_string())),
        Value::String(text) if text.is_empty() => Ok(Color::Reset),
        Value::String(text) if text.starts_with('#') => parse_hex(text),
        Value::String(text) if text.to_ascii_lowercase().starts_with("okhsl(") => parse_okhsl(text),
        Value::String(name) => {
            if visiting.contains(name) {
                return Err(ThemeError::CircularVariable(name.clone()));
            }
            let target = vars
                .and_then(|vars| vars.get(name))
                .ok_or_else(|| ThemeError::UnknownVariable(name.clone()))?;
            visiting.push(name.clone());
            let color = resolve(target, vars, visiting);
            visiting.pop();
            color
        }
        other => Err(ThemeError::InvalidColor(other.to_string())),
    }
}

fn parse_hex(text: &str) -> Result<Color, ThemeError> {
    let digits = text.trim_start_matches('#');
    let channel = |range: std::ops::Range<usize>| {
        digits
            .get(range)
            .and_then(|pair| u8::from_str_radix(pair, 16).ok())
    };
    match (digits.len(), channel(0..2), channel(2..4), channel(4..6)) {
        (6, Some(r), Some(g), Some(b)) => Ok(Color::Rgb(r, g, b)),
        _ => Err(ThemeError::InvalidColor(text.to_owned())),
    }
}

fn parse_okhsl(text: &str) -> Result<Color, ThemeError> {
    let invalid = || ThemeError::InvalidColor(text.to_owned());
    let inner = text
        .trim()
        .strip_suffix(')')
        .and_then(|rest| rest.split_once('('))
        .map(|(_, inner)| inner)
        .ok_or_else(invalid)?;
    let parts = inner
        .split_whitespace()
        .map(|part| part.trim_end_matches('%').parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid())?;
    let [hue, saturation, lightness] = parts[..] else {
        return Err(invalid());
    };
    let rgb = Srgb::from_color(Okhsl::new(hue, saturation / 100.0, lightness / 100.0));
    let byte = |channel: f32| (channel.clamp(0.0, 1.0) * 255.0).round() as u8;
    Ok(Color::Rgb(byte(rgb.red), byte(rgb.green), byte(rgb.blue)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_themes_resolve_every_token_to_a_color() {
        for theme in [Theme::dark(), Theme::light()] {
            for token in [
                "userMessageBg",
                "toolSuccessBg",
                "mdHeading",
                "toolDiffAdded",
            ] {
                assert!(matches!(theme.color(token), Color::Rgb(..)), "{token}");
            }
        }
        assert!(Theme::named("dark").is_some() && Theme::named("nope").is_none());
    }

    #[test]
    fn okhsl_converts_known_values_and_clamps_out_of_gamut_results() {
        // Zero saturation is a neutral gray with equal channels.
        let Color::Rgb(r, g, b) = parse_okhsl("okhsl(0 0% 50%)").unwrap() else {
            panic!("expected rgb")
        };
        assert!(r == g && g == b, "{r} {g} {b}");
        assert_eq!(parse_okhsl("okhsl(0 0% 0%)").unwrap(), Color::Rgb(0, 0, 0));
        assert_eq!(
            parse_okhsl("okhsl(0 0% 100%)").unwrap(),
            Color::Rgb(255, 255, 255)
        );
    }

    #[test]
    fn variable_references_resolve_and_report_pi_errors() {
        let theme =
            Theme::parse(r##"{"vars":{"a":"#112233","b":"a"},"colors":{"x":"b","y":"","z":7}}"##)
                .unwrap();
        assert_eq!(theme.color("x"), Color::Rgb(0x11, 0x22, 0x33));
        assert_eq!(theme.color("y"), Color::Reset);
        assert_eq!(theme.color("z"), Color::Indexed(7));
        assert_eq!(
            Theme::parse(r#"{"colors":{"x":"missing"}}"#).unwrap_err(),
            ThemeError::UnknownVariable("missing".into())
        );
        assert_eq!(
            Theme::parse(r#"{"vars":{"a":"b","b":"a"},"colors":{"x":"a"}}"#).unwrap_err(),
            ThemeError::CircularVariable("a".into())
        );
    }
}
