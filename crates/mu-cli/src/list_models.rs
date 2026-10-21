//! `--list-models [search]`, following Pi's `cli/list-models.ts`.
//!
//! Only models declared in `models.json` are listed; Pi's built-in provider catalog is not ported.

use serde_json::Value;

use crate::tui::fuzzy::fuzzy_filter;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    provider: String,
    model: String,
    context: String,
    max_out: String,
    thinking: &'static str,
    images: &'static str,
}

fn format_token_count(count: u64) -> String {
    let scaled = |divisor: f64, suffix: &str| {
        let value = count as f64 / divisor;
        if value.fract() == 0.0 {
            format!("{}{suffix}", value as u64)
        } else {
            format!("{value:.1}{suffix}")
        }
    };
    if count >= 1_000_000 {
        scaled(1_000_000.0, "M")
    } else if count >= 1_000 {
        scaled(1_000.0, "K")
    } else {
        count.to_string()
    }
}

fn rows(models_json: &Value) -> Vec<Row> {
    let Some(providers) = models_json.get("providers").and_then(Value::as_object) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for (provider, entry) in providers {
        for model in entry
            .get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(id) = model.get("id").and_then(Value::as_str) else {
                continue;
            };
            let number =
                |key: &str, default: u64| model.get(key).and_then(Value::as_u64).unwrap_or(default);
            let images = model
                .get("input")
                .and_then(Value::as_array)
                .is_some_and(|input| input.iter().any(|kind| kind == "image"));
            rows.push(Row {
                provider: provider.clone(),
                model: id.to_owned(),
                context: format_token_count(number("contextWindow", 128_000)),
                max_out: format_token_count(number("maxTokens", 16_384)),
                thinking: if model
                    .get("reasoning")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    "yes"
                } else {
                    "no"
                },
                images: if images { "yes" } else { "no" },
            });
        }
    }
    rows
}

/// The table printed by `--list-models`, or the message Pi prints when nothing matches.
pub fn render(models_json: &Value, search: Option<&str>) -> String {
    let all = rows(models_json);
    if all.is_empty() {
        return "No models available. Add models to models.json or run /login.".to_owned();
    }
    let mut filtered = match search {
        Some(pattern) => fuzzy_filter(&all, pattern, |row| {
            format!("{} {}", row.provider, row.model)
        }),
        None => all,
    };
    if filtered.is_empty() {
        return format!("No models matching \"{}\"", search.unwrap_or_default());
    }
    filtered.sort_by(|a, b| {
        a.provider
            .to_lowercase()
            .cmp(&b.provider.to_lowercase())
            .then(a.model.to_lowercase().cmp(&b.model.to_lowercase()))
    });
    let header = Row {
        provider: "provider".into(),
        model: "model".into(),
        context: "context".into(),
        max_out: "max-out".into(),
        thinking: "thinking",
        images: "images",
    };
    let cells = |row: &Row| {
        [
            row.provider.clone(),
            row.model.clone(),
            row.context.clone(),
            row.max_out.clone(),
            row.thinking.to_owned(),
            row.images.to_owned(),
        ]
    };
    let table = std::iter::once(cells(&header))
        .chain(filtered.iter().map(cells))
        .collect::<Vec<_>>();
    let widths = (0..6)
        .map(|column| {
            table
                .iter()
                .map(|row| row[column].chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    table
        .iter()
        .map(|row| {
            row.iter()
                .zip(&widths)
                .map(|(cell, width)| format!("{cell:<width$}"))
                .collect::<Vec<_>>()
                .join("  ")
                .trim_end()
                .to_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn sample() -> Value {
        json!({"providers":{
            "zeta":{"models":[{"id":"big","contextWindow":1_000_000,"maxTokens":64_000,"reasoning":true,"input":["text","image"]}]},
            "alpha":{"models":[{"id":"small","contextWindow":32768,"maxTokens":4096},{"id":"mid"}]}
        }})
    }

    #[test]
    fn lists_sorted_columns_with_humanized_token_counts() {
        assert_eq!(
            render(&sample(), None),
            "provider  model  context  max-out  thinking  images\n\
             alpha     mid    128K     16.4K    no        no\n\
             alpha     small  32.8K    4.1K     no        no\n\
             zeta      big    1M       64K      yes       yes"
        );
    }

    #[test]
    fn search_filters_fuzzily_and_reports_empty_results() {
        assert!(render(&sample(), Some("zeta big")).contains("zeta"));
        assert!(!render(&sample(), Some("zeta big")).contains("alpha"));
        assert_eq!(
            render(&sample(), Some("nomatch")),
            "No models matching \"nomatch\""
        );
        assert!(render(&json!({}), None).starts_with("No models available"));
    }
}
