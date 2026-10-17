use serde_json::{Value, json};

const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "with",
];

pub(crate) fn definition() -> Value {
    json!({
        "type":"function",
        "name":"tool_search",
        "description":"Search for additional MCP tools by name, description, and input fields. Matching tools will be available directly on the next turn.",
        "parameters":{
            "type":"object",
            "properties":{
                "query":{"type":"string","description":"What capability or data to find."},
                "limit":{"type":"integer","minimum":1,"maximum":30,"description":"Maximum number of tools. Defaults to 8."}
            },
            "required":["query"],
            "additionalProperties":false
        }
    })
}

pub(crate) fn rank(query: &str, tools: &[Value], limit: usize) -> Vec<Value> {
    let query_terms = tokenize(query);
    if query_terms.is_empty() || tools.is_empty() || limit == 0 {
        return Vec::new();
    }
    let documents = tools.iter().map(tool_text).collect::<Vec<_>>();
    let term_counts = documents
        .iter()
        .map(|text| term_counts(text))
        .collect::<Vec<_>>();
    let lengths = term_counts
        .iter()
        .map(|counts| counts.values().sum::<usize>())
        .collect::<Vec<_>>();
    let average_length = (lengths.iter().sum::<usize>() as f64 / lengths.len() as f64).max(1.0);
    let k1 = 1.2;
    let b = 0.75;
    let mut matches = tools
        .iter()
        .enumerate()
        .filter_map(|(index, tool)| {
            let mut score = 0.0;
            for term in &query_terms {
                let frequency = term_counts[index].get(term).copied().unwrap_or_default();
                if frequency == 0 {
                    continue;
                }
                let document_frequency = term_counts
                    .iter()
                    .filter(|counts| counts.contains_key(term))
                    .count() as f64;
                let idf = (1.0
                    + (tools.len() as f64 - document_frequency + 0.5) / (document_frequency + 0.5))
                    .ln();
                let normalized = k1 * (1.0 - b + b * lengths[index] as f64 / average_length);
                score += idf * (frequency as f64 * (k1 + 1.0)) / (frequency as f64 + normalized);
            }
            (score > 0.0).then_some((index, score, tool.clone()))
        })
        .collect::<Vec<_>>();
    matches.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    matches
        .into_iter()
        .take(limit)
        .map(|(_, _, tool)| tool)
        .collect()
}

fn tool_text(tool: &Value) -> String {
    let mut parts = vec![
        tool.get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        tool.get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .replace('_', " "),
        tool.get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
    ];
    schema_text(tool.get("parameters"), &mut parts);
    parts.join(" ")
}

fn schema_text(schema: Option<&Value>, parts: &mut Vec<String>) {
    let Some(schema) = schema.and_then(Value::as_object) else {
        return;
    };
    if let Some(description) = schema.get("description").and_then(Value::as_str) {
        parts.push(description.to_owned());
    }
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (name, property) in properties {
            parts.push(name.clone());
            schema_text(Some(property), parts);
        }
    }
    schema_text(schema.get("items"), parts);
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(variants) = schema.get(key).and_then(Value::as_array) {
            for variant in variants {
                schema_text(Some(variant), parts);
            }
        }
    }
}

fn term_counts(text: &str) -> std::collections::HashMap<String, usize> {
    let mut counts = std::collections::HashMap::new();
    for term in tokenize(text) {
        *counts.entry(term).or_default() += 1;
    }
    counts
}

fn tokenize(text: &str) -> Vec<String> {
    let mut split_camel_case = String::with_capacity(text.len());
    let mut previous = None;
    for character in text.chars() {
        if previous.is_some_and(|previous: char| {
            previous.is_ascii_lowercase() || previous.is_ascii_digit()
        }) && character.is_ascii_uppercase()
        {
            split_camel_case.push(' ');
        }
        split_camel_case.push(character.to_ascii_lowercase());
        previous = Some(character);
    }
    split_camel_case
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|term| !term.is_empty() && !STOP_WORDS.contains(term))
        .map(stem)
        .collect()
}

fn stem(term: &str) -> String {
    if term.len() > 4 && term.ends_with("ies") {
        return format!("{}y", &term[..term.len() - 3]);
    }
    if term.len() > 4
        && ["ches", "shes", "sses", "xes", "zes"]
            .iter()
            .any(|suffix| term.ends_with(suffix))
    {
        return term[..term.len() - 2].to_owned();
    }
    if term.len() > 3 && term.ends_with('s') && !term.ends_with("ss") {
        return term[..term.len() - 1].to_owned();
    }
    term.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_names_descriptions_and_schema_fields_with_stable_ties() {
        let tools = vec![
            json!({"name":"mcp__drive__list_files","description":"List files in a drive","parameters":{"properties":{"folderId":{"description":"folder identifier"}}}}),
            json!({"name":"mcp__issues__search","description":"Search issue tracker","parameters":{"properties":{"query":{"description":"issue query"}}}}),
        ];
        let found = rank("find issues by query", &tools, 8);
        assert_eq!(found[0]["name"], "mcp__issues__search");
        assert_eq!(rank("", &tools, 8).len(), 0);
        assert_eq!(
            rank("folder", &tools, 1)[0]["name"],
            "mcp__drive__list_files"
        );
    }
}
