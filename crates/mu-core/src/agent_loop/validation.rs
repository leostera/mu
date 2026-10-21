//! Tool-argument validation (`packages/ai/src/utils/validation.ts`).
//!
//! Arguments are normalized (optional nulls dropped), coerced to the schema's primitive types
//! the way Pi's validator does, then checked with the `jsonschema` crate.

use serde_json::{Map, Value};

fn schema_types(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn matches_json_type(value: &Value, kind: &str) -> bool {
    match kind {
        "number" => value.is_number(),
        "integer" => {
            value.is_i64()
                || value.is_u64()
                || value
                    .as_f64()
                    .is_some_and(|number| number.is_finite() && number.fract() == 0.0)
        }
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn is_valid(schema: &Value, value: &Value) -> bool {
    jsonschema::validator_for(schema).is_ok_and(|validator| validator.is_valid(value))
}

fn number_value(number: f64) -> Value {
    if number.fract() == 0.0 && number.abs() < 9.0e15 {
        Value::from(number as i64)
    } else {
        serde_json::Number::from_f64(number).map_or(Value::Null, Value::Number)
    }
}

/// JavaScript `Number(string)` for the finite cases Pi relies on.
fn parse_js_number(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
}

fn coerce_primitive(value: &Value, kind: &str) -> Value {
    match kind {
        "number" => match value {
            Value::Null => Value::from(0),
            Value::String(text) => {
                parse_js_number(text).map_or_else(|| value.clone(), number_value)
            }
            Value::Bool(flag) => Value::from(u8::from(*flag)),
            _ => value.clone(),
        },
        "integer" => match value {
            Value::Null => Value::from(0),
            Value::String(text) => parse_js_number(text)
                .filter(|number| number.fract() == 0.0)
                .map_or_else(|| value.clone(), number_value),
            Value::Bool(flag) => Value::from(u8::from(*flag)),
            _ => value.clone(),
        },
        "boolean" => match value {
            Value::Null => Value::Bool(false),
            Value::String(text) if text == "true" => Value::Bool(true),
            Value::String(text) if text == "false" => Value::Bool(false),
            Value::Number(number) if number.as_f64() == Some(1.0) => Value::Bool(true),
            Value::Number(number) if number.as_f64() == Some(0.0) => Value::Bool(false),
            _ => value.clone(),
        },
        "string" => match value {
            Value::Null => Value::String(String::new()),
            Value::Number(number) => Value::String(number.to_string()),
            Value::Bool(flag) => Value::String(flag.to_string()),
            _ => value.clone(),
        },
        "null" => match value {
            Value::String(text) if text.is_empty() => Value::Null,
            Value::Number(number) if number.as_f64() == Some(0.0) => Value::Null,
            Value::Bool(false) => Value::Null,
            _ => value.clone(),
        },
        _ => value.clone(),
    }
}

fn coerce_union(value: Value, schemas: &[Value]) -> Value {
    if schemas.iter().any(|schema| is_valid(schema, &value)) {
        return value;
    }
    for schema in schemas {
        let coerced = coerce_with_schema(value.clone(), schema);
        if is_valid(schema, &coerced) {
            return coerced;
        }
    }
    value
}

fn coerce_with_schema(mut value: Value, schema: &Value) -> Value {
    if let Some(nested) = schema.get("allOf").and_then(Value::as_array) {
        for schema in nested {
            value = coerce_with_schema(value, schema);
        }
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = schema.get(key).and_then(Value::as_array) {
            value = coerce_union(value, variants);
        }
    }

    let types = schema_types(schema);
    let matches_union_member =
        types.len() > 1 && types.iter().any(|kind| matches_json_type(&value, kind));
    if !types.is_empty() && !matches_union_member {
        for kind in &types {
            let candidate = coerce_primitive(&value, kind);
            if candidate != value {
                value = candidate;
                break;
            }
        }
    }

    if types.contains(&"object") {
        if let Value::Object(object) = &mut value {
            coerce_object(object, schema);
        }
    }
    if types.contains(&"array") {
        if let Value::Array(items) = &mut value {
            coerce_array(items, schema);
        }
    }
    value
}

fn coerce_object(object: &mut Map<String, Value>, schema: &Value) {
    let properties = schema.get("properties").and_then(Value::as_object);
    if let Some(properties) = properties {
        for (key, property_schema) in properties {
            if let Some(entry) = object.get_mut(key) {
                *entry = coerce_with_schema(entry.take(), property_schema);
            }
        }
    }
    if let Some(additional) = schema.get("additionalProperties").filter(|v| v.is_object()) {
        for (key, entry) in object.iter_mut() {
            if properties.is_some_and(|properties| properties.contains_key(key)) {
                continue;
            }
            *entry = coerce_with_schema(entry.take(), additional);
        }
    }
}

fn coerce_array(items: &mut [Value], schema: &Value) {
    match schema.get("items") {
        Some(Value::Array(item_schemas)) => {
            for (item, item_schema) in items.iter_mut().zip(item_schemas) {
                *item = coerce_with_schema(item.take(), item_schema);
            }
        }
        Some(item_schema @ Value::Object(_)) => {
            for item in items {
                *item = coerce_with_schema(item.take(), item_schema);
            }
        }
        _ => {}
    }
}

fn normalize_optional_nulls(value: &mut Value, schema: &Value) {
    match value {
        Value::Array(items) => match schema.get("items") {
            Some(Value::Array(item_schemas)) => {
                for (item, item_schema) in items.iter_mut().zip(item_schemas) {
                    normalize_optional_nulls(item, item_schema);
                }
            }
            Some(item_schema) => {
                for item in items {
                    normalize_optional_nulls(item, item_schema);
                }
            }
            None => {}
        },
        Value::Object(object) => {
            let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
                return;
            };
            let required = schema
                .get("required")
                .and_then(Value::as_array)
                .map(|keys| keys.iter().filter_map(Value::as_str).collect::<Vec<_>>())
                .unwrap_or_default();
            for (key, property_schema) in properties {
                let Some(entry) = object.get_mut(key) else {
                    continue;
                };
                if entry.is_null()
                    && !required.contains(&key.as_str())
                    && property_schema.get("$ref").is_none_or(|r| !r.is_string())
                    && !is_valid(property_schema, &Value::Null)
                {
                    object.remove(key);
                } else {
                    normalize_optional_nulls(entry, property_schema);
                }
            }
        }
        _ => {}
    }
}

fn format_path(path: &str) -> String {
    let path = path.trim_start_matches('/').replace('/', ".");
    if path.is_empty() { "root".into() } else { path }
}

/// Validate (and possibly coerce) `arguments` against `parameters`.
///
/// The error text matches Pi's `Validation failed for tool "<name>":` format.
pub fn validate_tool_arguments(
    tool_name: &str,
    parameters: &Value,
    arguments: &Value,
) -> Result<Value, String> {
    let mut args = arguments.clone();
    normalize_optional_nulls(&mut args, parameters);
    let args = coerce_with_schema(args, parameters);

    let validator = jsonschema::validator_for(parameters)
        .map_err(|error| format!("Invalid parameter schema for tool \"{tool_name}\": {error}"))?;
    let errors = validator
        .iter_errors(&args)
        .map(|error| {
            let location = error.instance_path().to_string();
            let path = match error.kind() {
                jsonschema::error::ValidationErrorKind::Required { property } => {
                    let property = property.as_str().unwrap_or_default();
                    match format_path(&location).as_str() {
                        "root" => property.to_owned(),
                        base => format!("{base}.{property}"),
                    }
                }
                _ => format_path(&location),
            };
            format!("  - {path}: {error}")
        })
        .collect::<Vec<_>>();
    if errors.is_empty() {
        return Ok(args);
    }
    Err(format!(
        "Validation failed for tool \"{tool_name}\":\n{}\n\nReceived arguments:\n{}",
        errors.join("\n"),
        serde_json::to_string_pretty(arguments).unwrap_or_default()
    ))
}
