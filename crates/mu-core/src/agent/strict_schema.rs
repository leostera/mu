//! Strict function-calling schemas: which models get them and how a JSON schema is tightened.

use std::collections::HashSet;

use serde_json::{Value, json};

pub(super) fn model_supports_strict_sampling(model: &Value) -> bool {
    model
        .pointer("/compat/supportsStrictMode")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub(super) fn mark_strict_builtin_tool(mut definition: Value) -> Value {
    let name = definition
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !matches!(name, "read" | "bash" | "edit" | "write") {
        return definition;
    }
    // Pi's constrained sampling is "prefer": an unsupported schema keeps the tool non-strict.
    if let Some(strict) = definition
        .get("parameters")
        .and_then(|parameters| strict_json_schema(parameters).ok())
    {
        definition["parameters"] = strict;
        definition["strict"] = Value::Bool(true);
    }
    definition
}

/// Keywords provider strict modes reject (Pi's `UNSUPPORTED_STRICT_SCHEMA_KEYS`).
pub(super) const UNSUPPORTED_STRICT_SCHEMA_KEYS: [&str; 16] = [
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
];

pub(super) fn is_structured_schema(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };
    let types: Vec<&str> = match object.get("type") {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    types.contains(&"object")
        || types.contains(&"array")
        || object.contains_key("properties")
        || object.contains_key("items")
}

/// Convert a tool schema to the strict subset (Pi's `makeStrictJsonSchema`): every property
/// required, optional ones made nullable, `additionalProperties: false`; unsupported shapes
/// are errors.
pub(super) fn strict_json_schema(schema: &Value) -> Result<Value, String> {
    let mut clone = schema.clone();
    if !clone.is_object() {
        return Err("root schema must have type object".into());
    }
    strictify_node(&mut clone)?;
    if clone.get("type").and_then(Value::as_str) != Some("object") {
        return Err("root schema must have type object".into());
    }
    Ok(clone)
}

pub(super) fn strictify_node(schema: &mut Value) -> Result<(), String> {
    let Some(object) = schema.as_object_mut() else {
        return Err("boolean schemas are unsupported".into());
    };
    for key in UNSUPPORTED_STRICT_SCHEMA_KEYS {
        if object.contains_key(key) {
            return Err(format!("{key} schemas are unsupported"));
        }
    }
    if let Some(any_of) = object.get_mut("anyOf") {
        let variants = any_of
            .as_array_mut()
            .filter(|variants| !variants.is_empty())
            .ok_or("anyOf must contain at least one schema")?;
        for variant in variants {
            if is_structured_schema(variant) {
                return Err("object and array unions are unsupported".into());
            }
            strictify_node(variant)?;
        }
    }
    if let Some(items) = object.get_mut("items") {
        if items.is_array() {
            return Err("tuple schemas are unsupported".into());
        }
        strictify_node(items)?;
    }
    let is_object = object.get("type").and_then(Value::as_str) == Some("object");
    if object.contains_key("properties") && !is_object {
        return Err("properties require type object".into());
    }
    if !is_object {
        return Ok(());
    }
    if object
        .get("additionalProperties")
        .is_some_and(|value| *value != Value::Bool(false))
    {
        return Err("schema-valued or true additionalProperties is unsupported".into());
    }
    let required = match object.get("required") {
        None => HashSet::new(),
        Some(Value::Array(keys)) => keys
            .iter()
            .map(|key| key.as_str().map(str::to_owned))
            .collect::<Option<HashSet<_>>>()
            .ok_or("object required must be a string array")?,
        Some(_) => return Err("object required must be a string array".into()),
    };
    let mut properties = match object.remove("properties") {
        None => serde_json::Map::new(),
        Some(Value::Object(properties)) => properties,
        Some(_) => return Err("object properties must be a schema map".into()),
    };
    if required.iter().any(|key| !properties.contains_key(key)) {
        return Err("required contains an unknown property".into());
    }
    let names = properties.keys().cloned().collect::<Vec<_>>();
    for (name, property) in &mut properties {
        strictify_node(property)?;
        if !required.contains(name) && !schema_allows_null(property) {
            *property = json!({"anyOf":[property.clone(),{"type":"null"}]});
        }
    }
    let object = schema.as_object_mut().expect("checked above");
    object.insert("properties".into(), Value::Object(properties));
    object.insert("required".into(), json!(names));
    object.insert("additionalProperties".into(), Value::Bool(false));
    Ok(())
}

pub(super) fn schema_allows_null(schema: &Value) -> bool {
    schema.get("type").and_then(Value::as_str) == Some("null")
        || schema
            .get("type")
            .and_then(Value::as_array)
            .is_some_and(|types| types.iter().any(|kind| kind.as_str() == Some("null")))
        || schema.get("const") == Some(&Value::Null)
        || schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| values.contains(&Value::Null))
        || schema
            .get("anyOf")
            .and_then(Value::as_array)
            .is_some_and(|variants| variants.iter().any(schema_allows_null))
}
