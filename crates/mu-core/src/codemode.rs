use std::{collections::HashMap, path::Path};

use codemode::{CodeMode, CodeModeConfig, CodeModeError, NativeFunctionRegistry, RunCode};
use serde_json::{Value, json};

use crate::{mcp::McpRegistry, tools};

const TOOL_NAME: &str = "codemode";
const CODEMODE_DESCRIPTION: &str = "Run JavaScript that calls other tools. Provide raw JavaScript as the code value; it runs as an async function body. Use `await tools.<name>({ ...args })` to call tools. `text(value)`, `console.log(...)`, and `return value` produce output. `searchTools(query)`, `describeTool(name)`, and `ALL_TOOLS` help discover tools.";

pub(crate) fn definition(mcp: &McpRegistry) -> Value {
    let callable_tools = callable_tool_definitions(mcp);
    let declarations = callable_tools
        .iter()
        .filter_map(|tool| {
            let name = tool.get("name")?.as_str()?;
            let description = tool
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Some(format!(
                "- `tools.{name}(args)`: {}",
                description.lines().next().unwrap_or_default()
            ))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let description = if declarations.is_empty() {
        CODEMODE_DESCRIPTION.to_owned()
    } else {
        format!("{CODEMODE_DESCRIPTION}\n\nCallable tools:\n{declarations}")
    };
    json!({
        "type": "function",
        "name": TOOL_NAME,
        "description": description,
        "parameters": {
            "type":"object",
            "properties":{"code":{"type":"string","description":"Raw JavaScript source to run as an async function body."}},
            "required":["code"],
            "additionalProperties":false
        }
    })
}

pub(crate) async fn execute(
    cwd: &Path,
    mcp: &McpRegistry,
    stored_values: &Value,
    arguments: &Value,
    shell_environment: &HashMap<String, String>,
) -> Result<(String, bool, Option<Value>), String> {
    let code = arguments
        .get("code")
        .and_then(Value::as_str)
        .ok_or_else(|| "`code` must be a string".to_owned())?;
    let tool_definitions = callable_tool_definitions(mcp);
    let metadata = tool_definitions
        .iter()
        .filter_map(|definition| {
            Some(json!({
                "name": definition.get("name")?.as_str()?,
                "description": definition.get("description").and_then(Value::as_str).unwrap_or_default(),
                "parameters": definition.get("parameters").cloned().unwrap_or_else(|| json!({})),
            }))
        })
        .collect::<Vec<_>>();
    let source = script_source(code, &metadata, stored_values)?;
    let mcp = mcp.clone();
    let cwd = cwd.to_owned();
    let shell_environment = shell_environment.clone();
    let searchable_tools = tool_definitions.clone();
    let native_functions = NativeFunctionRegistry::default()
        .add_function("call_tool", move |args: Value| {
            let mcp = mcp.clone();
            let cwd = cwd.clone();
            let shell_environment = shell_environment.clone();
            async move {
                let call = args.as_array().ok_or_else(|| CodeModeError::ExecuteCode {
                    reason: "native call_tool arguments must be an array".into(),
                })?;
                let name = call.first().and_then(Value::as_str).ok_or_else(|| {
                    CodeModeError::ExecuteCode {
                        reason: "native call_tool requires a tool name".into(),
                    }
                })?;
                let arguments = call.get(1).cloned().unwrap_or_else(|| json!({}));
                if let Some(result) = mcp.call(name, &arguments).await {
                    let result =
                        result.map_err(|error| CodeModeError::ExecuteCode { reason: error })?;
                    return Ok(result
                        .structured_content
                        .unwrap_or_else(|| Value::String(result.content)));
                }
                let result =
                    tools::execute_with_environment(&cwd, name, &arguments, &shell_environment)
                        .await
                        .map_err(|error| CodeModeError::ExecuteCode {
                            reason: error.to_string(),
                        })?;
                Ok(result
                    .structured_content
                    .unwrap_or_else(|| Value::String(result.content)))
            }
        })
        .add_function("search_tools", move |args: Value| {
            let tools = searchable_tools.clone();
            async move {
                let arguments = args.as_array().ok_or_else(|| CodeModeError::ExecuteCode {
                    reason: "native search_tools arguments must be an array".into(),
                })?;
                let query = arguments.first().and_then(Value::as_str).ok_or_else(|| {
                    CodeModeError::ExecuteCode {
                        reason: "searchTools() requires a string query".into(),
                    }
                })?;
                let options = arguments.get(1).and_then(Value::as_object);
                let namespace = options
                    .and_then(|options| options.get("namespace"))
                    .and_then(Value::as_str);
                let limit = options
                    .and_then(|options| options.get("limit"))
                    .and_then(Value::as_u64)
                    .filter(|limit| *limit > 0)
                    .unwrap_or(8)
                    .min(30) as usize;
                let candidates = tools
                    .into_iter()
                    .filter(|tool| {
                        namespace.is_none_or(|namespace| {
                            tool.get("name")
                                .and_then(Value::as_str)
                                .is_some_and(|name| {
                                    name.starts_with(&format!(
                                        "mcp__{}__",
                                        namespace.replace('-', "_")
                                    ))
                                })
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(Value::Array(crate::tool_search::rank(
                    query,
                    &candidates,
                    limit,
                )))
            }
        });
    let runtime = CodeMode::builder()
        .with_config(CodeModeConfig::default().multithreaded(true))
        .with_native_functions(native_functions)
        .build()
        .map_err(|error| error.to_string())?;
    let started = std::time::Instant::now();
    let result = runtime
        .run_code(RunCode {
            code: source,
            imports: Vec::new(),
        })
        .await;
    let elapsed = started.elapsed().as_secs_f64();
    match result {
        Ok(result) => {
            let ok = result
                .value
                .get("ok")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let mut output = result
                .value
                .get("output")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>();
            if let Some(value) = result.value.get("value").filter(|value| !value.is_null()) {
                output.push(value_text(value));
            }
            let output_text = if output.is_empty() {
                String::new()
            } else {
                output.join("\n")
            };
            let mut response = format!(
                "Script {}\nWall time {elapsed:.1} seconds\nOutput:\n",
                if ok { "completed" } else { "failed" }
            );
            if !output_text.is_empty() {
                response.push_str(&output_text);
            }
            if !ok {
                if let Some(error) = result.value.get("error").and_then(Value::as_str) {
                    response.push_str("\nScript error:\n");
                    response.push_str(error);
                }
            }
            let store_writes = result
                .value
                .get("storeWrites")
                .filter(|_| ok)
                .filter(|writes| {
                    writes
                        .get("set")
                        .and_then(Value::as_object)
                        .is_some_and(|set| !set.is_empty())
                })
                .cloned();
            Ok((response, !ok, store_writes))
        }
        Err(error) => Ok((
            format!(
                "Script failed\nWall time {elapsed:.1} seconds\nOutput:\nScript error:\n{error}"
            ),
            true,
            None,
        )),
    }
}

fn callable_tool_definitions(mcp: &McpRegistry) -> Vec<Value> {
    let mut definitions = tools::definitions();
    definitions.extend(mcp.codemode_definitions());
    definitions
}

fn script_source(code: &str, metadata: &[Value], stored_values: &Value) -> Result<String, String> {
    let metadata = serde_json::to_string(metadata).map_err(|error| error.to_string())?;
    let stored_values = if stored_values.is_object() {
        stored_values.to_string()
    } else {
        "{}".into()
    };
    let stored_values = serde_json::to_string(&stored_values).map_err(|error| error.to_string())?;
    Ok(format!(
        r#"async () => {{
  const __muTools = {metadata};
  const __muOutput = [];
  const __muStore = JSON.parse({stored_values});
  const __muStoreWrites = Object.create(null);
  const __muExit = Symbol("mu codemode exit");
  globalThis.tools = Object.create(null);
  for (const tool of __muTools) {{
    globalThis.tools[tool.name] = (args = {{}}) => native.call_tool(tool.name, args);
  }}
  globalThis.ALL_TOOLS = __muTools;
  // The crate enables these host capabilities by default; Pi codemode exposes neither.
  globalThis.fetch = undefined;
  globalThis.env = undefined;
  globalThis.Deno = undefined;
  globalThis.text = (value) => __muOutput.push(typeof value === "string" ? value : JSON.stringify(value));
  globalThis.image = (_value) => __muOutput.push("[image output is not supported by this Mu renderer]");
  globalThis.exit = () => {{ throw __muExit; }};
  globalThis.store = (key, value) => {{
    const cloned = JSON.parse(JSON.stringify(value));
    __muStore[String(key)] = cloned;
    __muStoreWrites[String(key)] = cloned;
  }};
  globalThis.load = (key) => __muStore[String(key)];
  globalThis.searchTools = async (query, options = {{}}) => native.search_tools(query, options);
  globalThis.describeTool = (name) => __muTools.find((tool) => tool.name === name);
  globalThis.describeNamespace = (name) => {{
    const prefix = "mcp__" + String(name).replaceAll("-", "_") + "__";
    const tools = __muTools.filter((tool) => tool.name.startsWith(prefix));
    return tools.length ? {{ name, tools: tools.map((tool) => tool.name) }} : undefined;
  }};
  globalThis.console = {{
    log: (...values) => __muOutput.push(values.map((value) => typeof value === "string" ? value : JSON.stringify(value)).join(" ")),
    info: (...values) => __muOutput.push(values.map((value) => typeof value === "string" ? value : JSON.stringify(value)).join(" ")),
    warn: (...values) => __muOutput.push(values.map((value) => typeof value === "string" ? value : JSON.stringify(value)).join(" ")),
    error: (...values) => __muOutput.push(values.map((value) => typeof value === "string" ? value : JSON.stringify(value)).join(" ")),
  }};
  try {{
    let value;
    try {{ value = await (async () => {{
{code}
}})(); }}
    catch (error) {{ if (error !== __muExit) throw error; }}
    return {{ ok: true, value: value === undefined ? null : value, output: __muOutput, storeWrites: {{ set: __muStoreWrites, delete: [] }} }};
  }} catch (error) {{
    return {{ ok: false, value: null, output: __muOutput, error: String(error && error.stack || error) }};
  }}
}}"#
    ))
}

fn value_text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[tokio::test]
    async fn script_can_call_mu_tools_and_return_values() {
        let cwd = std::env::temp_dir().join(format!("mu-codemode-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&cwd).unwrap();
        fs::write(cwd.join("sample.txt"), "the file content").unwrap();

        let (output, is_error, store_writes) = execute(
            &cwd,
            &McpRegistry::default(),
            &json!({"count":1}),
            &json!({"code":"const result = await tools.read({ path: 'sample.txt' }); text(result); const matches = await searchTools('read file'); text(matches[0].name); store('count', load('count') + 1); return { done: true, capabilities: [typeof fetch, typeof env, typeof Deno] };"}),
            &HashMap::new(),
        )
        .await
        .unwrap();
        assert!(!is_error, "{output}");
        assert!(output.contains("Script completed"));
        assert!(output.contains("the file content"));
        assert!(output.contains("\nread"));
        assert!(output.contains("\"done\":true"));
        assert!(output.contains("\"capabilities\":[\"undefined\",\"undefined\",\"undefined\"]"));
        assert_eq!(store_writes.unwrap()["set"]["count"], 2);
        fs::remove_dir_all(cwd).unwrap();
    }

    #[tokio::test]
    async fn script_passes_pi_session_environment_to_nested_bash() {
        let cwd = std::env::temp_dir().join(format!("mu-codemode-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&cwd).unwrap();
        let environment = HashMap::from([("PI_SESSION_ID".into(), "nested-session".into())]);

        let (output, is_error, _) = execute(
            &cwd,
            &McpRegistry::default(),
            &json!({}),
            &json!({"code":"return await tools.bash({ command: 'printf $PI_SESSION_ID' });"}),
            &environment,
        )
        .await
        .unwrap();

        assert!(!is_error, "{output}");
        assert!(output.contains("nested-session"), "{output}");
        fs::remove_dir_all(cwd).unwrap();
    }

    #[tokio::test]
    async fn script_reports_tool_failures_as_script_errors() {
        let cwd = std::env::temp_dir().join(format!("mu-codemode-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&cwd).unwrap();

        let (output, is_error, store_writes) = execute(
            &cwd,
            &McpRegistry::default(),
            &json!({}),
            &json!({"code":"store('lost', true); await tools.read({ path: 'missing.txt' });"}),
            &HashMap::new(),
        )
        .await
        .unwrap();
        assert!(is_error, "{output}");
        assert!(output.contains("Script failed"));
        assert!(output.contains("Script error:"));
        assert!(store_writes.is_none());
        fs::remove_dir_all(cwd).unwrap();
    }
}
