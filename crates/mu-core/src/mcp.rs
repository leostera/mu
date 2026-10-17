use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};

use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
    time::timeout,
};

const PROTOCOL_VERSION: &str = "2025-11-25";
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] =
    &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone)]
struct ServerConfig {
    name: String,
    command: String,
    args: Vec<String>,
    env: BTreeMap<String, String>,
    cwd: Option<PathBuf>,
}

struct RemoteTool {
    name: String,
    description: String,
    input_schema: Value,
    exposed_name: String,
}

struct ConnectedServer {
    process: Mutex<McpProcess>,
}

struct ExposedTool {
    server: Arc<ConnectedServer>,
    remote_name: String,
    definition: Value,
}

#[derive(Default)]
pub(crate) struct McpRegistry {
    tools: BTreeMap<String, ExposedTool>,
}

struct McpProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

pub(crate) struct McpToolResult {
    pub content: String,
    pub is_error: bool,
}

impl McpRegistry {
    pub(crate) async fn connect(agent_dir: &Path, cwd: &Path) -> Self {
        let path = agent_dir.join("mcp.json");
        let configs = match load_configs(&path, cwd) {
            Ok(configs) => configs,
            Err(error) => {
                eprintln!("mu: MCP configuration: {error}");
                return Self::default();
            }
        };

        let mut registry = Self::default();
        let mut used_names = HashSet::new();
        for config in configs {
            let name = config.name.clone();
            match connect_server(config, cwd, &mut used_names).await {
                Ok((server, tools)) => {
                    for tool in tools {
                        registry.tools.insert(
                            tool.exposed_name.clone(),
                            ExposedTool {
                                server: Arc::clone(&server),
                                remote_name: tool.name,
                                definition: json!({
                                    "type": "function",
                                    "name": tool.exposed_name,
                                    "description": tool.description,
                                    "parameters": tool.input_schema,
                                }),
                            },
                        );
                    }
                }
                Err(error) => eprintln!("mu: MCP server `{name}`: {error}"),
            }
        }
        registry
    }

    pub(crate) fn definitions(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|tool| tool.definition.clone())
            .collect()
    }

    pub(crate) async fn call(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Option<Result<McpToolResult, String>> {
        let tool = self.tools.get(name)?;
        Some(
            tool.server
                .process
                .lock()
                .await
                .call_tool(&tool.remote_name, arguments)
                .await,
        )
    }
}

impl Drop for McpProcess {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

impl McpProcess {
    async fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        let request = json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params});
        let mut encoded = serde_json::to_vec(&request).map_err(|error| error.to_string())?;
        encoded.push(b'\n');
        timeout(REQUEST_TIMEOUT, self.stdin.write_all(&encoded))
            .await
            .map_err(|_| format!("{method} request timed out while writing"))?
            .map_err(|error| format!("{method} request write failed: {error}"))?;

        loop {
            let mut line = String::new();
            let read = timeout(REQUEST_TIMEOUT, self.stdout.read_line(&mut line))
                .await
                .map_err(|_| format!("{method} request timed out waiting for a response"))?
                .map_err(|error| format!("{method} response read failed: {error}"))?;
            if read == 0 {
                return Err(format!("MCP server closed stdout while handling {method}"));
            }
            let message: Value = serde_json::from_str(&line)
                .map_err(|error| format!("MCP server sent invalid JSON: {error}"))?;
            if message.get("id").and_then(Value::as_u64) != Some(id) {
                // Servers may emit notifications or requests while a client request is pending.
                continue;
            }
            if let Some(error) = message.get("error") {
                return Err(format!(
                    "MCP {method} failed: {}",
                    error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error")
                ));
            }
            return message
                .get("result")
                .cloned()
                .ok_or_else(|| format!("MCP {method} response has no result"));
        }
    }

    async fn notify(&mut self, method: &str) -> Result<(), String> {
        let notification = json!({"jsonrpc":"2.0", "method":method});
        let mut encoded = serde_json::to_vec(&notification).map_err(|error| error.to_string())?;
        encoded.push(b'\n');
        self.stdin
            .write_all(&encoded)
            .await
            .map_err(|error| format!("MCP {method} notification failed: {error}"))
    }

    async fn call_tool(&mut self, name: &str, arguments: &Value) -> Result<McpToolResult, String> {
        let arguments = arguments
            .as_object()
            .ok_or_else(|| "MCP tool arguments must be a JSON object".to_owned())?;
        let result = self
            .request("tools/call", json!({"name":name, "arguments":arguments}))
            .await?;
        let blocks = result
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut content = blocks
            .iter()
            .filter_map(|block| {
                if block.get("type").and_then(Value::as_str) == Some("text") {
                    block.get("text").and_then(Value::as_str).map(str::to_owned)
                } else {
                    block
                        .get("type")
                        .and_then(Value::as_str)
                        .map(|kind| format!("[MCP {kind} content omitted]"))
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        if content.is_empty() {
            if let Some(structured) = result.get("structuredContent") {
                content = serde_json::to_string_pretty(structured)
                    .unwrap_or_else(|_| structured.to_string());
            }
        }
        Ok(McpToolResult {
            content,
            is_error: result
                .get("isError")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }
}

async fn connect_server(
    config: ServerConfig,
    cwd: &Path,
    used_names: &mut HashSet<String>,
) -> Result<(Arc<ConnectedServer>, Vec<RemoteTool>), String> {
    let mut command = Command::new(&config.command);
    command
        .args(&config.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    command.current_dir(config.cwd.as_deref().unwrap_or(cwd));
    for (name, value) in &config.env {
        command.env(name, expand_environment(value)?);
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("could not start `{}`: {error}", config.command))?;
    let stdin = child.stdin.take().ok_or("MCP server stdin was not piped")?;
    let stdout = child
        .stdout
        .take()
        .ok_or("MCP server stdout was not piped")?;
    let mut process = McpProcess {
        child,
        stdin,
        stdout: BufReader::new(stdout),
        next_id: 1,
    };

    let initialized = process
        .request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name":"mu", "version":env!("CARGO_PKG_VERSION")},
            }),
        )
        .await?;
    if !initialized
        .get("protocolVersion")
        .and_then(Value::as_str)
        .is_some_and(|version| SUPPORTED_PROTOCOL_VERSIONS.contains(&version))
        || initialized
            .get("capabilities")
            .and_then(Value::as_object)
            .is_none()
        || initialized
            .pointer("/serverInfo/name")
            .and_then(Value::as_str)
            .is_none()
    {
        return Err("MCP initialize result is missing required fields".into());
    }
    process.notify("notifications/initialized").await?;

    let mut tools = Vec::new();
    let mut cursor = None;
    let mut seen_cursors = HashSet::new();
    for _ in 0..100 {
        let params = cursor
            .as_ref()
            .map(|cursor| json!({"cursor":cursor}))
            .unwrap_or_else(|| json!({}));
        let page = process.request("tools/list", params).await?;
        let page_tools = page
            .get("tools")
            .and_then(Value::as_array)
            .ok_or("MCP tools/list result has no tools array")?;
        for tool in page_tools {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or("MCP tool has no name")?;
            let input_schema = tool
                .get("inputSchema")
                .filter(|schema| schema.is_object())
                .cloned()
                .ok_or_else(|| format!("MCP tool `{name}` has no object inputSchema"))?;
            let exposed_name = unique_tool_name(&config.name, name, used_names);
            tools.push(RemoteTool {
                name: name.to_owned(),
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .or_else(|| tool.get("title").and_then(Value::as_str))
                    .unwrap_or(name)
                    .to_owned(),
                input_schema,
                exposed_name,
            });
        }
        cursor = page
            .get("nextCursor")
            .and_then(Value::as_str)
            .filter(|cursor| !cursor.is_empty())
            .map(str::to_owned);
        if let Some(cursor) = &cursor {
            if !seen_cursors.insert(cursor.clone()) {
                return Err(format!(
                    "MCP tools/list returned duplicate cursor `{cursor}`"
                ));
            }
        }
        if cursor.is_none() {
            let server = Arc::new(ConnectedServer {
                process: Mutex::new(process),
            });
            return Ok((server, tools));
        }
    }
    Err("MCP tools/list exceeded 100 pages".into())
}

fn unique_tool_name(server: &str, tool: &str, used_names: &mut HashSet<String>) -> String {
    let base = format!("mcp__{server}__{tool}")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    let base = base.chars().take(64).collect::<String>();
    if used_names.insert(base.clone()) {
        return base;
    }
    for index in 2.. {
        let suffix = format!("_{index}");
        let shortened = base
            .chars()
            .take(64usize.saturating_sub(suffix.len()))
            .collect::<String>();
        let candidate = format!("{shortened}{suffix}");
        if used_names.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("an unused provider tool name is available")
}

fn load_configs(path: &Path, project_cwd: &Path) -> Result<Vec<ServerConfig>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let source =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let value: Value =
        serde_json::from_str(&source).map_err(|error| format!("{}: {error}", path.display()))?;
    let Some(servers) = value.get("mcpServers") else {
        return Ok(Vec::new());
    };
    let servers = servers
        .as_object()
        .ok_or_else(|| format!("{}: `mcpServers` must be an object", path.display()))?;
    let mut configs = Vec::new();
    for (name, raw) in servers {
        if name.is_empty()
            || !name.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '_' | '-')
            })
        {
            eprintln!("mu: skipping MCP server with invalid name `{name}`");
            continue;
        }
        let Some(object) = raw.as_object() else {
            eprintln!("mu: skipping MCP server `{name}`: config must be an object");
            continue;
        };
        if object.get("enabled").and_then(Value::as_bool) == Some(false) {
            continue;
        }
        if object.get("exposure").and_then(Value::as_str) == Some("hidden") {
            continue;
        }
        let Some(command) = object.get("command").and_then(Value::as_str) else {
            eprintln!(
                "mu: skipping MCP server `{name}`: only stdio `command` servers are supported currently"
            );
            continue;
        };
        let args = match object.get("args") {
            None => Vec::new(),
            Some(Value::Array(values)) => {
                let Some(values) = values.iter().map(Value::as_str).collect::<Option<Vec<_>>>()
                else {
                    eprintln!("mu: skipping MCP server `{name}`: args must be strings");
                    continue;
                };
                values.into_iter().map(str::to_owned).collect()
            }
            _ => {
                eprintln!("mu: skipping MCP server `{name}`: args must be an array");
                continue;
            }
        };
        let env = match object.get("env") {
            None => Some(BTreeMap::new()),
            Some(Value::Object(values)) => values
                .iter()
                .map(|(name, value)| Some((name.clone(), value.as_str()?.to_owned())))
                .collect::<Option<BTreeMap<_, _>>>(),
            _ => None,
        };
        let Some(env) = env else {
            eprintln!("mu: skipping MCP server `{name}`: env must map names to strings");
            continue;
        };
        let server_cwd = object.get("cwd").and_then(Value::as_str).map(|cwd| {
            let path = PathBuf::from(cwd);
            if path.is_absolute() {
                path
            } else {
                project_cwd.join(path)
            }
        });
        configs.push(ServerConfig {
            name: name.clone(),
            command: command.to_owned(),
            args,
            env,
            cwd: server_cwd,
        });
    }
    Ok(configs)
}

fn expand_environment(value: &str) -> Result<String, String> {
    let mut output = String::with_capacity(value.len());
    let mut remaining = value;
    while let Some(start) = remaining.find("${") {
        output.push_str(&remaining[..start]);
        let expression = &remaining[start + 2..];
        let Some(end) = expression.find('}') else {
            return Err(format!("unterminated environment variable in `{value}`"));
        };
        let name = &expression[..end];
        let replacement =
            std::env::var(name).map_err(|_| format!("environment variable `{name}` is not set"))?;
        output.push_str(&replacement);
        remaining = &expression[end + 1..];
    }
    output.push_str(remaining);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_stdio_servers_and_resolves_relative_working_directories() {
        let root = std::env::temp_dir().join(format!("mu-mcp-config-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let config = root.join("mcp.json");
        std::fs::write(
            &config,
            r#"{"mcpServers":{"filesystem":{"command":"node","args":["server.js"],"cwd":"tools","env":{"ROOT":"${HOME}"}},"disabled":{"command":"never","enabled":false}}}"#,
        )
        .unwrap();

        let servers = load_configs(&config, &root).unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "filesystem");
        assert_eq!(servers[0].command, "node");
        assert_eq!(servers[0].args, ["server.js"]);
        assert_eq!(
            servers[0].cwd.as_deref(),
            Some(root.join("tools").as_path())
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stdio_server_initializes_lists_and_executes_tools() {
        let script = r#"
while IFS= read -r line; do
  case "$line" in
    *initialize*)
      echo '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"test-server","version":"1"}}}' ;;
    *tools/list*)
      echo '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"echo","description":"Return text","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}]}}' ;;
    *tools/call*)
      echo '{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"hello from MCP"}]}}' ;;
  esac
done
"#;
        let config = ServerConfig {
            name: "test".into(),
            command: "sh".into(),
            args: vec!["-c".into(), script.into()],
            env: BTreeMap::new(),
            cwd: None,
        };
        let mut names = HashSet::new();
        let (server, tools) = connect_server(config, Path::new("."), &mut names)
            .await
            .expect("connect stdio MCP server");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].exposed_name, "mcp__test__echo");

        let result = server
            .process
            .lock()
            .await
            .call_tool("echo", &json!({"text":"hello"}))
            .await
            .expect("call MCP tool");
        assert_eq!(result.content, "hello from MCP");
        assert!(!result.is_error);
    }

    #[test]
    fn sanitizes_and_uniquifies_provider_tool_names() {
        let mut used = HashSet::new();
        let first = unique_tool_name("server-one", "read file", &mut used);
        let second = unique_tool_name("server_one", "read file", &mut used);
        assert_eq!(first, "mcp__server_one__read_file");
        assert_eq!(second, "mcp__server_one__read_file_2");
        assert!(second.len() <= 64);
    }
}
