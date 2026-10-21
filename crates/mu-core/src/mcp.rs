use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};

use futures_util::{StreamExt, future::join_all};
use reqwest::{
    Client, Url,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde_json::{Map, Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
    time::timeout,
};

const PROTOCOL_VERSION: &str = "2025-11-25";
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] =
    &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Exposure {
    Codemode,
    Deferred,
    Direct,
    Hidden,
}

impl Exposure {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "codemode" | "codemode-deferred" => Some(Self::Codemode),
            "deferred" => Some(Self::Deferred),
            "direct" => Some(Self::Direct),
            "hidden" => Some(Self::Hidden),
            _ => None,
        }
    }
}

struct ServerConfig {
    name: String,
    transport: ServerTransportConfig,
    exposure: Exposure,
    tool_exposure: Vec<(String, Exposure)>,
    timeout: Duration,
}

enum ServerTransportConfig {
    Stdio {
        command: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        cwd: Option<PathBuf>,
    },
    Http {
        endpoint: Url,
        headers: BTreeMap<String, String>,
    },
}

struct RemoteTool {
    name: String,
    description: String,
    input_schema: Value,
}

struct ConnectedServer {
    client: Mutex<McpClient>,
    /// The server advertised the `resources` capability at initialization.
    has_resources: bool,
}

enum McpClient {
    Stdio(McpProcess),
    Http(McpHttpClient),
}

#[derive(Clone)]
struct ExposedTool {
    server: Arc<ConnectedServer>,
    server_name: String,
    remote_name: String,
    exposure: Exposure,
    definition: Value,
}

#[derive(Clone)]
pub(crate) struct McpRegistry {
    tools: BTreeMap<String, ExposedTool>,
    /// Connected servers that offer resources, by configured name (for the resource tools).
    resource_servers: BTreeMap<String, Arc<ConnectedServer>>,
    server_status: Vec<String>,
    auto_enable_codemode: bool,
}

struct McpProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    request_timeout: Duration,
}

struct McpHttpClient {
    client: Client,
    endpoint: Url,
    headers: HeaderMap,
    session_id: Option<String>,
    next_id: u64,
    protocol_version: Option<String>,
    request_timeout: Duration,
}

pub(crate) struct McpToolResult {
    pub content: String,
    pub structured_content: Option<Value>,
    pub is_error: bool,
}

impl McpRegistry {
    pub(crate) async fn connect(agent_dir: &Path, cwd: &Path) -> Self {
        let (configs, auto_enable_codemode) = match load_config(agent_dir, cwd) {
            Ok(config) => config,
            Err(error) => {
                return Self {
                    server_status: vec![format!("MCP configuration: {error}")],
                    ..Self::default()
                };
            }
        };

        let mut registry = Self {
            auto_enable_codemode,
            ..Self::default()
        };
        let connected = join_all(configs.into_iter().map(|config| {
            let cwd = cwd.to_owned();
            async move {
                let name = config.name.clone();
                let server_exposure = config.exposure;
                let tool_exposure = config.tool_exposure.clone();
                let result = connect_server(config, &cwd).await;
                (name, server_exposure, tool_exposure, result)
            }
        }))
        .await;
        let mut used_names = HashSet::new();
        for (name, server_exposure, tool_exposure, result) in connected {
            match result {
                Ok((server, tools)) => {
                    if server.has_resources {
                        registry
                            .resource_servers
                            .insert(name.clone(), Arc::clone(&server));
                    }
                    registry
                        .server_status
                        .push(format!("{name}: connected ({} tools)", tools.len()));
                    for tool in tools {
                        let exposure =
                            tool_exposure_for(server_exposure, &tool_exposure, &tool.name);
                        if exposure == Exposure::Hidden {
                            continue;
                        }
                        let exposed_name = unique_tool_name(&name, &tool.name, &mut used_names);
                        registry.tools.insert(
                            exposed_name.clone(),
                            ExposedTool {
                                server: Arc::clone(&server),
                                server_name: name.clone(),
                                remote_name: tool.name,
                                exposure,
                                definition: json!({
                                    "type": "function",
                                    "name": exposed_name,
                                    "description": tool.description,
                                    "parameters": tool.input_schema,
                                }),
                            },
                        );
                    }
                }
                Err(error) => {
                    registry
                        .server_status
                        .push(format!("{name}: disconnected ({error})"));
                }
            }
        }
        registry
    }

    pub(crate) fn status_text(&self) -> String {
        if self.server_status.is_empty() {
            return "No MCP servers configured.".to_owned();
        }
        self.server_status.join("\n")
    }

    pub(crate) fn definitions_for_model(&self, active_tools: &HashSet<String>) -> Vec<Value> {
        self.tools
            .iter()
            .filter(|(name, tool)| {
                tool.exposure == Exposure::Direct || active_tools.contains(*name)
            })
            .map(|(_, tool)| tool.definition.clone())
            .chain(self.resource_tool_definitions())
            .collect()
    }

    /// `list_mcp_resources`, `list_mcp_resource_templates` and `read_mcp_resource`, offered
    /// whenever a connected server has resources (Pi's `extensions/mcp/resources.ts`).
    fn resource_tool_definitions(&self) -> Vec<Value> {
        if self.resource_servers.is_empty() {
            return Vec::new();
        }
        let list_parameters = json!({
            "type":"object",
            "properties":{
                "server":{"type":"string","description":"MCP server name. Omit to list every server with resources."},
                "cursor":{"type":"string","description":"Opaque cursor from a previous call with the same server; omit for the first page."},
            },
            "additionalProperties":false,
        });
        vec![
            json!({"type":"function","name":LIST_RESOURCES_TOOL,
                "description":"Lists resources provided by MCP servers. Resources allow servers to share data that provides context to language models, such as files, database schemas, or application-specific information. Prefer resources over web search when possible.",
                "parameters":list_parameters}),
            json!({"type":"function","name":LIST_TEMPLATES_TOOL,
                "description":"Lists resource templates provided by MCP servers. Parameterized resource templates allow servers to share data that takes parameters and provides context to language models, such as files, database schemas, or application-specific information. Prefer resource templates over web search when possible.",
                "parameters":list_parameters}),
            json!({"type":"function","name":READ_RESOURCE_TOOL,
                "description":"Read a specific resource from an MCP server given the server name and resource URI.",
                "parameters":{"type":"object","properties":{
                    "server":{"type":"string","description":"MCP server name exactly as configured. Must match the 'server' field returned by list_mcp_resources."},
                    "uri":{"type":"string","description":"Resource URI to read. Must be one of the URIs returned by list_mcp_resources."},
                },"required":["server","uri"],"additionalProperties":false}}),
        ]
    }

    pub(crate) fn has_deferred_tools(&self) -> bool {
        self.tools
            .values()
            .any(|tool| tool.exposure == Exposure::Deferred)
    }

    pub(crate) fn search_tools(
        &self,
        query: &str,
        limit: usize,
        active_tools: &HashSet<String>,
    ) -> Vec<Value> {
        let candidates = self
            .tools
            .iter()
            .filter(|(name, tool)| {
                matches!(tool.exposure, Exposure::Codemode | Exposure::Deferred)
                    && !active_tools.contains(*name)
            })
            .map(|(_, tool)| tool.definition.clone())
            .collect::<Vec<_>>();
        crate::tool_search::rank(query, &candidates, limit)
    }

    pub(crate) fn codemode_definitions(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|tool| tool.definition.clone())
            .chain(self.resource_tool_definitions())
            .collect()
    }

    pub(crate) fn codemode_enabled(&self) -> bool {
        self.auto_enable_codemode
            && self
                .tools
                .values()
                .any(|tool| tool.exposure == Exposure::Codemode)
    }

    pub(crate) async fn call(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Option<Result<McpToolResult, String>> {
        if !self.resource_servers.is_empty()
            && matches!(
                name,
                LIST_RESOURCES_TOOL | LIST_TEMPLATES_TOOL | READ_RESOURCE_TOOL
            )
        {
            let mut result = self.call_resource_tool(name, arguments).await;
            if let Ok(outcome) = &mut result {
                outcome.content = limit_mcp_text(&outcome.content, save_to_temp_file);
            }
            return Some(result);
        }
        let tool = self.tools.get(name)?;
        let mut result = tool
            .server
            .client
            .lock()
            .await
            .call_tool(&tool.remote_name, arguments)
            .await;
        // Pi: an error result without text still says which tool failed.
        if let Ok(outcome) = &mut result {
            name_empty_error(outcome, &tool.server_name, &tool.remote_name);
            outcome.content = limit_mcp_text(&outcome.content, save_to_temp_file);
        }
        Some(result)
    }
}

fn name_empty_error(outcome: &mut McpToolResult, server: &str, tool: &str) {
    if outcome.is_error && outcome.content.trim().is_empty() {
        outcome.content = format!("MCP tool {server}/{tool} returned an error");
    }
}

const MCP_OUTPUT_MAX_BYTES: usize = 20 * 1024;

/// Pi's `truncateMiddle`: keep the head and tail halves of `max_bytes`, on character boundaries.
fn truncate_middle(content: &str, max_bytes: usize) -> Option<(String, usize)> {
    if content.len() <= max_bytes {
        return None;
    }
    let mut head_end = max_bytes / 2;
    while !content.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = content.len() - (max_bytes - max_bytes / 2);
    while !content.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let removed = content[head_end..tail_start].chars().count();
    Some((
        format!(
            "{}…{removed} chars truncated…{}",
            &content[..head_end],
            &content[tail_start..]
        ),
        content.len(),
    ))
}

/// Pi's `limitMcpContent` for text: over 20 KiB, keep head and tail and save the full output.
fn limit_mcp_text(text: &str, save: impl FnOnce(&str) -> Result<String, String>) -> String {
    let Some((truncated, total_bytes)) = truncate_middle(text, MCP_OUTPUT_MAX_BYTES) else {
        return text.to_owned();
    };
    let where_ = match save(text) {
        Ok(path) => format!("[Full output: {path} (read it with offset/limit)]"),
        Err(error) => format!("[Could not save the full output: {error}]"),
    };
    let mut lines = text.split('\n').count();
    if text.ends_with('\n') {
        lines -= 1;
    }
    let tokens = total_bytes.div_ceil(4);
    format!(
        "Warning: truncated output (original token count: {tokens})\nTotal output lines: {lines}\n\n{truncated}\n\n{where_}"
    )
}

fn save_to_temp_file(text: &str) -> Result<String, String> {
    let path = std::env::temp_dir().join(format!("pi-mcp-{}.txt", uuid::Uuid::new_v4()));
    std::fs::write(&path, text).map_err(|error| error.to_string())?;
    Ok(path.display().to_string())
}

const LIST_RESOURCES_TOOL: &str = "list_mcp_resources";
const LIST_TEMPLATES_TOOL: &str = "list_mcp_resource_templates";
const READ_RESOURCE_TOOL: &str = "read_mcp_resource";

/// MCP App user interfaces (`ui://` or `profile=mcp-app`) are for hosts that render them.
fn is_mcp_app_resource(item: &Value) -> bool {
    let uri = item
        .get("uri")
        .or_else(|| item.get("uriTemplate"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mime = item
        .get("mimeType")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_lowercase();
    uri.starts_with("ui://")
        || mime.contains("profile=mcp-app")
        || mime.contains("profile=\"mcp-app\"")
}

/// A listed item without `_meta`/`icons`, tagged with its server.
fn listed_resource(server: &str, item: &Value) -> Value {
    let mut object = item.as_object().cloned().unwrap_or_default();
    object.remove("_meta");
    object.remove("icons");
    let mut tagged = Map::new();
    tagged.insert("server".into(), json!(server));
    tagged.extend(object);
    Value::Object(tagged)
}

impl McpRegistry {
    fn resource_server(&self, name: &str) -> Result<&Arc<ConnectedServer>, String> {
        self.resource_servers.get(name).ok_or_else(|| {
            let available = self
                .resource_servers
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "MCP server \"{name}\" has no resources{}",
                if available.is_empty() {
                    String::new()
                } else {
                    format!(". Servers with resources: {available}")
                }
            )
        })
    }

    async fn list_all(
        server: &ConnectedServer,
        method: &str,
        key: &str,
    ) -> Result<Vec<Value>, String> {
        let mut items = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..100 {
            let params = cursor
                .as_ref()
                .map_or_else(|| json!({}), |c| json!({"cursor": c}));
            let page = server.client.lock().await.request(method, params).await?;
            items.extend(
                page.get(key)
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
            );
            cursor = page
                .get("nextCursor")
                .and_then(Value::as_str)
                .filter(|c| !c.is_empty())
                .map(str::to_owned);
            if cursor.is_none() {
                return Ok(items);
            }
        }
        Err(format!("MCP {method} exceeded 100 pages"))
    }

    async fn call_resource_tool(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Result<McpToolResult, String> {
        let text = |key: &str| -> Result<Option<String>, String> {
            match arguments.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(value)) => {
                    Ok(Some(value.trim().to_owned()).filter(|v| !v.is_empty()))
                }
                Some(_) => Err(format!("{key} must be a string")),
            }
        };
        let payload = if name == READ_RESOURCE_TOOL {
            let server_name = text("server")?.ok_or("server must be provided")?;
            let uri = text("uri")?.ok_or("uri must be provided")?;
            let server = self.resource_server(&server_name)?;
            let result = server
                .client
                .lock()
                .await
                .request("resources/read", json!({"uri": uri}))
                .await?;
            let contents = result
                .get("contents")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let mut parts = Vec::new();
            for item in &contents {
                if contents.len() > 1 {
                    parts.push(format!("{}:", item["uri"].as_str().unwrap_or_default()));
                }
                match item.get("text").and_then(Value::as_str) {
                    Some(text) => parts.push(text.to_owned()),
                    None => parts.push(blob_resource_text(item)),
                }
            }
            let contents = contents
                .into_iter()
                .map(|mut item| {
                    if let Some(object) = item.as_object_mut() {
                        object.remove("_meta");
                    }
                    item
                })
                .collect::<Vec<_>>();
            let content = if parts.is_empty() {
                format!("Resource {uri} is empty.")
            } else {
                parts.join("\n")
            };
            return Ok(McpToolResult {
                content,
                structured_content: Some(
                    json!({"server": server_name, "uri": uri, "contents": contents}),
                ),
                is_error: false,
            });
        } else {
            let (method, key) = if name == LIST_RESOURCES_TOOL {
                ("resources/list", "resources")
            } else {
                ("resources/templates/list", "resourceTemplates")
            };
            let visible = |item: &&Value| !is_mcp_app_resource(item);
            match (text("server")?, text("cursor")?) {
                (Some(server_name), cursor) => {
                    let server = self.resource_server(&server_name)?;
                    let params = cursor.map_or_else(|| json!({}), |c| json!({"cursor": c}));
                    let page = server.client.lock().await.request(method, params).await?;
                    let items = page
                        .get(key)
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    let mut payload = json!({
                        "server": server_name,
                        key: items.iter().filter(visible).map(|i| listed_resource(&server_name, i)).collect::<Vec<_>>(),
                    });
                    if let Some(next) = page.get("nextCursor") {
                        payload["nextCursor"] = next.clone();
                    }
                    payload
                }
                (None, Some(_)) => {
                    return Err("cursor can only be used when a server is specified".into());
                }
                (None, None) => {
                    let (mut items, mut errors) = (Vec::new(), Vec::new());
                    for (server_name, server) in &self.resource_servers {
                        match Self::list_all(server, method, key).await {
                            Ok(found) => items.extend(
                                found
                                    .iter()
                                    .filter(visible)
                                    .map(|i| listed_resource(server_name, i)),
                            ),
                            Err(error) => {
                                errors.push(json!({"server": server_name, "error": error}))
                            }
                        }
                    }
                    let mut payload = json!({ key: items });
                    if !errors.is_empty() {
                        payload["errors"] = Value::Array(errors);
                    }
                    payload
                }
            }
        };
        Ok(McpToolResult {
            content: payload.to_string(),
            structured_content: Some(payload),
            is_error: false,
        })
    }
}

impl Default for McpRegistry {
    fn default() -> Self {
        Self {
            tools: BTreeMap::new(),
            resource_servers: BTreeMap::new(),
            server_status: Vec::new(),
            auto_enable_codemode: true,
        }
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
        timeout(self.request_timeout, self.stdin.write_all(&encoded))
            .await
            .map_err(|_| format!("{method} request timed out while writing"))?
            .map_err(|error| format!("{method} request write failed: {error}"))?;

        loop {
            let mut line = String::new();
            let read = timeout(self.request_timeout, self.stdout.read_line(&mut line))
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
        parse_tool_result(result)
    }
}

impl McpClient {
    async fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        match self {
            Self::Stdio(process) => process.request(method, params).await,
            Self::Http(client) => client.request(method, params).await,
        }
    }

    async fn notify(&mut self, method: &str) -> Result<(), String> {
        match self {
            Self::Stdio(process) => process.notify(method).await,
            Self::Http(client) => client.notify(method).await,
        }
    }

    async fn call_tool(&mut self, name: &str, arguments: &Value) -> Result<McpToolResult, String> {
        match self {
            Self::Stdio(process) => process.call_tool(name, arguments).await,
            Self::Http(client) => client.call_tool(name, arguments).await,
        }
    }

    fn set_protocol_version(&mut self, version: String) {
        if let Self::Http(client) = self {
            client.protocol_version = Some(version);
        }
    }
}

impl McpHttpClient {
    async fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        let request = json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params});
        let mut builder = self
            .client
            .post(self.endpoint.clone())
            .headers(self.headers.clone())
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .json(&request);
        if let Some(session_id) = &self.session_id {
            builder = builder.header("mcp-session-id", session_id);
        }
        if method != "initialize" {
            if let Some(version) = &self.protocol_version {
                builder = builder.header("mcp-protocol-version", version);
            }
        }
        let response = timeout(self.request_timeout, builder.send())
            .await
            .map_err(|_| format!("MCP HTTP {method} timed out"))?
            .map_err(|error| format!("MCP HTTP {method} failed: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("MCP HTTP {method} returned {status}"));
        }
        if let Some(session_id) = response.headers().get("mcp-session-id") {
            self.session_id = Some(
                session_id
                    .to_str()
                    .map_err(|error| format!("invalid MCP session id: {error}"))?
                    .to_owned(),
            );
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        if content_type
            .to_ascii_lowercase()
            .contains("text/event-stream")
        {
            let mut stream = response.bytes_stream();
            return timeout(self.request_timeout, async {
                let mut buffer = String::new();
                loop {
                    let Some(chunk) = stream.next().await else {
                        return Err(format!(
                            "MCP HTTP {method} event stream closed before its response"
                        ));
                    };
                    let chunk = chunk.map_err(|error| {
                        format!("MCP HTTP {method} event stream failed: {error}")
                    })?;
                    buffer.push_str(&String::from_utf8_lossy(&chunk));
                    if buffer.len() > 4 * 1024 * 1024 {
                        return Err(format!("MCP HTTP {method} event exceeded 4 MiB"));
                    }
                    while let Some((end, delimiter)) = event_boundary(&buffer) {
                        let event = buffer[..end].to_owned();
                        buffer.drain(..end + delimiter);
                        let data = event
                            .lines()
                            .filter_map(|line| line.strip_prefix("data:"))
                            .map(str::trim_start)
                            .collect::<Vec<_>>()
                            .join("\n");
                        if data.is_empty() || data == "[DONE]" {
                            continue;
                        }
                        let message: Value = serde_json::from_str(&data).map_err(|error| {
                            format!("MCP HTTP {method} event contained invalid JSON: {error}")
                        })?;
                        if message.get("id").and_then(Value::as_u64) == Some(id) {
                            return parse_rpc_response(message, id, method);
                        }
                    }
                }
            })
            .await
            .map_err(|_| format!("MCP HTTP {method} response timed out"))?;
        }
        let body = timeout(self.request_timeout, response.text())
            .await
            .map_err(|_| format!("MCP HTTP {method} response timed out"))?
            .map_err(|error| format!("MCP HTTP {method} response failed: {error}"))?;
        parse_http_response(&body, &content_type, id, method)
    }

    async fn notify(&mut self, method: &str) -> Result<(), String> {
        let notification = json!({"jsonrpc":"2.0", "method":method});
        let mut builder = self
            .client
            .post(self.endpoint.clone())
            .headers(self.headers.clone())
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .json(&notification);
        if let Some(session_id) = &self.session_id {
            builder = builder.header("mcp-session-id", session_id);
        }
        if let Some(version) = &self.protocol_version {
            builder = builder.header("mcp-protocol-version", version);
        }
        let response = timeout(self.request_timeout, builder.send())
            .await
            .map_err(|_| format!("MCP HTTP notification {method} timed out"))?
            .map_err(|error| format!("MCP HTTP notification {method} failed: {error}"))?;
        if !response.status().is_success() {
            return Err(format!(
                "MCP HTTP notification {method} returned {}",
                response.status()
            ));
        }
        Ok(())
    }

    async fn call_tool(&mut self, name: &str, arguments: &Value) -> Result<McpToolResult, String> {
        let arguments = arguments
            .as_object()
            .ok_or_else(|| "MCP tool arguments must be a JSON object".to_owned())?;
        parse_tool_result(
            self.request("tools/call", json!({"name":name, "arguments":arguments}))
                .await?,
        )
    }
}

fn event_boundary(buffer: &str) -> Option<(usize, usize)> {
    let lf = buffer.find("\n\n").map(|index| (index, 2));
    let crlf = buffer.find("\r\n\r\n").map(|index| (index, 4));
    match (lf, crlf) {
        (Some(lf), Some(crlf)) => Some(if lf.0 < crlf.0 { lf } else { crlf }),
        (Some(boundary), None) | (None, Some(boundary)) => Some(boundary),
        (None, None) => None,
    }
}

fn parse_rpc_response(message: Value, id: u64, method: &str) -> Result<Value, String> {
    if message.get("id").and_then(Value::as_u64) != Some(id) {
        return Err(format!(
            "MCP {method} returned a response for a different request"
        ));
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
    message
        .get("result")
        .cloned()
        .ok_or_else(|| format!("MCP {method} response has no result"))
}

fn parse_http_response(
    body: &str,
    content_type: &str,
    id: u64,
    method: &str,
) -> Result<Value, String> {
    let messages = if content_type
        .to_ascii_lowercase()
        .contains("text/event-stream")
    {
        body.split("\n\n")
            .filter_map(|event| {
                let data = event
                    .lines()
                    .filter_map(|line| line.strip_prefix("data:"))
                    .map(str::trim_start)
                    .collect::<Vec<_>>()
                    .join("\n");
                (!data.is_empty() && data != "[DONE]").then_some(data)
            })
            .collect::<Vec<_>>()
    } else {
        vec![body.to_owned()]
    };
    for encoded in messages {
        let message: Value = serde_json::from_str(&encoded)
            .map_err(|error| format!("MCP HTTP {method} returned invalid JSON: {error}"))?;
        if message.get("id").and_then(Value::as_u64) != Some(id) {
            continue;
        }
        return parse_rpc_response(message, id, method);
    }
    Err(format!(
        "MCP HTTP {method} response contained no matching JSON-RPC result"
    ))
}

fn is_text_mime(mime: Option<&str>) -> bool {
    let Some(mime) = mime else { return false };
    let kind = mime
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    kind.starts_with("text/")
        || kind == "application/json"
        || kind.ends_with("+json")
        || kind.ends_with("+xml")
}

/// File extension of a resource URI path (`.bin` when none), as Pi's `extensionOf`.
fn uri_extension(uri: &str) -> String {
    let path = uri.split(['?', '#']).next().unwrap_or(uri);
    path.rsplit_once('.')
        .map(|(_, ext)| ext)
        .filter(|ext| {
            (1..=8).contains(&ext.len()) && ext.chars().all(|c| c.is_ascii_alphanumeric())
        })
        .map_or_else(|| ".bin".to_owned(), |ext| format!(".{ext}"))
}

/// Pi's handling of a `blob` resource: text types decode to text, others are saved to a temp file.
fn blob_resource_text(resource: &Value) -> String {
    use base64::Engine;
    let uri = resource["uri"].as_str().unwrap_or_default();
    let mime = resource.get("mimeType").and_then(Value::as_str);
    let Some(blob) = resource.get("blob").and_then(Value::as_str) else {
        return "[MCP resource has no content]".to_owned();
    };
    let Ok(data) = base64::engine::general_purpose::STANDARD.decode(blob) else {
        return format!("[Binary resource {uri} could not be decoded]");
    };
    if is_text_mime(mime) {
        return String::from_utf8_lossy(&data).into_owned();
    }
    let kind = format!(
        "{}, {}",
        mime.unwrap_or("unknown type"),
        format_size(data.len() as u64)
    );
    let path = std::env::temp_dir().join(format!(
        "pi-mcp-{}{}",
        uuid::Uuid::new_v4(),
        uri_extension(uri)
    ));
    match std::fs::write(&path, &data) {
        Ok(()) => format!(
            "[Binary resource {uri} ({kind}) saved to {}]",
            path.display()
        ),
        Err(error) => format!("[Binary resource {uri} ({kind}) could not be saved: {error}]"),
    }
}

/// Pi's `formatSize`.
fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// A `resource_link` block as Pi shows it to the model.
fn resource_link_text(block: &Value) -> String {
    let text = |key: &str| block.get(key).and_then(Value::as_str);
    let mut details = Vec::new();
    details.extend(text("mimeType").map(str::to_owned));
    details.extend(block.get("size").and_then(Value::as_u64).map(format_size));
    let label = text("title").or_else(|| text("name")).unwrap_or_default();
    let details = if details.is_empty() {
        String::new()
    } else {
        format!(" ({})", details.join(", "))
    };
    let description = text("description")
        .map(|d| format!(": {d}"))
        .unwrap_or_default();
    format!(
        "[Resource {} \"{label}\"{details}{description}]",
        text("uri").unwrap_or_default()
    )
}

fn parse_tool_result(result: Value) -> Result<McpToolResult, String> {
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
            } else if block.get("type").and_then(Value::as_str) == Some("resource_link") {
                Some(resource_link_text(block))
            } else if block.get("type").and_then(Value::as_str) == Some("resource") {
                let resource = block.get("resource").unwrap_or(&Value::Null);
                Some(match resource.get("text").and_then(Value::as_str) {
                    Some(text) => text.to_owned(),
                    None => blob_resource_text(resource),
                })
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
            content =
                serde_json::to_string_pretty(structured).unwrap_or_else(|_| structured.to_string());
        }
    }
    Ok(McpToolResult {
        content,
        structured_content: result.get("structuredContent").cloned(),
        is_error: result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

async fn connect_server(
    config: ServerConfig,
    cwd: &Path,
) -> Result<(Arc<ConnectedServer>, Vec<RemoteTool>), String> {
    let mut client = match config.transport {
        ServerTransportConfig::Stdio {
            command: executable,
            args,
            env,
            cwd: server_cwd,
        } => {
            let mut command = Command::new(&executable);
            command
                .args(&args)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null());
            command.current_dir(server_cwd.as_deref().unwrap_or(cwd));
            for (name, value) in &env {
                command.env(name, expand_environment(value)?);
            }
            let mut child = command
                .spawn()
                .map_err(|error| format!("could not start `{executable}`: {error}"))?;
            let stdin = child.stdin.take().ok_or("MCP server stdin was not piped")?;
            let stdout = child
                .stdout
                .take()
                .ok_or("MCP server stdout was not piped")?;
            McpClient::Stdio(McpProcess {
                child,
                stdin,
                stdout: BufReader::new(stdout),
                next_id: 1,
                request_timeout: config.timeout,
            })
        }
        ServerTransportConfig::Http { endpoint, headers } => {
            let mut resolved_headers = HeaderMap::new();
            for (name, value) in headers {
                let name = HeaderName::from_bytes(name.as_bytes())
                    .map_err(|error| format!("invalid MCP HTTP header name `{name}`: {error}"))?;
                let value = HeaderValue::from_str(&expand_environment(&value)?)
                    .map_err(|error| format!("invalid MCP HTTP header `{name}`: {error}"))?;
                resolved_headers.insert(name, value);
            }
            let http = Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|error| format!("could not create MCP HTTP client: {error}"))?;
            McpClient::Http(McpHttpClient {
                client: http,
                endpoint,
                headers: resolved_headers,
                session_id: None,
                next_id: 1,
                protocol_version: None,
                request_timeout: config.timeout,
            })
        }
    };

    let initialized = client
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
    let has_resources = initialized.pointer("/capabilities/resources").is_some();
    let protocol_version = initialized
        .get("protocolVersion")
        .and_then(Value::as_str)
        .expect("validated protocol version")
        .to_owned();
    client.set_protocol_version(protocol_version);
    client.notify("notifications/initialized").await?;

    let mut tools = Vec::new();
    let mut cursor = None;
    let mut seen_cursors = HashSet::new();
    for _ in 0..100 {
        let params = cursor
            .as_ref()
            .map(|cursor| json!({"cursor":cursor}))
            .unwrap_or_else(|| json!({}));
        let page = client.request("tools/list", params).await?;
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
            tools.push(RemoteTool {
                name: name.to_owned(),
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .or_else(|| tool.get("title").and_then(Value::as_str))
                    .unwrap_or(name)
                    .to_owned(),
                input_schema,
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
                client: Mutex::new(client),
                has_resources,
            });
            return Ok((server, tools));
        }
    }
    Err("MCP tools/list exceeded 100 pages".into())
}

fn tool_exposure_for(default: Exposure, overrides: &[(String, Exposure)], tool: &str) -> Exposure {
    if let Some((_, exposure)) = overrides
        .iter()
        .find(|(pattern, _)| !pattern.contains('*') && pattern == tool)
    {
        return *exposure;
    }
    overrides
        .iter()
        .find(|(pattern, _)| pattern.contains('*') && wildcard_matches(pattern, tool))
        .map_or(default, |(_, exposure)| *exposure)
}

fn wildcard_matches(pattern: &str, value: &str) -> bool {
    let mut rest = value;
    let mut parts = pattern.split('*').peekable();
    if let Some(prefix) = parts.next() {
        if !pattern.starts_with('*') {
            let Some(after_prefix) = rest.strip_prefix(prefix) else {
                return false;
            };
            rest = after_prefix;
        }
    }
    while let Some(part) = parts.next() {
        if part.is_empty() {
            continue;
        }
        let Some(index) = rest.find(part) else {
            return false;
        };
        if parts.peek().is_none() && !pattern.ends_with('*') && index + part.len() != rest.len() {
            return false;
        }
        rest = &rest[index + part.len()..];
    }
    pattern.starts_with('*') || value.starts_with(pattern.split('*').next().unwrap_or_default())
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

fn read_mcp_config(path: &Path) -> Result<Value, String> {
    if !path.exists() {
        return Ok(json!({"mcpServers": {}}));
    }
    let source =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let source = source.strip_prefix('\u{feff}').unwrap_or(&source);
    let value: Value =
        serde_json::from_str(source).map_err(|error| format!("{}: {error}", path.display()))?;
    if !value.is_object() {
        return Err(format!("{}: expected a JSON object", path.display()));
    }
    Ok(value)
}

fn config_servers(path: &Path, config: &Value) -> Result<Map<String, Value>, String> {
    match config.get("mcpServers") {
        None => Ok(Map::new()),
        Some(Value::Object(servers)) => Ok(servers.clone()),
        Some(_) => Err(format!(
            "{}: `mcpServers` must be an object",
            path.display()
        )),
    }
}

fn parse_auto_enable_codemode(path: &Path, config: &Value) -> Result<bool, String> {
    match config.get("autoEnableCodemode") {
        None => Ok(true),
        Some(Value::Bool(enabled)) => Ok(*enabled),
        Some(_) => Err(format!(
            "{}: `autoEnableCodemode` must be a boolean",
            path.display()
        )),
    }
}

fn is_project_override(value: &Value) -> bool {
    value.as_object().is_some_and(|object| {
        !object.contains_key("command")
            && !object.contains_key("url")
            && !object.contains_key("type")
    })
}

/// `--approve` / `--no-approve` for this process: 1 = trust, 2 = ignore, 0 = use `trust.json`.
static TRUST_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Set the per-run project trust override (`Some(true)` trusts, `Some(false)` ignores).
pub(crate) fn set_project_trust_override(decision: Option<bool>) {
    let value = match decision {
        None => 0,
        Some(true) => 1,
        Some(false) => 2,
    };
    TRUST_OVERRIDE.store(value, std::sync::atomic::Ordering::Relaxed);
}

/// Persist a trust decision for `cwd` in `trust.json` (keyed by canonical path, like Pi).
pub(crate) fn save_project_trust(
    agent_dir: &Path,
    cwd: &Path,
    trusted: bool,
) -> Result<(), String> {
    let path = agent_dir.join("trust.json");
    let mut trust = match std::fs::read_to_string(&path) {
        Ok(source) => {
            serde_json::from_str::<Value>(source.strip_prefix('\u{feff}').unwrap_or(&source))
                .ok()
                .and_then(|value| value.as_object().cloned())
                .ok_or_else(|| format!("{}: expected a JSON object", path.display()))?
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => serde_json::Map::new(),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    let key = std::fs::canonicalize(cwd).unwrap_or_else(|_| cwd.to_owned());
    trust.insert(key.to_string_lossy().into_owned(), Value::Bool(trusted));
    std::fs::create_dir_all(agent_dir).map_err(|error| error.to_string())?;
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&Value::Object(trust)).unwrap_or_default(),
    )
    .map_err(|error| format!("{}: {error}", path.display()))
}

fn project_is_trusted(agent_dir: &Path, cwd: &Path) -> Result<bool, String> {
    match TRUST_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return Ok(true),
        2 => return Ok(false),
        _ => {}
    }
    let trust_path = agent_dir.join("trust.json");
    if !trust_path.exists() {
        return Ok(default_project_trust(agent_dir));
    }
    let source = std::fs::read_to_string(&trust_path)
        .map_err(|error| format!("{}: {error}", trust_path.display()))?;
    let trust: Value = serde_json::from_str(source.strip_prefix('\u{feff}').unwrap_or(&source))
        .map_err(|error| format!("{}: {error}", trust_path.display()))?;
    let trust = trust
        .as_object()
        .ok_or_else(|| format!("{}: expected a JSON object", trust_path.display()))?;
    if trust
        .values()
        .any(|value| !value.is_null() && !value.is_boolean())
    {
        return Err(format!(
            "{}: trust decisions must be booleans or null",
            trust_path.display()
        ));
    }

    let mut current = std::fs::canonicalize(cwd).unwrap_or_else(|_| cwd.to_owned());
    loop {
        if let Some(decision) = trust.get(current.to_string_lossy().as_ref()) {
            if let Some(decision) = decision.as_bool() {
                return Ok(decision);
            }
        }
        if !current.pop() {
            return Ok(default_project_trust(agent_dir));
        }
    }
}

/// Pi's global `defaultProjectTrust` when no decision is stored: `always` trusts, `never` and
/// `ask` do not (Mu has no startup prompt yet, so `ask` behaves like `never`).
fn default_project_trust(agent_dir: &Path) -> bool {
    let settings = crate::settings::Settings::load(agent_dir, agent_dir, false);
    settings
        .get("defaultProjectTrust")
        .as_ref()
        .and_then(Value::as_str)
        == Some("always")
}

fn load_config(agent_dir: &Path, project_cwd: &Path) -> Result<(Vec<ServerConfig>, bool), String> {
    let global_path = agent_dir.join("mcp.json");
    let global = read_mcp_config(&global_path)?;
    let mut auto_enable_codemode = parse_auto_enable_codemode(&global_path, &global)?;
    let mut servers = config_servers(&global_path, &global)?;

    let project_path = project_cwd.join(".pi/mcp.json");
    if project_path.exists()
        && project_is_trusted(agent_dir, project_cwd).unwrap_or_else(|error| {
            eprintln!("mu: MCP project trust: {error}");
            false
        })
    {
        match read_mcp_config(&project_path) {
            Ok(project) => {
                if let Some(value) = project.get("autoEnableCodemode") {
                    match value.as_bool() {
                        Some(enabled) => auto_enable_codemode = enabled,
                        None => eprintln!(
                            "mu: {}: `autoEnableCodemode` must be a boolean",
                            project_path.display()
                        ),
                    }
                }
                match config_servers(&project_path, &project) {
                    Ok(project_servers) => {
                        for (name, project_entry) in &project_servers {
                            if is_project_override(project_entry) {
                                let Some(global_entry) = servers.get_mut(name) else {
                                    eprintln!(
                                        "mu: {}: project override for `{name}` has no global server",
                                        project_path.display()
                                    );
                                    continue;
                                };
                                let Some(global_object) = global_entry.as_object_mut() else {
                                    eprintln!(
                                        "mu: {}: global server `{name}` cannot be overridden",
                                        project_path.display()
                                    );
                                    continue;
                                };
                                let Some(project_object) = project_entry.as_object() else {
                                    continue;
                                };
                                let allowed = ["enabled", "exposure", "toolExposure"];
                                if project_object
                                    .keys()
                                    .any(|key| !allowed.contains(&key.as_str()))
                                {
                                    eprintln!(
                                        "mu: {}: override for `{name}` may only set enabled, exposure, and toolExposure",
                                        project_path.display()
                                    );
                                    continue;
                                }
                                global_object.extend(project_object.clone());
                            } else {
                                servers.insert(name.clone(), project_entry.clone());
                            }
                        }
                    }
                    Err(error) => eprintln!("mu: ignoring project MCP config: {error}"),
                }
            }
            Err(error) => eprintln!("mu: ignoring project MCP config: {error}"),
        }
    }

    let value = json!({"mcpServers": servers});
    let servers = value
        .get("mcpServers")
        .and_then(Value::as_object)
        .expect("merged MCP server map is an object");
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
        if let Some(enabled) = object.get("enabled") {
            match enabled.as_bool() {
                Some(false) => continue,
                Some(true) => {}
                None => {
                    eprintln!("mu: skipping MCP server `{name}`: enabled must be a boolean");
                    continue;
                }
            }
        }
        if object
            .get("description")
            .is_some_and(|value| !value.is_string())
        {
            eprintln!("mu: skipping MCP server `{name}`: description must be a string");
            continue;
        }
        let exposure = match object.get("exposure") {
            None => Exposure::Codemode,
            Some(Value::String(value)) => match Exposure::parse(value) {
                Some(exposure) => exposure,
                None => {
                    eprintln!("mu: skipping MCP server `{name}`: unknown exposure `{value}`");
                    continue;
                }
            },
            Some(_) => {
                eprintln!("mu: skipping MCP server `{name}`: exposure must be a string");
                continue;
            }
        };
        let tool_exposure = match object.get("toolExposure") {
            None => Vec::new(),
            Some(Value::Object(entries)) => {
                let mut parsed = Vec::new();
                let mut invalid = false;
                for (pattern, raw_exposure) in entries {
                    let Some(raw_exposure) = raw_exposure.as_str() else {
                        invalid = true;
                        break;
                    };
                    let Some(exposure) = Exposure::parse(raw_exposure) else {
                        invalid = true;
                        break;
                    };
                    parsed.push((pattern.clone(), exposure));
                }
                if invalid {
                    eprintln!(
                        "mu: skipping MCP server `{name}`: toolExposure entries must use a known exposure"
                    );
                    continue;
                }
                parsed
            }
            Some(_) => {
                eprintln!("mu: skipping MCP server `{name}`: toolExposure must be an object");
                continue;
            }
        };
        let timeout = match object.get("timeout") {
            None => DEFAULT_REQUEST_TIMEOUT,
            Some(value) => {
                let Some(seconds) = value
                    .as_f64()
                    .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
                else {
                    eprintln!(
                        "mu: skipping MCP server `{name}`: timeout must be a positive number"
                    );
                    continue;
                };
                let Ok(timeout) = Duration::try_from_secs_f64(seconds) else {
                    eprintln!("mu: skipping MCP server `{name}`: timeout is too large");
                    continue;
                };
                timeout
            }
        };
        let transport = if let Some(raw_url) = object.get("url") {
            let type_is_http = match object.get("type") {
                None => true,
                Some(Value::String(kind)) => kind == "http" || kind == "streamable-http",
                Some(_) => false,
            };
            let Some(url) = raw_url.as_str().and_then(|value| Url::parse(value).ok()) else {
                eprintln!("mu: skipping MCP server `{name}`: url must be a valid http(s) URL");
                continue;
            };
            if !type_is_http || !matches!(url.scheme(), "http" | "https") {
                eprintln!(
                    "mu: skipping MCP server `{name}`: url transport must use type http and an http(s) URL"
                );
                continue;
            }
            if object.contains_key("auth") || object.contains_key("oauth") {
                eprintln!(
                    "mu: MCP server `{name}`: configured OAuth/provider authentication is not implemented yet"
                );
                continue;
            }
            let headers = match object.get("headers") {
                None => Some(BTreeMap::new()),
                Some(Value::Object(headers)) => headers
                    .iter()
                    .map(|(name, value)| Some((name.clone(), value.as_str()?.to_owned())))
                    .collect::<Option<BTreeMap<_, _>>>(),
                _ => None,
            };
            let Some(headers) = headers else {
                eprintln!("mu: skipping MCP server `{name}`: headers must map names to strings");
                continue;
            };
            ServerTransportConfig::Http {
                endpoint: url,
                headers,
            }
        } else {
            let type_is_stdio = match object.get("type") {
                None => true,
                Some(Value::String(kind)) => kind == "stdio",
                Some(_) => false,
            };
            if !type_is_stdio {
                eprintln!("mu: skipping MCP server `{name}`: stdio entries require type `stdio`");
                continue;
            }
            let Some(command) = object
                .get("command")
                .and_then(Value::as_str)
                .filter(|command| !command.is_empty())
            else {
                eprintln!("mu: skipping MCP server `{name}`: stdio server requires a command");
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
            let server_cwd = match object.get("cwd") {
                None => None,
                Some(Value::String(cwd)) => {
                    let path = PathBuf::from(cwd);
                    Some(if path.is_absolute() {
                        path
                    } else {
                        project_cwd.join(path)
                    })
                }
                Some(_) => {
                    eprintln!("mu: skipping MCP server `{name}`: cwd must be a string");
                    continue;
                }
            };
            ServerTransportConfig::Stdio {
                command: command.to_owned(),
                args,
                env,
                cwd: server_cwd,
            }
        };
        configs.push(ServerConfig {
            name: name.clone(),
            transport,
            exposure,
            tool_exposure,
            timeout,
        });
    }
    Ok((configs, auto_enable_codemode))
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
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn long_output_keeps_head_and_tail_and_reports_the_saved_file() {
        let short = "x".repeat(MCP_OUTPUT_MAX_BYTES);
        assert_eq!(limit_mcp_text(&short, |_| panic!("not saved")), short);
        let long = format!("{}\n{}", "a".repeat(15_000), "b".repeat(15_000));
        let out = limit_mcp_text(&long, |full| {
            assert_eq!(full, long);
            Ok("/tmp/full.txt".into())
        });
        assert!(out.starts_with(
            "Warning: truncated output (original token count: 7501)\nTotal output lines: 2\n\n"
        ));
        assert!(out.contains("…9"), "removed-chars marker present: {out}");
        assert!(out.contains("chars truncated…"));
        assert!(out.ends_with("[Full output: /tmp/full.txt (read it with offset/limit)]"));
        let failed = limit_mcp_text(&long, |_| Err("disk full".into()));
        assert!(failed.ends_with("[Could not save the full output: disk full]"));
    }

    #[test]
    fn truncation_respects_multibyte_boundaries() {
        let text = "é".repeat(MCP_OUTPUT_MAX_BYTES);
        let (truncated, total) = truncate_middle(&text, MCP_OUTPUT_MAX_BYTES).unwrap();
        assert_eq!(total, text.len());
        assert!(truncated.contains("chars truncated"));
    }

    #[test]
    fn resource_links_and_embedded_text_resources_become_model_text() {
        let result = parse_tool_result(json!({"content":[
            {"type":"resource_link","uri":"file:///a.txt","name":"a","title":"The A","mimeType":"text/plain","size":2048,"description":"notes"},
            {"type":"resource_link","uri":"file:///b","name":"b"},
            {"type":"resource","resource":{"uri":"file:///c","text":"embedded"}},
        ]})).unwrap();
        assert_eq!(
            result.content,
            "[Resource file:///a.txt \"The A\" (text/plain, 2.0KB): notes]\n[Resource file:///b \"b\"]\nembedded"
        );
        assert_eq!(format_size(10), "10B");
        assert_eq!(format_size(3 * 1024 * 1024), "3.0MB");
    }

    #[test]
    fn embedded_blob_resources_in_tool_results_are_decoded() {
        use base64::Engine;
        let blob = base64::engine::general_purpose::STANDARD.encode("hello");
        let result = parse_tool_result(json!({"content":[
            {"type":"resource","resource":{"uri":"file:///a.txt","mimeType":"text/plain","blob":blob}}
        ]})).unwrap();
        assert_eq!(result.content, "hello");
    }

    #[test]
    fn blob_resources_decode_text_or_save_binary_files() {
        use base64::Engine;
        let b64 = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
        let text = blob_resource_text(
            &json!({"uri":"file:///a.json","mimeType":"application/json","blob":b64(b"{\"a\":1}")}),
        );
        assert_eq!(text, "{\"a\":1}");
        let saved = blob_resource_text(
            &json!({"uri":"file:///pic.png?x=1","mimeType":"image/png","blob":b64(&[1,2,3])}),
        );
        assert!(
            saved.starts_with("[Binary resource file:///pic.png?x=1 (image/png, 3B) saved to "),
            "{saved}"
        );
        assert!(saved.ends_with(".png]"));
        assert!(
            blob_resource_text(&json!({"uri":"u","blob":"!!"})).contains("could not be decoded")
        );
        assert_eq!(uri_extension("file:///noext"), ".bin");
    }

    #[test]
    fn empty_error_results_name_the_failing_tool() {
        let result = |content: &str, is_error| McpToolResult {
            content: content.into(),
            structured_content: None,
            is_error,
        };
        let mut empty = result("  ", true);
        name_empty_error(&mut empty, "s", "t");
        assert_eq!(empty.content, "MCP tool s/t returned an error");
        let mut kept = result("boom", true);
        name_empty_error(&mut kept, "s", "t");
        assert_eq!(kept.content, "boom");
        let mut ok = result("", false);
        name_empty_error(&mut ok, "s", "t");
        assert_eq!(ok.content, "");
    }

    #[test]
    fn resource_helpers_hide_app_uis_and_tag_servers() {
        assert!(is_mcp_app_resource(&json!({"uri":"ui://widget"})));
        assert!(is_mcp_app_resource(
            &json!({"uri":"file:///a","mimeType":"text/html;profile=mcp-app"})
        ));
        assert!(!is_mcp_app_resource(
            &json!({"uri":"file:///a","mimeType":"text/plain"})
        ));
        let tagged = listed_resource("s", &json!({"uri":"u","name":"n","_meta":{},"icons":[]}));
        assert_eq!(tagged, json!({"server":"s","uri":"u","name":"n"}));
    }

    #[tokio::test]
    async fn resource_tools_appear_only_with_resource_servers_and_validate_arguments() {
        let mut registry = McpRegistry::default();
        assert!(registry.resource_tool_definitions().is_empty());
        assert!(
            registry
                .call("read_mcp_resource", &json!({}))
                .await
                .is_none()
        );
        let client = McpClient::Http(McpHttpClient {
            client: Client::new(),
            endpoint: Url::parse("http://127.0.0.1:1/").unwrap(),
            headers: HeaderMap::new(),
            session_id: None,
            next_id: 1,
            protocol_version: None,
            request_timeout: Duration::from_secs(1),
        });
        registry.resource_servers.insert(
            "docs".into(),
            Arc::new(ConnectedServer {
                client: Mutex::new(client),
                has_resources: true,
            }),
        );
        let names = registry
            .resource_tool_definitions()
            .iter()
            .map(|d| d["name"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "list_mcp_resources",
                "list_mcp_resource_templates",
                "read_mcp_resource"
            ]
        );
        let missing = registry
            .call("read_mcp_resource", &json!({"server":"docs"}))
            .await
            .unwrap();
        assert_eq!(missing.err().as_deref(), Some("uri must be provided"));
        let unknown = registry
            .call("list_mcp_resources", &json!({"server":"nope"}))
            .await
            .unwrap();
        assert!(
            unknown
                .err()
                .unwrap()
                .contains("Servers with resources: docs")
        );
        let cursor = registry
            .call("list_mcp_resources", &json!({"cursor":"c"}))
            .await
            .unwrap();
        assert_eq!(
            cursor.err().as_deref(),
            Some("cursor can only be used when a server is specified")
        );
    }

    #[test]
    fn status_reports_when_no_servers_are_configured() {
        assert_eq!(
            McpRegistry::default().status_text(),
            "No MCP servers configured."
        );
    }

    #[test]
    fn loads_stdio_servers_and_resolves_relative_working_directories() {
        let root = std::env::temp_dir().join(format!("mu-mcp-config-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let config = root.join("mcp.json");
        std::fs::write(
            &config,
            r#"{"autoEnableCodemode":false,"mcpServers":{"filesystem":{"command":"node","args":["server.js"],"cwd":"tools","env":{"ROOT":"${HOME}"}},"direct":{"command":"node","exposure":"direct","toolExposure":{"read*":"hidden","read_file":"direct"},"timeout":2.5},"disabled":{"command":"never","enabled":false}}}"#,
        )
        .unwrap();

        let (servers, auto_enable) = load_config(&root, &root).unwrap();
        assert!(!auto_enable);
        let (_, default_auto_enable) = load_config(&root.join("empty-agent"), &root).unwrap();
        assert!(default_auto_enable);
        assert_eq!(servers.len(), 2);
        let filesystem = servers
            .iter()
            .find(|server| server.name == "filesystem")
            .unwrap();
        let ServerTransportConfig::Stdio { command, args, .. } = &filesystem.transport else {
            panic!("expected stdio transport");
        };
        assert_eq!(command, "node");
        assert_eq!(args, &["server.js"]);
        assert_eq!(filesystem.exposure, Exposure::Codemode);
        let ServerTransportConfig::Stdio { cwd, .. } = &filesystem.transport else {
            panic!("expected stdio transport");
        };
        assert_eq!(cwd.as_deref(), Some(root.join("tools").as_path()));
        let direct = servers
            .iter()
            .find(|server| server.name == "direct")
            .unwrap();
        assert_eq!(direct.exposure, Exposure::Direct);
        assert_eq!(direct.timeout, Duration::from_secs_f64(2.5));
        assert_eq!(
            tool_exposure_for(direct.exposure, &direct.tool_exposure, "read_file"),
            Exposure::Direct
        );
        assert_eq!(
            tool_exposure_for(direct.exposure, &direct.tool_exposure, "read_directory"),
            Exposure::Hidden
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn preserves_mcp_structured_content_with_tool_result() {
        let result = parse_tool_result(json!({
            "content":[{"type":"text","text":"answer"}],
            "structuredContent":{"answer":42},
            "isError":false
        }))
        .unwrap();
        assert_eq!(result.content, "answer");
        assert_eq!(result.structured_content, Some(json!({"answer":42})));
    }

    #[test]
    fn config_accepts_streamable_http_with_headers() {
        let root =
            std::env::temp_dir().join(format!("mu-mcp-http-config-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("mcp.json"),
            r#"{"mcpServers":{"remote":{"type":"http","url":"https://example.test/mcp","headers":{"Authorization":"Bearer ${TOKEN}"}}}}"#,
        )
        .unwrap();
        let (servers, _) = load_config(&root, &root).unwrap();
        assert_eq!(servers.len(), 1);
        let ServerTransportConfig::Http { endpoint, headers } = &servers[0].transport else {
            panic!("expected Streamable HTTP transport");
        };
        assert_eq!(endpoint.as_str(), "https://example.test/mcp");
        assert_eq!(
            headers.get("Authorization").map(String::as_str),
            Some("Bearer ${TOKEN}")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn streamable_http_negotiates_session_and_calls_tools() {
        use std::{
            io::{BufRead, BufReader as StdBufReader, Read, Write},
            net::TcpListener,
            thread,
        };

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/mcp", listener.local_addr().unwrap());
        let server_thread = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = StdBufReader::new(stream.try_clone().unwrap());
            for request_index in 0..4 {
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                assert!(
                    !request_line.is_empty(),
                    "client closed before request {request_index}"
                );
                let mut headers = BTreeMap::new();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line == "\n" {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        headers.insert(name.to_ascii_lowercase(), value.trim().to_owned());
                    }
                }
                let content_length = headers
                    .get("content-length")
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or_default();
                let mut body = vec![0; content_length];
                reader.read_exact(&mut body).unwrap();
                let request: Value = serde_json::from_slice(&body).unwrap();
                if request_index > 0 {
                    assert_eq!(
                        headers.get("mcp-session-id").map(String::as_str),
                        Some("test-session")
                    );
                    assert_eq!(
                        headers.get("mcp-protocol-version").map(String::as_str),
                        Some("2025-11-25")
                    );
                }
                let method = request.get("method").and_then(Value::as_str).unwrap();
                let (status, response_body) = match method {
                    "initialize" => ("200 OK", json!({
                        "jsonrpc":"2.0","id":request["id"],"result":{
                            "protocolVersion":"2025-11-25","capabilities":{"tools":{}},
                            "serverInfo":{"name":"http-test","version":"1"}
                        }
                    }).to_string()),
                    "notifications/initialized" => ("202 Accepted", String::new()),
                    "tools/list" => ("200 OK", json!({
                        "jsonrpc":"2.0","id":request["id"],"result":{"tools":[{
                            "name":"echo","description":"Echo text",
                            "inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}
                        }]}
                    }).to_string()),
                    "tools/call" => ("200 OK", json!({
                        "jsonrpc":"2.0","id":request["id"],"result":{"content":[{"type":"text","text":"HTTP MCP result"}]}
                    }).to_string()),
                    other => panic!("unexpected MCP method {other}"),
                };
                let (content_type, wire_body) = if method == "tools/list" {
                    (
                        "text/event-stream",
                        format!("event: message\ndata: {response_body}\n\n"),
                    )
                } else {
                    ("application/json", response_body)
                };
                let session_header = if request_index == 0 {
                    "Mcp-Session-Id: test-session\r\n"
                } else {
                    ""
                };
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{session_header}Connection: keep-alive\r\n\r\n{wire_body}",
                    wire_body.len()
                ).unwrap();
                stream.flush().unwrap();
            }
        });

        let config = ServerConfig {
            name: "http-test".into(),
            transport: ServerTransportConfig::Http {
                endpoint: Url::parse(&endpoint).unwrap(),
                headers: BTreeMap::new(),
            },
            exposure: Exposure::Direct,
            tool_exposure: Vec::new(),
            timeout: Duration::from_secs(5),
        };
        let (server, tools) = connect_server(config, Path::new("."))
            .await
            .expect("connect Streamable HTTP MCP server");
        assert_eq!(tools.len(), 1);
        let result = server
            .client
            .lock()
            .await
            .call_tool("echo", &json!({"text":"hello"}))
            .await
            .unwrap();
        assert_eq!(result.content, "HTTP MCP result");
        server_thread.join().unwrap();
    }

    #[test]
    fn parses_streamable_http_sse_response() {
        let body =
            "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"ok\":true}}\n\n";
        let response = parse_http_response(body, "text/event-stream", 7, "tools/list").unwrap();
        assert_eq!(response["ok"], true);
    }

    #[test]
    fn saved_trust_decisions_round_trip_through_trust_json() {
        let dir = std::env::temp_dir().join(format!("mu-savetrust-{}", uuid::Uuid::new_v4()));
        let project = dir.join("proj");
        std::fs::create_dir_all(&project).unwrap();
        assert_eq!(project_is_trusted(&dir, &project), Ok(false));
        save_project_trust(&dir, &project, true).unwrap();
        assert_eq!(project_is_trusted(&dir, &project), Ok(true));
        save_project_trust(&dir, &project, false).unwrap();
        assert_eq!(project_is_trusted(&dir, &project), Ok(false));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn default_project_trust_always_applies_only_without_a_stored_decision() {
        let dir = std::env::temp_dir().join(format!("mu-deftrust-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("settings.json"),
            r#"{"defaultProjectTrust":"always"}"#,
        )
        .unwrap();
        assert_eq!(project_is_trusted(&dir, &dir), Ok(true));
        save_project_trust(&dir, &dir, false).unwrap();
        assert_eq!(
            project_is_trusted(&dir, &dir),
            Ok(false),
            "stored decision wins"
        );
        std::fs::write(
            dir.join("settings.json"),
            r#"{"defaultProjectTrust":"never"}"#,
        )
        .unwrap();
        std::fs::remove_file(dir.join("trust.json")).unwrap();
        assert_eq!(project_is_trusted(&dir, &dir), Ok(false));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn trust_override_beats_trust_json_and_resets() {
        let dir = std::env::temp_dir().join(format!("mu-trust-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(project_is_trusted(&dir, &dir), Ok(false));
        set_project_trust_override(Some(true));
        assert_eq!(project_is_trusted(&dir, &dir), Ok(true));
        set_project_trust_override(Some(false));
        assert_eq!(project_is_trusted(&dir, &dir), Ok(false));
        set_project_trust_override(None);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn project_mcp_config_requires_trust_and_only_overrides_safe_fields() {
        let root = std::env::temp_dir().join(format!("mu-mcp-project-{}", uuid::Uuid::new_v4()));
        let agent_dir = root.join("agent");
        let project = root.join("project");
        std::fs::create_dir_all(agent_dir.join("sessions")).unwrap();
        std::fs::create_dir_all(project.join(".pi")).unwrap();
        std::fs::write(
            agent_dir.join("mcp.json"),
            serde_json::to_vec(&json!({
                "mcpServers": {
                    "internal": {
                        "command":"global-server",
                        "env":{"TOKEN":"global-secret"},
                        "exposure":"codemode"
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            project.join(".pi/mcp.json"),
            serde_json::to_vec(&json!({
                "autoEnableCodemode":false,
                "mcpServers": {
                    "internal":{"exposure":"direct"},
                    "local":{"command":"project-server"},
                    "missing":{"enabled":false}
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let (untrusted_servers, untrusted_auto_enable) = load_config(&agent_dir, &project).unwrap();
        assert!(untrusted_auto_enable);
        assert_eq!(untrusted_servers.len(), 1);
        assert_eq!(untrusted_servers[0].name, "internal");
        assert_eq!(untrusted_servers[0].exposure, Exposure::Codemode);

        let canonical_project = std::fs::canonicalize(&project).unwrap();
        std::fs::write(
            agent_dir.join("trust.json"),
            serde_json::to_vec(&json!({canonical_project.to_string_lossy().as_ref(): true}))
                .unwrap(),
        )
        .unwrap();
        let (trusted_servers, trusted_auto_enable) = load_config(&agent_dir, &project).unwrap();
        assert!(!trusted_auto_enable);
        assert_eq!(trusted_servers.len(), 2);
        let internal = trusted_servers
            .iter()
            .find(|server| server.name == "internal")
            .unwrap();
        let ServerTransportConfig::Stdio { command, env, .. } = &internal.transport else {
            panic!("expected stdio transport");
        };
        assert_eq!(command, "global-server");
        assert_eq!(env.get("TOKEN").map(String::as_str), Some("global-secret"));
        assert_eq!(internal.exposure, Exposure::Direct);
        assert!(trusted_servers.iter().any(|server| server.name == "local"));
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
      echo '{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"hello from MCP"}],"structuredContent":{"answer":42},"isError":true}}' ;;
  esac
done
"#;
        let config = ServerConfig {
            name: "test".into(),
            transport: ServerTransportConfig::Stdio {
                command: "sh".into(),
                args: vec!["-c".into(), script.into()],
                env: BTreeMap::new(),
                cwd: None,
            },
            exposure: Exposure::Codemode,
            tool_exposure: Vec::new(),
            timeout: DEFAULT_REQUEST_TIMEOUT,
        };
        let (server, tools) = connect_server(config, Path::new("."))
            .await
            .expect("connect stdio MCP server");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");

        let tool_name = "mcp__test__echo";
        let definition = json!({
            "type":"function",
            "name":tool_name,
            "description":"Return text",
            "parameters":{"type":"object","properties":{"text":{"type":"string"}}}
        });
        let mut registry = McpRegistry::default();
        registry.tools.insert(
            tool_name.into(),
            ExposedTool {
                server,
                server_name: "test".into(),
                remote_name: "echo".into(),
                exposure: Exposure::Codemode,
                definition,
            },
        );
        assert!(registry.codemode_enabled());
        assert!(registry.definitions_for_model(&HashSet::new()).is_empty());
        assert_eq!(registry.codemode_definitions().len(), 1);

        let mut deferred_registry = registry.clone();
        deferred_registry.tools.get_mut(tool_name).unwrap().exposure = Exposure::Deferred;
        assert!(deferred_registry.has_deferred_tools());
        let matches = deferred_registry.search_tools("return text", 8, &HashSet::new());
        assert_eq!(matches[0]["name"], tool_name);
        assert!(
            deferred_registry
                .definitions_for_model(&HashSet::new())
                .is_empty()
        );
        assert_eq!(
            deferred_registry
                .definitions_for_model(&HashSet::from([tool_name.to_owned()]))
                .len(),
            1
        );

        let (output, is_error, store_writes) = crate::codemode::execute(
            Path::new("."),
            &registry,
            &json!({}),
            &json!({"code":"const result = await tools.mcp__test__echo({ text: 'hello' }); text(result.answer); return result;"}),
            &HashMap::new(),
        )
        .await
        .expect("execute MCP tool through codemode");
        assert!(!is_error, "{output}");
        assert!(output.contains("42"), "{output}");
        assert!(store_writes.is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn registry_connects_all_configured_servers_concurrently() {
        let root = std::env::temp_dir().join(format!("mu-mcp-parallel-{}", uuid::Uuid::new_v4()));
        let ready = root.join("ready");
        std::fs::create_dir_all(&ready).unwrap();
        let mut servers = serde_json::Map::new();
        for (name, other) in [("alpha", "beta"), ("beta", "alpha")] {
            let script = r#"
while IFS= read -r line; do
  case "$line" in
    *initialize*)
      touch "$READY_DIR/$SERVER_NAME"
      i=0
      while [ ! -e "$READY_DIR/$OTHER_NAME" ] && [ "$i" -lt 200 ]; do
        i=$((i + 1)); sleep 0.01
      done
      [ -e "$READY_DIR/$OTHER_NAME" ] || exit 2
      echo '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"parallel-test","version":"1"}}}' ;;
    *tools/list*)
      echo '{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}' ;;
  esac
done
"#;
            servers.insert(
                name.to_owned(),
                json!({
                    "type":"stdio",
                    "command":"sh",
                    "args":["-c",script],
                    "env":{
                        "READY_DIR":ready.to_string_lossy(),
                        "SERVER_NAME":name,
                        "OTHER_NAME":other
                    }
                }),
            );
        }
        std::fs::write(
            root.join("mcp.json"),
            json!({"mcpServers":servers}).to_string(),
        )
        .unwrap();

        let registry =
            tokio::time::timeout(Duration::from_secs(5), McpRegistry::connect(&root, &root))
                .await
                .expect("parallel startup completed");
        assert!(registry.status_text().contains("alpha: connected"));
        assert!(registry.status_text().contains("beta: connected"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_tool_exposure_overrides_server_and_wildcards() {
        let overrides = vec![
            ("read*".to_owned(), Exposure::Hidden),
            ("read_file".to_owned(), Exposure::Direct),
        ];
        assert_eq!(
            tool_exposure_for(Exposure::Codemode, &overrides, "read_file"),
            Exposure::Direct
        );
        assert_eq!(
            tool_exposure_for(Exposure::Codemode, &overrides, "read_directory"),
            Exposure::Hidden
        );
        assert_eq!(
            tool_exposure_for(Exposure::Codemode, &overrides, "write_file"),
            Exposure::Codemode
        );
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
