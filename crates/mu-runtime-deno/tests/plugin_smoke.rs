use mu_runtime_deno::PluginRuntime;

#[test]
fn evaluates_a_javascript_plugin_source() {
    let mut runtime = PluginRuntime::new();

    runtime
        .evaluate(
            "file:///extensions/smoke.js",
            "globalThis.pluginLoaded = true;",
        )
        .expect("plugin source should execute");
}

#[test]
fn strips_typescript_types_before_evaluating_a_plugin_script() {
    let mut runtime = PluginRuntime::new();

    runtime
        .evaluate(
            "file:///extensions/smoke.ts",
            "interface Pair { left: number; right: number }\nconst add = (pair: Pair): number => pair.left + pair.right;\nglobalThis.pluginResult = add({ left: 20, right: 22 });",
        )
        .expect("TypeScript types should be removed before execution");
    runtime
        .evaluate(
            "file:///extensions/check.js",
            "if (globalThis.pluginResult !== 42) throw new Error('wrong TypeScript result');",
        )
        .expect("transpiled TypeScript should execute with its expected value");
}

#[test]
fn typescript_parse_errors_are_reported_as_runtime_errors() {
    let mut runtime = PluginRuntime::new();

    let error = runtime
        .evaluate("file:///extensions/broken.ts", "const value: = 1;")
        .expect_err("invalid TypeScript should not execute");

    assert!(error.to_string().contains("broken.ts"));
}
