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
