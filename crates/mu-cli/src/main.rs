#[tokio::main]
async fn main() {
    if let Err(error) = mu_core::run_cli().await {
        eprintln!("mu: {error}");
        std::process::exit(1);
    }
}
