flamegraph:
    cargo build --profile profiling --example profile --target aarch64-apple-darwin
    samply record ./target/aarch64-apple-darwin/profiling/examples/profile
