# wasmscanify

Rust WASM port of [jscanify](https://github.com/puffinsoft/jscanify) - a mobile document scanner library.

This is a Rust implementation that compiles to WebAssembly, with the goal of providing the same document scanning capabilities as jscanify.

## Installation

### Prerequisites

1. Install Rust: https://rustup.rs/
2. Install wasm-pack:
```bash
cargo install wasm-pack
```

### Building

```bash
# For web
wasm-pack build --target web --release

# For Node.js
wasm-pack build --target nodejs --release

# For bundlers (webpack, etc.)
wasm-pack build --target bundler --release
```

## License

MIT License - same as the original jscanify project.

## Roadmap

- Add multi-threading support with Web Workers
- Add gpu support
- Benchmarks
