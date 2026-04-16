# CatBoost Rust Bindings

Rust bindings for [CatBoost](https://catboost.ai/), a gradient boosting library for machine learning. This crate provides a safe and ergonomic Rust interface to CatBoost's C API.

## Features

- **Cross-platform**: Works on Linux, macOS, and Windows
- **Self-contained**: Downloads CatBoost binaries at runtime - no system dependencies required
- **Version control**: Specify different CatBoost versions via environment variable
- **Safe Rust API**: Memory-safe wrapper around CatBoost's C API
- **Multiple feature types**: Support for numeric, categorical, text, and embedding features
- **GPU support**: Optional GPU acceleration (requires `gpu` feature)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
catboost-rust = "0.2.0"
```

For GPU support:

```toml
[dependencies]
catboost-rust = { version = "0.2.0", features = ["gpu"] }
```

## Quick Start

```rust
use catboost_rust::{Model, ObjectsOrderFeatures};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a trained CatBoost model
    let model = Model::load("path/to/model.cbm")?;
    
    // Make predictions with numeric features
    let features = ObjectsOrderFeatures::new()
        .with_float_features(&[
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[2.0, 3.0, 4.0, 5.0, 6.0],
        ]);
    
    let predictions = model.predict(features)?;
    println!("Predictions: {:?}", predictions);
    
    Ok(())
}
```

## Usage Examples

### Basic Usage

```rust
use catboost_rust::{Model, ObjectsOrderFeatures};

// Load model from file
let model = Model::load("model.cbm")?;

// Simple numeric features prediction
let features = ObjectsOrderFeatures::new()
    .with_float_features(&[&[1.0, 2.0, 3.0, 4.0, 5.0]]);

let predictions = model.predict(features)?;
```

### Categorical Features

```rust
use catboost_rust::{Model, ObjectsOrderFeatures};

let model = Model::load("model.cbm")?;

// Mixed numeric and categorical features
let features = ObjectsOrderFeatures::new()
    .with_float_features(&[&[1.0, 2.0, 3.0]])
    .with_cat_features(&[&["A", "B", "C"]]);

let predictions = model.predict(features)?;
```

### Text Features

```rust
use catboost_rust::{Model, ObjectsOrderFeatures};
use std::ffi::CString;

let model = Model::load("model.cbm")?;

let text_features = vec![
    CString::new("This is a sample text").unwrap(),
    CString::new("Another text sample").unwrap(),
];

let features = ObjectsOrderFeatures::new()
    .with_float_features(&[&[1.0, 2.0]])
    .with_text_features(&[&text_features]);

let predictions = model.predict(features)?;
```

### Embedding Features

```rust
use catboost_rust::{Model, ObjectsOrderFeatures};

let model = Model::load("model.cbm")?;

let embeddings = vec![
    vec![0.1, 0.2, 0.3, 0.4], // First embedding
    vec![0.5, 0.6, 0.7, 0.8], // Second embedding
];

let features = ObjectsOrderFeatures::new()
    .with_float_features(&[&[1.0, 2.0]])
    .with_embedding_features(&[&embeddings]);

let predictions = model.predict(features)?;
```

### Zero-Copy Buffer Loading (Recommended)

`Model::load_buffer_zero_copy` is the recommended way to load models from memory. Unlike `load_buffer`, it avoids copying the model data, resulting in lower memory usage, faster loading, and no internal memory pool leaks. Requires CatBoost v1.2.9+ (the default).

```rust
use catboost_rust::Model;
use std::fs;

let buffer = fs::read("model.cbm")?;
let model = Model::load_buffer_zero_copy(buffer)?;
```

The buffer is owned by the `Model` and freed automatically when it is dropped.

## Configuration

### CatBoost Version

You can specify which version of CatBoost to use by setting the `CATBOOST_VERSION` environment variable:

```bash
export CATBOOST_VERSION=1.2.8
cargo build
```

The default version is `1.2.8`.

### GPU Support

To enable GPU acceleration, compile with the `gpu` feature:

```bash
cargo build --features gpu
```

Then enable GPU evaluation in your code:

```rust
let model = Model::load("model.cbm")?;
model.enable_gpu_evaluation()?;
```

## Model Information

You can inspect model properties:

```rust
let model = Model::load("model.cbm")?;

println!("Float features: {}", model.get_float_features_count());
println!("Categorical features: {}", model.get_cat_features_count());
println!("Text features: {}", model.get_text_features_count());
println!("Embedding features: {}", model.get_embedding_features_count());
println!("Trees: {}", model.get_tree_count());
println!("Dimensions: {}", model.get_dimensions_count());
```

## Error Handling

The crate provides comprehensive error handling:

```rust
use catboost_rust::{Model, CatBoostError, CatBoostResult};

fn load_and_predict() -> CatBoostResult<Vec<f64>> {
    let model = Model::load("model.cbm")?;
    
    let features = ObjectsOrderFeatures::new()
        .with_float_features(&[&[1.0, 2.0, 3.0]]);
    
    model.predict(features)
}

match load_and_predict() {
    Ok(predictions) => println!("Success: {:?}", predictions),
    Err(CatBoostError { description }) => println!("Error: {}", description),
}
```

## Platform Support

This crate automatically downloads the appropriate CatBoost binary for your platform:

- **Linux**: x86_64, aarch64
- **macOS**: Universal binary (x86_64 + arm64)
- **Windows**: x86_64

## Building from Source

The crate downloads CatBoost binaries at runtime, so no system dependencies are required. However, if you want to build from source, you can set the `CATBOOST_BUILD_FROM_SOURCE` environment variable.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Examples

See the `examples/` directory for more detailed usage examples:

- `basic_usage.rs` - Simple prediction examples
- `advanced_usage.rs` - Advanced features and model inspection

Run examples with:

```bash
cargo run --example basic_usage
cargo run --example advanced_usage
```
