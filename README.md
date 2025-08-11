# catboost-rs

Rust bindings for CatBoost, a gradient boosting library for machine learning.

## Features

- **Binary Download**: Downloads pre-compiled CatBoost binaries instead of building from source
- **Multi-platform Support**: Works on Linux, macOS, and Windows
- **Fallback to Source Build**: Automatically falls back to source compilation if binary download fails
- **GPU Support**: Optional GPU support via feature flag
- **Comprehensive API**: Full access to CatBoost's model interface
- **Type Safety**: Rust's type system ensures safe usage

## Quick Start

### Prerequisites

- Rust (1.70 or later)
- Python (optional, for creating sample models)

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
catboost = { git = "https://github.com/your-username/catboost-rs" }

# For GPU support:
# catboost = { git = "https://github.com/your-username/catboost-rs", features = ["gpu"] }
```

### Basic Usage

```rust
use catboost::{Model, CatBoostError};

fn main() -> Result<(), CatBoostError> {
    // Load a model
    let model = Model::load("path/to/model.bin")?;
    
    // Make predictions
    let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let predictions = model.calc_model_prediction(features, vec![Vec::<String>::new()])?;
    
    println!("Prediction: {}", predictions[0]);
    Ok(())
}
```

## Examples

This repository includes comprehensive examples:

```bash
# Create sample models
python examples/create_sample_model.py

# Run examples
cargo run --example basic_usage
cargo run --example advanced_usage
```

See the [examples/](examples/) directory for detailed examples and documentation.

## Configuration

### CatBoost Version

You can specify which version of CatBoost to download by setting the `CATBOOST_VERSION` environment variable:

```bash
export CATBOOST_VERSION=1.2.8
cargo build
```

If not specified, it defaults to version `1.2.8`.

### Features

- `gpu`: Enables GPU support (requires CUDA for source build fallback)

## Supported Platforms

- **Linux x86_64**: Downloads `catboost-linux-x86_64-{version}`
- **Linux aarch64**: Downloads `catboost-linux-aarch64-{version}`
- **macOS x86_64/aarch64**: Downloads `catboost-darwin-universal2-{version}` (universal binary)
- **Windows x86_64**: Downloads `catboost-windows-x86_64-{version}.exe`

## How It Works

1. **Binary Download**: The build script attempts to download the appropriate pre-compiled binary for your platform from CatBoost's GitHub releases
2. **Extraction**: Downloads are automatically extracted to the build output directory
3. **Fallback**: If download fails (e.g., network issues, unsupported platform), it falls back to building from source using the original Python build script
4. **Binding Generation**: Rust bindings are generated using `bindgen` from the C++ headers

## API Reference

### Model

```rust
pub struct Model {
    // ...
}

impl Model {
    /// Load a model from a file
    pub fn load<P: AsRef<Path>>(path: P) -> CatBoostResult<Self>
    
    /// Load a model from a buffer
    pub fn load_buffer<P: AsRef<Vec<u8>>>(buffer: P) -> CatBoostResult<Self>
    
    /// Calculate model predictions
    pub fn calc_model_prediction<TFloatFeature, TFloatFeatures, TString, TCatFeature, TCatFeatures>(
        &self,
        float_features: TFloatFeatures,
        cat_features: TCatFeatures,
    ) -> CatBoostResult<Vec<f64>>
    
    /// Get number of float features
    pub fn get_float_features_count(&self) -> usize
    
    /// Get number of categorical features
    pub fn get_cat_features_count(&self) -> usize
    
    /// Get number of trees
    pub fn get_tree_count(&self) -> usize
    
    /// Get number of dimensions
    pub fn get_dimensions_count(&self) -> usize
}
```

## Building from Source

If you need to build from source (e.g., for custom CatBoost modifications):

1. Clone the CatBoost repository
2. Build the model interface library
3. Set up the build environment
4. Use the fallback source build

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [CatBoost](https://github.com/catboost/catboost) - The original gradient boosting library
- [bindgen](https://github.com/rust-lang/rust-bindgen) - For generating Rust bindings from C++ headers

## Troubleshooting

### Common Issues

1. **"No model file found"**
   - Make sure you've created a model using the Python script
   - Check that the model file exists in the expected location

2. **"Failed to load model"**
   - Ensure the model file is a valid CatBoost binary format
   - Check file permissions

3. **"Feature count mismatch"**
   - Make sure your input features match the model's expected feature count
   - Check the model's `num_features` property

4. **"Binary download failed"**
   - Check your internet connection
   - Verify the CatBoost version exists in GitHub releases
   - The build will automatically fall back to source compilation

### Getting Help

- Check the [examples/](examples/) directory for usage examples
- Review the API documentation in the source code
- Run `cargo test` to verify the library is working correctly
- Open an issue on GitHub for bugs or feature requests
