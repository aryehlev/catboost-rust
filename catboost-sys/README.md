# catboost-sys

This crate provides Rust bindings for CatBoost's C++ model interface library.

## Features

- **Binary Download**: By default, this crate downloads pre-compiled CatBoost binaries from GitHub releases instead of building from source
- **Fallback to Source Build**: If binary download fails, it falls back to the original source build approach
- **Multi-platform Support**: Supports Linux, macOS, and Windows on x86_64 and aarch64 architectures
- **GPU Support**: Optional GPU support via the `gpu` feature

## Configuration

### CatBoost Version

You can specify which version of CatBoost to download by setting the `CATBOOST_VERSION` environment variable:

```bash
export CATBOOST_VERSION=1.2.2
cargo build
```

If not specified, it defaults to version `1.2.8`.

### Features

- `gpu`: Enables GPU support (requires CUDA for source build fallback)

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
catboost-sys = { path = "catboost-sys" }

# For GPU support:
# catboost-sys = { path = "catboost-sys", features = ["gpu"] }
```

## How it Works

1. **Binary Download**: The build script attempts to download the appropriate pre-compiled binary for your platform from CatBoost's GitHub releases
2. **Extraction**: Downloads are automatically extracted to the build output directory
3. **Fallback**: If download fails (e.g., network issues, unsupported platform), it falls back to building from source using the original Python build script
4. **Binding Generation**: Rust bindings are generated using `bindgen` from the C++ headers

## Supported Platforms

- **Linux x86_64**: Downloads `catboost-linux-x86_64-{version}`
- **Linux aarch64**: Downloads `catboost-linux-aarch64-{version}`
- **macOS x86_64/aarch64**: Downloads `catboost-darwin-universal2-{version}` (universal binary)
- **Windows x86_64**: Downloads `catboost-windows-x86_64-{version}.exe`

## Troubleshooting

If you encounter issues with binary downloads:

1. Check your internet connection
2. Verify the CatBoost version exists in GitHub releases
3. The build will automatically fall back to source compilation
4. For unsupported platforms, the fallback source build will be used

## Dependencies

The build process requires these additional dependencies:
- `ureq`: For HTTP downloads
- `flate2`: For decompressing tar.gz files
- `tar`: For extracting tar archives
- `zip`: For extracting zip files (Windows)
