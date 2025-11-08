extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};
use std::fs;
use std::io;

fn get_catboost_version() -> String {
    env::var("CATBOOST_VERSION").unwrap_or_else(|_| "1.2.8".to_string())
}

fn get_platform_info() -> (String, String) {
    let target = env::var("TARGET").unwrap();
    
    // Determine OS
    let os = if target.contains("apple-darwin") {
        "darwin"
    } else if target.contains("linux") {
        "linux"
    } else if target.contains("windows") {
        "windows"
    } else {
        panic!("Unsupported target: {}", target);
    };
    
    // Determine architecture
    let arch = if target.contains("x86_64") {
        "x86_64"
    } else if target.contains("aarch64") || target.contains("arm64") {
        "aarch64"
    } else if target.contains("i686") || target.contains("i586") {
        "i686"
    } else {
        panic!("Unsupported architecture for target: {}", target);
    };
    
    (os.to_string(), arch.to_string())
}

fn download_model_interface_headers(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let version = get_catboost_version();
    
    // Create the model_interface directory
    let model_interface_dir = out_dir.join("libs/model_interface");
    fs::create_dir_all(&model_interface_dir)?;
    
    // Download the c_api.h file
    let c_api_url = format!(
        "https://raw.githubusercontent.com/catboost/catboost/v{}/catboost/libs/model_interface/c_api.h",
        version
    );
    
    println!("cargo:warning=Downloading c_api.h from: {}", c_api_url);
    
    let response = ureq::get(&c_api_url).call()?;
    let status = response.status();
    if status < 200 || status >= 300 {
        return Err(format!("Failed to download c_api.h: HTTP {}", status).into());
    }
    
    let c_api_path = model_interface_dir.join("c_api.h");
    let mut file = fs::File::create(&c_api_path)?;
    io::copy(&mut response.into_reader(), &mut file)?;
    
    Ok(())
}

fn download_compiled_library(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (os, arch) = get_platform_info();
    let version = get_catboost_version();

    // Parse version to determine URL format
    // v1.0.x - v1.1.x use simple filenames
    // v1.2+ use platform-specific versioned filenames
    let version_parts: Vec<&str> = version.split('.').collect();
    let major: u32 = version_parts.get(0).and_then(|s| s.parse().ok()).unwrap_or(1);
    let minor: u32 = version_parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    let use_new_format = major > 1 || (major == 1 && minor >= 2);

    // Determine download URL based on version and platform
    let (lib_filename, download_url) = if use_new_format {
        // v1.2+ format with platform and version in filename
        match (os.as_str(), arch.as_str()) {
            ("linux", "x86_64") => (
                "libcatboostmodel.so".to_string(),
                format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/libcatboostmodel-linux-x86_64-{}.so",
                    version, version
                ),
            ),
            ("linux", "aarch64") => (
                "libcatboostmodel.so".to_string(),
                format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/libcatboostmodel-linux-aarch64-{}.so",
                    version, version
                ),
            ),
            ("darwin", "x86_64") | ("darwin", "aarch64") => (
                "libcatboostmodel.dylib".to_string(),
                format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/libcatboostmodel-darwin-universal2-{}.dylib",
                    version, version
                ),
            ),
            ("windows", "x86_64") => (
                "catboostmodel.dll".to_string(),
                format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/catboostmodel.dll",
                    version
                ),
            ),
            _ => return Err(format!("Unsupported platform: {}-{}", os, arch).into()),
        }
    } else {
        // v1.0.x - v1.1.x format with simple filenames
        match os.as_str() {
            "linux" => (
                "libcatboostmodel.so".to_string(),
                format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/libcatboostmodel.so",
                    version
                ),
            ),
            "darwin" => (
                "libcatboostmodel.dylib".to_string(),
                format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/libcatboostmodel.dylib",
                    version
                ),
            ),
            "windows" => (
                "catboostmodel.dll".to_string(),
                format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/catboostmodel.dll",
                    version
                ),
            ),
            _ => return Err(format!("Unsupported platform: {}", os).into()),
        }
    };

    println!(
        "cargo:warning=Downloading CatBoost v{} library from: {}",
        version, download_url
    );

    // Create the library directory
    let lib_dir = out_dir.join("libs");
    fs::create_dir_all(&lib_dir)?;

    // Download the library directly into the `libs` directory with its correct name
    let lib_path = lib_dir.join(&lib_filename);
    let mut dest = fs::File::create(&lib_path)?;

    let response = ureq::get(&download_url).call()?;
    let status = response.status();
    if status < 200 || status >= 300 {
        return Err(format!("Failed to download library: HTTP {}", status).into());
    }

    // SIMPLIFIED: No need for extraction, just copy the downloaded content
    io::copy(&mut response.into_reader(), &mut dest)?;

    println!(
        "cargo:warning=Downloaded CatBoost library to: {}",
        lib_path.display()
    );

    Ok(())
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cb_model_interface_root = out_dir.join("libs/model_interface");

    // Parse version for feature detection
    let version = get_catboost_version();
    let version_parts: Vec<&str> = version.split('.').collect();
    let major: u32 = version_parts.get(0).and_then(|s| s.parse().ok()).unwrap_or(1);
    let minor: u32 = version_parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let patch: u32 = version_parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    // Emit cfg flags for version-specific features
    // v1.1.1+: Embedding features support
    if major > 1 || (major == 1 && minor > 1) || (major == 1 && minor == 1 && patch >= 1) {
        println!("cargo:rustc-cfg=catboost_embeddings");
    }

    // v1.2+: Text features count function
    if major > 1 || (major == 1 && minor >= 2) {
        println!("cargo:rustc-cfg=catboost_text_count");
    }

    // v1.2.3+: Staged predictions and feature indices
    if major > 1 || (major == 1 && minor > 2) || (major == 1 && minor == 2 && patch >= 3) {
        println!("cargo:rustc-cfg=catboost_staged_prediction");
        println!("cargo:rustc-cfg=catboost_feature_indices");
    }

    // Download the model interface headers
    if let Err(e) = download_model_interface_headers(&out_dir) {
        eprintln!("Failed to download model interface headers: {}", e);
        panic!("Cannot proceed without headers");
    }

    // Download the compiled library
    if let Err(e) = download_compiled_library(&out_dir) {
        eprintln!("Failed to download compiled library: {}", e);
        panic!("Cannot proceed without compiled library");
    }

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", cb_model_interface_root.display()))
        .size_t_is_usize(true)
        .rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate bindings.");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    // 1. Get platform info using your existing function
    let (os, _arch) = get_platform_info();

    // 2. Determine the library filename based on the OS
    let lib_filename = match os.as_str() {
        "windows" => "catboostmodel.dll",
        "darwin" => "libcatboostmodel.dylib", // "darwin" comes from your function
        _ => "libcatboostmodel.so", // Default to Linux/Unix
    };

    // 3. Copy the library from OUT_DIR/libs to the final target directory
    let lib_source_path = out_dir.join("libs").join(lib_filename);

    // Find the final output directory (e.g., target/release)
    let target_dir = out_dir.ancestors().find(|p| p.ends_with("target")).unwrap().join(env::var("PROFILE").unwrap());

    let lib_dest_path = target_dir.join(lib_filename);
    fs::copy(&lib_source_path, &lib_dest_path).expect("Failed to copy library to target directory");

    // 4. Set the library search path for the build-time linker
    let lib_search_path = out_dir.join("libs");
    println!(
        "cargo:rustc-link-search=native={}",
        lib_search_path.display()
    );

    // 5. Set the rpath for the run-time linker based on the OS
    match os.as_str() {
        "darwin" => {
            // For macOS, add multiple rpath entries for IDE compatibility
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../..");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_search_path.display());
            // Add the target directory to rpath as well
            if let Some(target_root) = out_dir.ancestors().find(|p| p.ends_with("target")) {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}/debug", target_root.display());
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}/release", target_root.display());
            }
        },
        "linux" => {
            // For Linux, use $ORIGIN
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../..");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_search_path.display());
        },
        _ => {} // No rpath needed for Windows
    }

    println!("cargo:rustc-link-lib=dylib=catboostmodel");
}
