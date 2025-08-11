extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};
use std::fs;
use std::io;
use std::process::Command;

fn get_catboost_version() -> String {
    env::var("CATBOOST_VERSION").unwrap_or_else(|_| "1.2.8".to_string())
}

fn get_platform_info() -> (String, String) {
    let target = env::var("TARGET").unwrap();
    
    // Determine OS
    let os = if target.contains("apple-darwin") {
        "macos"
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

fn download_and_extract_binary(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (os, arch) = get_platform_info();
    let version = get_catboost_version();
    
    // Construct download URL based on platform
    let download_url = match (os.as_str(), arch.as_str()) {
        ("linux", "x86_64") => format!(
            "https://github.com/catboost/catboost/releases/download/v{}/catboost-linux-x86_64-{}",
            version, version
        ),
        ("linux", "aarch64") => format!(
            "https://github.com/catboost/catboost/releases/download/v{}/catboost-linux-aarch64-{}",
            version, version
        ),
        ("macos", "x86_64") => format!(
            "https://github.com/catboost/catboost/releases/download/v{}/catboost-darwin-universal2-{}",
            version, version
        ),
        ("macos", "aarch64") => format!(
            "https://github.com/catboost/catboost/releases/download/v{}/catboost-darwin-universal2-{}",
            version, version
        ),
        ("windows", "x86_64") => format!(
            "https://github.com/catboost/catboost/releases/download/v{}/catboost-windows-x86_64-{}.exe",
            version, version
        ),
        _ => return Err(format!("Unsupported platform: {}-{}", os, arch).into()),
    };
    
    println!("cargo:warning=Downloading CatBoost v{} binary from: {}", version, download_url);
    
    // Create download directory
    let download_dir = out_dir.join("download");
    fs::create_dir_all(&download_dir)?;
    
    // Download the binary
    let response = ureq::get(&download_url).call()?;
    let status = response.status();
    if status < 200 || status >= 300 {
        return Err(format!("Failed to download binary: HTTP {}", status).into());
    }
    
    let archive_path = download_dir.join("catboost-archive");
    let mut file = fs::File::create(&archive_path)?;
    io::copy(&mut response.into_reader(), &mut file)?;
    
    // Extract the archive
    let extract_dir = out_dir.join("catboost");
    fs::create_dir_all(&extract_dir)?;
    
    if download_url.ends_with(".tar.gz") {
        let file = fs::File::open(&archive_path)?;
        let gz = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(gz);
        archive.unpack(&extract_dir)?;
    } else if download_url.ends_with(".zip") {
        let file = fs::File::open(&archive_path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        archive.extract(&extract_dir)?;
    } else {
        // For uncompressed files, just copy to the extract directory
        let final_path = extract_dir.join("catboost");
        fs::copy(&archive_path, &final_path)?;
        // Make executable on Unix systems
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&final_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&final_path, perms)?;
        }
    }
    
    // Clean up download
    fs::remove_file(archive_path)?;
    fs::remove_dir_all(download_dir)?;
    
    Ok(())
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cb_model_interface_root = Path::new("../../libs/model_interface/")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("."));

    // Download and extract the binary
    if let Err(e) = download_and_extract_binary(&out_dir) {
        eprintln!("Failed to download CatBoost binary: {}", e);
        eprintln!("Falling back to source build...");
        
        // Fallback to original source build
        let debug = env::var("DEBUG").unwrap();
        let mut build_native_args = vec![
            "../../../build/build_native.py",
            "--targets",
            "catboostmodel",
            "--build-root-dir",
            out_dir.to_str().unwrap(),
        ];
        if debug == "true" {
            build_native_args.push("--build-type=Debug");
        } else {
            build_native_args.push("--build-type=Release");
        }

        #[cfg(feature = "gpu")]
        build_native_args.push("--have-cuda");

        let build_cmd_status = Command::new("python")
            .args(&build_native_args)
            .status()
            .unwrap_or_else(|e| {
                panic!("Failed to run build_native.py : {}", e);
            });

        if !build_cmd_status.success() {
            panic!("Building with build_native.py failed");
        }
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

    // Try to find the library in the downloaded/extracted location first
    let lib_search_path = out_dir.join("catboost/libs/model_interface");
    if lib_search_path.exists() {
        println!(
            "cargo:rustc-link-search={}",
            lib_search_path.display()
        );
    } else {
        // Fallback to original path
        println!(
            "cargo:rustc-link-search={}",
            out_dir.join("catboost/libs/model_interface").display()
        );
    }

    println!("cargo:rustc-link-lib=dylib=catboostmodel");
}
