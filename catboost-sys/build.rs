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
    
    // Construct download URL for the compiled library
    let download_url = match (os.as_str(), arch.as_str()) {
        ("linux", "x86_64") => format!(
            "https://github.com/catboost/catboost/releases/download/v{}/catboost-linux-x86_64-{}",
            version, version
        ),
        ("linux", "aarch64") => format!(
            "https://github.com/catboost/catboost/releases/download/v{}/catboost-linux-aarch64-{}",
            version, version
        ),
        ("darwin", "x86_64") => format!(
            "https://github.com/catboost/catboost/releases/download/v{}/catboost-darwin-universal2-{}",
            version, version
        ),
        ("darwin", "aarch64") => format!(
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
    
    let archive_path = download_dir.join("catboost-binary");
    let mut file = fs::File::create(&archive_path)?;
    io::copy(&mut response.into_reader(), &mut file)?;
    
    // Extract or copy the binary to the appropriate location
    let lib_dir = out_dir.join("libs");
    fs::create_dir_all(&lib_dir)?;
    
    if download_url.ends_with(".tar.gz") {
        let file = fs::File::open(&archive_path)?;
        let gz = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(gz);
        archive.unpack(&lib_dir)?;
    } else if download_url.ends_with(".zip") {
        let file = fs::File::open(&archive_path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        archive.extract(&lib_dir)?;
    } else {
        // For uncompressed files, just copy to the lib directory
        let final_path = lib_dir.join("catboost");
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
    let cb_model_interface_root = out_dir.join("libs/model_interface");

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

    // Set up library search path
    let lib_search_path = out_dir.join("libs");
    if lib_search_path.exists() {
        println!(
            "cargo:rustc-link-search={}",
            lib_search_path.display()
        );
    }

    println!("cargo:rustc-link-lib=dylib=catboostmodel");
}
