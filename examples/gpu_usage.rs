use catboost_rust::{Model, ObjectsOrderFeatures};

#[cfg(catboost_zero_copy)]
fn load_model(path: &str) -> Result<Model, Box<dyn std::error::Error>> {
    println!("  (using zero-copy buffer loading)");
    let buffer = std::fs::read(path)?;
    Ok(Model::load_buffer_zero_copy(buffer)?)
}

#[cfg(not(catboost_zero_copy))]
fn load_model(path: &str) -> Result<Model, Box<dyn std::error::Error>> {
    println!("  (using file loading - zero-copy not available in this CatBoost version)");
    Ok(Model::load(path)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CatBoost Rust Example - GPU Usage");
    println!("==================================");

    // Load a model (prefer zero-copy when available)
    println!("Loading model from tmp/model.bin...");
    let model = load_model("tmp/model.bin")?;
    println!("Model loaded successfully!");

    // Display model information
    println!("Model info:");
    println!(
        "  - Number of float features: {}",
        model.get_float_features_count()
    );
    println!(
        "  - Number of categorical features: {}",
        model.get_cat_features_count()
    );
    println!("  - Number of trees: {}", model.get_tree_count());
    println!("  - Number of dimensions: {}", model.get_dimensions_count());
    println!();

    // Enable GPU evaluation
    println!("Enabling GPU evaluation...");
    match model.enable_gpu_evaluation() {
        Ok(()) => println!("GPU evaluation enabled successfully!"),
        Err(e) => {
            println!("Warning: Failed to enable GPU evaluation: {}", e);
            println!("This is normal if no GPU is available or CUDA is not installed.");
            println!("The model will continue to use CPU evaluation.");
        }
    }
    println!();

    // Example 1: Basic prediction with GPU
    println!("Example 1: Basic prediction with GPU acceleration");
    let features = ObjectsOrderFeatures::new().with_float_features(&[
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[2.0, 3.0, 4.0, 5.0, 6.0],
        &[3.0, 4.0, 5.0, 6.0, 7.0],
    ]);

    let predictions = model.predict(features)?;
    println!(
        "  Sample 1: [1.0, 2.0, 3.0, 4.0, 5.0] -> {:.6}",
        predictions[0]
    );
    println!(
        "  Sample 2: [2.0, 3.0, 4.0, 5.0, 6.0] -> {:.6}",
        predictions[1]
    );
    println!(
        "  Sample 3: [3.0, 4.0, 5.0, 6.0, 7.0] -> {:.6}",
        predictions[2]
    );
    println!();

    // Example 2: Batch prediction with mixed features
    println!("Example 2: Batch prediction with mixed features (GPU accelerated)");
    let features = ObjectsOrderFeatures::new()
        .with_float_features(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]])
        .with_cat_features(&[&["A", "B", "C"], &["D", "E", "F"], &["G", "H", "I"]]);

    let predictions = model.predict(features)?;
    println!(
        "  Sample 1: [1.0, 2.0, 3.0] + [\"A\", \"B\", \"C\"] -> {:.6}",
        predictions[0]
    );
    println!(
        "  Sample 2: [4.0, 5.0, 6.0] + [\"D\", \"E\", \"F\"] -> {:.6}",
        predictions[1]
    );
    println!(
        "  Sample 3: [7.0, 8.0, 9.0] + [\"G\", \"H\", \"I\"] -> {:.6}",
        predictions[2]
    );
    println!();

    println!("All examples completed successfully!");
    println!();
    println!("Note: GPU acceleration provides significant speedup for large batch predictions.");
    println!("The actual speedup depends on your GPU hardware and the size of your data.");

    Ok(())
}
