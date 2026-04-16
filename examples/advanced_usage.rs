use catboost_rust::{CatBoostError, Model};
use std::fs;

fn main() -> Result<(), CatBoostError> {
    println!("CatBoost Rust Example - Advanced Usage");
    println!("======================================");

    // Check if we have a model file to load
    let model_path = "tmp/model.bin";
    if !fs::metadata(model_path).is_ok() {
        println!(
            "No model file found at {}. Please create a model first.",
            model_path
        );
        return Ok(());
    }

    // Load the model (prefer zero-copy when available)
    println!("Loading model from {}...", model_path);
    let model = load_model(model_path)?;

    println!("Model loaded successfully!");

    // Display comprehensive model information
    display_model_info(&model)?;

    // Example 1: Get model statistics
    println!("\n=== Model Statistics ===");
    println!("Model statistics:");
    println!(
        "  - Number of float features: {}",
        model.get_float_features_count()
    );
    println!(
        "  - Number of categorical features: {}",
        model.get_cat_features_count()
    );
    println!(
        "  - Number of text features: {}",
        model.get_text_features_count()
    );
    println!(
        "  - Number of embedding features: {}",
        model.get_embedding_features_count()
    );
    println!("  - Number of trees: {}", model.get_tree_count());
    println!("  - Number of dimensions: {}", model.get_dimensions_count());

    // Example 2: Prediction with different feature types
    println!("\n=== Feature Type Examples ===");

    // Numeric features only
    let numeric_features = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]];
    let prediction = model.calc_model_prediction(numeric_features, vec![Vec::<String>::new()])?;
    println!("Numeric features only:");
    println!("  Features: {:?}", vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    println!("  Prediction: {:.6}", prediction[0]);

    // Mixed numeric and categorical features
    let numeric_features = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5]];
    let categorical_features = vec![vec![
        String::from("A"),
        String::from("B"),
        String::from("C"),
    ]];
    let prediction = model.calc_model_prediction(numeric_features, categorical_features)?;
    println!("Mixed features:");
    println!("  Numeric: {:?}", vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    println!("  Categorical: {:?}", vec!["A", "B", "C"]);
    println!("  Prediction: {:.6}", prediction[0]);

    // Example 3: Batch predictions with error handling
    println!("\n=== Batch Predictions ===");
    let batch_features = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![2.0, 3.0, 4.0, 5.0, 6.0],
        vec![3.0, 4.0, 5.0, 6.0, 7.0],
        vec![4.0, 5.0, 6.0, 7.0, 8.0],
    ];

    match model.calc_model_prediction(
        batch_features,
        vec![
            Vec::<String>::new(),
            Vec::<String>::new(),
            Vec::<String>::new(),
            Vec::<String>::new(),
        ],
    ) {
        Ok(predictions) => {
            for (i, pred) in predictions.iter().enumerate() {
                println!(
                    "  Sample {}: {:?} -> {:.6}",
                    i + 1,
                    vec![
                        1.0 + i as f32,
                        2.0 + i as f32,
                        3.0 + i as f32,
                        4.0 + i as f32,
                        5.0 + i as f32
                    ],
                    pred
                );
            }
        }
        Err(e) => println!("  Batch prediction error: {}", e),
    }

    // Example 4: Model validation
    println!("\n=== Model Validation ===");
    validate_model(&model)?;

    println!("\nAdvanced examples completed successfully!");
    Ok(())
}

#[cfg(catboost_zero_copy)]
fn load_model(path: &str) -> Result<Model, CatBoostError> {
    println!("  (using zero-copy buffer loading)");
    let buffer = fs::read(path).map_err(|e| CatBoostError {
        description: format!("could not read file into memory: {}", e),
    })?;
    Model::load_buffer_zero_copy(buffer)
}

#[cfg(not(catboost_zero_copy))]
fn load_model(path: &str) -> Result<Model, CatBoostError> {
    println!("  (using file loading - zero-copy not available in this CatBoost version)");
    Model::load(path)
}

fn display_model_info(model: &Model) -> Result<(), CatBoostError> {
    println!("Model Information:");
    println!(
        "  - Number of float features: {}",
        model.get_float_features_count()
    );
    println!(
        "  - Number of categorical features: {}",
        model.get_cat_features_count()
    );
    println!(
        "  - Number of text features: {}",
        model.get_text_features_count()
    );
    println!(
        "  - Number of embedding features: {}",
        model.get_embedding_features_count()
    );
    println!("  - Number of trees: {}", model.get_tree_count());
    println!("  - Number of dimensions: {}", model.get_dimensions_count());

    Ok(())
}

fn validate_model(model: &Model) -> Result<(), CatBoostError> {
    println!("Validating model...");

    // Test with valid features (should succeed)
    let num_features = model.get_float_features_count();
    let valid_features = vec![vec![0.0; num_features]];
    match model.calc_model_prediction(valid_features, vec![Vec::<String>::new()]) {
        Ok(predictions) => println!(
            "  ✅ Valid features accepted, prediction: {:.6}",
            predictions[0]
        ),
        Err(e) => println!("  ❌ Valid features failed: {}", e),
    }

    Ok(())
}
