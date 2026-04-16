use catboost_rust::{CatBoostError, Model};
use std::fs;

fn main() -> Result<(), CatBoostError> {
    println!("CatBoost Rust Example - Basic Usage");
    println!("===================================");

    // Check if we have a model file to load
    let model_path = "tmp/model.bin";
    if !fs::metadata(model_path).is_ok() {
        println!(
            "No model file found at {}. Creating a simple example...",
            model_path
        );
        create_simple_example()?;
        return Err(CatBoostError {
            description: "No model file found at {}. Creating a simple example..".to_string(),
        });
    }

    // Load the model (prefer zero-copy when available)
    println!("Loading model from {}...", model_path);
    let model = load_model(model_path)?;

    println!("Model loaded successfully!");
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

    // Example 1: Simple prediction with numeric features
    println!("\nExample 1: Numeric features prediction");
    let numeric_features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let prediction = model.calc_model_prediction(numeric_features, vec![Vec::<String>::new()])?;
    println!("  Input features: {:?}", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("  Prediction: {:.6}", prediction[0]);

    // Example 2: Prediction with categorical features
    println!("\nExample 2: Categorical features prediction");
    let numeric_features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let categorical_features = vec![vec![
        String::from("A"),
        String::from("B"),
        String::from("C"),
    ]];
    let prediction = model.calc_model_prediction(numeric_features, categorical_features)?;
    println!("  Numeric features: {:?}", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("  Categorical features: {:?}", vec!["A", "B", "C"]);
    println!("  Prediction: {:.6}", prediction[0]);

    // Example 3: Batch prediction
    println!("\nExample 3: Batch prediction");
    let batch_features = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![2.0, 3.0, 4.0, 5.0, 6.0],
        vec![3.0, 4.0, 5.0, 6.0, 7.0],
    ];

    let predictions = model.calc_model_prediction(
        batch_features,
        vec![
            Vec::<String>::new(),
            Vec::<String>::new(),
            Vec::<String>::new(),
        ],
    )?;
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

    println!("\nAll examples completed successfully!");
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

fn create_simple_example() -> Result<(), CatBoostError> {
    println!("Since no model file is available, here's how you would use the library:");
    println!();
    println!("1. Train a CatBoost model using Python or other tools");
    println!("2. Save it as 'tmp/model.bin'");
    println!("3. Run this example again");
    println!();
    println!("Example Python code to create a model:");
    println!("```python");
    println!("from catboost import CatBoostRegressor");
    println!("import numpy as np");
    println!();
    println!("# Create sample data");
    println!("X = np.random.rand(100, 5)");
    println!("y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 100)");
    println!();
    println!("# Train model");
    println!("model = CatBoostRegressor(iterations=100, depth=3, verbose=False)");
    println!("model.fit(X, y)");
    println!();
    println!("# Save model");
    println!("model.save_model('tmp/model.bin')");
    println!("```");
    Ok(())
}
