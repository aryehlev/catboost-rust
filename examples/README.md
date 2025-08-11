# CatBoost Rust Examples

This directory contains example programs demonstrating how to use the CatBoost Rust package.

## Examples Overview

### 1. Basic Usage (`basic_usage.rs`)
A simple example that shows:
- Loading a CatBoost model from file
- Making predictions with numeric features
- Making predictions with categorical features
- Batch predictions
- Basic error handling

### 2. Advanced Usage (`advanced_usage.rs`)
A more comprehensive example that demonstrates:
- Detailed model information and statistics
- Different types of feature inputs
- Batch predictions with error handling
- Model validation
- Advanced error handling patterns

## Getting Started

### Prerequisites

1. **Rust**: Make sure you have Rust installed
2. **Python**: For creating sample models (optional)
3. **CatBoost Python package**: For model creation

### Step 1: Create Sample Models

First, create some sample models to test with:

```bash
# Install CatBoost Python package (if not already installed)
pip install catboost numpy pandas

# Create sample models
python examples/create_sample_model.py
```

This will create:
- `tmp/model.bin` - A regression model
- `tmp/classification_model.bin` - A classification model

### Step 2: Run the Examples

```bash
# Run basic usage example
cargo run --example basic_usage

# Run advanced usage example
cargo run --example advanced_usage
```

## Example Output

### Basic Usage Example
```
CatBoost Rust Example - Basic Usage
===================================
Loading model from tmp/model.bin...
Model loaded successfully!
Model info:
  - Number of features: 5
  - Number of trees: 100
  - Model type: Regression

Example 1: Numeric features prediction
  Input features: [1.0, 2.0, 3.0, 4.0, 5.0]
  Prediction: 4.123456

Example 2: Categorical features prediction
  Numeric features: [1.0, 2.0, 3.0]
  Categorical features: [Some("category1"), Some("category2"), None]
  Prediction: 2.987654

Example 3: Batch prediction
  Sample 1: [1.0, 2.0, 3.0, 4.0, 5.0] -> 4.123456
  Sample 2: [2.0, 3.0, 4.0, 5.0, 6.0] -> 5.234567
  Sample 3: [3.0, 4.0, 5.0, 6.0, 7.0] -> 6.345678

All examples completed successfully!
```

### Advanced Usage Example
```
CatBoost Rust Example - Advanced Usage
======================================
Loading model from tmp/model.bin...
Model loaded successfully!
Model Information:
  - Number of features: 5
  - Number of trees: 100
  - Model type: Regression
  - Prediction dimension: 1
  - Model type (from stats): Regression

=== Model Statistics ===
Model statistics:
  - Number of features: 5
  - Number of trees: 100
  - Model type: Regression
  - Prediction dimension: 1

=== Feature Type Examples ===
Numeric features only:
  Features: [0.1, 0.2, 0.3, 0.4, 0.5]
  Prediction: 0.987654

Mixed features:
  Numeric: [0.1, 0.2, 0.3]
  Categorical: [Some("A"), Some("B"), Some("C")]
  Prediction: 0.456789

=== Batch Predictions ===
  Sample 1: [1.0, 2.0, 3.0, 4.0, 5.0] -> 4.123456
  Sample 2: [2.0, 3.0, 4.0, 5.0, 6.0] -> 5.234567
  Sample 3: [3.0, 4.0, 5.0, 6.0, 7.0] -> 6.345678
  Sample 4: [4.0, 5.0, 6.0, 7.0, 8.0] -> 7.456789

=== Model Validation ===
Validating model...
  ✅ Empty features correctly rejected
  ✅ Too many features correctly rejected
  ✅ Valid features accepted, prediction: 0.000000

Advanced examples completed successfully!
```

## Using Your Own Models

To use your own CatBoost models:

1. **Train a model** using CatBoost Python, R, or other tools
2. **Save the model** in CatBoost binary format (`.bin` file)
3. **Place the model file** in the `tmp/` directory
4. **Update the model path** in the example code if needed
5. **Run the examples**

### Example Python Code for Model Creation

```python
from catboost import CatBoostRegressor
import numpy as np

# Create sample data
X = np.random.rand(100, 5)
y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 100)

# Train model
model = CatBoostRegressor(iterations=100, depth=3, verbose=False)
model.fit(X, y)

# Save model
model.save_model('tmp/my_model.bin')
```

## Troubleshooting

### Common Issues

1. **"No model file found"**
   - Make sure you've created a model using the Python script
   - Check that the model file exists in the `tmp/` directory

2. **"Failed to load model"**
   - Ensure the model file is a valid CatBoost binary format
   - Check file permissions

3. **"Feature count mismatch"**
   - Make sure your input features match the model's expected feature count
   - Check the model's `num_features` property

4. **"Categorical feature error"**
   - Ensure categorical features are provided as strings
   - Use `None` for missing categorical values

### Getting Help

- Check the main project README for more information
- Review the API documentation in the source code
- Run `cargo test` to verify the library is working correctly
