#!/usr/bin/env python3
"""
Script to create a sample CatBoost model for testing the Rust examples.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
import os

def create_regression_model():
    """Create a sample regression model."""
    print("Creating sample regression model...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create target (non-linear function of features)
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.5 + 
         X[:, 0] * X[:, 1] * 0.3 + np.random.normal(0, 0.1, n_samples))
    
    # Train model
    model = CatBoostRegressor(
        iterations=100,
        depth=4,
        learning_rate=0.1,
        loss_function='RMSE',
        verbose=False
    )
    
    model.fit(X, y)
    
    # Save model
    os.makedirs('tmp', exist_ok=True)
    model.save_model('tmp/model.bin')
    print("Regression model saved to tmp/model.bin")
    
    # Test prediction
    test_features = [1.0, 2.0, 3.0, 4.0, 5.0]
    prediction = model.predict([test_features])[0]
    print(f"Test prediction for {test_features}: {prediction:.6}")
    
    return model

def create_classification_model():
    """Create a sample classification model."""
    print("\nCreating sample classification model...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create categorical features
    categorical_features = np.random.choice(['A', 'B', 'C'], size=(n_samples, 3))
    
    # Create target (binary classification)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Combine numeric and categorical features
    X_combined = np.column_stack([X, categorical_features])
    
    # Train model
    model = CatBoostClassifier(
        iterations=100,
        depth=4,
        learning_rate=0.1,
        loss_function='Logloss',
        verbose=False
    )
    
    model.fit(X_combined, y, cat_features=[5, 6, 7])
    
    # Save model
    model.save_model('tmp/classification_model.bin')
    print("Classification model saved to tmp/classification_model.bin")
    
    # Test prediction
    test_features = [1.0, 2.0, 3.0, 4.0, 5.0]
    test_cat_features = ['A', 'B', 'C']
    test_combined = test_features + test_cat_features
    prediction = model.predict_proba([test_combined])[0]
    print(f"Test prediction for {test_combined}: {prediction}")
    
    return model

def main():
    """Main function to create sample models."""
    print("CatBoost Sample Model Generator")
    print("===============================")
    
    try:
        # Create regression model
        regression_model = create_regression_model()
        
        # Create classification model
        classification_model = create_classification_model()
        
        print("\n✅ Sample models created successfully!")
        print("\nYou can now run the Rust examples:")
        print("  cargo run --example basic_usage")
        print("  cargo run --example advanced_usage")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Please install CatBoost Python package:")
        print("  pip install catboost")
    except Exception as e:
        print(f"❌ Error creating models: {e}")

if __name__ == "__main__":
    main()
