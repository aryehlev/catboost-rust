#!/bin/bash

# catboost-rs setup script
# This script helps set up the development environment for catboost-rs

set -e

echo "🚀 Setting up catboost-rs development environment..."
echo "=================================================="

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust is not installed. Please install Rust first:"
    echo "   https://rustup.rs/"
    exit 1
fi

echo "✅ Rust is installed: $(cargo --version)"

# Check if Python is installed (optional)
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 is installed: $(python3 --version)"
    
    # Check if catboost Python package is installed
    if python3 -c "import catboost" &> /dev/null; then
        echo "✅ CatBoost Python package is installed"
    else
        echo "⚠️  CatBoost Python package not found. Installing..."
        pip3 install catboost numpy pandas
        echo "✅ CatBoost Python package installed"
    fi
else
    echo "⚠️  Python 3 not found. You won't be able to create sample models."
    echo "   Install Python 3 to use the example scripts."
fi

# Build the project
echo "🔨 Building catboost-rs..."
cargo build

echo "✅ Build successful!"

# Run tests
echo "🧪 Running tests..."
cargo test

echo "✅ All tests passed!"

# Create sample models if Python is available
if command -v python3 &> /dev/null && python3 -c "import catboost" &> /dev/null; then
    echo "📊 Creating sample models..."
    python3 examples/create_sample_model.py
    
    echo "✅ Sample models created!"
    echo ""
    echo "🎉 Setup complete! You can now run the examples:"
    echo "   cargo run --example basic_usage"
    echo "   cargo run --example advanced_usage"
else
    echo ""
    echo "🎉 Setup complete! (without sample models)"
    echo "   Install Python 3 and catboost package to create sample models."
fi

echo ""
echo "📚 Documentation:"
echo "   - README.md - Main documentation"
echo "   - examples/ - Example programs"
echo "   - cargo doc --open - API documentation"
