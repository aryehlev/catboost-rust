# Use the standard Rust image
FROM rust:1.85


RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev libclang-dev clang cmake


# Set working directory
WORKDIR /app

# Copy the project files
COPY . .

# Build the project in release mode
RUN cargo build -v --release

# Build and cache examples
RUN cargo build --release --example basic_usage
RUN cargo build --release --example advanced_usage

# Set the default command to an interactive shell for development
CMD ["bash"]