# DNN Library - Industrial-Grade Deep Neural Network Framework

A high-performance, industrial-grade deep neural network library implemented in modern C++20 with comprehensive features and best practices.

## Features

- **Modern C++20 Implementation**: Uses latest C++ features for optimal performance and maintainability
- **Comprehensive Layer Types**: Dense, Conv2D, Pooling, Normalization, and more
- **Advanced Optimizers**: SGD, Adam, RMSprop, AdamW with momentum and regularization
- **Robust Model Persistence**: Complete save/load functionality with integrity checks
- **Industrial Build System**: Modern CMake with comprehensive testing and packaging
- **Memory Efficient**: Optimized memory management with copy-on-write semantics
- **Numerically Stable**: Carefully implemented mathematical operations with overflow protection
- **Extensible Architecture**: Clean interfaces for adding new layers, activations, and optimizers

## Architecture Overview

```
dnn/
├── include/                 # Public headers
│   ├── tensor.hpp           # Core tensor implementation
│   ├── layers.hpp           # Layer interfaces and implementations
│   ├── optimizers.hpp       # Optimizer interfaces and implementations
│   ├── model_serializer.hpp # Model serialization utilities
│   └── utils.hpp            # Utility functions and helpers
├── src/                     # Implementation files
│   ├── tensor.cpp           # Tensor implementation
│   ├── layers.cpp           # Layer implementations
│   ├── optimizers.cpp       # Optimizer implementations
│   └── model_serializer.cpp # Model serialization implementation
├── examples/                # Usage examples
│   ├── xor_example.cpp      # Simple XOR classification
│   ├── mnist_example.cpp    # MNIST digit recognition
│   └── layer_example.cpp    # Layer composition example
├── apps/                    # Main application
│   └── main.cpp             # Entry point
├── tests/                   # Unit and integration tests
├── cmake/                   # CMake modules
│   ├── CompilerWarnings.cmake
│   ├── Dependencies.cmake
│   ├── Options.cmake
│   ├── Sanitizers.cmake
│   └── StandardProjectSettings.cmake
├── docs/                    # Documentation
└── CMakeLists.txt           # Build configuration
```

## Build Instructions

### Prerequisites

- C++20 compatible compiler (GCC 10+, Clang 10+, MSVC 2019+)
- CMake 3.20 or higher
- Git (for dependencies, if using Conan)

### Building

```bash
# Clone the repository
git clone <repository-url>
cd dnn

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build . --config Release

# Run tests (optional)
ctest --verbose

# Install (optional)
cmake --install . --prefix /install/prefix
```

### CMake Options

- `DNN_BUILD_TESTS`: Build unit tests (default: ON)
- `DNN_BUILD_EXAMPLES`: Build examples (default: ON)
- `DNN_BUILD_BENCHMARKS`: Build benchmarks (default: OFF)
- `DNN_ENABLE_COVERAGE`: Enable code coverage (default: OFF)
- `DNN_WARNINGS_AS_ERRORS`: Treat warnings as errors (default: OFF)
- `DNN_ENABLE_LTO`: Enable Link Time Optimization (default: ON for Release builds)
- `DNN_USE_CLANG_TIDY`: Enable clang-tidy static analysis (default: OFF)
- `DNN_USE_CPPCHECK`: Enable Cppcheck static analysis (default: OFF)

## Usage Example

```cpp
#include <dnn.hpp>

int main() {
    // Create a simple neural network
    dnn::Model model;
    
    // Add layers
    model.add(std::make_unique<dnn::Dense>(784, 128, dnn::Activation::ReLU));
    model.add(std::make_unique<dnn::Dense>(128, 10, dnn::Activation::Softmax));
    
    // Compile with optimizer
    model.compile(std::make_unique<dnn::Adam>(0.001f));
    
    // Train the model
    // model.fit(X_train, y_train, epochs, dnn::LossFunction::CrossEntropy);
    
    // Make predictions
    // auto predictions = model.predict(X_test);
    
    return 0;
}
```

## Code Quality Standards

This project follows industrial-grade code quality standards:

- Consistent naming conventions (PascalCase for classes, camelCase for functions)
- Comprehensive error handling with exceptions and error codes
- RAII memory management with smart pointers
- Doxygen-style documentation for all public interfaces
- High test coverage (>85%) with unit and integration tests
- Static analysis with Clang-Tidy and Cppcheck
- Sanitizer support for AddressSanitizer and UBSan

## Performance Optimizations

- SIMD instruction usage for mathematical operations
- Cache-friendly memory access patterns
- Memory pooling for temporary tensors
- Copy-on-write semantics for tensor sharing
- Optimized matrix multiplication with parallelization
- Link Time Optimization (LTO) enabled for release builds

## Contributing

1. Follow the code quality standards documented in `docs/code_quality_standards.md`
2. Write comprehensive tests for new functionality
3. Document all public interfaces with Doxygen comments
4. Ensure all tests pass before submitting a pull request
5. Run static analysis tools and fix any issues

## License

This project is licensed under [specify license type] - see the LICENSE file for details.

## Acknowledgments

- Modern CMake practices inspired by various C++ community resources
- Tensor implementation with shared data management
- Numerical stability considerations for deep learning operations