# Tensor System Documentation

## Overview
The tensor system forms the foundation of the DNN library, providing multi-dimensional array operations with efficient memory management and mathematical operations.

## Core Concepts

### Tensor Template
The `Tensor` template class provides a flexible, multi-dimensional array implementation:

```cpp
template<std::size_t NumDims = 2>
struct Tensor {
    std::array<std::size_t, NumDims> shape;
    std::vector<double> data;
    std::size_t size;
    // ...
};
```

### Key Features
- **Multi-dimensional**: Supports any number of dimensions (default is 2 for matrices)
- **Row-major storage**: Elements stored in row-major order for cache efficiency
- **Automatic memory management**: Uses RAII principles with std::vector
- **Bounds checking**: Runtime bounds checking with std::out_of_range exceptions

## Tensor Operations

### Construction
```cpp
// Create a 3x4 matrix filled with zeros
dnn::Tensor<2> matrix({3, 4});

// Create a 3x4 matrix filled with a specific value
dnn::Tensor<2> filled_matrix({3, 4}, 5.0);

// Create a 2D matrix using the Matrix alias
dnn::Matrix mat(3, 4);  // 3 rows, 4 columns
```

### Element Access
```cpp
dnn::Matrix mat(3, 4);

// Access elements using parentheses
mat(1, 2) = 5.0;  // Set element at row 1, column 2
double val = mat(1, 2);  // Get element at row 1, column 2

// Linear access using operator[]
mat[5] = 3.0;  // Set 6th element (0-indexed)
```

### Utility Methods
- `zeros()`: Create tensor filled with zeros
- `ones()`: Create tensor filled with ones
- `filled()`: Create tensor filled with a specific value
- `random_normal()`: Create tensor with normally distributed random values
- `random_uniform()`: Create tensor with uniformly distributed random values
- `reshape()`: Change tensor dimensions while preserving data
- `flatten()`: Convert to 2D matrix (rows, 1)
- `fill()`: Fill all elements with a specific value
- `any_nonfinite()`: Check for non-finite values (NaN, infinity)

## Matrix Operations
The library provides several matrix-specific operations:

### Basic Operations
- `matmul()`: Matrix multiplication with optional parallelization
- `transpose()`: Matrix transposition
- `add()`: Matrix addition
- `sub()`: Matrix subtraction
- `hadamard()`: Element-wise multiplication
- `scalar_mul()`: Scalar multiplication
- `sum_rows()`: Sum along rows

### Memory Management
- Data stored in contiguous `std::vector<double>` for cache efficiency
- Automatic memory allocation and deallocation
- Copy and move semantics properly implemented
- Temporary matrices are efficiently managed during operations

## Performance Considerations

### Vectorization
- Uses C++23 features for potential compiler optimizations
- Parallel execution for large matrix operations when `Config::USE_VECTORIZATION` is true
- Configurable thread count with `Config::MAX_THREADS`

### Memory Layout
- Row-major ordering for better cache locality
- Contiguous memory allocation for optimal performance
- Minimized memory allocations during operations

## Error Handling
- Bounds checking with `std::out_of_range` exceptions
- Shape validation for operations
- Finite value checking to prevent numerical issues

## Best Practices
1. Use the `Matrix` alias for 2D operations (more readable)
2. Prefer the tensor constructor that takes dimensions as parameters for 2D tensors
3. Use the static methods (zeros, ones, etc.) for common initialization patterns
4. Check for non-finite values during training to catch numerical issues early
5. Use appropriate initialization methods (Xavier/Glorot for Dense layers, He for ReLU networks)