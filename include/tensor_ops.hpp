#ifndef TENSOR_OPS_HPP
#define TENSOR_OPS_HPP

// tensor_ops.hpp — tensor operations for the improved tensor system
// Provides mathematical operations and utilities for tensors

#include "tensor_improved.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <thread>
#include <execution>

namespace dnn {

// ---------------- Tensor Operations ----------------

// Transpose a 2D tensor
template<typename T>
Tensor<T> transpose(const Tensor<T>& A);

// Matrix multiplication
template<typename T>
Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B);

// Add a row vector to each row of a matrix
template<typename T>
void add_rowwise_inplace(Tensor<T>& A, const Tensor<T>& rowvec);

// Element-wise addition
template<typename T>
Tensor<T> add(const Tensor<T>& A, const Tensor<T>& B);

// Element-wise subtraction
template<typename T>
Tensor<T> sub(const Tensor<T>& A, const Tensor<T>& B);

// Element-wise Hadamard product (element-wise multiplication)
template<typename T>
Tensor<T> hadamard(const Tensor<T>& A, const Tensor<T>& B);

// Scalar multiplication
template<typename T>
Tensor<T> scalar_mul(const Tensor<T>& A, T s);

// Sum along rows (reduce columns)
template<typename T>
Tensor<T> sum_rows(const Tensor<T>& A);

// Compute broadcast shape for two tensors
template<typename T>
std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);

// Stable sigmoid function
template<typename T>
T stable_sigmoid(T x) {
    if constexpr (std::is_floating_point_v<T>) {
        if (x >= T(0)) {
            const T e = std::exp(-x);
            return T(1) / (T(1) + e);
        }
        const T e = std::exp(x);
        return e / (T(1) + e);
    } else {
        // For integer types, convert to floating point for computation
        using FloatType = typename std::conditional<std::is_same_v<T, int>, float, double>::type;
        FloatType fx = static_cast<FloatType>(x);
        if (fx >= FloatType(0)) {
            const FloatType e = std::exp(-fx);
            return static_cast<T>(FloatType(1) / (FloatType(1) + e));
        }
        const FloatType e = std::exp(fx);
        return static_cast<T>(e / (FloatType(1) + e));
    }
}

// Stable softplus function
template<typename T>
T stable_softplus(T x) {
    if constexpr (std::is_floating_point_v<T>) {
        if (x > T(0)) {
            return x + std::log1p(std::exp(-x));
        }
        return std::log1p(std::exp(x));
    } else {
        // For integer types, convert to floating point for computation
        using FloatType = typename std::conditional<std::is_same_v<T, int>, float, double>::type;
        FloatType fx = static_cast<FloatType>(x);
        if (fx > FloatType(0)) {
            return static_cast<T>(fx + std::log1p(std::exp(-fx)));
        }
        return static_cast<T>(std::log1p(std::exp(fx)));
    }
}

// Safe logarithm function
template<typename T>
T safe_log(T x) {
    const T epsilon = NumericalLimits<T>::epsilon();
    if (x < epsilon) {
        return std::log(epsilon);
    }
    return std::log(x);
}

// Clamp probabilities to prevent log(0)
template<typename T>
void clamp_probabilities(Tensor<T>& tensor) {
    const T epsilon = NumericalLimits<T>::epsilon();
    const T min_val = epsilon;
    const T max_val = T(1) - epsilon;
    
    T* data = tensor.data();
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (data[i] < min_val) data[i] = min_val;
        else if (data[i] > max_val) data[i] = max_val;
    }
}

} // namespace dnn

#endif // TENSOR_OPS_HPP