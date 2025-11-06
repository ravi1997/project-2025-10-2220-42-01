// tensor_improved.cpp — implementation of the improved tensor system
#include "tensor_improved.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <execution>

namespace dnn {

// ---------------- TensorF Implementation ----------------
// Explicit instantiation of Tensor<float>
template class Tensor<float>;

// ---------------- TensorD Implementation ----------------
// Explicit instantiation of Tensor<double>
template class Tensor<double>;

// ---------------- TensorI Implementation ----------------
// Explicit instantiation of Tensor<int>
template class Tensor<int>;

// ---------------- TensorL Implementation ----------------
// Explicit instantiation of Tensor<long>
template class Tensor<long>;

// ---------------- TensorB Implementation ----------------
// Explicit instantiation of Tensor<bool>
template class Tensor<bool>;

// ---------------- Numerical Stability Utilities Implementation ----------------
template<>
float NumericalLimits<float>::epsilon() {
    return 1e-7f;
}

template<>
double NumericalLimits<double>::epsilon() {
    return 1e-15;
}

template<>
int NumericalLimits<int>::epsilon() {
    return 0;
}

template<>
long NumericalLimits<long>::epsilon() {
    return 0;
}

template<>
bool NumericalLimits<bool>::epsilon() {
    return false;
}

template<>
float NumericalLimits<float>::max_value() {
    return std::numeric_limits<float>::max();
}

template<>
double NumericalLimits<double>::max_value() {
    return std::numeric_limits<double>::max();
}

template<>
int NumericalLimits<int>::max_value() {
    return std::numeric_limits<int>::max();
}

template<>
long NumericalLimits<long>::max_value() {
    return std::numeric_limits<long>::max();
}

template<>
bool NumericalLimits<bool>::max_value() {
    return true;
}

template<>
float NumericalLimits<float>::min_value() {
    return std::numeric_limits<float>::min();
}

template<>
double NumericalLimits<double>::min_value() {
    return std::numeric_limits<double>::min();
}

template<>
int NumericalLimits<int>::min_value() {
    return std::numeric_limits<int>::min();
}

template<>
long NumericalLimits<long>::min_value() {
    return std::numeric_limits<long>::min();
}

template<>
bool NumericalLimits<bool>::min_value() {
    return false;
}

template<>
bool NumericalStability<float>::is_valid(float value) {
    return std::isfinite(value);
}

template<>
bool NumericalStability<double>::is_valid(double value) {
    return std::isfinite(value);
}

template<>
bool NumericalStability<int>::is_valid(int /*value*/) {
    return true;
}

template<>
bool NumericalStability<long>::is_valid(long /*value*/) {
    return true;
}

template<>
bool NumericalStability<bool>::is_valid(bool /*value*/) {
    return true;
}

template<>
bool NumericalStability<float>::is_valid_tensor(const float* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (!std::isfinite(data[i])) {
            return false;
        }
    }
    return true;
}

template<>
bool NumericalStability<double>::is_valid_tensor(const double* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (!std::isfinite(data[i])) {
            return false;
        }
    }
    return true;
}

template<>
bool NumericalStability<int>::is_valid_tensor(const int* /*data*/, size_t /*size*/) {
    return true;
}

template<>
bool NumericalStability<long>::is_valid_tensor(const long* /*data*/, size_t /*size*/) {
    return true;
}

template<>
bool NumericalStability<bool>::is_valid_tensor(const bool* /*data*/, size_t /*size*/) {
    return true;
}

template<>
float NumericalStability<float>::clamp(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

template<>
double NumericalStability<double>::clamp(double value, double min_val, double max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

template<>
int NumericalStability<int>::clamp(int value, int min_val, int max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

template<>
long NumericalStability<long>::clamp(long value, long min_val, long max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

template<>
bool NumericalStability<bool>::clamp(bool value, bool min_val, bool max_val) {
    // For boolean values, clamping doesn't make sense, so just return the value
    return value;
}


template<>
float NumericalStability<float>::stable_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = std::exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = std::exp(x);
    return e / (1.0f + e);
}

template<>
double NumericalStability<double>::stable_sigmoid(double x) {
    if (x >= 0.0) {
        const double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    }
    const double e = std::exp(x);
    return e / (1.0 + e);
}

template<>
float NumericalStability<float>::stable_softplus(float x) {
    if (x > 0.0f) {
        return x + std::log1p(std::exp(-x));
    }
    return std::log1p(std::exp(x));
}

template<>
double NumericalStability<double>::stable_softplus(double x) {
    if (x > 0.0) {
        return x + std::log1p(std::exp(-x));
    }
    return std::log1p(std::exp(x));
}

template<>
float NumericalStability<float>::safe_log(float x) {
    const float epsilon = NumericalLimits<float>::epsilon();
    if (x < epsilon) {
        return std::log(epsilon);
    }
    return std::log(x);
}

template<>
double NumericalStability<double>::safe_log(double x) {
    const double epsilon = NumericalLimits<double>::epsilon();
    if (x < epsilon) {
        return std::log(epsilon);
    }
    return std::log(x);
}

template<>
int NumericalStability<int>::safe_log(int x) {
    if (x <= 0) {
        return 0; // Return 0 for invalid input
    }
    return static_cast<int>(std::log(static_cast<double>(x)));
}

template<>
long NumericalStability<long>::safe_log(long x) {
    if (x <= 0) {
        return 0; // Return 0 for invalid input
    }
    return static_cast<long>(std::log(static_cast<double>(x)));
}

template<>
bool NumericalStability<bool>::safe_log(bool x) {
    // Logarithm of boolean doesn't make sense, return false
    return false;
}

} // namespace dnn