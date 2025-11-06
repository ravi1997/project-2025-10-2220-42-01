// tensor_ops.cpp — implementation of tensor operations for the improved tensor system
#include "tensor_ops.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <execution>

namespace dnn {

// ---------------- Tensor Operations Implementation ----------------

// Compute broadcast shape for two tensors
std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    size_t ndim1 = shape1.size();
    size_t ndim2 = shape2.size();
    size_t result_ndim = std::max(ndim1, ndim2);
    
    std::vector<size_t> result_shape(result_ndim);
    
    for (size_t i = 0; i < result_ndim; ++i) {
        size_t dim1_idx = (ndim1 > i) ? ndim1 - 1 - i : 0;
        size_t dim2_idx = (ndim2 > i) ? ndim2 - 1 - i : 0;
        
        size_t size1 = (ndim1 > i) ? shape1[dim1_idx] : 1;
        size_t size2 = (ndim2 > i) ? shape2[dim2_idx] : 1;
        
        if (size1 == 1) {
            result_shape[result_ndim - 1 - i] = size2;
        } else if (size2 == 1) {
            result_shape[result_ndim - 1 - i] = size1;
        } else if (size1 == size2) {
            result_shape[result_ndim - 1 - i] = size1;
        } else {
            throw DimensionMismatchException("Shapes " + std::to_string(size1) + " and " +
                                          std::to_string(size2) + " are not broadcastable");
        }
    }
    
    return result_shape;
}

// Transpose a 2D tensor
template<typename T>
Tensor<T> transpose(const Tensor<T>& A) {
    if (A.ndim() != 2) {
        throw InvalidOperation("Transpose is only implemented for 2D tensors");
    }
    
    std::vector<size_t> transposed_shape = {A.shape()[1], A.shape()[0]};
    Tensor<T> result(transposed_shape, A.layout());
    
    // Use parallel execution for larger tensors
    if (A.size() >= 100) {
        std::vector<std::thread> threads;
        const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), static_cast<size_t>(4));
        const size_t rows_per_thread = A.shape()[0] / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const size_t start_row = t * rows_per_thread;
                const size_t end_row = (t == num_threads - 1) ? A.shape()[0] : (t + 1) * rows_per_thread;
                
                for (size_t r = start_row; r < end_row; ++r) {
                    for (size_t c = 0; c < A.shape()[1]; ++c) {
                        result(c, r) = A(r, c);
                    }
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Sequential implementation for smaller tensors
        for (size_t r = 0; r < A.shape()[0]; ++r) {
            for (size_t c = 0; c < A.shape()[1]; ++c) {
                result(c, r) = A(r, c);
            }
        }
    }
    
    return result;
}

// Matrix multiplication
template<typename T>
Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B) {
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw InvalidOperation("Matrix multiplication is only implemented for 2D tensors");
    }
    
    if (A.shape()[1] != B.shape()[0]) {
        throw DimensionMismatchException("Cannot multiply matrices with incompatible dimensions: (" +
                                       std::to_string(A.shape()[0]) + "x" + std::to_string(A.shape()[1]) +
                                       ") and (" + std::to_string(B.shape()[0]) + "x" +
                                       std::to_string(B.shape()[1]) + ")");
    }
    
    std::vector<size_t> result_shape = {A.shape()[0], B.shape()[1]};
    Tensor<T> C(result_shape);
    
    // Use parallel execution for larger matrices
    if (A.shape()[0] >= 100) {
        std::vector<std::thread> threads;
        const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), static_cast<size_t>(4));
        const size_t rows_per_thread = A.shape()[0] / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const size_t start_row = t * rows_per_thread;
                const size_t end_row = (t == num_threads - 1) ? A.shape()[0] : (t + 1) * rows_per_thread;
                
                for (size_t i = start_row; i < end_row; ++i) {
                    for (size_t k = 0; k < A.shape()[1]; ++k) {
                        const T a = A(i, k);
                        // Check for numerical stability
                        if constexpr (std::is_floating_point_v<T>) {
                            if (!NumericalStability<T>::is_valid(a)) {
                                throw NumericalStabilityException("Matrix multiplication: non-finite value in first matrix");
                            }
                        }
                        
                        for (size_t j = 0; j < B.shape()[1]; ++j) {
                            const T b = B(k, j);
                            // Check for numerical stability
                            if constexpr (std::is_floating_point_v<T>) {
                                if (!NumericalStability<T>::is_valid(b)) {
                                    throw NumericalStabilityException("Matrix multiplication: non-finite value in second matrix");
                                }
                            }
                            
                            C(i, j) += a * b;
                            
                            // Check for overflow
                            if constexpr (std::is_floating_point_v<T>) {
                                if (!NumericalStability<T>::is_valid(C(i, j))) {
                                    throw NumericalStabilityException("Matrix multiplication: overflow detected");
                                }
                            }
                        }
                    }
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Sequential implementation for smaller matrices
        for (size_t i = 0; i < A.shape()[0]; ++i) {
            for (size_t k = 0; k < A.shape()[1]; ++k) {
                const T a = A(i, k);
                // Check for numerical stability
                if constexpr (std::is_floating_point_v<T>) {
                    if (!NumericalStability<T>::is_valid(a)) {
                        throw NumericalStabilityException("Matrix multiplication: non-finite value in first matrix");
                    }
                }
                
                for (size_t j = 0; j < B.shape()[1]; ++j) {
                    const T b = B(k, j);
                    // Check for numerical stability
                    if constexpr (std::is_floating_point_v<T>) {
                        if (!NumericalStability<T>::is_valid(b)) {
                            throw NumericalStabilityException("Matrix multiplication: non-finite value in second matrix");
                        }
                    }
                    
                    C(i, j) += a * b;
                    
                    // Check for overflow
                    if constexpr (std::is_floating_point_v<T>) {
                        if (!NumericalStability<T>::is_valid(C(i, j))) {
                            throw NumericalStabilityException("Matrix multiplication: overflow detected");
                        }
                    }
                }
            }
        }
    }
    
    return C;
}

// Add a row vector to each row of a matrix
template<typename T>
void add_rowwise_inplace(Tensor<T>& A, const Tensor<T>& rowvec) {
    if (rowvec.ndim() != 2 || rowvec.shape()[0] != 1 || rowvec.shape()[1] != A.shape()[1]) {
        throw InvalidOperation("add_rowwise_inplace: row vector must have shape (1, A.cols)");
    }
    
    T* A_data = A.mutable_data();
    const T* rowvec_data = rowvec.data();
    
    // Use parallel execution for larger tensors
    if (A.shape()[0] >= 100) {
        std::vector<std::thread> threads;
        const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), static_cast<size_t>(4));
        const size_t rows_per_thread = A.shape()[0] / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const size_t start_row = t * rows_per_thread;
                const size_t end_row = (t == num_threads - 1) ? A.shape()[0] : (t + 1) * rows_per_thread;
                
                for (size_t r = start_row; r < end_row; ++r) {
                    for (size_t c = 0; c < A.shape()[1]; ++c) {
                        size_t idx = r * A.shape()[1] + c;
                        A_data[idx] += rowvec_data[c];
                    }
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Sequential implementation for smaller tensors
        for (size_t r = 0; r < A.shape()[0]; ++r) {
            for (size_t c = 0; c < A.shape()[1]; ++c) {
                size_t idx = r * A.shape()[1] + c;
                A_data[idx] += rowvec_data[c];
            }
        }
    }
}

// Element-wise addition
template<typename T>
Tensor<T> add(const Tensor<T>& A, const Tensor<T>& B) {
    if (A.shape() != B.shape()) {
        // Try broadcasting if shapes don't match
        auto broadcasted_shape = compute_broadcast_shape(A.shape(), B.shape());
        auto A_broadcasted = A.broadcast_to(broadcasted_shape);
        auto B_broadcasted = B.broadcast_to(broadcasted_shape);
        
        Tensor<T> C(broadcasted_shape, A.layout());
        T* C_data = C.data();
        const T* A_data = A_broadcasted.data();
        const T* B_data = B_broadcasted.data();
        
        // Use parallel execution for larger tensors
        if (C.size() >= 1000) {
            std::vector<std::thread> threads;
            const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), static_cast<size_t>(4));
            const size_t elements_per_thread = C.size() / num_threads;
            
            for (size_t t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    const size_t start_idx = t * elements_per_thread;
                    const size_t end_idx = (t == num_threads - 1) ? C.size() : (t + 1) * elements_per_thread;
                    
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        T sum = A_data[i] + B_data[i];
                        
                        // Check for numerical stability
                        if constexpr (std::is_floating_point_v<T>) {
                            if (!NumericalStability<T>::is_valid(sum)) {
                                throw NumericalStabilityException("Addition: non-finite result");
                            }
                        }
                        
                        C_data[i] = sum;
                    }
                });
            }
            
            for (auto& thread : threads) {
                thread.join();
            }
        } else {
            // Sequential implementation for smaller tensors
            for (size_t i = 0; i < C.size(); ++i) {
                T sum = A_data[i] + B_data[i];
                
                // Check for numerical stability
                if constexpr (std::is_floating_point_v<T>) {
                    if (!NumericalStability<T>::is_valid(sum)) {
                        throw NumericalStabilityException("Addition: non-finite result");
                    }
                }
                
                C_data[i] = sum;
            }
        }
        
        return C;
    }
    
    Tensor<T> C(A.shape(), A.layout());
    T* C_data = C.data();
    const T* A_data = A.data();
    const T* B_data = B.data();
    
    // Use parallel execution for larger tensors
    if (A.size() >= 1000) {
        std::vector<std::thread> threads;
        const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), static_cast<size_t>(4));
        const size_t elements_per_thread = A.size() / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const size_t start_idx = t * elements_per_thread;
                const size_t end_idx = (t == num_threads - 1) ? A.size() : (t + 1) * elements_per_thread;
                
                for (size_t i = start_idx; i < end_idx; ++i) {
                    T sum = A_data[i] + B_data[i];
                    
                    // Check for numerical stability
                    if constexpr (std::is_floating_point_v<T>) {
                        if (!NumericalStability<T>::is_valid(sum)) {
                            throw NumericalStabilityException("Addition: non-finite result");
                        }
                    }
                    
                    C_data[i] = sum;
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Sequential implementation for smaller tensors
        for (size_t i = 0; i < A.size(); ++i) {
            T sum = A_data[i] + B_data[i];
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!NumericalStability<T>::is_valid(sum)) {
                    throw NumericalStabilityException("Addition: non-finite result");
                }
            }
            
            C_data[i] = sum;
        }
    }
    
    return C;
}

// Element-wise subtraction
template<typename T>
Tensor<T> sub(const Tensor<T>& A, const Tensor<T>& B) {
    if (A.shape() != B.shape()) {
        // Try broadcasting if shapes don't match
        auto broadcasted_shape = compute_broadcast_shape(A.shape(), B.shape());
        auto A_broadcasted = A.broadcast_to(broadcasted_shape);
        auto B_broadcasted = B.broadcast_to(broadcasted_shape);
        
        Tensor<T> C(broadcasted_shape, A.layout());
        T* C_data = C.data();
        const T* A_data = A_broadcasted.data();
        const T* B_data = B_broadcasted.data();
        
        // Use parallel execution for larger tensors
        if (C.size() >= 1000) {
            std::vector<std::thread> threads;
            const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), static_cast<size_t>(4));
            const size_t elements_per_thread = C.size() / num_threads;
            
            for (size_t t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    const size_t start_idx = t * elements_per_thread;
                    const size_t end_idx = (t == num_threads - 1) ? C.size() : (t + 1) * elements_per_thread;
                    
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        T diff = A_data[i] - B_data[i];
                        
                        // Check for numerical stability
                        if constexpr (std::is_floating_point_v<T>) {
                            if (!NumericalStability<T>::is_valid(diff)) {
                                throw NumericalStabilityException("Subtraction: non-finite result");
                            }
                        }
                        
                        C_data[i] = diff;
                    }
                });
            }
            
            for (auto& thread : threads) {
                thread.join();
            }
        } else {
            // Sequential implementation for smaller tensors
            for (size_t i = 0; i < C.size(); ++i) {
                T diff = A_data[i] - B_data[i];
                
                // Check for numerical stability
                if constexpr (std::is_floating_point_v<T>) {
                    if (!NumericalStability<T>::is_valid(diff)) {
                        throw NumericalStabilityException("Subtraction: non-finite result");
                    }
                }
                
                C_data[i] = diff;
            }
        }
        
        return C;
    }
    
    Tensor<T> C(A.shape(), A.layout());
    T* C_data = C.data();
    const T* A_data = A.data();
    const T* B_data = B.data();
    
    // Use parallel execution for larger tensors
    if (A.size() >= 1000) {
        std::vector<std::thread> threads;
        const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), static_cast<size_t>(4));
        const size_t elements_per_thread = A.size() / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const size_t start_idx = t * elements_per_thread;
                const size_t end_idx = (t == num_threads - 1) ? A.size() : (t + 1) * elements_per_thread;
                
                for (size_t i = start_idx; i < end_idx; ++i) {
                    T diff = A_data[i] - B_data[i];
                    
                    // Check for numerical stability
                    if constexpr (std::is_floating_point_v<T>) {
                        if (!NumericalStability<T>::is_valid(diff)) {
                            throw NumericalStabilityException("Subtraction: non-finite result");
                        }
                    }
                    
                    C_data[i] = diff;
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Sequential implementation for smaller tensors
        for (size_t i = 0; i < A.size(); ++i) {
            T diff = A_data[i] - B_data[i];
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!NumericalStability<T>::is_valid(diff)) {
                    throw NumericalStabilityException("Subtraction: non-finite result");
                }
            }
            
            C_data[i] = diff;
        }
    }
    
    return C;
}

// Element-wise Hadamard product (element-wise multiplication)
template<typename T>
Tensor<T> hadamard(const Tensor<T>& A, const Tensor<T>& B) {
    if (A.shape() != B.shape()) {
        // Try broadcasting if shapes don't match
        auto broadcasted_shape = compute_broadcast_shape(A.shape(), B.shape());
        auto A_broadcasted = A.broadcast_to(broadcasted_shape);
        auto B_broadcasted = B.broadcast_to(broadcasted_shape);
        
        Tensor<T> C(broadcasted_shape, A.layout());
        T* C_data = C.data();
        const T* A_data = A_broadcasted.data();
        const T* B_data = B_broadcasted.data();
        
        // Use parallel execution for larger tensors
        if (C.size() >= 1000) {
            std::vector<std::thread> threads;
            const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), static_cast<size_t>(4));
            const size_t elements_per_thread = C.size() / num_threads;
            
            for (size_t t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    const size_t start_idx = t * elements_per_thread;
                    const size_t end_idx = (t == num_threads - 1) ? C.size() : (t + 1) * elements_per_thread;
                    
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        T product = A_data[i] * B_data[i];
                        
                        // Check for numerical stability
                        if constexpr (std::is_floating_point_v<T>) {
                            if (!NumericalStability<T>::is_valid(product)) {
                                throw NumericalStabilityException("Hadamard product: non-finite result");
                            }
                        }
                        
                        C_data[i] = product;
                    }
                });
            }
            
            for (auto& thread : threads) {
                thread.join();
            }
        } else {
            // Sequential implementation for smaller tensors
            for (size_t i = 0; i < C.size(); ++i) {
                T product = A_data[i] * B_data[i];
                
                // Check for numerical stability
                if constexpr (std::is_floating_point_v<T>) {
                    if (!NumericalStability<T>::is_valid(product)) {
                        throw NumericalStabilityException("Hadamard product: non-finite result");
                    }
                }
                
                C_data[i] = product;
            }
        }
        
        return C;
    }
    
    Tensor<T> C(A.shape(), A.layout());
    T* C_data = C.data();
    const T* A_data = A.data();
    const T* B_data = B.data();
    
    // Use parallel execution for larger tensors
    if (A.size() >= 1000) {
        std::vector<std::thread> threads;
        const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), static_cast<size_t>(4));
        const size_t elements_per_thread = A.size() / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const size_t start_idx = t * elements_per_thread;
                const size_t end_idx = (t == num_threads - 1) ? A.size() : (t + 1) * elements_per_thread;
                
                for (size_t i = start_idx; i < end_idx; ++i) {
                    T product = A_data[i] * B_data[i];
                    
                    // Check for numerical stability
                    if constexpr (std::is_floating_point_v<T>) {
                        if (!NumericalStability<T>::is_valid(product)) {
                            throw NumericalStabilityException("Hadamard product: non-finite result");
                        }
                    }
                    
                    C_data[i] = product;
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Sequential implementation for smaller tensors
        for (size_t i = 0; i < A.size(); ++i) {
            T product = A_data[i] * B_data[i];
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!NumericalStability<T>::is_valid(product)) {
                    throw NumericalStabilityException("Hadamard product: non-finite result");
                }
            }
            
            C_data[i] = product;
        }
    }
    
    return C;
}

// Scalar multiplication
template<typename T>
Tensor<T> scalar_mul(const Tensor<T>& A, T s) {
    Tensor<T> C(A.shape(), A.layout());
    T* C_data = C.data();
    const T* A_data = A.data();
    
    // Use parallel execution for larger tensors
    if (A.size() >= 1000) {
        std::vector<std::thread> threads;
        const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), static_cast<size_t>(4));
        const size_t elements_per_thread = A.size() / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const size_t start_idx = t * elements_per_thread;
                const size_t end_idx = (t == num_threads - 1) ? A.size() : (t + 1) * elements_per_thread;
                
                for (size_t i = start_idx; i < end_idx; ++i) {
                    T product = A_data[i] * s;
                    
                    // Check for numerical stability
                    if constexpr (std::is_floating_point_v<T>) {
                        if (!NumericalStability<T>::is_valid(product)) {
                            throw NumericalStabilityException("Scalar multiplication: non-finite result");
                        }
                    }
                    
                    C_data[i] = product;
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Sequential implementation for smaller tensors
        for (size_t i = 0; i < A.size(); ++i) {
            T product = A_data[i] * s;
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!NumericalStability<T>::is_valid(product)) {
                    throw NumericalStabilityException("Scalar multiplication: non-finite result");
                }
            }
            
            C_data[i] = product;
        }
    }
    
    return C;
}

// Sum along rows (reduce columns)
template<typename T>
Tensor<T> sum_rows(const Tensor<T>& A) {
    if (A.ndim() != 2) {
        throw InvalidOperation("sum_rows is only implemented for 2D tensors");
    }
    
    Tensor<T> v({1, A.shape()[1]}, T(0));
    T* v_data = v.data();
    const T* A_data = A.data();
    
    // Use parallel execution for larger tensors
    if (A.shape()[0] >= 100) {
        std::vector<std::thread> threads;
        const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), static_cast<size_t>(4));
        const size_t cols_per_thread = A.shape()[1] / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const size_t start_col = t * cols_per_thread;
                const size_t end_col = (t == num_threads - 1) ? A.shape()[1] : (t + 1) * cols_per_thread;
                
                for (size_t c = start_col; c < end_col; ++c) {
                    T sum = T(0);
                    for (size_t r = 0; r < A.shape()[0]; ++r) {
                        size_t idx = r * A.shape()[1] + c;
                        sum += A_data[idx];
                        
                        // Check for numerical stability
                        if constexpr (std::is_floating_point_v<T>) {
                            if (!NumericalStability<T>::is_valid(sum)) {
                                throw NumericalStabilityException("Row sum: non-finite result");
                            }
                        }
                    }
                    v_data[c] = sum;
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Sequential implementation for smaller tensors
        for (size_t c = 0; c < A.shape()[1]; ++c) {
            T sum = T(0);
            for (size_t r = 0; r < A.shape()[0]; ++r) {
                size_t idx = r * A.shape()[1] + c;
                sum += A_data[idx];
                
                // Check for numerical stability
                if constexpr (std::is_floating_point_v<T>) {
                    if (!NumericalStability<T>::is_valid(sum)) {
                        throw NumericalStabilityException("Row sum: non-finite result");
                    }
                }
            }
            v_data[c] = sum;
        }
    }
    
    return v;
}

// Explicit template instantiations for common types
// No template instantiation needed for compute_broadcast_shape since it's not a template function

template TensorF transpose(const TensorF& A);
template TensorD transpose(const TensorD& A);
template TensorI transpose(const TensorI& A);
template TensorL transpose(const TensorL& A);
template TensorB transpose(const TensorB& A);

template TensorF matmul(const TensorF& A, const TensorF& B);
template TensorD matmul(const TensorD& A, const TensorD& B);
template TensorI matmul(const TensorI& A, const TensorI& B);
template TensorL matmul(const TensorL& A, const TensorL& B);
template TensorB matmul(const TensorB& A, const TensorB& B);

template void add_rowwise_inplace(TensorF& A, const TensorF& rowvec);
template void add_rowwise_inplace(TensorD& A, const TensorD& rowvec);
template void add_rowwise_inplace(TensorI& A, const TensorI& rowvec);
template void add_rowwise_inplace(TensorL& A, const TensorL& rowvec);
template void add_rowwise_inplace(TensorB& A, const TensorB& rowvec);

template TensorF add(const TensorF& A, const TensorF& B);
template TensorD add(const TensorD& A, const TensorD& B);
template TensorI add(const TensorI& A, const TensorI& B);
template TensorL add(const TensorL& A, const TensorL& B);
template TensorB add(const TensorB& A, const TensorB& B);

template TensorF sub(const TensorF& A, const TensorF& B);
template TensorD sub(const TensorD& A, const TensorD& B);
template TensorI sub(const TensorI& A, const TensorI& B);
template TensorL sub(const TensorL& A, const TensorL& B);
template TensorB sub(const TensorB& A, const TensorB& B);

template TensorF hadamard(const TensorF& A, const TensorF& B);
template TensorD hadamard(const TensorD& A, const TensorD& B);
template TensorI hadamard(const TensorI& A, const TensorI& B);
template TensorL hadamard(const TensorL& A, const TensorL& B);
template TensorB hadamard(const TensorB& A, const TensorB& B);

template TensorF scalar_mul(const TensorF& A, float s);
template TensorD scalar_mul(const TensorD& A, double s);
template TensorI scalar_mul(const TensorI& A, int s);
template TensorL scalar_mul(const TensorL& A, long s);
template TensorB scalar_mul(const TensorB& A, bool s);

template TensorF sum_rows(const TensorF& A);
template TensorD sum_rows(const TensorD& A);
template TensorI sum_rows(const TensorI& A);
template TensorL sum_rows(const TensorL& A);
template TensorB sum_rows(const TensorB& A);

} // namespace dnn