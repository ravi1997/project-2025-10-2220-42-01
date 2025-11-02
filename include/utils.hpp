#pragma once
// utils.hpp - Utility functions for neural network operations
// Includes error handling, validation, and batch processing support

#include "tensor.hpp"
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

namespace dnn {

// Exception classes for neural network operations
class NeuralNetworkException : public std::runtime_error {
public:
    explicit NeuralNetworkException(const std::string& msg) 
        : std::runtime_error("Neural Network Error: " + msg) {}
};

class ShapeMismatchException : public NeuralNetworkException {
public:
    explicit ShapeMismatchException(const std::string& msg) 
        : NeuralNetworkException("Shape Mismatch: " + msg) {}
};

class InvalidValueException : public NeuralNetworkException {
public:
    explicit InvalidValueException(const std::string& msg) 
        : NeuralNetworkException("Invalid Value: " + msg) {}
};

class LayerConfigurationException : public NeuralNetworkException {
public:
    explicit LayerConfigurationException(const std::string& msg) 
        : NeuralNetworkException("Layer Configuration Error: " + msg) {}
};

// Utility functions for validation
class Validation {
public:
    // Validate tensor dimensions for a specific operation
    static void validate_dimensions(const TensorF& tensor, size_t expected_dims, 
                                   const std::string& operation = "") {
        if (tensor.ndim() != expected_dims) {
            throw ShapeMismatchException(
                "Expected " + std::to_string(expected_dims) + " dimensions for " + 
                operation + ", got " + std::to_string(tensor.ndim()));
        }
    }
    
    // Validate that two tensors have compatible shapes for an operation
    static void validate_compatible_shapes(const TensorF& a, const TensorF& b, 
                                          const std::string& operation = "") {
        if (a.shape() != b.shape()) {
            throw ShapeMismatchException(
                "Tensors have incompatible shapes for " + operation + 
                ". Tensor A: " + shape_to_string(a.shape()) + 
                ", Tensor B: " + shape_to_string(b.shape()));
        }
    }
    
    // Validate that a tensor shape is compatible with expected dimensions
    static void validate_shape_compatibility(const std::vector<size_t>& actual,
                                            const std::vector<size_t>& expected,
                                            const std::string& operation = "") {
        if (actual.size() != expected.size()) {
            throw ShapeMismatchException(
                "Shape size mismatch in " + operation + 
                ". Expected " + std::to_string(expected.size()) + 
                " dimensions, got " + std::to_string(actual.size()));
        }
        
        for (size_t i = 0; i < actual.size(); ++i) {
            if (expected[i] != 0 && actual[i] != expected[i]) {  // 0 means "any size"
                throw ShapeMismatchException(
                    "Dimension " + std::to_string(i) + " mismatch in " + operation + 
                    ". Expected " + std::to_string(expected[i]) + 
                    ", got " + std::to_string(actual[i]));
            }
        }
    }
    
    // Check for NaN or infinity values in tensor
    static bool has_nan_or_inf(const TensorF& tensor) {
        for (size_t i = 0; i < tensor.size(); ++i) {
            float val = tensor[i];
            if (std::isnan(val) || std::isinf(val)) {
                return true;
            }
        }
        return false;
    }
    
    // Validate tensor values are within reasonable bounds
    static void validate_finite_values(const TensorF& tensor, 
                                      const std::string& tensor_name = "") {
        if (has_nan_or_inf(tensor)) {
            throw InvalidValueException(
                "Tensor " + tensor_name + " contains NaN or infinity values");
        }
    }
    
    // Convert shape vector to string for error messages
    static std::string shape_to_string(const std::vector<size_t>& shape) {
        std::string result = "(";
        for (size_t i = 0; i < shape.size(); ++i) {
            result += std::to_string(shape[i]);
            if (i < shape.size() - 1) {
                result += ", ";
            }
        }
        result += ")";
        return result;
    }
};

// Batch processing utilities
class BatchProcessor {
public:
    // Split a large batch into smaller batches
    static std::vector<TensorF> split_batch(const TensorF& input, size_t batch_size) {
        if (input.ndim() == 0) {
            throw NeuralNetworkException("Cannot split empty tensor");
        }
        
        size_t total_samples = input.shape()[0];
        std::vector<TensorF> batches;
        
        for (size_t start = 0; start < total_samples; start += batch_size) {
            size_t end = std::min(start + batch_size, total_samples);
            size_t current_batch_size = end - start;
            
            // Create a new tensor for the batch
            std::vector<size_t> batch_shape = input.shape();
            batch_shape[0] = current_batch_size;
            
            TensorF batch_tensor(batch_shape);
            
            // Copy data from original tensor to batch tensor
            size_t elements_per_sample = input.size() / total_samples;
            for (size_t i = 0; i < current_batch_size; ++i) {
                for (size_t j = 0; j < elements_per_sample; ++j) {
                    batch_tensor[i * elements_per_sample + j] = 
                        input[(start + i) * elements_per_sample + j];
                }
            }
            
            batches.push_back(batch_tensor);
        }
        
        return batches;
    }
    
    // Concatenate multiple batches into a single tensor
    static TensorF concatenate_batches(const std::vector<TensorF>& batches) {
        if (batches.empty()) {
            throw NeuralNetworkException("Cannot concatenate empty batch vector");
        }
        
        if (batches.size() == 1) {
            return batches[0];
        }
        
        // Verify all batches have compatible shapes (except the first dimension)
        std::vector<size_t> sample_shape = batches[0].shape();
        sample_shape[0] = 1;  // Remove batch dimension
        
        for (size_t i = 1; i < batches.size(); ++i) {
            std::vector<size_t> current_sample_shape = batches[i].shape();
            current_sample_shape[0] = 1;  // Remove batch dimension
            
            if (sample_shape != current_sample_shape) {
                throw ShapeMismatchException(
                    "Batch " + std::to_string(i) + " has incompatible shape with batch 0. " +
                    "Expected: " + Validation::shape_to_string(sample_shape) + 
                    ", Got: " + Validation::shape_to_string(current_sample_shape));
            }
        }
        
        // Calculate total size
        size_t total_samples = 0;
        size_t elements_per_sample = 1;
        for (size_t i = 1; i < batches[0].shape().size(); ++i) {
            elements_per_sample *= batches[0].shape()[i];
        }
        
        for (const auto& batch : batches) {
            total_samples += batch.shape()[0];
        }
        
        // Create result tensor
        std::vector<size_t> result_shape = batches[0].shape();
        result_shape[0] = total_samples;
        
        TensorF result(result_shape);
        
        // Copy data from all batches
        size_t output_idx = 0;
        for (const auto& batch : batches) {
            for (size_t i = 0; i < batch.size(); ++i) {
                result[output_idx++] = batch[i];
            }
        }
        
        return result;
    }
    
    // Process tensor in chunks to save memory
    static TensorF process_in_chunks(const TensorF& input,
                                    std::function<TensorF(const TensorF&)> processor,
                                    size_t chunk_size = 32) {
        std::vector<TensorF> input_chunks = split_batch(input, chunk_size);
        std::vector<TensorF> output_chunks;
        
        for (const auto& chunk : input_chunks) {
            TensorF processed_chunk = processor(chunk);
            output_chunks.push_back(processed_chunk);
        }
        
        return concatenate_batches(output_chunks);
    }
};

// Gradient checking utilities for validation
class GradientChecker {
public:
    // Numerically compute gradient for validation
    static TensorF numerical_gradient(const std::function<float(const TensorF&)>& func,
                                     const TensorF& input,
                                     float epsilon = 1e-5f) {
        TensorF grad(input.shape());
        
        for (size_t i = 0; i < input.size(); ++i) {
            // Compute f(x + epsilon)
            TensorF input_plus = input;
            input_plus[i] += epsilon;
            float f_plus = func(input_plus);
            
            // Compute f(x - epsilon)
            TensorF input_minus = input;
            input_minus[i] -= epsilon;
            float f_minus = func(input_minus);
            
            // Central difference
            grad[i] = (f_plus - f_minus) / (2.0f * epsilon);
        }
        
        return grad;
    }
    
    // Compare analytical and numerical gradients
    static float gradient_difference(const TensorF& analytical_grad,
                                   const TensorF& numerical_grad) {
        if (analytical_grad.size() != numerical_grad.size()) {
            throw ShapeMismatchException("Gradient tensors have different sizes");
        }
        
        float diff_sum = 0.0f;
        for (size_t i = 0; i < analytical_grad.size(); ++i) {
            float diff = analytical_grad[i] - numerical_grad[i];
            diff_sum += diff * diff;
        }
        
        return std::sqrt(diff_sum);
    }
};

} // namespace dnn