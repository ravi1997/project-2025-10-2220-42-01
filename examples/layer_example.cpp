#include "../include/layers.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <memory>

int main() {
    std::cout << "Neural Network Layer Implementation Example" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        // Initialize random number generator
        std::random_device rd;
        std::mt19937 rng(rd());
        
        // Example 1: Dense Layer
        std::cout << "\n1. Dense Layer Example:" << std::endl;
        {
            dnn::Dense dense_layer(10, 5, "dense1");
            dense_layer.initialize_parameters(rng);
            
            // Create input tensor (batch_size=3, features=10)
            dnn::TensorF input({3, 10});
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = static_cast<float>(i) / 10.0f;
            }
            
            // Forward pass
            dnn::TensorF output = dense_layer.forward(input);
            std::cout << "Input shape: (" << input.shape()[0] << ", " << input.shape()[1] << ")" << std::endl;
            std::cout << "Output shape: (" << output.shape()[0] << ", " << output.shape()[1] << ")" << std::endl;
            
            // Create gradient output for backward pass
            dnn::TensorF grad_output(output.shape());
            for (size_t i = 0; i < grad_output.size(); ++i) {
                grad_output[i] = 0.1f;
            }
            
            // Backward pass
            dnn::TensorF grad_input = dense_layer.backward(grad_output);
            std::cout << "Gradient input shape: (" << grad_input.shape()[0] << ", " << grad_input.shape()[1] << ")" << std::endl;
            
            // Update parameters
            dense_layer.update_parameters(0.01f);
            std::cout << "Dense layer parameters updated." << std::endl;
        }
        
        // Example 2: Conv2D Layer
        std::cout << "\n2. Conv2D Layer Example:" << std::endl;
        {
            dnn::Conv2D conv_layer(3, 16, 3, 3, 1, 1, 1, 1, "conv1");
            conv_layer.initialize_parameters(rng);
            
            // Create input tensor (batch_size=2, channels=3, height=32, width=32)
            dnn::TensorF input({2, 3, 32, 32});
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = static_cast<float>(i % 100) / 100.0f;
            }
            
            // Forward pass
            dnn::TensorF output = conv_layer.forward(input);
            std::cout << "Input shape: (" << input.shape()[0] << ", " << input.shape()[1] 
                      << ", " << input.shape()[2] << ", " << input.shape()[3] << ")" << std::endl;
            std::cout << "Output shape: (" << output.shape()[0] << ", " << output.shape()[1] 
                      << ", " << output.shape()[2] << ", " << output.shape()[3] << ")" << std::endl;
            
            // Create gradient output for backward pass
            dnn::TensorF grad_output(output.shape());
            for (size_t i = 0; i < grad_output.size(); ++i) {
                grad_output[i] = 0.01f;
            }
            
            // Backward pass
            dnn::TensorF grad_input = conv_layer.backward(grad_output);
            std::cout << "Gradient input shape: (" << grad_input.shape()[0] << ", " << grad_input.shape()[1] 
                      << ", " << grad_input.shape()[2] << ", " << grad_input.shape()[3] << ")" << std::endl;
            
            // Update parameters
            conv_layer.update_parameters(0.01f);
            std::cout << "Conv2D layer parameters updated." << std::endl;
        }
        
        // Example 3: Activation Layers
        std::cout << "\n3. Activation Layers Example:" << std::endl;
        {
            dnn::ReLU relu_layer("relu1");
            dnn::Sigmoid sigmoid_layer("sigmoid1");
            dnn::Tanh tanh_layer("tanh1");
            
            // Create input tensor
            dnn::TensorF input({2, 5});
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = static_cast<float>(i - 5) / 5.0f;  // Values from -1 to 1
            }
            
            // Test ReLU
            dnn::TensorF relu_output = relu_layer.forward(input);
            dnn::TensorF relu_grad = relu_layer.backward(relu_output);
            std::cout << "ReLU: Input max=" << input[0] << ", Output max=" << relu_output[0] << std::endl;
            
            // Test Sigmoid
            dnn::TensorF sigmoid_output = sigmoid_layer.forward(input);
            dnn::TensorF sigmoid_grad = sigmoid_layer.backward(sigmoid_output);
            std::cout << "Sigmoid: Input max=" << input[0] << ", Output max=" << sigmoid_output[0] << std::endl;
            
            // Test Tanh
            dnn::TensorF tanh_output = tanh_layer.forward(input);
            dnn::TensorF tanh_grad = tanh_layer.backward(tanh_output);
            std::cout << "Tanh: Input max=" << input[0] << ", Output max=" << tanh_output[0] << std::endl;
        }
        
        // Example 4: Normalization Layers
        std::cout << "\n4. Normalization Layers Example:" << std::endl;
        {
            dnn::BatchNorm batch_norm(10, 0.1f, 1e-5f, "batch_norm1");
            batch_norm.initialize_parameters(rng);
            
            // Create input tensor
            dnn::TensorF input({5, 10});  // batch_size=5, features=10
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = static_cast<float>(i) / 10.0f;
            }
            
            // Forward pass
            dnn::TensorF output = batch_norm.forward(input);
            std::cout << "BatchNorm input shape: (" << input.shape()[0] << ", " << input.shape()[1] << ")" << std::endl;
            std::cout << "BatchNorm output shape: (" << output.shape()[0] << ", " << output.shape()[1] << ")" << std::endl;
            
            // Create gradient output
            dnn::TensorF grad_output(output.shape());
            for (size_t i = 0; i < grad_output.size(); ++i) {
                grad_output[i] = 0.1f;
            }
            
            // Backward pass
            dnn::TensorF grad_input = batch_norm.backward(grad_output);
            
            // Update parameters
            batch_norm.update_parameters(0.01f);
            std::cout << "BatchNorm parameters updated." << std::endl;
        }
        
        // Example 5: Dropout Layer
        std::cout << "\n5. Dropout Layer Example:" << std::endl;
        {
            dnn::Dropout dropout_layer(0.5f, "dropout1");
            
            // Create input tensor
            dnn::TensorF input({2, 10});
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = 1.0f;
            }
            
            // Set to training mode
            dropout_layer.set_training_mode(true);
            
            // Forward pass
            dnn::TensorF output = dropout_layer.forward(input);
            std::cout << "Dropout input sum: " << input.size() << ", output sum (approx): ";
            
            float sum = 0.0f;
            for (size_t i = 0; i < output.size(); ++i) {
                sum += output[i];
            }
            std::cout << sum << std::endl;
            
            // Backward pass
            dnn::TensorF grad_output = output;  // Use same values for gradient
            dnn::TensorF grad_input = dropout_layer.backward(grad_output);
            std::cout << "Dropout backward completed." << std::endl;
        }
        
        // Example 6: Batch Processing with Validation
        std::cout << "\n6. Batch Processing and Validation Example:" << std::endl;
        {
            // Create a batch of inputs
            dnn::TensorF large_input({100, 10});  // 100 samples, 10 features each
            for (size_t i = 0; i < large_input.size(); ++i) {
                large_input[i] = static_cast<float>(i % 100) / 100.0f;
            }
            
            // Validate the input
            dnn::Validation::validate_finite_values(large_input, "large_input");
            std::cout << "Input validation passed." << std::endl;
            
            // Process in smaller batches using BatchProcessor
            std::vector<dnn::TensorF> batches = dnn::BatchProcessor::split_batch(large_input, 32);
            std::cout << "Split large input into " << batches.size() << " batches." << std::endl;
            
            // Process each batch with a simple operation (this simulates layer processing)
            std::vector<dnn::TensorF> processed_batches;
            for (size_t i = 0; i < batches.size(); ++i) {
                // Simulate some processing (e.g., apply a simple transformation)
                dnn::TensorF processed_batch(batches[i].shape());
                for (size_t j = 0; j < batches[i].size(); ++j) {
                    processed_batch[j] = batches[i][j] * 2.0f + 0.1f;
                }
                processed_batches.push_back(processed_batch);
            }
            
            // Concatenate results back
            dnn::TensorF final_result = dnn::BatchProcessor::concatenate_batches(processed_batches);
            std::cout << "Final result shape: (" << final_result.shape()[0] << ", " 
                      << final_result.shape()[1] << ")" << std::endl;
            
            // Validate the result
            dnn::Validation::validate_finite_values(final_result, "final_result");
            std::cout << "Result validation passed." << std::endl;
        }
        
        // Example 7: Error Handling Demonstration
        std::cout << "\n7. Error Handling Example:" << std::endl;
        {
            dnn::Dense layer(10, 5);
            
            // Try to pass incorrectly sized input
            dnn::TensorF wrong_input({3, 8});  // Should be {3, 10}
            for (size_t i = 0; i < wrong_input.size(); ++i) {
                wrong_input[i] = 0.5f;
            }
            
            try {
                dnn::TensorF result = layer.forward(wrong_input);
                std::cout << "This should not print due to error." << std::endl;
            } catch (const dnn::NeuralNetworkException& e) {
                std::cout << "Caught expected error: " << e.what() << std::endl;
            }
            
            std::cout << "Error handling demonstrated successfully." << std::endl;
        }
        
        std::cout << "\nAll examples completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}