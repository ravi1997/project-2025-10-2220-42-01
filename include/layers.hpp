#pragma once
// layers.hpp - Neural Network Layer Implementations
// Comprehensive layer implementations with forward/backward propagation

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <string>
#include <random>

namespace dnn {

// Base Layer class with virtual interface for forward/backward propagation
class Layer {
public:
    std::string name;
    bool trainable;
    
    explicit Layer(std::string layer_name = "layer", bool is_trainable = true)
        : name(std::move(layer_name)), trainable(is_trainable) {}
    
    virtual ~Layer() = default;
    
    // Forward propagation - compute output from input
    virtual TensorF forward(const TensorF& input) = 0;
    
    // Backward propagation - compute gradients and return gradient w.r.t input
    virtual TensorF backward(const TensorF& grad_output) = 0;
    
    // Update parameters using optimizer
    virtual void update_parameters(float learning_rate) = 0;
    
    // Get total number of parameters in the layer
    virtual size_t get_parameter_count() const = 0;
    
    // Zero gradients of parameters
    virtual void zero_gradients() = 0;
    
    // Initialize parameters using specified method
    virtual void initialize_parameters(std::mt19937& rng) = 0;
};

// Dense/Linear layer with proper forward/backward propagation
class Dense : public Layer {
private:
    size_t in_features, out_features;
    TensorF weights;      // [in_features, out_features]
    TensorF bias;         // [1, out_features]
    TensorF grad_weights; // [in_features, out_features]
    TensorF grad_bias;    // [1, out_features]
    
    // Cache for backward pass
    TensorF input_cache;
    
    // Initialization methods
    void xavier_initialization(std::mt19937& rng);
    void he_initialization(std::mt19937& rng);

public:
    explicit Dense(size_t in_features, size_t out_features, 
                   std::string layer_name = "dense");
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(float learning_rate) override;
    size_t get_parameter_count() const override;
    void zero_gradients() override;
    void initialize_parameters(std::mt19937& rng) override;
    
    // Getters for parameters
    const TensorF& get_weights() const { return weights; }
    const TensorF& get_bias() const { return bias; }
    const TensorF& get_grad_weights() const { return grad_weights; }
    const TensorF& get_grad_bias() const { return grad_bias; }
};

// Convolutional layer with proper forward/backward propagation
class Conv2D : public Layer {
private:
    size_t in_channels, out_channels;
    size_t kernel_height, kernel_width;
    size_t stride_h, stride_w;
    size_t padding_h, padding_w;
    
    TensorF weights;      // [out_channels, in_channels, kernel_height, kernel_width]
    TensorF bias;         // [out_channels]
    TensorF grad_weights; // [out_channels, in_channels, kernel_height, kernel_width]
    TensorF grad_bias;    // [out_channels]
    
    // Cache for backward pass
    TensorF input_cache;
    
public:
    Conv2D(size_t in_channels, size_t out_channels,
           size_t kernel_height, size_t kernel_width,
           size_t stride_h = 1, size_t stride_w = 1,
           size_t padding_h = 0, size_t padding_w = 0,
           std::string layer_name = "conv2d");
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(float learning_rate) override;
    size_t get_parameter_count() const override;
    void zero_gradients() override;
    void initialize_parameters(std::mt19937& rng) override;
    
    // Getters for parameters
    const TensorF& get_weights() const { return weights; }
    const TensorF& get_bias() const { return bias; }
    const TensorF& get_grad_weights() const { return grad_weights; }
    const TensorF& get_grad_bias() const { return grad_bias; }
};

// Activation layers (ReLU, Sigmoid, Tanh, etc.)
class ReLU : public Layer {
public:
    explicit ReLU(std::string layer_name = "relu");
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(float learning_rate) override;
    size_t get_parameter_count() const override;
    void zero_gradients() override;
    void initialize_parameters(std::mt19937& rng) override;
    
private:
    TensorF mask_cache; // Cache for backward pass
};

class Sigmoid : public Layer {
public:
    explicit Sigmoid(std::string layer_name = "sigmoid");
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(float learning_rate) override;
    size_t get_parameter_count() const override;
    void zero_gradients() override;
    void initialize_parameters(std::mt19937& rng) override;
    
private:
    TensorF output_cache; // Cache for backward pass
};

class Tanh : public Layer {
public:
    explicit Tanh(std::string layer_name = "tanh");
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(float learning_rate) override;
    size_t get_parameter_count() const override;
    void zero_gradients() override;
    void initialize_parameters(std::mt19937& rng) override;
    
private:
    TensorF output_cache; // Cache for backward pass
};

class LeakyReLU : public Layer {
private:
    float alpha;

public:
    explicit LeakyReLU(float alpha = 0.01f, std::string layer_name = "leaky_relu");
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(float learning_rate) override;
    size_t get_parameter_count() const override;
    void zero_gradients() override;
    void initialize_parameters(std::mt19937& rng) override;
    
private:
    TensorF mask_cache; // Cache for backward pass
};

// Pooling layers (MaxPool, AvgPool)
class MaxPool2D : public Layer {
private:
    size_t pool_height, pool_width;
    size_t stride_h, stride_w;
    size_t padding_h, padding_w;
    
    // Cache for backward pass
    TensorF input_cache;
    TensorF mask_cache; // To track where max values came from

public:
    MaxPool2D(size_t pool_height, size_t pool_width,
              size_t stride_h = 1, size_t stride_w = 1,
              size_t padding_h = 0, size_t padding_w = 0,
              std::string layer_name = "maxpool2d");
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(float learning_rate) override;
    size_t get_parameter_count() const override;
    void zero_gradients() override;
    void initialize_parameters(std::mt19937& rng) override;
};

class AvgPool2D : public Layer {
private:
    size_t pool_height, pool_width;
    size_t stride_h, stride_w;
    size_t padding_h, padding_w;
    
    // Cache for backward pass
    TensorF input_cache;

public:
    AvgPool2D(size_t pool_height, size_t pool_width,
              size_t stride_h = 1, size_t stride_w = 1,
              size_t padding_h = 0, size_t padding_w = 0,
              std::string layer_name = "avgpool2d");
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(float learning_rate) override;
    size_t get_parameter_count() const override;
    void zero_gradients() override;
    void initialize_parameters(std::mt19937& rng) override;
};

// Normalization layers (BatchNorm, LayerNorm)
class BatchNorm : public Layer {
private:
    size_t features;
    float momentum;
    float epsilon;
    
    TensorF gamma;           // Scale parameter
    TensorF beta;            // Shift parameter
    TensorF running_mean;    // Running mean for inference
    TensorF running_var;     // Running variance for inference
    TensorF grad_gamma;      // Gradient of gamma
    TensorF grad_beta;       // Gradient of beta
    
    // Cache for backward pass
    TensorF input_cache;
    TensorF x_norm_cache;    // Normalized input
    TensorF x_centered_cache; // Centered input
    TensorF inv_std_cache;   // Inverse standard deviation

public:
    BatchNorm(size_t features, 
              float momentum = 0.1f, 
              float epsilon = 1e-5f,
              std::string layer_name = "batchnorm");
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(float learning_rate) override;
    size_t get_parameter_count() const override;
    void zero_gradients() override;
    void initialize_parameters(std::mt19937& rng) override;
};

class LayerNorm : public Layer {
private:
    size_t features;
    float epsilon;
    
    TensorF gamma;           // Scale parameter
    TensorF beta;            // Shift parameter
    TensorF grad_gamma;      // Gradient of gamma
    TensorF grad_beta;       // Gradient of beta
    
    // Cache for backward pass
    TensorF input_cache;
    TensorF x_norm_cache;    // Normalized input
    TensorF mean_cache;      // Mean for each sample
    TensorF var_cache;       // Variance for each sample

public:
    LayerNorm(size_t features, 
              float epsilon = 1e-5f,
              std::string layer_name = "layernorm");
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(float learning_rate) override;
    size_t get_parameter_count() const override;
    void zero_gradients() override;
    void initialize_parameters(std::mt19937& rng) override;
};

// Dropout layer
class Dropout : public Layer {
private:
    float rate;
    bool training_mode;
    
    // Cache for backward pass
    TensorF mask_cache;

public:
    explicit Dropout(float dropout_rate = 0.5f, std::string layer_name = "dropout");
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(float learning_rate) override;
    size_t get_parameter_count() const override;
    void zero_gradients() override;
    void initialize_parameters(std::mt19937& rng) override;
    
    // Set training/inference mode
    void set_training_mode(bool mode) { training_mode = mode; }
    bool is_training() const { return training_mode; }
};

// Embedding layer
class Embedding : public Layer {
private:
    size_t num_embeddings;  // Vocabulary size
    size_t embedding_dim;   // Embedding dimension
    
    TensorF weights;        // [num_embeddings, embedding_dim]
    TensorF grad_weights;   // [num_embeddings, embedding_dim]

public:
    Embedding(size_t num_embeddings, size_t embedding_dim,
              std::string layer_name = "embedding");
    
    TensorF forward(const TensorF& input) override;  // Input is indices tensor
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(float learning_rate) override;
    size_t get_parameter_count() const override;
    void zero_gradients() override;
    void initialize_parameters(std::mt19937& rng) override;
    
    // Getters for parameters
    const TensorF& get_weights() const { return weights; }
    const TensorF& get_grad_weights() const { return grad_weights; }
};

} // namespace dnn