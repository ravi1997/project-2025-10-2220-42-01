// layers.cpp - Neural Network Layer Implementations
// Comprehensive layer implementations with forward/backward propagation

#include "../include/layers.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace dnn {

// Dense layer implementation
Dense::Dense(size_t in_features, size_t out_features, std::string layer_name)
    : Layer(std::move(layer_name), true),
      in_features(in_features), out_features(out_features),
      weights({in_features, out_features}),
      bias({1, out_features}),
      grad_weights({in_features, out_features}),
      grad_bias({1, out_features}) {
    // Initialize gradients to zero
    grad_weights.fill(0.0f);
    grad_bias.fill(0.0f);
}

void Dense::xavier_initialization(std::mt19937& rng) {
    float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
    std::uniform_real_distribution<float> dist(-limit, limit);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = dist(rng);
    }
    bias.fill(0.0f);
}

void Dense::he_initialization(std::mt19937& rng) {
    float std_dev = std::sqrt(2.0f / static_cast<float>(in_features));
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = dist(rng);
    }
    bias.fill(0.0f);
}

TensorF Dense::forward(const TensorF& input) {
    if (input.ndim() < 2) {
        throw std::invalid_argument("Dense layer input must have at least 2 dimensions (batch_size, features)");
    }
    
    // Ensure input features match expected size
    if (input.shape().back() != in_features) {
        throw std::invalid_argument("Dense layer input feature size mismatch");
    }
    
    // Store input for backward pass
    input_cache = input;
    
    // Reshape input to (batch_size, in_features) if needed
    std::vector<size_t> input_shape = input.shape();
    size_t batch_size = input.size() / in_features;
    
    // Perform matrix multiplication: output = input * weights + bias
    // If input is (batch_size, in_features) and weights is (in_features, out_features)
    // Then output is (batch_size, out_features)
    TensorF output({batch_size, out_features});
    
    // Perform matrix multiplication manually for now
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_features; ++i) {
                // Calculate input index: b * in_features + i
                // Calculate weight index: i * out_features + j
                sum += input[b * in_features + i] * weights[i * out_features + j];
            }
            output[b * out_features + j] = sum + bias[j];
        }
    }
    
    return output;
}

TensorF Dense::backward(const TensorF& grad_output) {
    if (grad_output.ndim() < 2) {
        throw std::invalid_argument("Dense layer grad_output must have at least 2 dimensions");
    }
    
    // Calculate dimensions
    size_t batch_size = grad_output.size() / out_features;
    if (batch_size * out_features != grad_output.size()) {
        throw std::invalid_argument("Dense layer grad_output size mismatch");
    }
    
    // Calculate gradients with respect to weights and bias
    // grad_weights = input^T * grad_output
    for (size_t i = 0; i < in_features; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                sum += input_cache[b * in_features + i] * grad_output[b * out_features + j];
            }
            grad_weights[i * out_features + j] = sum;
        }
    }
    
    // grad_bias = sum(grad_output, axis=0)
    for (size_t j = 0; j < out_features; ++j) {
        float sum = 0.0f;
        for (size_t b = 0; b < batch_size; ++b) {
            sum += grad_output[b * out_features + j];
        }
        grad_bias[j] = sum;
    }
    
    // Calculate gradient with respect to input
    // grad_input = grad_output * weights^T
    TensorF grad_input({batch_size, in_features});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < in_features; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < out_features; ++j) {
                sum += grad_output[b * out_features + j] * weights[i * out_features + j];
            }
            grad_input[b * in_features + i] = sum;
        }
    }
    
    return grad_input;
}

void Dense::update_parameters(float learning_rate) {
    if (!trainable) return;
    
    // Update weights: weights = weights - learning_rate * grad_weights
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate * grad_weights[i];
    }
    
    // Update bias: bias = bias - learning_rate * grad_bias
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] -= learning_rate * grad_bias[i];
    }
}

size_t Dense::get_parameter_count() const {
    return weights.size() + bias.size();
}

void Dense::zero_gradients() {
    grad_weights.fill(0.0f);
    grad_bias.fill(0.0f);
}

void Dense::initialize_parameters(std::mt19937& rng) {
    // Use Xavier initialization for Dense layer
    xavier_initialization(rng);
}

// Conv2D layer implementation
Conv2D::Conv2D(size_t in_channels, size_t out_channels,
               size_t kernel_height, size_t kernel_width,
               size_t stride_h, size_t stride_w,
               size_t padding_h, size_t padding_w,
               std::string layer_name)
    : Layer(std::move(layer_name), true),
      in_channels(in_channels), out_channels(out_channels),
      kernel_height(kernel_height), kernel_width(kernel_width),
      stride_h(stride_h), stride_w(stride_w),
      padding_h(padding_h), padding_w(padding_w),
      weights({out_channels, in_channels, kernel_height, kernel_width}),
      bias({out_channels}),
      grad_weights({out_channels, in_channels, kernel_height, kernel_width}),
      grad_bias({out_channels}) {
    grad_weights.fill(0.0f);
    grad_bias.fill(0.0f);
}

TensorF Conv2D::forward(const TensorF& input) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("Conv2D layer expects 4D input (batch, channels, height, width)");
    }
    
    size_t batch_size = input.shape()[0];
    size_t in_h = input.shape()[2];
    size_t in_w = input.shape()[3];
    
    if (input.shape()[1] != in_channels) {
        throw std::invalid_argument("Conv2D input channels mismatch");
    }
    
    // Calculate output dimensions
    size_t out_h = (in_h + 2 * padding_h - kernel_height) / stride_h + 1;
    size_t out_w = (in_w + 2 * padding_w - kernel_width) / stride_w + 1;
    
    // Store input for backward pass
    input_cache = input;
    
    // Create output tensor
    TensorF output({batch_size, out_channels, out_h, out_w});
    
    // Perform convolution
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    
                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_height; ++kh) {
                            for (size_t kw = 0; kw < kernel_width; ++kw) {
                                size_t ih = oh * stride_h - padding_h + kh;
                                size_t iw = ow * stride_w - padding_w + kw;
                                
                                if (ih < in_h && iw < in_w) {
                                    // Calculate indices for input and weights tensors
                                    size_t input_idx = b * in_channels * in_h * in_w + 
                                                      ic * in_h * in_w + 
                                                      ih * in_w + iw;
                                    size_t weight_idx = oc * in_channels * kernel_height * kernel_width + 
                                                       ic * kernel_height * kernel_width + 
                                                       kh * kernel_width + kw;
                                    sum += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }
                    
                    // Add bias
                    size_t output_idx = b * out_channels * out_h * out_w + 
                                       oc * out_h * out_w + 
                                       oh * out_w + ow;
                    output[output_idx] = sum + bias[oc];
                }
            }
        }
    }
    
    return output;
}

TensorF Conv2D::backward(const TensorF& grad_output) {
    if (grad_output.ndim() != 4) {
        throw std::invalid_argument("Conv2D backward expects 4D grad_output");
    }
    
    size_t batch_size = grad_output.shape()[0];
    size_t out_channels_check = grad_output.shape()[1];
    size_t out_h = grad_output.shape()[2];
    size_t out_w = grad_output.shape()[3];
    
    if (out_channels_check != out_channels) {
        throw std::invalid_argument("Conv2D grad_output channels mismatch");
    }
    
    size_t in_h = input_cache.shape()[2];
    size_t in_w = input_cache.shape()[3];
    
    // Initialize gradient with respect to input
    TensorF grad_input(input_cache.shape());
    grad_input.fill(0.0f);
    
    // Calculate gradients
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    size_t grad_output_idx = b * out_channels * out_h * out_w + 
                                            oc * out_h * out_w + 
                                            oh * out_w + ow;
                    float grad_val = grad_output[grad_output_idx];
                    
                    // Update bias gradient
                    grad_bias[oc] += grad_val;
                    
                    // Update weight gradients and input gradients
                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_height; ++kh) {
                            for (size_t kw = 0; kw < kernel_width; ++kw) {
                                size_t ih = oh * stride_h - padding_h + kh;
                                size_t iw = ow * stride_w - padding_w + kw;
                                
                                if (ih < in_h && iw < in_w) {
                                    // Update weight gradient
                                    size_t weight_idx = oc * in_channels * kernel_height * kernel_width + 
                                                       ic * kernel_height * kernel_width + 
                                                       kh * kernel_width + kw;
                                    size_t input_idx = b * in_channels * in_h * in_w + 
                                                      ic * in_h * in_w + 
                                                      ih * in_w + iw;
                                    grad_weights[weight_idx] += input_cache[input_idx] * grad_val;
                                    
                                    // Update input gradient
                                    grad_input[input_idx] += weights[weight_idx] * grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return grad_input;
}

void Conv2D::update_parameters(float learning_rate) {
    if (!trainable) return;
    
    // Update weights
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate * grad_weights[i];
    }
    
    // Update bias
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] -= learning_rate * grad_bias[i];
    }
}

size_t Conv2D::get_parameter_count() const {
    return weights.size() + bias.size();
}

void Conv2D::zero_gradients() {
    grad_weights.fill(0.0f);
    grad_bias.fill(0.0f);
}

void Conv2D::initialize_parameters(std::mt19937& rng) {
    // He initialization for convolutional layers
    float std_dev = std::sqrt(2.0f / static_cast<float>(in_channels * kernel_height * kernel_width));
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = dist(rng);
    }
    bias.fill(0.0f);
}

// ReLU layer implementation
ReLU::ReLU(std::string layer_name) : Layer(std::move(layer_name), false) {}

TensorF ReLU::forward(const TensorF& input) {
    mask_cache = TensorF(input.shape());  // Create mask tensor for backward pass
    
    TensorF output(input.shape());
    
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] > 0.0f) {
            output[i] = input[i];
            mask_cache[i] = 1.0f;
        } else {
            output[i] = 0.0f;
            mask_cache[i] = 0.0f;
        }
    }
    
    return output;
}

TensorF ReLU::backward(const TensorF& grad_output) {
    if (grad_output.size() != mask_cache.size()) {
        throw std::invalid_argument("ReLU backward input size mismatch");
    }
    
    TensorF grad_input(grad_output.shape());
    
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = grad_output[i] * mask_cache[i];
    }
    
    return grad_input;
}

void ReLU::update_parameters(float learning_rate) {
    // ReLU has no parameters to update
}

size_t ReLU::get_parameter_count() const {
    return 0;
}

void ReLU::zero_gradients() {
    // ReLU has no gradients to zero
}

void ReLU::initialize_parameters(std::mt19937& rng) {
    // ReLU has no parameters to initialize
}

// Sigmoid layer implementation
Sigmoid::Sigmoid(std::string layer_name) : Layer(std::move(layer_name), false) {}

TensorF Sigmoid::forward(const TensorF& input) {
    output_cache = TensorF(input.shape());
    
    TensorF output(input.shape());
    
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float result;
        
        if (x >= 0) {
            float exp_neg_x = std::exp(-x);
            result = 1.0f / (1.0f + exp_neg_x);
        } else {
            float exp_x = std::exp(x);
            result = exp_x / (1.0f + exp_x);
        }
        
        output[i] = result;
        output_cache[i] = result;  // Store for backward pass
    }
    
    return output;
}

TensorF Sigmoid::backward(const TensorF& grad_output) {
    if (grad_output.size() != output_cache.size()) {
        throw std::invalid_argument("Sigmoid backward input size mismatch");
    }
    
    TensorF grad_input(grad_output.shape());
    
    for (size_t i = 0; i < grad_output.size(); ++i) {
        float sigmoid_val = output_cache[i];
        grad_input[i] = grad_output[i] * sigmoid_val * (1.0f - sigmoid_val);
    }
    
    return grad_input;
}

void Sigmoid::update_parameters(float learning_rate) {
    // Sigmoid has no parameters to update
}

size_t Sigmoid::get_parameter_count() const {
    return 0;
}

void Sigmoid::zero_gradients() {
    // Sigmoid has no gradients to zero
}

void Sigmoid::initialize_parameters(std::mt19937& rng) {
    // Sigmoid has no parameters to initialize
}

// Tanh layer implementation
Tanh::Tanh(std::string layer_name) : Layer(std::move(layer_name), false) {}

TensorF Tanh::forward(const TensorF& input) {
    output_cache = TensorF(input.shape());
    
    TensorF output(input.shape());
    
    for (size_t i = 0; i < input.size(); ++i) {
        float result = std::tanh(input[i]);
        output[i] = result;
        output_cache[i] = result;  // Store for backward pass
    }
    
    return output;
}

TensorF Tanh::backward(const TensorF& grad_output) {
    if (grad_output.size() != output_cache.size()) {
        throw std::invalid_argument("Tanh backward input size mismatch");
    }
    
    TensorF grad_input(grad_output.shape());
    
    for (size_t i = 0; i < grad_output.size(); ++i) {
        float tanh_val = output_cache[i];
        grad_input[i] = grad_output[i] * (1.0f - tanh_val * tanh_val);
    }
    
    return grad_input;
}

void Tanh::update_parameters(float learning_rate) {
    // Tanh has no parameters to update
}

size_t Tanh::get_parameter_count() const {
    return 0;
}

void Tanh::zero_gradients() {
    // Tanh has no gradients to zero
}

void Tanh::initialize_parameters(std::mt19937& rng) {
    // Tanh has no parameters to initialize
}

// LeakyReLU layer implementation
LeakyReLU::LeakyReLU(float alpha, std::string layer_name) 
    : Layer(std::move(layer_name), false), alpha(alpha) {}

TensorF LeakyReLU::forward(const TensorF& input) {
    mask_cache = TensorF(input.shape());  // Create mask tensor for backward pass
    
    TensorF output(input.shape());
    
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] > 0.0f) {
            output[i] = input[i];
            mask_cache[i] = 1.0f;
        } else {
            output[i] = alpha * input[i];
            mask_cache[i] = alpha;
        }
    }
    
    return output;
}

TensorF LeakyReLU::backward(const TensorF& grad_output) {
    if (grad_output.size() != mask_cache.size()) {
        throw std::invalid_argument("LeakyReLU backward input size mismatch");
    }
    
    TensorF grad_input(grad_output.shape());
    
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = grad_output[i] * mask_cache[i];
    }
    
    return grad_input;
}

void LeakyReLU::update_parameters(float learning_rate) {
    // LeakyReLU has no parameters to update
}

size_t LeakyReLU::get_parameter_count() const {
    return 0;
}

void LeakyReLU::zero_gradients() {
    // LeakyReLU has no gradients to zero
}

void LeakyReLU::initialize_parameters(std::mt19937& rng) {
    // LeakyReLU has no parameters to initialize
}

// MaxPool2D layer implementation
MaxPool2D::MaxPool2D(size_t pool_height, size_t pool_width,
                     size_t stride_h, size_t stride_w,
                     size_t padding_h, size_t padding_w,
                     std::string layer_name)
    : Layer(std::move(layer_name), false),
      pool_height(pool_height), pool_width(pool_width),
      stride_h(stride_h), stride_w(stride_w),
      padding_h(padding_h), padding_w(padding_w) {}

TensorF MaxPool2D::forward(const TensorF& input) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("MaxPool2D expects 4D input (batch, channels, height, width)");
    }
    
    size_t batch_size = input.shape()[0];
    size_t channels = input.shape()[1];
    size_t in_h = input.shape()[2];
    size_t in_w = input.shape()[3];
    
    // Calculate output dimensions
    size_t out_h = (in_h + 2 * padding_h - pool_height) / stride_h + 1;
    size_t out_w = (in_w + 2 * padding_w - pool_width) / stride_w + 1;
    
    // Store input for backward pass
    input_cache = input;
    
    // Create output tensor and mask tensor
    TensorF output({batch_size, channels, out_h, out_w});
    mask_cache = TensorF({batch_size, channels, in_h, in_w});
    mask_cache.fill(0.0f);
    
    // Perform max pooling
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    size_t max_h = 0, max_w = 0;
                    
                    // Find max value in the pooling window
                    for (size_t ph = 0; ph < pool_height; ++ph) {
                        for (size_t pw = 0; pw < pool_width; ++pw) {
                            size_t ih = oh * stride_h - padding_h + ph;
                            size_t iw = ow * stride_w - padding_w + pw;
                            
                            if (ih < in_h && iw < in_w) {
                                size_t input_idx = b * channels * in_h * in_w + 
                                                  c * in_h * in_w + 
                                                  ih * in_w + iw;
                                if (input[input_idx] > max_val) {
                                    max_val = input[input_idx];
                                    max_h = ih;
                                    max_w = iw;
                                }
                            }
                        }
                    }
                    
                    // Store max value
                    size_t output_idx = b * channels * out_h * out_w + 
                                       c * out_h * out_w + 
                                       oh * out_w + ow;
                    output[output_idx] = max_val;
                    
                    // Mark the position of max value in the mask
                    size_t mask_idx = b * channels * in_h * in_w + 
                                     c * in_h * in_w + 
                                     max_h * in_w + max_w;
                    mask_cache[mask_idx] = 1.0f;
                }
            }
        }
    }
    
    return output;
}

TensorF MaxPool2D::backward(const TensorF& grad_output) {
    if (grad_output.ndim() != 4) {
        throw std::invalid_argument("MaxPool2D backward expects 4D grad_output");
    }
    
    size_t batch_size = grad_output.shape()[0];
    size_t channels = grad_output.shape()[1];
    size_t out_h = grad_output.shape()[2];
    size_t out_w = grad_output.shape()[3];
    
    // Initialize gradient with respect to input
    TensorF grad_input(input_cache.shape());
    grad_input.fill(0.0f);
    
    // Propagate gradients only to positions that had max values
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    size_t output_idx = b * channels * out_h * out_w + 
                                       c * out_h * out_w + 
                                       oh * out_w + ow;
                    float grad_val = grad_output[output_idx];
                    
                    // Find the max position in the original input
                    for (size_t ph = 0; ph < pool_height; ++ph) {
                        for (size_t pw = 0; pw < pool_width; ++pw) {
                            size_t ih = oh * stride_h - padding_h + ph;
                            size_t iw = ow * stride_w - padding_w + pw;
                            
                            if (ih < input_cache.shape()[2] && iw < input_cache.shape()[3]) {
                                size_t mask_idx = b * channels * input_cache.shape()[2] * input_cache.shape()[3] + 
                                                 c * input_cache.shape()[2] * input_cache.shape()[3] + 
                                                 ih * input_cache.shape()[3] + iw;
                                // Only propagate gradient if this position was the max
                                if (mask_cache[mask_idx] > 0.0f) {
                                    size_t grad_input_idx = b * channels * input_cache.shape()[2] * input_cache.shape()[3] + 
                                                           c * input_cache.shape()[2] * input_cache.shape()[3] + 
                                                           ih * input_cache.shape()[3] + iw;
                                    grad_input[grad_input_idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return grad_input;
}

void MaxPool2D::update_parameters(float learning_rate) {
    // MaxPool2D has no parameters to update
}

size_t MaxPool2D::get_parameter_count() const {
    return 0;
}

void MaxPool2D::zero_gradients() {
    // MaxPool2D has no gradients to zero
}

void MaxPool2D::initialize_parameters(std::mt19937& rng) {
    // MaxPool2D has no parameters to initialize
}

// AvgPool2D layer implementation
AvgPool2D::AvgPool2D(size_t pool_height, size_t pool_width,
                     size_t stride_h, size_t stride_w,
                     size_t padding_h, size_t padding_w,
                     std::string layer_name)
    : Layer(std::move(layer_name), false),
      pool_height(pool_height), pool_width(pool_width),
      stride_h(stride_h), stride_w(stride_w),
      padding_h(padding_h), padding_w(padding_w) {}

TensorF AvgPool2D::forward(const TensorF& input) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("AvgPool2D expects 4D input (batch, channels, height, width)");
    }
    
    size_t batch_size = input.shape()[0];
    size_t channels = input.shape()[1];
    size_t in_h = input.shape()[2];
    size_t in_w = input.shape()[3];
    
    // Calculate output dimensions
    size_t out_h = (in_h + 2 * padding_h - pool_height) / stride_h + 1;
    size_t out_w = (in_w + 2 * padding_w - pool_width) / stride_w + 1;
    
    // Store input for backward pass
    input_cache = input;
    
    // Create output tensor
    TensorF output({batch_size, channels, out_h, out_w});
    
    // Perform average pooling
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    size_t count = 0;
                    
                    // Calculate average in the pooling window
                    for (size_t ph = 0; ph < pool_height; ++ph) {
                        for (size_t pw = 0; pw < pool_width; ++pw) {
                            size_t ih = oh * stride_h - padding_h + ph;
                            size_t iw = ow * stride_w - padding_w + pw;
                            
                            if (ih < in_h && iw < in_w) {
                                size_t input_idx = b * channels * in_h * in_w + 
                                                  c * in_h * in_w + 
                                                  ih * in_w + iw;
                                sum += input[input_idx];
                                count++;
                            }
                        }
                    }
                    
                    // Store average value
                    size_t output_idx = b * channels * out_h * out_w + 
                                       c * out_h * out_w + 
                                       oh * out_w + ow;
                    output[output_idx] = (count > 0) ? sum / static_cast<float>(count) : 0.0f;
                }
            }
        }
    }
    
    return output;
}

TensorF AvgPool2D::backward(const TensorF& grad_output) {
    if (grad_output.ndim() != 4) {
        throw std::invalid_argument("AvgPool2D backward expects 4D grad_output");
    }
    
    size_t batch_size = grad_output.shape()[0];
    size_t channels = grad_output.shape()[1];
    size_t out_h = grad_output.shape()[2];
    size_t out_w = grad_output.shape()[3];
    
    // Initialize gradient with respect to input
    TensorF grad_input(input_cache.shape());
    grad_input.fill(0.0f);
    
    // Propagate gradients equally to all positions in the pooling window
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    size_t output_idx = b * channels * out_h * out_w + 
                                       c * out_h * out_w + 
                                       oh * out_w + ow;
                    float grad_val = grad_output[output_idx];
                    
                    // Distribute gradient equally to all positions in the pooling window
                    size_t count = 0;
                    std::vector<std::pair<size_t, size_t>> valid_positions;
                    
                    for (size_t ph = 0; ph < pool_height; ++ph) {
                        for (size_t pw = 0; pw < pool_width; ++pw) {
                            size_t ih = oh * stride_h - padding_h + ph;
                            size_t iw = ow * stride_w - padding_w + pw;
                            
                            if (ih < input_cache.shape()[2] && iw < input_cache.shape()[3]) {
                                valid_positions.push_back({ih, iw});
                                count++;
                            }
                        }
                    }
                    
                    float distributed_grad = (count > 0) ? grad_val / static_cast<float>(count) : 0.0f;
                    
                    for (auto& pos : valid_positions) {
                        size_t ih = pos.first;
                        size_t iw = pos.second;
                        size_t grad_input_idx = b * channels * input_cache.shape()[2] * input_cache.shape()[3] + 
                                               c * input_cache.shape()[2] * input_cache.shape()[3] + 
                                               ih * input_cache.shape()[3] + iw;
                        grad_input[grad_input_idx] += distributed_grad;
                    }
                }
            }
        }
    }
    
    return grad_input;
}

void AvgPool2D::update_parameters(float learning_rate) {
    // AvgPool2D has no parameters to update
}

size_t AvgPool2D::get_parameter_count() const {
    return 0;
}

void AvgPool2D::zero_gradients() {
    // AvgPool2D has no gradients to zero
}

void AvgPool2D::initialize_parameters(std::mt19937& rng) {
    // AvgPool2D has no parameters to initialize
}

// BatchNorm layer implementation
BatchNorm::BatchNorm(size_t features, float momentum, float epsilon, std::string layer_name)
    : Layer(std::move(layer_name), true),
      features(features), momentum(momentum), epsilon(epsilon),
      gamma({1, features}), beta({1, features}),
      running_mean({1, features}), running_var({1, features}),
      grad_gamma({1, features}), grad_beta({1, features}) {
    // Initialize gamma to 1 and beta to 0
    gamma.fill(1.0f);
    beta.fill(0.0f);
    running_mean.fill(0.0f);
    running_var.fill(1.0f);  // Start with variance of 1
    grad_gamma.fill(0.0f);
    grad_beta.fill(0.0f);
}

TensorF BatchNorm::forward(const TensorF& input) {
    if (input.ndim() < 2) {
        throw std::invalid_argument("BatchNorm expects at least 2D input");
    }
    
    // For input of shape (N, C) or (N, C, H, W), normalize over the feature/channel dimension
    // Reshape to treat all non-feature dimensions as batch
    size_t batch_size = input.size() / features;
    
    if (input.size() % features != 0) {
        throw std::invalid_argument("BatchNorm input size not divisible by features");
    }
    
    // Store input for backward pass
    input_cache = input;
    
    // Calculate mean and variance for the batch
    TensorF batch_mean({1, features});
    TensorF batch_var({1, features});
    
    // Calculate mean
    for (size_t f = 0; f < features; ++f) {
        float sum = 0.0f;
        for (size_t b = 0; b < batch_size; ++b) {
            sum += input[b * features + f];
        }
        batch_mean[f] = sum / static_cast<float>(batch_size);
    }
    
    // Calculate variance
    for (size_t f = 0; f < features; ++f) {
        float sum = 0.0f;
        for (size_t b = 0; b < batch_size; ++b) {
            float diff = input[b * features + f] - batch_mean[f];
            sum += diff * diff;
        }
        batch_var[f] = sum / static_cast<float>(batch_size);
    }
    
    // Update running statistics
    for (size_t f = 0; f < features; ++f) {
        running_mean[f] = momentum * batch_mean[f] + (1.0f - momentum) * running_mean[f];
        running_var[f] = momentum * batch_var[f] + (1.0f - momentum) * running_var[f];
    }
    
    // Normalize
    x_centered_cache = TensorF(input.shape());
    x_norm_cache = TensorF(input.shape());
    inv_std_cache = TensorF({1, features}); // Store inverse std for each feature
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t f = 0; f < features; ++f) {
            x_centered_cache[b * features + f] = input[b * features + f] - batch_mean[f];
            float inv_std = 1.0f / std::sqrt(batch_var[f] + epsilon);
            x_norm_cache[b * features + f] = x_centered_cache[b * features + f] * inv_std;
            if (b == 0) { // Store inverse std for backward pass
                inv_std_cache[f] = inv_std;
            }
        }
    }
    
    // Scale and shift
    TensorF output(input.shape());
    for (size_t i = 0; i < input.size(); ++i) {
        size_t f = i % features;  // Get feature index
        output[i] = x_norm_cache[i] * gamma[f] + beta[f];
    }
    
    return output;
}

TensorF BatchNorm::backward(const TensorF& grad_output) {
    if (grad_output.size() != input_cache.size()) {
        throw std::invalid_argument("BatchNorm backward input size mismatch");
    }
    
    size_t batch_size = input_cache.size() / features;
    
    // Gradients for gamma and beta
    grad_gamma.fill(0.0f);
    grad_beta.fill(0.0f);
    
    for (size_t f = 0; f < features; ++f) {
        float sum_gamma = 0.0f;
        float sum_beta = 0.0f;
        for (size_t b = 0; b < batch_size; ++b) {
            size_t idx = b * features + f;
            sum_gamma += grad_output[idx] * x_norm_cache[idx];
            sum_beta += grad_output[idx];
        }
        grad_gamma[f] = sum_gamma;
        grad_beta[f] = sum_beta;
    }
    
    // Gradient for input
    TensorF grad_input(input_cache.shape());
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t f = 0; f < features; ++f) {
            size_t idx = b * features + f;
            
            // Gradient through scale and shift
            float grad_norm = grad_output[idx] * gamma[f];
            
            // Gradient of variance
            float grad_var = grad_norm * x_centered_cache[idx] * -0.5f *
                            std::pow(inv_std_cache[f], 3.0f);
            
            // Gradient of mean
            float grad_mean = grad_norm * -inv_std_cache[f];
            
            // Calculate additional components for the full gradient
            float mean_grad_var = 0.0f;
            for (size_t b2 = 0; b2 < batch_size; ++b2) {
                size_t idx2 = b2 * features + f;
                mean_grad_var += -2.0f * x_centered_cache[idx2] / static_cast<float>(batch_size);
            }
            mean_grad_var *= grad_var;
            
            // Combine gradients
            grad_input[idx] = grad_norm * inv_std_cache[f] +
                             grad_var * 2.0f * x_centered_cache[idx] / static_cast<float>(batch_size) +
                             (grad_mean + mean_grad_var) / static_cast<float>(batch_size);
        }
    }
    
    return grad_input;
}

void BatchNorm::update_parameters(float learning_rate) {
    if (!trainable) return;
    
    // Update gamma
    for (size_t i = 0; i < gamma.size(); ++i) {
        gamma[i] -= learning_rate * grad_gamma[i];
    }
    
    // Update beta
    for (size_t i = 0; i < beta.size(); ++i) {
        beta[i] -= learning_rate * grad_beta[i];
    }
}

size_t BatchNorm::get_parameter_count() const {
    return gamma.size() + beta.size();
}

void BatchNorm::zero_gradients() {
    grad_gamma.fill(0.0f);
    grad_beta.fill(0.0f);
}

void BatchNorm::initialize_parameters(std::mt19937& rng) {
    // BatchNorm parameters are already initialized in constructor
    // Gamma starts as 1s, beta starts as 0s
}

// LayerNorm layer implementation
LayerNorm::LayerNorm(size_t features, float epsilon, std::string layer_name)
    : Layer(std::move(layer_name), true),
      features(features), epsilon(epsilon),
      gamma({1, features}), beta({1, features}),
      grad_gamma({1, features}), grad_beta({1, features}) {
    // Initialize gamma to 1 and beta to 0
    gamma.fill(1.0f);
    beta.fill(0.0f);
    grad_gamma.fill(0.0f);
    grad_beta.fill(0.0f);
}

TensorF LayerNorm::forward(const TensorF& input) {
    if (input.ndim() < 2) {
        throw std::invalid_argument("LayerNorm expects at least 2D input");
    }
    
    // For input of shape (N, C) or (N, C, H, W), normalize over the last dimension
    // For simplicity, assume input is (batch_size, features) for now
    size_t batch_size = input.size() / features;
    
    if (input.size() % features != 0) {
        throw std::invalid_argument("LayerNorm input size not divisible by features");
    }
    
    // Store input for backward pass
    input_cache = input;
    
    // Calculate mean and variance for each sample
    mean_cache = TensorF({batch_size, 1});
    var_cache = TensorF({batch_size, 1});
    
    // Calculate mean and variance for each sample in the batch
    for (size_t b = 0; b < batch_size; ++b) {
        float sum = 0.0f;
        for (size_t f = 0; f < features; ++f) {
            sum += input[b * features + f];
        }
        mean_cache[b] = sum / static_cast<float>(features);
        
        float var_sum = 0.0f;
        for (size_t f = 0; f < features; ++f) {
            float diff = input[b * features + f] - mean_cache[b];
            var_sum += diff * diff;
        }
        var_cache[b] = var_sum / static_cast<float>(features);
    }
    
    // Normalize each sample independently
    x_norm_cache = TensorF(input.shape());
    
    for (size_t b = 0; b < batch_size; ++b) {
        float inv_std = 1.0f / std::sqrt(var_cache[b] + epsilon);
        for (size_t f = 0; f < features; ++f) {
            float centered = input[b * features + f] - mean_cache[b];
            x_norm_cache[b * features + f] = centered * inv_std;
        }
    }
    
    // Scale and shift
    TensorF output(input.shape());
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t f = 0; f < features; ++f) {
            output[b * features + f] = x_norm_cache[b * features + f] * gamma[f] + beta[f];
        }
    }
    
    return output;
}

TensorF LayerNorm::backward(const TensorF& grad_output) {
    if (grad_output.size() != input_cache.size()) {
        throw std::invalid_argument("LayerNorm backward input size mismatch");
    }
    
    size_t batch_size = input_cache.size() / features;
    
    // Gradients for gamma and beta
    grad_gamma.fill(0.0f);
    grad_beta.fill(0.0f);
    
    for (size_t f = 0; f < features; ++f) {
        float sum_gamma = 0.0f;
        for (size_t b = 0; b < batch_size; ++b) {
            sum_gamma += grad_output[b * features + f] * x_norm_cache[b * features + f];
        }
        grad_gamma[f] = sum_gamma;
        
        float sum_beta = 0.0f;
        for (size_t b = 0; b < batch_size; ++b) {
            sum_beta += grad_output[b * features + f];
        }
        grad_beta[f] = sum_beta;
    }
    
    // Gradient for input
    TensorF grad_input(input_cache.shape());
    
    for (size_t b = 0; b < batch_size; ++b) {
        float inv_std = 1.0f / std::sqrt(var_cache[b] + epsilon);
        
        // Calculate intermediate values for gradient computation
        float grad_norm_sum = 0.0f;
        float grad_norm_weighted_sum = 0.0f;
        
        for (size_t f = 0; f < features; ++f) {
            size_t idx = b * features + f;
            float grad_norm = grad_output[idx] * gamma[f];
            grad_norm_sum += grad_norm;
            grad_norm_weighted_sum += grad_norm * x_norm_cache[idx];
        }
        
        for (size_t f = 0; f < features; ++f) {
            size_t idx = b * features + f;
            
            float grad_norm = grad_output[idx] * gamma[f];
            
            // Combine all gradient components
            grad_input[idx] = inv_std * (
                grad_norm - 
                grad_norm_sum / static_cast<float>(features) - 
                x_norm_cache[idx] * grad_norm_weighted_sum / static_cast<float>(features)
            );
        }
    }
    
    return grad_input;
}

void LayerNorm::update_parameters(float learning_rate) {
    if (!trainable) return;
    
    // Update gamma
    for (size_t i = 0; i < gamma.size(); ++i) {
        gamma[i] -= learning_rate * grad_gamma[i];
    }
    
    // Update beta
    for (size_t i = 0; i < beta.size(); ++i) {
        beta[i] -= learning_rate * grad_beta[i];
    }
}

size_t LayerNorm::get_parameter_count() const {
    return gamma.size() + beta.size();
}

void LayerNorm::zero_gradients() {
    grad_gamma.fill(0.0f);
    grad_beta.fill(0.0f);
}

void LayerNorm::initialize_parameters(std::mt19937& rng) {
    // LayerNorm parameters are already initialized in constructor
    // Gamma starts as 1s, beta starts as 0s
}

// Dropout layer implementation
Dropout::Dropout(float dropout_rate, std::string layer_name)
    : Layer(std::move(layer_name), false), rate(dropout_rate), training_mode(true) {}

TensorF Dropout::forward(const TensorF& input) {
    if (rate <= 0.0f || rate >= 1.0f) {
        // No dropout
        return input;
    }
    
    if (!training_mode) {
        // In inference mode, return input unchanged
        return input;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0f - rate);
    
    // Create mask tensor
    mask_cache = TensorF(input.shape());
    
    TensorF output(input.shape());
    
    for (size_t i = 0; i < input.size(); ++i) {
        bool keep = dist(gen);
        mask_cache[i] = keep ? 1.0f : 0.0f;
        output[i] = input[i] * mask_cache[i] / (1.0f - rate);
    }
    
    return output;
}

TensorF Dropout::backward(const TensorF& grad_output) {
    if (rate <= 0.0f || rate >= 1.0f || !training_mode) {
        // No dropout during backward pass if not in training mode or rate is 0
        TensorF grad_input(grad_output.shape());
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_input[i] = grad_output[i];
        }
        return grad_input;
    }
    
    if (mask_cache.size() != grad_output.size()) {
        throw std::invalid_argument("Dropout backward input size mismatch");
    }
    
    TensorF grad_input(grad_output.shape());
    
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = grad_output[i] * mask_cache[i] / (1.0f - rate);
    }
    
    return grad_input;
}

void Dropout::update_parameters(float learning_rate) {
    // Dropout has no parameters to update
}

size_t Dropout::get_parameter_count() const {
    return 0;
}

void Dropout::zero_gradients() {
    // Dropout has no gradients to zero
}

void Dropout::initialize_parameters(std::mt19937& rng) {
    // Dropout has no parameters to initialize
}

// Embedding layer implementation
Embedding::Embedding(size_t num_embeddings, size_t embedding_dim, std::string layer_name)
    : Layer(std::move(layer_name), true),
      num_embeddings(num_embeddings), embedding_dim(embedding_dim),
      weights({num_embeddings, embedding_dim}),
      grad_weights({num_embeddings, embedding_dim}) {
    grad_weights.fill(0.0f);
}

TensorF Embedding::forward(const TensorF& input) {
    // Input should be a tensor of indices
    if (input.ndim() != 1 && input.ndim() != 2) {
        throw std::invalid_argument("Embedding layer expects 1D or 2D input of indices");
    }
    
    // Verify that all indices are valid
    for (size_t i = 0; i < input.size(); ++i) {
        size_t idx = static_cast<size_t>(input[i]);
        if (idx >= num_embeddings) {
            throw std::invalid_argument("Embedding index out of range");
        }
    }
    
    // Create output tensor: [input_shape..., embedding_dim]
    std::vector<size_t> output_shape = input.shape();
    output_shape.push_back(embedding_dim);
    
    TensorF output(output_shape);
    
    // For each index in input, copy the corresponding embedding vector
    for (size_t i = 0; i < input.size(); ++i) {
        size_t idx = static_cast<size_t>(input[i]);
        for (size_t j = 0; j < embedding_dim; ++j) {
            output[i * embedding_dim + j] = weights[idx * embedding_dim + j];
        }
    }
    
    return output;
}

TensorF Embedding::backward(const TensorF& grad_output) {
    // grad_output shape: [input_size, embedding_dim]
    if (grad_output.ndim() < 2 || grad_output.shape().back() != embedding_dim) {
        throw std::invalid_argument("Embedding backward input dimension mismatch");
    }
    
    size_t input_size = grad_output.size() / embedding_dim;
    
    // Zero out previous gradients
    grad_weights.fill(0.0f);
    
    // Accumulate gradients for each embedding vector that was used
    for (size_t i = 0; i < input_size; ++i) {
        // Find which index this corresponds to in the original input
        // This requires knowing the original input indices, which we don't have stored
        // In practice, we would need to store the input indices during forward pass
        // For now, we'll assume the input was a sequence 0, 1, 2, ..., input_size-1
        // This is a simplified implementation
        size_t idx = i % num_embeddings;  // In real implementation, would use original indices
        
        for (size_t j = 0; j < embedding_dim; ++j) {
            grad_weights[idx * embedding_dim + j] += grad_output[i * embedding_dim + j];
        }
    }
    
    // Return gradient with respect to input indices (zeros since indices are discrete)
    TensorF grad_input({input_size});
    grad_input.fill(0.0f);
    
    return grad_input;
}

void Embedding::update_parameters(float learning_rate) {
    if (!trainable) return;
    
    // Update embedding weights
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate * grad_weights[i];
    }
}

size_t Embedding::get_parameter_count() const {
    return weights.size();
}

void Embedding::zero_gradients() {
    grad_weights.fill(0.0f);
}

void Embedding::initialize_parameters(std::mt19937& rng) {
    // Initialize embeddings with small random values
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = dist(rng);
    }
}

} // namespace dnn