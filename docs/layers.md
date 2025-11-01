# Layer Components Documentation

## Overview
The DNN library provides a comprehensive set of neural network layers that can be combined to create complex architectures. All layers inherit from the base `Layer` class and follow a consistent interface for forward and backward propagation.

## Layer Hierarchy

### Base Layer Class
```cpp
struct Layer {
    std::string name;
    bool trainable;
    
    explicit Layer(std::string layer_name = "layer", bool is_trainable = true);
    virtual ~Layer() = default;
    
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& grad_output) = 0;
    virtual void update_parameters(const Optimizer& opt) = 0;
    virtual std::size_t get_parameter_count() const = 0;
};
```

The base `Layer` class defines the interface that all layers must implement:
- `forward()`: Computes the output given an input
- `backward()`: Computes gradients for backpropagation
- `update_parameters()`: Updates trainable parameters using the optimizer
- `get_parameter_count()`: Returns the number of trainable parameters

## Implemented Layers

### Dense Layer
The `Dense` layer implements a fully connected neural network layer.

#### Constructor
```cpp
Dense(std::size_t in, std::size_t out,
      Activation act = Activation::ReLU,
      std::string layer_name = "dense",
      std::mt19937* rng = nullptr);
```

#### Features
- Matrix multiplication: `output = input @ weights + bias`
- Configurable activation functions
- Xavier/Glorot weight initialization
- Parameter and gradient caching for backpropagation
- Optimizer state management (momentum, RMS, etc.)

#### Usage Example
```cpp
// Create a dense layer with 10 input features and 20 output features
auto dense_layer = std::make_unique<dnn::Dense>(10, 20, dnn::Activation::ReLU);
```

### Conv2D Layer
The `Conv2D` layer implements 2D convolution operations.

#### Constructor
```cpp
Conv2D(std::size_t in_ch, std::size_t out_ch,
       std::size_t kh, std::size_t kw,
       std::size_t s_h = 1, std::size_t s_w = 1,
       std::size_t p_h = 0, std::size_t p_w = 0,
       Activation act = Activation::ReLU,
       std::string layer_name = "conv2d",
       std::mt19937* rng = nullptr);
```

#### Features
- Configurable kernel size, stride, and padding
- Flattened weight representation for efficient computation
- He initialization for ReLU networks
- Parameter and gradient caching
- Optimizer state management

#### Usage Example
```cpp
// Create a Conv2D layer: 3 input channels, 16 output channels, 3x3 kernel
auto conv_layer = std::make_unique<dnn::Conv2D>(3, 16, 3, 3, 1, 1, 1, 1, dnn::Activation::ReLU);
```

### MaxPool2D Layer
The `MaxPool2D` layer implements 2D max pooling operations.

#### Constructor
```cpp
MaxPool2D(std::size_t ph, std::size_t pw,
          std::size_t s_h = 1, std::size_t s_w = 1,
          std::string layer_name = "maxpool2d");
```

#### Features
- Configurable pool size and stride
- Non-trainable layer (no parameters to update)
- Position tracking for gradient computation
- Memory efficient implementation

#### Usage Example
```cpp
// Create a MaxPool2D layer with 2x2 pooling and stride 2
auto pool_layer = std::make_unique<dnn::MaxPool2D>(2, 2, 2, 2);
```

### Dropout Layer
The `Dropout` layer implements dropout regularization.

#### Constructor
```cpp
Dropout(double dropout_rate = 0.5, std::string layer_name = "dropout");
```

#### Features
- Configurable dropout rate
- Random mask generation during forward pass
- Consistent mask application during backward pass
- Non-trainable layer
- Different behavior during training vs inference

#### Usage Example
```cpp
// Create a dropout layer with 50% dropout rate
auto dropout_layer = std::make_unique<dnn::Dropout>(0.5);
```

### BatchNorm Layer
The `BatchNorm` layer implements batch normalization.

#### Constructor
```cpp
BatchNorm(std::size_t feat, 
          double mom = 0.1, 
          double eps = 1e-5,
          std::string layer_name = "batchnorm");
```

#### Features
- Learnable scale (gamma) and shift (beta) parameters
- Running statistics for inference
- Momentum for running statistics update
- Proper gradient computation through normalization
- Optimizer state management for gamma/beta

#### Usage Example
```cpp
// Create a batch normalization layer for 64 features
auto batch_norm_layer = std::make_unique<dnn::BatchNorm>(64);
```

## Layer Composition
Layers are combined using the `Model` class:

```cpp
dnn::Model model;
model.add(std::make_unique<dnn::Dense>(784, 128, dnn::Activation::ReLU));
model.add(std::make_unique<dnn::Dropout>(0.2));
model.add(std::make_unique<dnn::Dense>(128, 10, dnn::Activation::Softmax));
```

## Forward Pass Process
1. Input is passed to the first layer
2. Each layer processes the input and passes output to the next layer
3. Internal states and intermediate values are cached for backward pass

## Backward Pass Process
1. Gradient of loss with respect to output is computed
2. Gradients flow backward through each layer
3. Each layer computes gradients w.r.t. its inputs and parameters
4. Trainable parameters are updated via the optimizer

## Memory Management
- Each layer caches necessary values for backpropagation
- Memory is automatically managed through RAII
- Temporary computations are efficiently handled

## Best Practices
1. Use appropriate initialization methods for different layer types
2. Consider the order of layers (e.g., BatchNorm before activation vs after)
3. Use dropout appropriately to prevent overfitting
4. Monitor parameter counts to understand model complexity
5. Choose activation functions based on the problem type