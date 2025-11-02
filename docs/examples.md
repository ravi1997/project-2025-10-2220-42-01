# Examples and Tutorials

## Overview
This document provides practical examples and tutorials for using the DNN library. Each example demonstrates different aspects of the library and can be used as a starting point for your own projects.

## Getting Started Example

### Basic XOR Problem
This example demonstrates a simple neural network that learns the XOR function.

```cpp
#include "dnn.hpp"
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::cout << "XOR Example with DNN Library\n";
    std::cout << "=============================\n";
    
    // Create XOR dataset
    dnn::Matrix X({4, 2});
    X(0, 0) = 0.0; X(0, 1) = 0.0;
    X(1, 0) = 0.0; X(1, 1) = 1.0;
    X(2, 0) = 1.0; X(2, 1) = 0.0;
    X(3, 0) = 1.0; X(3, 1) = 1.0;
    
    dnn::Matrix y({4, 1});
    y(0, 0) = 0.0;
    y(1, 0) = 1.0;
    y(2, 0) = 1.0;
    y(3, 0) = 0.0;
    
    // Create model
    dnn::Config config;
    dnn::Model model(config);
    
    // Add layers
    model.add(std::make_unique<dnn::Dense>(2, 8, dnn::Activation::ReLU));
    model.add(std::make_unique<dnn::Dense>(8, 1, dnn::Activation::Sigmoid));
    
    // Compile model
    auto optimizer = std::make_unique<dnn::SGD>(0.1, 0.9);
    model.compile(std::move(optimizer));
    
    // Train model
    std::mt19937 rng(42);
    model.fit(X, y, 1000, dnn::LossFunction::MSE, rng, 0.0, true);
    
    // Test model
    std::cout << "\nTesting:\n";
    dnn::Matrix predictions = model.predict(X);
    for (std::size_t i = 0; i < X.shape[0]; ++i) {
        std::cout << X(i, 0) << " XOR " << X(i, 1) << " = " << predictions(i, 0) << " (expected: " << y(i, 0) << ")\n";
    }
    
    return 0;
}
```

### Key Concepts Demonstrated
- Creating input/output data matrices
- Building a neural network with Dense layers
- Using ReLU and Sigmoid activations
- Training with SGD optimizer
- Making predictions

## MNIST Classification Example

### Synthetic MNIST Example
This example demonstrates a more complex network suitable for image classification.

```cpp
#include "dnn.hpp"
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::cout << "MNIST Digit Classification Example with DNN Library\n";
    std::cout << "==================================================\n";
    
    // Create a simple synthetic MNIST-like dataset (28x28 = 784 pixels)
    const std::size_t num_samples = 100;
    const std::size_t input_size = 784;  // 28x28
    const std::size_t num_classes = 10;
    
    // Generate synthetic data
    dnn::Matrix X({num_samples, input_size});
    std::vector<int> labels(num_samples);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> label_dist(0, 9);
    std::uniform_real_distribution<> pixel_dist(0.0, 1.0);
    
    // Fill with random data
    for (std::size_t i = 0; i < num_samples; ++i) {
        labels[i] = label_dist(gen);
        for (std::size_t j = 0; j < input_size; ++j) {
            X(i, j) = pixel_dist(gen);
        }
    }
    
    // Convert labels to one-hot encoding
    dnn::Matrix y = dnn::one_hot(labels, static_cast<int>(num_classes));
    
    // Split data into train/test
    auto [X_train, y_train] = dnn::train_test_split(X, y, 0.2, gen);
    
    // Create model
    dnn::Config config;
    dnn::Model model(config);
    
    // Add layers for MNIST classification
    model.add(std::make_unique<dnn::Dense>(784, 128, dnn::Activation::ReLU));
    model.add(std::make_unique<dnn::Dropout>(0.2));
    model.add(std::make_unique<dnn::Dense>(128, 64, dnn::Activation::ReLU));
    model.add(std::make_unique<dnn::Dropout>(0.2));
    model.add(std::make_unique<dnn::Dense>(64, 10, dnn::Activation::Softmax));
    
    // Compile model with Adam optimizer
    auto optimizer = std::make_unique<dnn::Adam>(0.001, 0.9, 0.999);
    model.compile(std::move(optimizer));
    
    // Print model summary
    model.print_summary();
    
    // Train model
    std::cout << "\nTraining model...\n";
    model.fit(X_train, y_train, 50, dnn::LossFunction::CrossEntropy, gen, 0.1, true);
    
    // Evaluate model
    std::cout << "\nEvaluating model...\n";
    double train_accuracy = dnn::accuracy(model.predict(X_train), labels);
    std::cout << "Train Accuracy: " << train_accuracy << "\n";
    
    return 0;
}
```

### Key Concepts Demonstrated
- Creating synthetic image data
- Using dropout for regularization
- One-hot encoding for classification
- Adam optimizer for training
- Model evaluation and accuracy calculation

## Advanced Examples

### Convolutional Network
Example of using Conv2D layers for image processing:

```cpp
// Note: This is conceptual as the input format for Conv2D needs to be properly structured
dnn::Model cnn_model;

// Add convolutional layers
cnn_model.add(std::make_unique<dnn::Conv2D>(1, 32, 3, 3, 1, 1, 1, 1, dnn::Activation::ReLU));
cnn_model.add(std::make_unique<dnn::MaxPool2D>(2, 2, 2, 2));
cnn_model.add(std::make_unique<dnn::Conv2D>(32, 64, 3, 3, 1, 1, 1, 1, dnn::Activation::ReLU));
cnn_model.add(std::make_unique<dnn::MaxPool2D>(2, 2, 2, 2));

// Add dense layers for classification
cnn_model.add(std::make_unique<dnn::Dense>(3136, 128, dnn::Activation::ReLU));  // Adjust size as needed
cnn_model.add(std::make_unique<dnn::Dense>(128, 10, dnn::Activation::Softmax));
```

### Custom Training Loop
Example of implementing a custom training loop for more control:

```cpp
void custom_train(dnn::Model& model, 
                  const dnn::Matrix& X, 
                  const dnn::Matrix& y, 
                  int epochs) {
    std::mt19937 rng(std::random_device{}());
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        dnn::Matrix predictions = model.forward(X);
        
        // Compute loss
        double loss = model.compute_loss(predictions, y, dnn::LossFunction::CrossEntropy);
        
        // Manual backward pass
        dnn::LossResult loss_result = dnn::compute_loss(y, predictions, dnn::LossFunction::CrossEntropy);
        model.backward(loss_result.gradient);
        
        // Manual parameter update (if needed)
        // Note: This is typically handled by the training loop in fit()
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
    }
}
```

## Common Patterns and Best Practices

### Data Preparation
```cpp
// Normalize input data
dnn::Matrix normalized_X = dnn::normalize(X, mean, stddev);

// Create one-hot encoded labels
dnn::Matrix one_hot_y = dnn::one_hot(labels, num_classes);

// Split data for training and validation
auto [X_train, X_val] = dnn::train_test_split(X, y, 0.2, rng);
```

### Model Architecture Patterns
```cpp
// Feedforward network
dnn::Model ff_model;
ff_model.add(std::make_unique<dnn::Dense>(input_size, hidden1, dnn::Activation::ReLU));
ff_model.add(std::make_unique<dnn::Dropout>(0.2));
ff_model.add(std::make_unique<dnn::Dense>(hidden1, hidden2, dnn::Activation::ReLU));
ff_model.add(std::make_unique<dnn::Dropout>(0.2));
ff_model.add(std::make_unique<dnn::Dense>(hidden2, output_size, dnn::Activation::Softmax));

// With batch normalization
dnn::Model bn_model;
ff_model.add(std::make_unique<dnn::Dense>(input_size, hidden1, dnn::Activation::Linear));
ff_model.add(std::make_unique<dnn::BatchNorm>(hidden1));
ff_model.add(std::make_unique<dnn::Dense>(hidden1, dnn::Activation::ReLU));
```

### Training Configuration
```cpp
// Configure model for training
dnn::Config config;
config.learning_rate = 0.001;
config.batch_size = 32;
config.max_epochs = 10;

dnn::Model model(config);

// Choose appropriate optimizer
auto optimizer = std::make_unique<dnn::Adam>(0.001, 0.9, 0.9);
// Or for SGD with momentum
// auto optimizer = std::make_unique<dnn::SGD>(0.01, 0.9);
// Or for RMSprop
// auto optimizer = std::make_unique<dnn::RMSprop>(0.001, 0.9, 1e-8);
// Or for AdamW
// auto optimizer = std::make_unique<dnn::AdamW>(0.001, 0.9, 0.99, 1e-4);
```

## Troubleshooting Common Issues

### Vanishing Gradients
- Use ReLU or its variants instead of sigmoid/tanh for deep networks
- Consider batch normalization
- Use skip connections (not yet implemented in this library)

### Overfitting
- Add dropout layers
- Use L2 regularization (weight decay in optimizers)
- Implement early stopping
- Increase training data

### Slow Convergence
- Try different optimizers (Adam often works well)
- Adjust learning rate
- Use appropriate weight initialization
- Normalize input data

## Building and Running Examples

### CMake Configuration
The examples are built as part of the main CMake project:

```cmake
add_executable(xor_example examples/xor_example.cpp)
target_link_libraries(xor_example PRIVATE dnn)

add_executable(mnist_example examples/mnist_example.cpp)
target_link_libraries(mnist_example PRIVATE dnn)
```

### Compilation
```bash
mkdir build
cd build
cmake ..
make xor_example
make mnist_example
```

### Running
```bash
./xor_example
./mnist_example
```

These examples provide a solid foundation for building your own neural networks using the DNN library. Start with the XOR example to understand the basics, then move to more complex examples as needed for your specific use case.