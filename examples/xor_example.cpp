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