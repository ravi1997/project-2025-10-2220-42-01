#include "dnn.hpp"
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::cout << "MNIST Digit Classification Example with DNN Library\n";
    std::cout << "==================================================\n";
    
    // Create a simple synthetic MNIST-like dataset (28x28 = 784 pixels)
    // In a real application, you would load actual MNIST data
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