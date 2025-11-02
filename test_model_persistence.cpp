#include "dnn.hpp"
#include "model_serializer.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

int main() {
    std::cout << "Testing Model Persistence..." << std::endl;
    
    // Create a simple model
    dnn::Config config;
    dnn::Model model(config);
    
    // Add layers to the model
    model.add(std::make_unique<dnn::Dense>(2, 8, dnn::Activation::ReLU));
    model.add(std::make_unique<dnn::Dense>(8, 1, dnn::Activation::Sigmoid));
    
    // Compile with optimizer
    auto optimizer = std::make_unique<dnn::SGD>(0.01, 0.9);
    model.compile(std::move(optimizer));
    
    // Create some dummy training data
    dnn::Matrix X({4, 2});
    X.data[0] = 0.0; X.data[1] = 0.0;
    X.data[2] = 0.0; X.data[3] = 1.0;
    X.data[4] = 1.0; X.data[5] = 0.0;
    X.data[6] = 1.0; X.data[7] = 1.0;
    
    dnn::Matrix y({4, 1});
    y.data[0] = 0.0;
    y.data[1] = 1.0;
    y.data[2] = 1.0;
    y.data[3] = 0.0;
    
    // Train the model briefly to set some parameters
    std::mt19937 rng(42);
    std::cout << "Training model..." << std::endl;
    model.fit(X, y, 10, dnn::LossFunction::MSE, rng, 0.0, false); // 10 epochs, no verbose
    
    // Save the model
    std::string filepath = "test_model.dnn";
    std::cout << "Saving model to " << filepath << std::endl;
    model.save(filepath);
    
    // Create a new model and load the saved one
    dnn::Model loaded_model(config);
    std::cout << "Loading model from " << filepath << std::endl;
    loaded_model.load(filepath);
    
    // Verify that the loaded model has the same structure
    assert(loaded_model.layers.size() == model.layers.size());
    std::cout << "Layer count matches: " << loaded_model.layers.size() << std::endl;
    
    // Test predictions match
    dnn::Matrix original_pred = model.predict(X);
    dnn::Matrix loaded_pred = loaded_model.predict(X);
    
    bool predictions_match = true;
    for (size_t i = 0; i < original_pred.size; ++i) {
        if (std::abs(original_pred.data[i] - loaded_pred.data[i]) > 1e-6) {
            predictions_match = false;
            break;
        }
    }
    
    if (predictions_match) {
        std::cout << "SUCCESS: Model loaded correctly, predictions match!" << std::endl;
    } else {
        std::cout << "FAILURE: Predictions don't match after loading" << std::endl;
        return 1;
    }
    
    // Test that both models have same parameter count
    size_t original_params = model.get_parameter_count();
    size_t loaded_params = loaded_model.get_parameter_count();
    
    if (original_params == loaded_params) {
        std::cout << "SUCCESS: Parameter counts match (" << original_params << ")" << std::endl;
    } else {
        std::cout << "FAILURE: Parameter counts don't match (" 
                  << original_params << " vs " << loaded_params << ")" << std::endl;
        return 1;
    }
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}