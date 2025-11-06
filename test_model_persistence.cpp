#include "dnn.hpp"
#include "model_serializer.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <numeric>
#include <cmath>

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
    X.data()[0] = 0.0; X.data()[1] = 0.0;
    X.data()[2] = 0.0; X.data()[3] = 1.0;
    X.data()[4] = 1.0; X.data()[5] = 0.0;
    X.data()[6] = 1.0; X.data()[7] = 1.0;

    dnn::Matrix y({4, 1});
    y.data()[0] = 0.0;
    y.data()[1] = 1.0;
    y.data()[2] = 1.0;
    y.data()[3] = 0.0;
    
    // Train the model briefly to set some parameters
    std::mt19937 rng(42);
    std::cout << "Training model..." << std::endl;
    std::cout << "Initial parameter count: " << model.get_parameter_count() << std::endl;
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
    assert(loaded_model.layer_count() == model.layer_count());
    std::cout << "Layer count matches: " << loaded_model.layer_count() << std::endl;

    assert(loaded_model.get_optimizer() != nullptr);
    std::cout << "Optimizer restored successfully." << std::endl;
    
    // Test predictions match
    dnn::Matrix original_pred = model.predict(X);
    dnn::Matrix loaded_pred = loaded_model.predict(X);
    
    bool predictions_match = true;
    for (size_t i = 0; i < original_pred.size(); ++i) {
        if (std::abs(original_pred.data()[i] - loaded_pred.data()[i]) > 1e-6) {
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
    std::cout << "Original parameter count: " << original_params << std::endl;
    std::cout << "Loaded parameter count: " << loaded_params << std::endl;
    
    if (original_params == loaded_params) {
        std::cout << "SUCCESS: Parameter counts match (" << original_params << ")" << std::endl;
    } else {
        std::cout << "FAILURE: Parameter counts don't match (" 
                  << original_params << " vs " << loaded_params << ")" << std::endl;
        return 1;
    }

    // Verify Conv2D gradients propagate correctly
    {
        dnn::Conv2D conv_layer(1, 1, 1, 1);
        dnn::Matrix conv_input({1, 4});
        conv_input.data = {1.0, 2.0, 3.0, 4.0};
        dnn::Matrix conv_output = conv_layer.forward(conv_input);
        dnn::Matrix grad_out(conv_output.shape()[0], conv_output.shape()[1]);
        std::fill(grad_out.data(), grad_out.data() + grad_out.size(), 1.0);
        dnn::Matrix grad_in = conv_layer.backward(grad_out);
        (void)grad_in; // silence unused warning in case optimization removes it
        double expected_weight_grad = std::accumulate(conv_input.data(), conv_input.data() + conv_input.size(), 0.0);
        double expected_bias_grad = static_cast<double>(grad_out.size());
        assert(std::abs(conv_layer.weight_velocity.data()[0] - expected_weight_grad) < 1e-9);
        assert(std::abs(conv_layer.bias_velocity.data()[0] - expected_bias_grad) < 1e-9);
    }

    // Numerical stability smoke checks
    {
        dnn::Matrix logits({1, 2});
        logits(0, 0) = 1000.0;
        logits(0, 1) = -1000.0;

        dnn::Matrix softplus_out = dnn::apply_activation(logits, dnn::Activation::Softplus);
        assert(std::isfinite(softplus_out(0, 0)));
        assert(std::isfinite(softplus_out(0, 1)));

        dnn::Matrix softmax_out = dnn::apply_activation(logits, dnn::Activation::Softmax);
        for (size_t i = 0; i < softmax_out.size(); ++i) {
            double v = softmax_out.data()[i];
            assert(std::isfinite(v));
        }

        dnn::Matrix target({1, 2});
        target(0, 0) = 1.0;
        target(0, 1) = 0.0;

        dnn::LossResult loss = dnn::compute_loss(target, softmax_out, dnn::LossFunction::CrossEntropy);
        assert(std::isfinite(loss.value));
        for (size_t i = 0; i < loss.gradient.size(); ++i) {
            double v = loss.gradient.data()[i];
            assert(std::isfinite(v));
        }

        dnn::Matrix bce_pred({1, 1});
        dnn::Matrix bce_true({1, 1});
        bce_pred(0, 0) = 1.0;  // extreme probability
        bce_true(0, 0) = 0.0;
        auto bce_loss = dnn::compute_loss(bce_true, bce_pred, dnn::LossFunction::BinaryCrossEntropy);
        assert(std::isfinite(bce_loss.value));
        for (size_t i = 0; i < bce_loss.gradient.size(); ++i) {
            double v = bce_loss.gradient.data()[i];
            assert(std::isfinite(v));
        }

        dnn::Matrix kl_p({1, 2});
        dnn::Matrix kl_q({1, 2});
        kl_p(0, 0) = 1.0; kl_p(0, 1) = 0.0;
        kl_q(0, 0) = 0.0; kl_q(0, 1) = 1.0;
        auto kl_loss = dnn::compute_loss(kl_p, kl_q, dnn::LossFunction::KLDivergence);
        assert(std::isfinite(kl_loss.value));
        for (size_t i = 0; i < kl_loss.gradient.size(); ++i) {
            double v = kl_loss.gradient.data()[i];
            assert(std::isfinite(v));
        }
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
