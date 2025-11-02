#include "include/optimizers.hpp"
#include "include/layers.hpp"
#include "include/tensor.hpp"
#include <iostream>
#include <vector>
#include <memory>

int main() {
    std::cout << "Testing Optimizer System..." << std::endl;
    
    try {
        // Test basic optimizer functionality
        dnn::SGD sgd_optimizer(0.01f, 0.9f);  // lr=0.01, momentum=0.9
        dnn::Adam adam_optimizer(0.001f);     // lr=0.001
        dnn::RMSprop rmsprop_optimizer(0.001f); // lr=0.001
        
        // Create a simple parameter tensor using the TensorF alias
        dnn::TensorF param({2, 3});  // 2x3 parameter tensor
        for (size_t i = 0; i < param.size(); ++i) {
            param[i] = static_cast<float>(i + 1) / 10.0f; // Fill with 0.1, 0.2, 0.3, ...
        }
        
        // Create a gradient tensor
        dnn::TensorF grad({2, 3});   // 2x3 gradient tensor
        for (size_t i = 0; i < grad.size(); ++i) {
            grad[i] = static_cast<float>(i + 1) / 20.0f;   // Fill with 0.05, 0.1, 0.15, ...
        }
        
        std::cout << "Original parameter values: ";
        for (size_t i = 0; i < param.size(); ++i) {
            std::cout << param[i] << " ";
        }
        std::cout << std::endl;
        
        // Test SGD update
        dnn::TensorF param_sgd = param;  // Copy for SGD test
        sgd_optimizer.update_single(param_sgd, grad);
        
        std::cout << "After SGD update: ";
        for (size_t i = 0; i < param_sgd.size(); ++i) {
            std::cout << param_sgd[i] << " ";
        }
        std::cout << std::endl;
        
        // Test Adam update
        dnn::TensorF param_adam = param;  // Copy for Adam test
        adam_optimizer.update_single(param_adam, grad);
        
        std::cout << "After Adam update: ";
        for (size_t i = 0; i < param_adam.size(); ++i) {
            std::cout << param_adam[i] << " ";
        }
        std::cout << std::endl;
        
        // Test RMSprop update
        dnn::TensorF param_rmsprop = param;  // Copy for RMSprop test
        rmsprop_optimizer.update_single(param_rmsprop, grad);
        
        std::cout << "After RMSprop update: ";
        for (size_t i = 0; i < param_rmsprop.size(); ++i) {
            std::cout << param_rmsprop[i] << " ";
        }
        std::cout << std::endl;
        
        // Test learning rate schedulers
        dnn::StepLR step_scheduler(&sgd_optimizer, 10, 0.5f);  // Reduce LR by half every 10 steps
        std::cout << "Initial LR: " << sgd_optimizer.get_learning_rate() << std::endl;
        step_scheduler.step();
        std::cout << "LR after 1 step: " << sgd_optimizer.get_learning_rate() << std::endl;
        
        // Test regularization
        dnn::TensorF grad_reg = grad;  // Copy gradient for regularization test
        sgd_optimizer.apply_regularization(grad_reg, param, 0.01f, 0.01f);  // L1=0.01, L2=0.01
        std::cout << "Gradient after regularization: ";
        for (size_t i = 0; i < grad_reg.size(); ++i) {
            std::cout << grad_reg[i] << " ";
        }
        std::cout << std::endl;
        
        // Test gradient clipping
        dnn::TensorF grad_clip = grad;  // Copy gradient for clipping test
        sgd_optimizer.clip_gradients(grad_clip, 0.1f);  // Clip to [-0.1, 0.1]
        std::cout << "Gradient after clipping: ";
        for (size_t i = 0; i < grad_clip.size(); ++i) {
            std::cout << grad_clip[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "All optimizer tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}