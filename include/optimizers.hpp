#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include "tensor.hpp"
#include "utils.hpp"
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iterator>

namespace dnn {

// Base Optimizer class with virtual interface for parameter updates
class Optimizer {
protected:
    float learning_rate;
    float initial_lr;
    
public:
    explicit Optimizer(float lr = 0.001f) : learning_rate(lr), initial_lr(lr) {}
    virtual ~Optimizer() = default;
    
    // Virtual interface for updating parameters
    virtual void update(const std::vector<TensorF*>& params, 
                       const std::vector<TensorF*>& gradients) = 0;
    
    // Method to update a single parameter tensor with its gradient
    virtual void update_single(TensorF& param, const TensorF& grad) = 0;
    
    // Learning rate management
    void set_learning_rate(float lr) { learning_rate = lr; }
    float get_learning_rate() const { return learning_rate; }
    void reset_learning_rate() { learning_rate = initial_lr; }
    
    // Apply regularization to gradients
    virtual void apply_regularization(TensorF& grad, const TensorF& param, 
                                    float l1_reg = 0.0f, float l2_reg = 0.0f) {
        if (l1_reg > 0.0f) {
            // Add L1 regularization term to gradient: sign(param) * l1_reg
            for (size_t i = 0; i < param.size(); ++i) {
                float param_val = param[i];
                float sign = (param_val > 0.0f) ? 1.0f : (param_val < 0.0f) ? -1.0f : 0.0f;
                grad[i] += sign * l1_reg;
            }
        }
        if (l2_reg > 0.0f) {
            // Add L2 regularization term to gradient: param * l2_reg
            for (size_t i = 0; i < param.size(); ++i) {
                grad[i] += param[i] * l2_reg;
            }
        }
    }
    
    // Gradient clipping to prevent exploding gradients
    virtual void clip_gradients(TensorF& grad, float clip_value) {
        for (size_t i = 0; i < grad.size(); ++i) {
            if (grad[i] > clip_value) grad[i] = clip_value;
            else if (grad[i] < -clip_value) grad[i] = -clip_value;
        }
    }
    
    // Gradient clipping by norm
    virtual void clip_gradients_by_norm(TensorF& grad, float max_norm) {
        float norm = 0.0f;
        for (size_t i = 0; i < grad.size(); ++i) {
            norm += grad[i] * grad[i];
        }
        norm = std::sqrt(norm);
        
        if (norm > max_norm) {
            float scale = max_norm / norm;
            for (size_t i = 0; i < grad.size(); ++i) {
                grad[i] *= scale;
            }
        }
    }
};

// Stochastic Gradient Descent optimizer with momentum
class SGD : public Optimizer {
private:
    float momentum;
    std::vector<TensorF> velocity;
    bool nesterov;
    
public:
    SGD(float lr = 0.001f, float momentum = 0.0f, bool nesterov = false)
        : Optimizer(lr), momentum(momentum), nesterov(nesterov) {}
    
    void update(const std::vector<TensorF*>& params, 
                const std::vector<TensorF*>& gradients) override {
        // Ensure we have velocity tensors for all parameters
        if (velocity.size() != params.size()) {
            velocity.clear();
            for (const auto& param : params) {
                velocity.emplace_back(param->shape());
                velocity.back().fill(0.0f);
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            update_single_impl(*params[i], *gradients[i], velocity[i]);
        }
    }
    
    void update_single(TensorF& param, const TensorF& grad) override {
        // Find or create velocity tensor for this parameter
        auto it = std::find_if(velocity.begin(), velocity.end(),
            [&param](const TensorF& v) { return v.size() == param.size(); });
            
        TensorF* vel_ptr;
        if (it != velocity.end()) {
            vel_ptr = &(*it);
        } else {
            velocity.emplace_back(param.shape());
            velocity.back().fill(0.0f);
            vel_ptr = &velocity.back();
        }
        
        update_single_impl(param, grad, *vel_ptr);
    }
    
private:
    void update_single_impl(TensorF& param, const TensorF& grad, TensorF& vel) {
        if (momentum > 0.0f) {
            // Update velocity: v = momentum * v - learning_rate * grad
            for (size_t i = 0; i < vel.size(); ++i) {
                vel[i] = momentum * vel[i] - learning_rate * grad[i];
            }
            
            // Update parameters: param += v
            if (nesterov) {
                // Nesterov momentum: param += momentum * v - learning_rate * grad
                for (size_t i = 0; i < param.size(); ++i) {
                    param[i] += momentum * vel[i] - learning_rate * grad[i];
                }
            } else {
                for (size_t i = 0; i < param.size(); ++i) {
                    param[i] += vel[i];
                }
            }
        } else {
            // Standard SGD: param -= learning_rate * grad
            for (size_t i = 0; i < param.size(); ++i) {
                param[i] -= learning_rate * grad[i];
            }
        }
    }
};

// Adam optimizer with bias correction
class Adam : public Optimizer {
private:
    float beta1, beta2;
    float epsilon;
    float t; // Timestep
    
    std::vector<TensorF> m; // First moment estimates
    std::vector<TensorF> v; // Second moment estimates
    
public:
    Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : Optimizer(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}
    
    void update(const std::vector<TensorF*>& params, 
                const std::vector<TensorF*>& gradients) override {
        t++;
        
        // Ensure we have moment tensors for all parameters
        if (m.size() != params.size()) {
            m.clear();
            v.clear();
            for (const auto& param : params) {
                m.emplace_back(param->shape());
                v.emplace_back(param->shape());
                m.back().fill(0.0f);
                v.back().fill(0.0f);
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            update_single_impl(*params[i], *gradients[i], m[i], v[i]);
        }
    }
    
    void update_single(TensorF& param, const TensorF& grad) override {
        // Find or create moment tensors for this parameter
        auto mit = std::find_if(m.begin(), m.end(),
            [&param](const TensorF& moment) { return moment.size() == param.size(); });
        auto vit = std::find_if(v.begin(), v.end(),
            [&param](const TensorF& moment) { return moment.size() == param.size(); });
            
        TensorF* m_ptr, *v_ptr;
        if (mit != m.end() && vit != v.end()) {
            m_ptr = &(*mit);
            v_ptr = &(*vit);
        } else {
            m.emplace_back(param.shape());
            v.emplace_back(param.shape());
            m.back().fill(0.0f);
            v.back().fill(0.0f);
            m_ptr = &m.back();
            v_ptr = &v.back();
        }
        
        update_single_impl(param, grad, *m_ptr, *v_ptr);
    }
    
private:
    void update_single_impl(TensorF& param, const TensorF& grad, TensorF& m, TensorF& v) {
        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
        for (size_t i = 0; i < m.size(); ++i) {
            m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        }
        
        // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
        }
        
        // Compute bias-corrected first moment estimate
        float bias_correction1 = 1.0f - std::pow(beta1, t);
        // Compute bias-corrected second raw moment estimate
        float bias_correction2 = 1.0f - std::pow(beta2, t);
        
        // Update parameters: param -= learning_rate * m_corr / (sqrt(v_corr) + epsilon)
        for (size_t i = 0; i < param.size(); ++i) {
            float m_corr = m[i] / bias_correction1;
            float v_corr = v[i] / bias_correction2;
            param[i] -= learning_rate * m_corr / (std::sqrt(v_corr) + epsilon);
        }
    }
};

// RMSprop optimizer
class RMSprop : public Optimizer {
private:
    float alpha; // Decay rate
    float epsilon;
    std::vector<TensorF> cache;
    
public:
    explicit RMSprop(float lr = 0.01f, float alpha = 0.99f, float epsilon = 1e-8f)
        : Optimizer(lr), alpha(alpha), epsilon(epsilon) {}
    
    void update(const std::vector<TensorF*>& params, 
                const std::vector<TensorF*>& gradients) override {
        // Ensure we have cache tensors for all parameters
        if (cache.size() != params.size()) {
            cache.clear();
            for (const auto& param : params) {
                cache.emplace_back(param->shape());
                cache.back().fill(0.0f);
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            update_single_impl(*params[i], *gradients[i], cache[i]);
        }
    }
    
    void update_single(TensorF& param, const TensorF& grad) override {
        // Find or create cache tensor for this parameter
        auto it = std::find_if(cache.begin(), cache.end(),
            [&param](const TensorF& c) { return c.size() == param.size(); });
            
        TensorF* cache_ptr;
        if (it != cache.end()) {
            cache_ptr = &(*it);
        } else {
            cache.emplace_back(param.shape());
            cache.back().fill(0.0f);
            cache_ptr = &cache.back();
        }
        
        update_single_impl(param, grad, *cache_ptr);
    }
    
private:
    void update_single_impl(TensorF& param, const TensorF& grad, TensorF& cache) {
        // Update cache: cache = alpha * cache + (1 - alpha) * grad^2
        for (size_t i = 0; i < cache.size(); ++i) {
            cache[i] = alpha * cache[i] + (1.0f - alpha) * grad[i] * grad[i];
        }
        
        // Update parameters: param -= learning_rate * grad / (sqrt(cache) + epsilon)
        for (size_t i = 0; i < param.size(); ++i) {
            param[i] -= learning_rate * grad[i] / (std::sqrt(cache[i]) + epsilon);
        }
    }
};

// AdaGrad optimizer
class AdaGrad : public Optimizer {
private:
    float epsilon;
    std::vector<TensorF> cache;
    
public:
    explicit AdaGrad(float lr = 0.001f, float epsilon = 1e-8f)
        : Optimizer(lr), epsilon(epsilon) {}
    
    void update(const std::vector<TensorF*>& params, 
                const std::vector<TensorF*>& gradients) override {
        // Ensure we have cache tensors for all parameters
        if (cache.size() != params.size()) {
            cache.clear();
            for (const auto& param : params) {
                cache.emplace_back(param->shape());
                cache.back().fill(0.0f);
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            update_single_impl(*params[i], *gradients[i], cache[i]);
        }
    }
    
    void update_single(TensorF& param, const TensorF& grad) override {
        // Find or create cache tensor for this parameter
        auto it = std::find_if(cache.begin(), cache.end(),
            [&param](const TensorF& c) { return c.size() == param.size(); });
            
        TensorF* cache_ptr;
        if (it != cache.end()) {
            cache_ptr = &(*it);
        } else {
            cache.emplace_back(param.shape());
            cache.back().fill(0.0f);
            cache_ptr = &cache.back();
        }
        
        update_single_impl(param, grad, *cache_ptr);
    }
    
private:
    void update_single_impl(TensorF& param, const TensorF& grad, TensorF& cache) {
        // Update cache: cache += grad^2
        for (size_t i = 0; i < cache.size(); ++i) {
            cache[i] += grad[i] * grad[i];
        }
        
        // Update parameters: param -= learning_rate * grad / (sqrt(cache) + epsilon)
        for (size_t i = 0; i < param.size(); ++i) {
            param[i] -= learning_rate * grad[i] / (std::sqrt(cache[i]) + epsilon);
        }
    }
};

// AdaDelta optimizer
class AdaDelta : public Optimizer {
private:
    float rho; // Decay rate
    float epsilon;
    std::vector<TensorF> cache_grad; // Running average of squared gradients
    std::vector<TensorF> cache_delta; // Running average of squared parameter updates
    
public:
    explicit AdaDelta(float rho = 0.95f, float epsilon = 1e-6f)
        : Optimizer(1.0f), rho(rho), epsilon(epsilon) {} // Learning rate is not used in AdaDelta
    
    void update(const std::vector<TensorF*>& params, 
                const std::vector<TensorF*>& gradients) override {
        // Ensure we have cache tensors for all parameters
        if (cache_grad.size() != params.size() || cache_delta.size() != params.size()) {
            cache_grad.clear();
            cache_delta.clear();
            for (const auto& param : params) {
                cache_grad.emplace_back(param->shape());
                cache_delta.emplace_back(param->shape());
                cache_grad.back().fill(0.0f);
                cache_delta.back().fill(0.0f);
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            update_single_impl(*params[i], *gradients[i], cache_grad[i], cache_delta[i]);
        }
    }
    
    void update_single(TensorF& param, const TensorF& grad) override {
        // Find or create cache tensors for this parameter
        auto git = std::find_if(cache_grad.begin(), cache_grad.end(),
            [&param](const TensorF& c) { return c.size() == param.size(); });
        auto dit = std::find_if(cache_delta.begin(), cache_delta.end(),
            [&param](const TensorF& c) { return c.size() == param.size(); });
            
        TensorF* grad_cache_ptr, *delta_cache_ptr;
        if (git != cache_grad.end() && dit != cache_delta.end()) {
            grad_cache_ptr = &(*git);
            delta_cache_ptr = &(*dit);
        } else {
            cache_grad.emplace_back(param.shape());
            cache_delta.emplace_back(param.shape());
            cache_grad.back().fill(0.0f);
            cache_delta.back().fill(0.0f);
            grad_cache_ptr = &cache_grad.back();
            delta_cache_ptr = &cache_delta.back();
        }
        
        update_single_impl(param, grad, *grad_cache_ptr, *delta_cache_ptr);
    }
    
private:
    void update_single_impl(TensorF& param, const TensorF& grad, TensorF& cache_g, TensorF& cache_d) {
        // Update gradient cache: cache_g = rho * cache_g + (1 - rho) * grad^2
        for (size_t i = 0; i < cache_g.size(); ++i) {
            cache_g[i] = rho * cache_g[i] + (1.0f - rho) * grad[i] * grad[i];
        }
        
        // Calculate parameter update: delta = -sqrt(cache_d + epsilon) / sqrt(cache_g + epsilon) * grad
        TensorF delta(param.shape());
        for (size_t i = 0; i < delta.size(); ++i) {
            delta[i] = -std::sqrt(cache_d[i] + epsilon) / std::sqrt(cache_g[i] + epsilon) * grad[i];
        }
        
        // Update parameters: param += delta
        for (size_t i = 0; i < param.size(); ++i) {
            param[i] += delta[i];
        }
        
        // Update delta cache: cache_d = rho * cache_d + (1 - rho) * delta^2
        for (size_t i = 0; i < cache_d.size(); ++i) {
            cache_d[i] = rho * cache_d[i] + (1.0f - rho) * delta[i] * delta[i];
        }
    }
};

// Base class for learning rate schedulers
class LearningRateScheduler {
protected:
    Optimizer* optimizer;
    
public:
    explicit LearningRateScheduler(Optimizer* opt) : optimizer(opt) {}
    virtual ~LearningRateScheduler() = default;
    
    // Virtual method to update the learning rate based on epoch/call count
    virtual void step() = 0;
    
    // Method to get current learning rate
    virtual float get_lr() const = 0;
};

// Step decay scheduler
class StepLR : public LearningRateScheduler {
private:
    float gamma; // Multiplicative factor
    int step_size; // Period of learning rate decay
    int current_step;
    
public:
    StepLR(Optimizer* opt, int step_size, float gamma = 0.1f) 
        : LearningRateScheduler(opt), gamma(gamma), step_size(step_size), current_step(0) {}
    
    void step() override {
        current_step++;
        if (current_step % step_size == 0) {
            float new_lr = optimizer->get_learning_rate() * gamma;
            optimizer->set_learning_rate(new_lr);
        }
    }
    
    float get_lr() const override {
        int num_steps = current_step / step_size;
        return optimizer->get_learning_rate() * std::pow(gamma, num_steps);
    }
};

// Exponential decay scheduler
class ExponentialLR : public LearningRateScheduler {
private:
    float gamma; // Multiplicative factor
    int current_step;
    
public:
    explicit ExponentialLR(Optimizer* opt, float gamma) 
        : LearningRateScheduler(opt), gamma(gamma), current_step(0) {}
    
    void step() override {
        current_step++;
        float new_lr = optimizer->get_learning_rate() * gamma;
        optimizer->set_learning_rate(new_lr);
    }
    
    float get_lr() const override {
        return optimizer->get_learning_rate() * std::pow(gamma, current_step);
    }
};

// Polynomial decay scheduler
class PolynomialLR : public LearningRateScheduler {
private:
    float power; // Decay power
    int max_steps; // Total number of steps
    float initial_lr;
    int current_step;
    
public:
    PolynomialLR(Optimizer* opt, int max_steps, float power = 1.0f) 
        : LearningRateScheduler(opt), power(power), max_steps(max_steps), 
          initial_lr(opt->get_learning_rate()), current_step(0) {}
    
    void step() override {
        current_step++;
        if (current_step >= max_steps) current_step = max_steps; // Clamp to max_steps
        
        float factor = std::pow(1.0f - static_cast<float>(current_step) / max_steps, power);
        float new_lr = initial_lr * factor;
        optimizer->set_learning_rate(new_lr);
    }
    
    float get_lr() const override {
        if (current_step >= max_steps) return 0.0f;
        float factor = std::pow(1.0f - static_cast<float>(current_step) / max_steps, power);
        return initial_lr * factor;
    }
};

// Cosine annealing scheduler
class CosineAnnealingLR : public LearningRateScheduler {
private:
    int T_max; // Maximum number of iterations
    float eta_min; // Minimum learning rate
    float eta_max; // Maximum learning rate
    int current_step;
    
public:
    CosineAnnealingLR(Optimizer* opt, int T_max, float eta_min = 0.0f) 
        : LearningRateScheduler(opt), T_max(T_max), eta_min(eta_min), 
          eta_max(opt->get_learning_rate()), current_step(0) {}
    
    void step() override {
        current_step++;
        if (current_step >= T_max) current_step = T_max; // Clamp to T_max
        
        float lr = eta_min + 0.5f * (eta_max - eta_min) * (1 + std::cos(M_PI * current_step / T_max));
        optimizer->set_learning_rate(lr);
    }
    
    float get_lr() const override {
        if (current_step >= T_max) return eta_min;
        return eta_min + 0.5f * (eta_max - eta_min) * (1 + std::cos(M_PI * current_step / T_max));
    }
};

// Reduce learning rate on plateau
class ReduceLROnPlateau : public LearningRateScheduler {
private:
    int mode; // 0 for min, 1 for max
    float factor; // Factor by which to reduce lr
    int patience; // Number of epochs with no improvement
    float threshold; // Threshold for measuring new optimum
    int threshold_mode; // 0 for rel, 1 for abs
    int cooldown; // Number of epochs to wait before resuming normal operation
    int current_step;
    int num_bad_epochs;
    float best;
    int cooldown_counter;
    
public:
    ReduceLROnPlateau(Optimizer* opt, float factor = 0.1f, int patience = 10, 
                      float threshold = 1e-4f, int threshold_mode = 0, 
                      int cooldown = 0, int mode = 0) 
        : LearningRateScheduler(opt), factor(factor), patience(patience), 
          threshold(threshold), threshold_mode(threshold_mode), cooldown(cooldown),
          current_step(0), num_bad_epochs(0), cooldown_counter(0) {
        if (mode == 0) { // min mode
            best = std::numeric_limits<float>::infinity();
        } else { // max mode
            best = -std::numeric_limits<float>::infinity();
        }
    }
    
    void step(float metric) {
        bool is_better;
        if (mode == 0) { // min mode
            if (threshold_mode == 0) { // rel
                is_better = metric < best * (1.0f - threshold);
            } else { // abs
                is_better = metric < best - threshold;
            }
        } else { // max mode
            if (threshold_mode == 0) { // rel
                is_better = metric > best * (1.0f + threshold);
            } else { // abs
                is_better = metric > best + threshold;
            }
        }
        
        if (is_better) {
            best = metric;
            num_bad_epochs = 0;
        } else {
            num_bad_epochs++;
        }
        
        if (cooldown_counter > 0) {
            cooldown_counter--;
            num_bad_epochs = 0; // Reset bad epochs during cooldown
        }
        
        if (num_bad_epochs > patience && cooldown_counter == 0) {
            float new_lr = optimizer->get_learning_rate() * factor;
            optimizer->set_learning_rate(new_lr);
            cooldown_counter = cooldown;
            num_bad_epochs = 0;
        }
        
        current_step++;
    }
    
    void step() override {
        // Default to no improvement if no metric provided
        step(mode == 0 ? best + 1 : best - 1);
    }
    
    float get_lr() const override {
        return optimizer->get_learning_rate();
    }
};

} // namespace dnn

#endif // OPTIMIZERS_HPP