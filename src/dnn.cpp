// dnn.cpp — comprehensive implementation of DNN library
#include "dnn.hpp"
#include "model_serializer.hpp"
#include "tensor_ops.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <execution>
#include <random>
#include <array>

namespace dnn {

// --- Forward declarations for functions used in Dense layer ---
Matrix apply_activation(const Matrix& z, Activation act);
Matrix apply_activation_derivative(const Matrix& a, const Matrix& grad, Activation act);

// --- Dense Layer Private Methods ---
Matrix Dense::apply_activation(const Matrix& z, Activation act) {
    return dnn::apply_activation(z, act);
}

Matrix Dense::apply_activation_derivative(const Matrix& a, const Matrix& grad, Activation act) {
    return dnn::apply_activation_derivative(a, grad, act);
}

// --- small utils (internal) ---
static inline bool is_finite(double x) noexcept {
    return std::isfinite(x) != 0;
}

static constexpr double kProbEpsilon = 1e-12;

static inline double clamp_probability(double x) noexcept {
    if (x < kProbEpsilon) {
        return kProbEpsilon;
    }
    double upper = 1.0 - kProbEpsilon;
    if (x > upper) {
        return upper;
    }
    return x;
}

static inline void clamp_matrix_probabilities(Matrix& m) {
    for (std::size_t i = 0; i < m.size(); ++i) {
        m.data()[i] = clamp_probability(m.data()[i]);
    }
}

static inline double safe_log(double x) noexcept {
    return std::log(clamp_probability(x));
}

static inline double stable_sigmoid(double x) noexcept {
    if (x >= 0.0) {
        const double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    }
    const double e = std::exp(x);
    return e / (1.0 + e);
}

static inline double stable_softplus(double x) noexcept {
    if (x > 0) {
        return x + std::log1p(std::exp(-x));
    }
    return std::log1p(std::exp(x));
}

static inline std::string now_time() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32]{};
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

static void clip_gradient(Matrix& grad, const Optimizer& opt) {
    if (!opt.use_gradient_clipping) {
        return;
    }
    if (opt.max_gradient_norm > 0.0) {
        double norm_sq = 0.0;
        for (std::size_t i = 0; i < grad.size(); ++i) {
            double value = grad.data()[i];
            norm_sq += value * value;
        }
        double norm = std::sqrt(norm_sq);
        if (norm > opt.max_gradient_norm && norm > 0.0) {
            double scale = opt.max_gradient_norm / (norm + 1e-12);
            for (std::size_t i = 0; i < grad.size(); ++i) {
                grad.data()[i] *= scale;
            }
        }
    } else if (opt.clip_value > 0.0) {
        const double clip = opt.clip_value;
        for (std::size_t i = 0; i < grad.size(); ++i) {
            double& value = grad.data()[i];
            if (value > clip) value = clip;
            else if (value < -clip) value = -clip;
        }
    }
}

static void add_regularization(Matrix& grad, const Matrix& param, const Optimizer& opt) {
    if (opt.l1_lambda > 0.0) {
        for (std::size_t i = 0; i < grad.size(); ++i) {
            const double w = param.data()[i];
            grad.data()[i] += (w > 0.0 ? 1.0 : (w < 0.0 ? -1.0 : 0.0)) * opt.l1_lambda;
        }
    }
    if (opt.l2_lambda > 0.0) {
        for (std::size_t i = 0; i < grad.size(); ++i) {
            grad.data()[i] += param.data()[i] * opt.l2_lambda;
        }
    }
}

static void sgd_update(const SGD& sgd, const Optimizer& opt, Matrix& param,
                       Matrix& grad, Matrix& momentum) {
    if (sgd.weight_decay > 0.0) {
        for (std::size_t i = 0; i < grad.size(); ++i) {
            grad.data()[i] += sgd.weight_decay * param.data()[i];
        }
    }
    if (sgd.momentum > 0.0) {
        for (std::size_t i = 0; i < momentum.size; ++i) {
            double v = sgd.momentum * momentum.data()[i] + (1.0 - sgd.dampening) * grad.data()[i];
            double update = sgd.nesterov ? (grad.data()[i] + sgd.momentum * v) : v;
            param.data()[i] -= opt.learning_rate * update;
            momentum.data()[i] = v;
        }
    } else {
        for (std::size_t i = 0; i < param.size; ++i) {
            param.data()[i] -= opt.learning_rate * grad.data()[i];
        }
    }
}

static void rmsprop_update(const RMSprop& rms, const Optimizer& opt,
                           Matrix& param, Matrix& grad, Matrix& rms_buffer) {
    for (std::size_t i = 0; i < rms_buffer.size; ++i) {
        if (rms.weight_decay > 0.0) {
            grad.data()[i] += rms.weight_decay * param.data()[i];
        }
        rms_buffer.data()[i] = rms.alpha * rms_buffer.data()[i] + (1.0 - rms.alpha) * grad.data()[i] * grad.data()[i];
        param.data()[i] -= opt.learning_rate * grad.data()[i] /
                         (std::sqrt(rms_buffer.data()[i]) + rms.epsilon);
    }
}

static void adam_update(const Adam& adam, const Optimizer& opt,
                        Matrix& param, Matrix& grad,
                        Matrix& momentum, Matrix& rms_buffer) {
    if (adam.weight_decay > 0.0) {
        for (std::size_t i = 0; i < grad.size; ++i) {
            grad.data()[i] += adam.weight_decay * param.data()[i];
        }
    }
    for (std::size_t i = 0; i < momentum.size(); ++i) {
        momentum.data()[i] = adam.beta1 * momentum.data()[i] + (1.0 - adam.beta1) * grad.data()[i];
        rms_buffer.data()[i] = adam.beta2 * rms_buffer.data()[i] +
                             (1.0 - adam.beta2) * grad.data()[i] * grad.data()[i];
        double m_hat = momentum.data()[i] / (1.0 - std::pow(adam.beta1, static_cast<double>(adam.step_count)) + 1e-12);
        double v_hat = rms_buffer.data()[i] / (1.0 - std::pow(adam.beta2, static_cast<double>(adam.step_count)) + 1e-12);
        param.data()[i] -= opt.learning_rate * m_hat / (std::sqrt(v_hat) + opt.epsilon);
    }
}

static void adamw_update(const AdamW& adamw, const Optimizer& opt,
                         Matrix& param, Matrix& grad,
                         Matrix& momentum, Matrix& rms_buffer) {
    for (std::size_t i = 0; i < momentum.size; ++i) {
        momentum.data()[i] = adamw.beta1 * momentum.data()[i] + (1.0 - adamw.beta1) * grad.data()[i];
        rms_buffer.data()[i] = adamw.beta2 * rms_buffer.data()[i] +
                             (1.0 - adamw.beta2) * grad.data()[i] * grad.data()[i];
        double m_hat = momentum.data()[i] / (1.0 - std::pow(adamw.beta1, static_cast<double>(adamw.step_count)) + 1e-12);
        double v_hat = rms_buffer.data()[i] / (1.0 - std::pow(adamw.beta2, static_cast<double>(adamw.step_count)) + 1e-12);
        double update = opt.learning_rate * m_hat / (std::sqrt(v_hat) + opt.epsilon);
        param.data()[i] -= update;
        if (adamw.weight_decay > 0.0) {
            param.data()[i] -= opt.learning_rate * adamw.weight_decay * param.data()[i];
        }
    }
}

static void apply_optimizer_update(const Optimizer& opt,
                                   Matrix& param, Matrix& grad,
                                   Matrix& momentum, Matrix& rms_buffer) {
    switch (opt.type) {
        case OptimizerType::SGD: {
            const auto* sgd = dynamic_cast<const SGD*>(&opt);
            if (!sgd) {
                throw std::runtime_error("Optimizer type mismatch for SGD");
            }
            sgd_update(*sgd, opt, param, grad, momentum);
            break;
        }
        case OptimizerType::Adam: {
            const auto* adam = dynamic_cast<const Adam*>(&opt);
            if (!adam) {
                throw std::runtime_error("Optimizer type mismatch for Adam");
            }
            adam_update(*adam, opt, param, grad, momentum, rms_buffer);
            break;
        }
        case OptimizerType::RMSprop: {
            const auto* rms = dynamic_cast<const RMSprop*>(&opt);
            if (!rms) {
                throw std::runtime_error("Optimizer type mismatch for RMSprop");
            }
            rmsprop_update(*rms, opt, param, grad, rms_buffer);
            break;
        }
        case OptimizerType::AdamW: {
            const auto* adamw = dynamic_cast<const AdamW*>(&opt);
            if (!adamw) {
                throw std::runtime_error("Optimizer type mismatch for AdamW");
            }
            adamw_update(*adamw, opt, param, grad, momentum, rms_buffer);
            break;
        }
        default:
            throw std::runtime_error("Unsupported optimizer type in parameter update");
    }
}


// ---------------- Tensor Operations ----------------
// Use the new tensor operations from tensor_ops.hpp
Matrix transpose(const Matrix& A) {
    return dnn::transpose(A);
}

Matrix matmul(const Matrix& A, const Matrix& B) {
    return dnn::matmul(A, B);
}

void add_rowwise_inplace(Matrix& A, const Matrix& rowvec) {
    dnn::add_rowwise_inplace(A, rowvec);
}

Matrix add(const Matrix& A, const Matrix& B) {
    return dnn::add(A, B);
}

Matrix sub(const Matrix& A, const Matrix& B) {
    return dnn::sub(A, B);
}

Matrix hadamard(const Matrix& A, const Matrix& B) {
    return dnn::hadamard(A, B);
}

Matrix scalar_mul(const Matrix& A, double s) {
    return dnn::scalar_mul(A, static_cast<float>(s));
}

Matrix sum_rows(const Matrix& A) {
    return dnn::sum_rows(A);
}

// ---------------- Activation Functions Implementation ----------------
Matrix apply_activation(const Matrix& z, Activation act) {
    switch (act) {
        case Activation::Linear:
            return z;
            
        case Activation::ReLU: {
            Matrix a(z.shape(), z.layout());
            for (std::size_t i = 0; i < z.size(); ++i) {
                a.data()[i] = (z.data()[i] > 0.0f ? z.data()[i] : 0.0f);
            }
            return a;
        }
            
        case Activation::LeakyReLU: {
            Matrix a(z.shape(), z.layout());
            const float alpha = 0.01f;
            for (std::size_t i = 0; i < z.size(); ++i) {
                a.data()[i] = (z.data()[i] > 0.0f ? z.data()[i] : alpha * z.data()[i]);
            }
            return a;
        }
        
        case Activation::ELU: {
            Matrix a(z.shape(), z.layout());
            const float alpha = 1.0f;
            for (std::size_t i = 0; i < z.size(); ++i) {
                const float x = z.data()[i];
                a.data()[i] = (x > 0.0f ? x : alpha * (std::exp(x) - 1.0f));
            }
            return a;
        }
        
        case Activation::Sigmoid: {
            Matrix a(z.shape(), z.layout());
            for (std::size_t i = 0; i < z.size(); ++i) {
                a.data()[i] = dnn::stable_sigmoid(z.data()[i]);
            }
            return a;
        }
            
        case Activation::Tanh: {
            Matrix a(z.shape(), z.layout());
            for (std::size_t i = 0; i < z.size(); ++i) {
                a.data()[i] = std::tanh(z.data()[i]);
            }
            return a;
        }
            
        case Activation::Softmax: {
            Matrix a(z.shape(), z.layout());
            for (std::size_t r = 0; r < z.shape()[0]; ++r) {
                float max_val = -std::numeric_limits<float>::infinity();
                
                // Find max value in the row
                for (std::size_t c = 0; c < z.shape()[1]; ++c) {
                    max_val = std::max(max_val, z(r, c));
                }
                
                float sum = 0.0f;
                // Compute exp and sum
                for (std::size_t c = 0; c < z.shape()[1]; ++c) {
                    const float e = std::exp(z(r, c) - max_val);
                    a(r, c) = e;
                    sum += e;
                }
                
                // Normalize
                const float denom = (sum > 0.0f ? sum : 1.0f);
                for (std::size_t c = 0; c < z.shape()[1]; ++c) {
                    a(r, c) /= denom;
                }
            }
            return a;
        }
        
        case Activation::Swish: {
            Matrix a(z.shape(), z.layout());
            for (std::size_t i = 0; i < z.size(); ++i) {
                const float x = z.data()[i];
                const float sigmoid_x = dnn::stable_sigmoid(x);
                a.data()[i] = x * sigmoid_x;
            }
            return a;
        }
            
        case Activation::GELU: {
            Matrix a(z.shape(), z.layout());
            for (std::size_t i = 0; i < z.size(); ++i) {
                const float x = z.data()[i];
                a.data()[i] = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
            }
            return a;
        }
            
        case Activation::Softplus: {
            Matrix a(z.shape(), z.layout());
            for (std::size_t i = 0; i < z.size(); ++i) {
                a.data()[i] = dnn::stable_softplus(z.data()[i]);
            }
            return a;
        }
    }
    
    // Should never reach here
    return z;
}

Matrix apply_activation_derivative(const Matrix& a, const Matrix& grad, Activation act) {
    if (a.shape() != grad.shape()) {
        throw DimensionMismatchException("apply_activation_derivative: shape mismatch");
    }
    
    Matrix dZ(a.shape(), a.layout());
    
    switch (act) {
        case Activation::Linear:
            dZ = grad;
            break;
            
        case Activation::ReLU:
            for (std::size_t i = 0; i < a.size(); ++i) {
                dZ.data()[i] = (a.data()[i] > 0.0f) ? grad.data()[i] : 0.0f;
            }
            break;
            
        case Activation::LeakyReLU: {
            const float alpha = 0.01f;
            for (std::size_t i = 0; i < a.size(); ++i) {
                dZ.data()[i] = (a.data()[i] > 0.0f) ? grad.data()[i] : alpha * grad.data()[i];
            }
            break;
        }
        
        case Activation::ELU: {
            const float alpha = 1.0f;
            for (std::size_t i = 0; i < a.size(); ++i) {
                const float x = a.data()[i];
                if (x > 0.0f) {
                    dZ.data()[i] = grad.data()[i];
                } else {
                    dZ.data()[i] = grad.data()[i] * alpha * std::exp(x);
                }
            }
            break;
        }
        
        case Activation::Sigmoid:
            for (std::size_t i = 0; i < a.size(); ++i) {
                const float s = dnn::stable_sigmoid(a.data()[i]);
                dZ.data()[i] = grad.data()[i] * s * (1.0f - s);
            }
            break;
            
        case Activation::Tanh:
            for (std::size_t i = 0; i < a.size(); ++i) {
                const float t = std::tanh(a.data()[i]);
                dZ.data()[i] = grad.data()[i] * (1.0f - t * t);
            }
            break;
            
        case Activation::Softmax: {
            // For softmax, we need to compute the full Jacobian for each row
            for (std::size_t r = 0; r < a.shape()[0]; ++r) {
                Matrix row_a(1, a.shape()[1]);
                Matrix row_softmax(1, a.shape()[1]);
                
                // Extract the row
                for (std::size_t c = 0; c < a.shape()[1]; ++c) {
                    row_a(0, c) = a(r, c);
                }
                
                // Compute softmax for the row
                float max_val = -std::numeric_limits<float>::infinity();
                for (std::size_t c = 0; c < a.shape()[1]; ++c) {
                    max_val = std::max(max_val, row_a(0, c));
                }
                
                float sum = 0.0f;
                for (std::size_t c = 0; c < a.shape()[1]; ++c) {
                    const float e = std::exp(row_a(0, c) - max_val);
                    row_softmax(0, c) = e;
                    sum += e;
                }
                
                const float denom = (sum > 0.0f ? sum : 1.0f);
                for (std::size_t c = 0; c < a.shape()[1]; ++c) {
                    row_softmax(0, c) /= denom;
                }
                
                // Compute gradient: grad_output * softmax * (1 - softmax) for diagonal
                // and -grad_output * softmax_i * softmax_j for off-diagonal
                for (std::size_t c = 0; c < a.shape()[1]; ++c) {
                    float grad_sum = 0.0f;
                    for (std::size_t k = 0; k < a.shape()[1]; ++k) {
                        if (k == c) {
                            grad_sum += grad(r, k) * row_softmax(0, c) * (1.0f - row_softmax(0, c));
                        } else {
                            grad_sum += grad(r, k) * (-row_softmax(0, c) * row_softmax(0, k));
                        }
                    }
                    dZ(r, c) = grad_sum;
                }
            }
            break;
        }
        
        case Activation::Swish: {
            for (std::size_t i = 0; i < a.size(); ++i) {
                const float x = a.data()[i];
                const float sigmoid_x = dnn::stable_sigmoid(x);
                const float swish_derivative = sigmoid_x + x * sigmoid_x * (1.0f - sigmoid_x);
                dZ.data()[i] = grad.data()[i] * swish_derivative;
            }
            break;
        }
        
        case Activation::GELU: {
            for (std::size_t i = 0; i < a.size(); ++i) {
                const float x = a.data()[i];
                const float phi_x = 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
                const float phi_prime_x = std::exp(-0.5f * x * x) / std::sqrt(2.0f * static_cast<float>(M_PI));
                dZ.data()[i] = grad.data()[i] * (phi_x + x * phi_prime_x);
            }
            break;
        }
        
        case Activation::Softplus:
            for (std::size_t i = 0; i < a.size(); ++i) {
                const float s = dnn::stable_sigmoid(a.data()[i]);
                dZ.data()[i] = grad.data()[i] * s;
            }
            break;
    }
    
    return dZ;
}

// ---------------- Loss Functions Implementation ----------------
LossResult compute_loss(const Matrix& y_true, const Matrix& y_pred, LossFunction loss_fn) {
    if (y_true.shape() != y_pred.shape()) {
        throw DimensionMismatchException("compute_loss: shape mismatch");
    }
    
    LossResult result;
    
    switch (loss_fn) {
        case LossFunction::MSE: {
            float sum = 0.0f;
            for (std::size_t i = 0; i < y_true.size(); ++i) {
                const float diff = y_pred.data()[i] - y_true.data()[i];
                sum += diff * diff;
            }
            result.value = sum / static_cast<float>(y_true.shape()[0]);
            
            // Gradient for MSE: 2*(y_pred - y_true)/n
            result.gradient = scalar_mul(sub(y_pred, y_true), 2.0f / static_cast<float>(y_true.shape()[0]));
            break;
        }
        
        case LossFunction::CrossEntropy: {
            Matrix softmax_pred = apply_activation(y_pred, Activation::Softmax);
            clamp_probabilities(softmax_pred);
            float sum = 0.0f;
            
            for (std::size_t r = 0; r < y_true.shape()[0]; ++r) {
                for (std::size_t c = 0; c < y_true.shape()[1]; ++c) {
                    if (y_true(r, c) > 0.0f) {
                        sum += -y_true(r, c) * safe_log(softmax_pred(r, c));
                    }
                }
            }
            
            result.value = sum / static_cast<float>(y_true.shape()[0]);
            
            // Gradient for cross-entropy with softmax: (softmax_pred - y_true) / n
            result.gradient = scalar_mul(sub(softmax_pred, y_true), 1.0f / static_cast<float>(y_true.shape()[0]));
            break;
        }
        
        case LossFunction::BinaryCrossEntropy: {
            Matrix clipped_pred = y_pred;
            clamp_probabilities(clipped_pred);
            
            float sum = 0.0f;
            for (std::size_t i = 0; i < y_true.size(); ++i) {
                sum += -(y_true.data()[i] * safe_log(clipped_pred.data()[i]) +
                         (1.0f - y_true.data()[i]) * safe_log(1.0f - clipped_pred.data()[i]));
            }
            
            result.value = sum / static_cast<float>(y_true.shape()[0]);
            
            // Gradient for binary cross-entropy: (y_pred - y_true) / n
            result.gradient = scalar_mul(sub(clipped_pred, y_true), 1.0f / static_cast<float>(y_true.shape()[0]));
            break;
        }
        
        case LossFunction::Hinge: {
           float sum = 0.0f;
           for (std::size_t r = 0; r < y_true.shape()[0]; ++r) {
               for (std::size_t c = 0; c < y_true.shape()[1]; ++c) {
                   if (y_true(r, c) > 0.0f) {  // This is the true class
                       for (std::size_t k = 0; k < y_true.shape()[1]; ++k) {
                           if (k != c) {  // For all other classes
                               const float margin = 1.0f - (y_pred(r, c) - y_pred(r, k));
                               sum += std::max(0.0f, margin);
                           }
                       }
                   }
               }
           }
           
           result.value = sum / static_cast<float>(y_true.shape()[0]);
           // Gradient for hinge loss: 1 if margin > 0, 0 otherwise
           Matrix grad(y_true.shape(), 0.0f);
           for (std::size_t r = 0; r < y_true.shape()[0]; ++r) {
               for (std::size_t c = 0; c < y_true.shape()[1]; ++c) {
                   if (y_true(r, c) > 0.0f) { // This is the true class
                       for (std::size_t k = 0; k < y_true.shape()[1]; ++k) {
                           if (k != c) {  // For all other classes
                               const float margin = 1.0f - (y_pred(r, c) - y_pred(r, k));
                               if (margin > 0.0f) {
                                   grad(r, c) -= 1.0f;  // Gradient for true class
                                   grad(r, k) += 1.0f;  // Gradient for false class
                               }
                           }
                       }
                   }
               }
           }
           result.gradient = scalar_mul(grad, 1.0f / static_cast<float>(y_true.shape()[0]));
           break;
       }
        
        case LossFunction::Huber: {
            const float delta = 1.0f;
            float sum = 0.0f;
            Matrix grad(y_true.shape(), 0.0f);
            
            for (std::size_t i = 0; i < y_true.size(); ++i) {
                const float diff = y_pred.data()[i] - y_true.data()[i];
                const float abs_diff = std::abs(diff);
                
                if (abs_diff <= delta) {
                    sum += 0.5f * diff * diff;
                    grad.data()[i] = diff;
                } else {
                    sum += delta * abs_diff - 0.5f * delta * delta;
                    grad.data()[i] = (diff > 0.0f) ? delta : -delta;
                }
            }
            
            result.value = sum / static_cast<float>(y_true.shape()[0]);
            result.gradient = scalar_mul(grad, 1.0f / static_cast<float>(y_true.shape()[0]));
            break;
        }
        
        case LossFunction::KLDivergence: {
            Matrix clipped_true = y_true;
            Matrix clipped_pred = y_pred;
            clamp_probabilities(clipped_true);
            clamp_probabilities(clipped_pred);
            
            float sum = 0.0f;
            for (std::size_t i = 0; i < y_true.size(); ++i) {
                sum += clipped_true.data()[i] * safe_log(clipped_true.data()[i] / clipped_pred.data()[i]);
            }
            
            result.value = sum / static_cast<float>(y_true.shape()[0]);
            
            // Gradient for KL divergence: (log(q) - log(p) + 1) where p is true and q is pred
            Matrix grad(y_true.shape());
            for (std::size_t i = 0; i < y_true.size(); ++i) {
                grad.data()[i] = std::log(clipped_pred.data()[i]) - std::log(clipped_true.data()[i]) + 1.0f;
            }
            result.gradient = scalar_mul(grad, 1.0f / static_cast<float>(y_true.shape()[0]));
            break;
        }
    }
    
    return result;
}

// ---------------- Dense Layer Methods ----------------
void Dense::update_parameters(const Optimizer& opt) {
    clip_gradient(weight_velocity, opt);
    clip_gradient(bias_velocity, opt);
    add_regularization(weight_velocity, weights, opt);
    add_regularization(bias_velocity, bias, opt);
    apply_optimizer_update(opt, weights, weight_velocity, weight_momentum, weight_rms);
    apply_optimizer_update(opt, bias, bias_velocity, bias_momentum, bias_rms);
}

// ---------------- Conv2D Layer Methods ----------------
Matrix Conv2D::forward(const Matrix& input) {
    // Store input for backward pass
    // For Conv2D, we need to define input_cache in the class or use a different approach
    // Since input_cache is not defined in Conv2D class, I'll fix the implementation to use the correct member
    input_cache = input;
    
    // Input shape: (batch_size, flattened_features) where flattened_features = height * width * in_channels
    // For this implementation, we'll assume input is (batch_size, height * width * in_channels)
    // Calculate spatial dimensions
    std::size_t total_spatial_size = input.shape[1] / in_channels;
    std::size_t input_height = static_cast<std::size_t>(std::sqrt(total_spatial_size));
    std::size_t input_width = total_spatial_size / input_height;
    
    // Calculate output dimensions
    std::size_t out_height = (input_height - kernel_height + 2 * padding_h) / stride_h + 1;
    std::size_t out_width = (input_width - kernel_width + 2 * padding_w) / stride_w + 1;
    last_input_height = input_height;
    last_input_width = input_width;
    
    // Initialize output matrix
    Matrix output(input.shape[0], out_height * out_width * out_channels);
    
    // Perform convolution for each sample in the batch
    for (std::size_t b = 0; b < input.shape[0]; ++b) {
        for (std::size_t oc = 0; oc < out_channels; ++oc) {
            for (std::size_t oh = 0; oh < out_height; ++oh) {
                for (std::size_t ow = 0; ow < out_width; ++ow) {
                    double sum = 0.0;
                    
                    // Convolution operation
                    for (std::size_t ic = 0; ic < in_channels; ++ic) {
                        for (std::size_t kh = 0; kh < kernel_height; ++kh) {
                            for (std::size_t kw = 0; kw < kernel_width; ++kw) {
                                std::size_t ih = oh * stride_h - padding_h + kh;
                                std::size_t iw = ow * stride_w - padding_w + kw;
                                
                                // Check bounds
                                if (ih < input_height && iw < input_width) {
                                    std::size_t input_idx = ic * (input_height * input_width) + ih * input_width + iw;
                                    std::size_t weight_idx = oc * (in_channels * kernel_height * kernel_width) +
                                                           ic * (kernel_height * kernel_width) +
                                                           kh * kernel_width + kw;
                                    
                                    sum += input(b, input_idx) * weights(oc, ic * kernel_height * kernel_width + kh * kernel_width + kw);
                                }
                            }
                        }
                    }
                    
                    // Add bias
                    sum += bias(0, oc);
                    
                    // Store result
                    std::size_t output_idx = oc * (out_height * out_width) + oh * out_width + ow;
                    output(b, output_idx) = sum;
                }
            }
        }
    }
    
    // Apply activation function
    pre_activation_cache = output;
    Matrix activated = apply_activation(output, activation);
    post_activation_cache = activated;
    return activated;
}

Matrix Conv2D::backward(const Matrix& grad_output) {
    std::size_t input_height = last_input_height;
    std::size_t input_width = last_input_width;
    if (input_height == 0 || input_width == 0) {
        std::size_t total_spatial_size = input_cache.shape[1] / in_channels;
        input_height = static_cast<std::size_t>(std::sqrt(total_spatial_size));
        input_width = total_spatial_size / input_height;
    }
    std::size_t out_height = (input_height - kernel_height + 2 * padding_h) / stride_h + 1;
    std::size_t out_width = (input_width - kernel_width + 2 * padding_w) / stride_w + 1;

    Matrix grad_act = apply_activation_derivative(post_activation_cache, grad_output, activation);
    Matrix grad_input(input_cache.shape[0], input_cache.shape[1], 0.0);
    Matrix grad_weights_local(out_channels, in_channels * kernel_height * kernel_width, 0.0);
    Matrix grad_bias_local(1, out_channels, 0.0);

    for (std::size_t b = 0; b < grad_act.shape[0]; ++b) {
        for (std::size_t oc = 0; oc < out_channels; ++oc) {
            for (std::size_t oh = 0; oh < out_height; ++oh) {
                for (std::size_t ow = 0; ow < out_width; ++ow) {
                    std::size_t output_idx = oc * (out_height * out_width) + oh * out_width + ow;
                    double grad_val = grad_act(b, output_idx);
                    grad_bias_local(0, oc) += grad_val;

                    for (std::size_t ic = 0; ic < in_channels; ++ic) {
                        for (std::size_t kh = 0; kh < kernel_height; ++kh) {
                            for (std::size_t kw = 0; kw < kernel_width; ++kw) {
                                std::size_t ih = oh * stride_h - padding_h + kh;
                                std::size_t iw = ow * stride_w - padding_w + kw;
                                if (ih < input_height && iw < input_width) {
                                    std::size_t input_idx = ic * (input_height * input_width) + ih * input_width + iw;
                                    std::size_t weight_col = ic * kernel_height * kernel_width + kh * kernel_width + kw;
                                    grad_weights_local(oc, weight_col) += input_cache(b, input_idx) * grad_val;
                                    grad_input(b, input_idx) += weights(oc, weight_col) * grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    grad_weights = grad_weights_local;
    grad_bias = grad_bias_local;
    weight_velocity = grad_weights_local;
    bias_velocity = grad_bias_local;
    return grad_input;
}

void Conv2D::update_parameters(const Optimizer& opt) {
    clip_gradient(weight_velocity, opt);
    clip_gradient(bias_velocity, opt);
    add_regularization(weight_velocity, weights, opt);
    add_regularization(bias_velocity, bias, opt);
    apply_optimizer_update(opt, weights, weight_velocity, weight_momentum, weight_rms);
    apply_optimizer_update(opt, bias, bias_velocity, bias_momentum, bias_rms);
}

// ---------------- MaxPool2D Layer Methods ----------------
Matrix MaxPool2D::forward(const Matrix& input) {
    // Store input for backward pass
    input_cache = input;
    
    // Input shape: (batch_size, flattened_features) where flattened_features = height * width * channels
    std::size_t total_spatial_size = input.shape[1];
    std::size_t input_channels = 1; // For this simplified implementation, assume single channel
    std::size_t input_height = static_cast<std::size_t>(std::sqrt(total_spatial_size));
    std::size_t input_width = total_spatial_size / input_height;
    
    // Calculate output dimensions
    std::size_t out_height = (input_height - pool_height) / stride_h + 1;
    std::size_t out_width = (input_width - pool_width) / stride_w + 1;
    
    // Initialize output matrix
    Matrix output(input.shape[0], out_height * out_width * input_channels);
    
    // Initialize mask cache to track max positions
    mask_cache = Matrix(input.shape[0], input.shape[1], 0.0);
    
    // Perform max pooling for each sample in the batch
    for (std::size_t b = 0; b < input.shape[0]; ++b) {
        for (std::size_t h = 0; h < out_height; ++h) {
            for (std::size_t w = 0; w < out_width; ++w) {
                double max_val = -std::numeric_limits<double>::infinity();
                std::size_t max_h = 0, max_w = 0;
                
                // Find max value in the pooling window
                for (std::size_t ph = 0; ph < pool_height; ++ph) {
                    for (std::size_t pw = 0; pw < pool_width; ++pw) {
                        std::size_t ih = h * stride_h + ph;
                        std::size_t iw = w * stride_w + pw;
                        
                        if (ih < input_height && iw < input_width) {
                            std::size_t input_idx = ih * input_width + iw;
                            if (input(b, input_idx) > max_val) {
                                max_val = input(b, input_idx);
                                max_h = ih;
                                max_w = iw;
                            }
                        }
                    }
                }
                
                // Store max value and mark position in mask
                std::size_t output_idx = h * out_width + w;
                output(b, output_idx) = max_val;
                
                // Mark the position of max value in the mask
                std::size_t mask_idx = max_h * input_width + max_w;
                mask_cache(b, mask_idx) = 1.0;
            }
        }
    }
    
    return output;
}

Matrix MaxPool2D::backward(const Matrix& grad_output) {
    // Calculate dimensions
    std::size_t total_spatial_size = input_cache.shape[1];
    std::size_t input_channels = 1; // For this simplified implementation
    std::size_t input_height = static_cast<std::size_t>(std::sqrt(total_spatial_size));
    std::size_t input_width = total_spatial_size / input_height;
    
    std::size_t out_height = (input_height - pool_height) / stride_h + 1;
    std::size_t out_width = (input_width - pool_width) / stride_w + 1;
    
    // Initialize gradient matrix
    Matrix grad_input(input_cache.shape[0], input_cache.shape[1], 0.0);
    
    // Propagate gradients only to positions that had max values
    for (std::size_t b = 0; b < grad_output.shape[0]; ++b) {
        for (std::size_t h = 0; h < out_height; ++h) {
            for (std::size_t w = 0; w < out_width; ++w) {
                // Get the gradient for this pooling region
                std::size_t output_idx = h * out_width + w;
                double grad_val = grad_output(b, output_idx);
                
                // Find the max position in the original input
                for (std::size_t ph = 0; ph < pool_height; ++ph) {
                    for (std::size_t pw = 0; pw < pool_width; ++pw) {
                        std::size_t ih = h * stride_h + ph;
                        std::size_t iw = w * stride_w + pw;
                        
                        if (ih < input_height && iw < input_width) {
                            std::size_t input_idx = ih * input_width + iw;
                            // Only propagate gradient if this position was the max
                            if (mask_cache(b, input_idx) > 0.0) {
                                grad_input(b, input_idx) += grad_val;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return grad_input;
}

// ---------------- Dropout Layer Methods ----------------
Matrix Dropout::forward(const Matrix& input) {
    if (rate <= 0.0) {
        return input;  // No dropout
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0 - rate);
    
    mask_cache.resize(input.size());
    Matrix output(input.shape[0], input.shape[1]);
    
    for (std::size_t i = 0; i < input.size(); ++i) {
        mask_cache[i] = dist(gen);
        output.data()[i] = input.data()[i] * (mask_cache[i] ? 1.0 : 0.0) / (1.0 - rate);
    }
    
    return output;
}

Matrix Dropout::backward(const Matrix& grad_output) {
    if (rate <= 0.0) {
        return grad_output;  // No dropout during backward pass
    }
    
    Matrix grad_input(grad_output.shape[0], grad_output.shape[1]);
    
    for (std::size_t i = 0; i < grad_output.size(); ++i) {
        grad_input.data()[i] = grad_output.data()[i] * (mask_cache[i] ? 1.0 : 0.0) / (1.0 - rate);
    }
    
    return grad_input;
}

// ---------------- BatchNorm Layer Methods ----------------
Matrix BatchNorm::forward(const Matrix& input) {
    if (input.shape[1] != features) {
        throw std::invalid_argument("BatchNorm: input features mismatch");
    }
    
    // Store input for backward pass
    input_cache = input;
    
    std::size_t batch_size = input.shape[0];
    
    // Calculate mean and variance for the batch
    Matrix batch_mean(1, features);
    Matrix batch_var(1, features);
    
    // Calculate mean
    for (std::size_t f = 0; f < features; ++f) {
        double sum = 0.0;
        for (std::size_t b = 0; b < batch_size; ++b) {
            sum += input(b, f);
        }
        batch_mean(0, f) = sum / static_cast<double>(batch_size);
    }
    
    // Calculate variance
    for (std::size_t f = 0; f < features; ++f) {
        double sum = 0.0;
        for (std::size_t b = 0; b < batch_size; ++b) {
            double diff = input(b, f) - batch_mean(0, f);
            sum += diff * diff;
        }
        batch_var(0, f) = sum / static_cast<double>(batch_size);
    }
    
    // Update running statistics
    running_mean = add(scalar_mul(running_mean, 1.0 - momentum), scalar_mul(batch_mean, momentum));
    running_var = add(scalar_mul(running_var, 1.0 - momentum), scalar_mul(batch_var, momentum));
    
    // Normalize
    x_centered_cache = Matrix(batch_size, features);
    x_norm_cache = Matrix(batch_size, features);
    inv_std_cache = Matrix(1, features); // Store inverse std for each feature
    
    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t f = 0; f < features; ++f) {
            x_centered_cache(b, f) = input(b, f) - batch_mean(0, f);
            double inv_std = 1.0 / std::sqrt(batch_var(0, f) + epsilon);
            x_norm_cache(b, f) = x_centered_cache(b, f) * inv_std;
            if (b == 0) { // Store inverse std for backward pass
                inv_std_cache(0, f) = inv_std;
            }
        }
    }
    
    // Scale and shift
    Matrix output(batch_size, features);
    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t f = 0; f < features; ++f) {
            output(b, f) = x_norm_cache(b, f) * gamma(0, f) + beta(0, f);
        }
    }
    
    return output;
}

Matrix BatchNorm::backward(const Matrix& grad_output) {
    std::size_t batch_size = grad_output.shape[0];
    
    // Gradients for gamma and beta
    Matrix grad_gamma(1, features);
    Matrix grad_beta(1, features);
    
    for (std::size_t f = 0; f < features; ++f) {
        double sum_gamma = 0.0;
        double sum_beta = 0.0;
        for (std::size_t b = 0; b < batch_size; ++b) {
            sum_gamma += grad_output(b, f) * x_norm_cache(b, f);
            sum_beta += grad_output(b, f);
        }
        grad_gamma(0, f) = sum_gamma;
        grad_beta(0, f) = sum_beta;
    }
    
    // Store gradients for optimizer update
    weight_velocity = grad_gamma; // Using weight_velocity for gamma gradient
    bias_velocity = grad_beta;    // Using bias_velocity for beta gradient
    
    // Gradient for input
    Matrix grad_input(batch_size, features);
    
    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t f = 0; f < features; ++f) {
            // Gradient through scale and shift
            double grad_norm = grad_output(b, f) * gamma(0, f);
            
            // Gradient of variance
            double grad_var = grad_norm * x_centered_cache(b, f) * -0.5 *
                             std::pow(inv_std_cache(0, f), 3.0);
            
            // Gradient of mean
            double grad_mean = grad_norm * -inv_std_cache(0, f);
            
            // Combine gradients
            grad_input(b, f) = grad_norm * inv_std_cache(0, f) +
                              grad_var * 2.0 * x_centered_cache(b, f) / static_cast<double>(batch_size) +
                              grad_mean / static_cast<double>(batch_size);
        }
    }
    
    return grad_input;
}

void BatchNorm::update_parameters(const Optimizer& opt) {
    clip_gradient(weight_velocity, opt);
    clip_gradient(bias_velocity, opt);
    add_regularization(weight_velocity, gamma, opt);
    add_regularization(bias_velocity, beta, opt);
    apply_optimizer_update(opt, gamma, weight_velocity, weight_momentum, weight_rms);
    apply_optimizer_update(opt, beta, bias_velocity, bias_momentum, bias_rms);
}

// ---------------- Optimizer Methods ----------------
// Base Optimizer methods
void Optimizer::enable_gradient_clipping(double max_norm) {
    use_gradient_clipping = true;
    max_gradient_norm = max_norm;
}

void Optimizer::enable_gradient_clipping_by_value(double clip_val) {
    use_gradient_clipping = true;
    clip_value = clip_val;
}

void Optimizer::disable_gradient_clipping() {
    use_gradient_clipping = false;
    max_gradient_norm = 0.0;
    clip_value = 0.0;
}

void Optimizer::set_regularization(double l1_reg, double l2_reg) {
    l1_lambda = l1_reg;
    l2_lambda = l2_reg;
}

void Optimizer::set_lr_scheduler(std::unique_ptr<LRScheduler> scheduler) {
    lr_scheduler = std::move(scheduler);
}

void Optimizer::update_learning_rate() {
    if (lr_scheduler) {
        lr_scheduler->step();
    }
}

void SGD::step() {
    // This method is called after all gradients have been computed
    // Actual parameter updates are done in each layer's update_parameters method
    // This is just a placeholder to match the interface
}

void SGD::zero_grad() {
    // This method zeros the gradients in all layers
    // This would be called by the model before each forward pass
}

// ---------------- Learning Rate Scheduler Methods ----------------
void StepLR::step() {
    last_epoch++;
    if (last_epoch % step_size == 0) {
        optimizer.learning_rate = initial_lr * std::pow(gamma, static_cast<double>(last_epoch / step_size));
    }
}

void ExponentialLR::step() {
    last_epoch++;
    optimizer.learning_rate = initial_lr * std::pow(gamma, static_cast<double>(last_epoch));
}

void PolynomialLR::step() {
    last_epoch++;
    double progress = static_cast<double>(last_epoch) / static_cast<double>(max_epochs);
    progress = std::min(progress, 1.0);
    
    double lr = initial_lr + (end_lr - initial_lr) * std::pow(progress, power);
    optimizer.learning_rate = lr;
}

void CosineAnnealingLR::step() {
    last_epoch++;
    double progress = static_cast<double>(last_epoch % t_max) / static_cast<double>(t_max);
    
    double lr = eta_min + (initial_lr - eta_min) * (1.0 + std::cos(progress * M_PI)) / 2.0;
    optimizer.learning_rate = lr;
}

void ReduceLROnPlateau::step() {
    // This scheduler requires a metric to be provided, so this is a placeholder
    // In practice, this would be implemented to use an internal metric
    last_epoch++;
}

void ReduceLROnPlateau::step(double metric) {
    last_epoch++;
    
    bool is_better = false;
    if (threshold_mode == 0) { // relative threshold
        if (mode_min > 0) { // minimizing
            is_better = metric < best * (1.0 - threshold);
        } else { // maximizing
            is_better = metric > best * (1.0 + threshold);
        }
    } else { // absolute threshold
        if (mode_min > 0) { // minimizing
            is_better = metric < best - threshold;
        } else { // maximizing
            is_better = metric > best + threshold;
        }
    }
    
    if (is_better) {
        best = metric;
        num_bad_epochs = 0;
        in_cooldown = false;
    } else {
        num_bad_epochs++;
        if (in_cooldown) {
            cooldown--;
            if (cooldown == 0) {
                in_cooldown = false;
            }
        }
        
        if (num_bad_epochs > patience && !in_cooldown) {
            double old_lr = optimizer.learning_rate;
            optimizer.learning_rate = old_lr * factor;
            cooldown = patience;
            in_cooldown = true;
            num_bad_epochs = 0;
        }
    }
}

void Adam::step() {
    // This method is called after all gradients have been computed
    // Actual parameter updates are done in each layer's update_parameters method
    // Increment step count for bias correction
    step_count++;
}

void Adam::zero_grad() {
    // This method zeros the gradients in all layers
    // This would be called by the model before each forward pass
}

// ---------------- Model Implementation ----------------
Model::Model(const Config& cfg) : config(cfg) {}

void Model::add(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

Matrix Model::forward(const Matrix& input) {
    Matrix current = input;
    for (auto& layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

double Model::compute_loss(const Matrix& predictions, const Matrix& targets, LossFunction loss_fn) {
    LossResult result = dnn::compute_loss(targets, predictions, loss_fn);
    return result.value;
}

void Model::backward(const Matrix& loss_gradient) {
    Matrix current_grad = loss_gradient;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        current_grad = (*it)->backward(current_grad);
    }
}

void Model::train_step(const Matrix& inputs, const Matrix& targets, LossFunction loss_fn) {
    if (!optimizer) {
        throw std::runtime_error("Model must be compiled with an optimizer before training");
    }
    
    // Forward pass
    Matrix predictions = forward(inputs);
    
    // Compute loss
    LossResult loss_result = dnn::compute_loss(targets, predictions, loss_fn);
    
    // Backward pass
    backward(loss_result.gradient);
    
    // Update parameters
    optimizer->step();
    for (auto& layer : layers) {
        if (layer->trainable) {
            layer->update_parameters(*optimizer);
        }
    }
    optimizer->update_learning_rate();
    
    // Zero gradients after update
    optimizer->zero_grad();
}

void Model::compile(std::unique_ptr<Optimizer> opt) {
    optimizer = std::move(opt);
}

void Model::fit(const Matrix& X, const Matrix& y,
                int epochs, LossFunction loss_fn,
                std::mt19937& rng,
                double validation_split,
                bool verbose) {
    if (!optimizer) {
        throw std::runtime_error("Model must be compiled with an optimizer before training");
    }
    
    // Split data if validation is requested
    Matrix X_train = X, y_train = y, X_val, y_val;
    
    if (validation_split > 0.0) {
        std::size_t val_size = static_cast<std::size_t>(X.shape[0] * validation_split);
        std::size_t train_size = X.shape[0] - val_size;
        
        X_train = Matrix(train_size, X.shape[1]);
        y_train = Matrix(train_size, y.shape[1]);
        X_val = Matrix(val_size, X.shape[1]);
        y_val = Matrix(val_size, y.shape[1]);
        
        // Create shuffled indices for validation split
        std::vector<std::size_t> indices(X.shape[0]);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        // Copy data according to shuffled indices
        for (std::size_t i = 0; i < train_size; ++i) {
            std::size_t idx = indices[i];
            for (std::size_t j = 0; j < X.shape[1]; ++j) {
                X_train(i, j) = X(idx, j);
            }
            for (std::size_t j = 0; j < y.shape[1]; ++j) {
                y_train(i, j) = y(idx, j);
            }
        }
        
        for (std::size_t i = 0; i < val_size; ++i) {
            std::size_t idx = indices[train_size + i];
            for (std::size_t j = 0; j < X.shape[1]; ++j) {
                X_val(i, j) = X(idx, j);
            }
            for (std::size_t j = 0; j < y.shape[1]; ++j) {
                y_val(i, j) = y(idx, j);
            }
        }
    }
    
    // Training loop with enhanced features
    double best_val_loss = std::numeric_limits<double>::infinity();
    int patience_counter = 0;
    const int patience = 50; // Early stopping patience
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle training data at the beginning of each epoch
        std::vector<std::size_t> train_indices(X_train.shape[0]);
        std::iota(train_indices.begin(), train_indices.end(), 0);
        std::shuffle(train_indices.begin(), train_indices.end(), rng);
        
        // Process in batches
        const std::size_t batch_size = config.batch_size;
        for (std::size_t start_idx = 0; start_idx < X_train.shape[0]; start_idx += batch_size) {
            std::size_t end_idx = std::min(start_idx + batch_size, X_train.shape[0]);
            
            // Create batch indices
            std::vector<std::size_t> batch_indices(train_indices.begin() + static_cast<std::ptrdiff_t>(start_idx),
                                                   train_indices.begin() + static_cast<std::ptrdiff_t>(end_idx));
            
            // Create batch matrices
            Matrix X_batch(batch_indices.size(), X_train.shape[1]);
            Matrix y_batch(batch_indices.size(), y_train.shape[1]);
            
            for (std::size_t i = 0; i < batch_indices.size(); ++i) {
                std::size_t idx = batch_indices[i];
                for (std::size_t j = 0; j < X_train.shape[1]; ++j) {
                    X_batch(i, j) = X_train(idx, j);
                }
                for (std::size_t j = 0; j < y_train.shape[1]; ++j) {
                    y_batch(i, j) = y_train(idx, j);
                }
            }
            
            train_step(X_batch, y_batch, loss_fn);
        }
        
        // Evaluate at specified intervals
        if (verbose && (epoch % 10 == 0 || epoch == epochs - 1)) {
            double train_loss = compute_loss(forward(X_train), y_train, loss_fn);
            std::cout << "[" << now_time() << "] Epoch " << epoch << ", Train Loss: " << std::fixed << std::setprecision(6) << train_loss;
            
            if (validation_split > 0.0) {
                double val_loss = compute_loss(forward(X_val), y_val, loss_fn);
                std::cout << ", Val Loss: " << std::fixed << std::setprecision(6) << val_loss;
                
                // Early stopping logic
                if (val_loss < best_val_loss) {
                    best_val_loss = val_loss;
                    patience_counter = 0;
                } else {
                    patience_counter++;
                    if (patience_counter >= patience) {
                        std::cout << " [Early Stopping triggered]" << std::endl;
                        break;
                    }
                }
            }
            std::cout << std::endl;
        }
    }
}

Matrix Model::predict(const Matrix& input) {
    return forward(input);
}

double Model::evaluate(const Matrix& X, const Matrix& y, LossFunction loss_fn) {
    Matrix predictions = forward(X);
    return compute_loss(predictions, y, loss_fn);
}

void Model::save(const std::string& filepath) const {
    if (!ModelSerializer::save_model(*this, filepath)) {
        throw std::runtime_error("Failed to save model to " + filepath);
    }
}

void Model::load(const std::string& filepath) {
    if (!ModelSerializer::load_model(*this, filepath)) {
        throw std::runtime_error("Failed to load model from " + filepath);
    }
}

std::size_t Model::get_parameter_count() const {
    std::size_t total = 0;
    for (const auto& layer : layers) {
        if (auto dense = dynamic_cast<const Dense*>(layer.get())) {
            total += dense->weights.size() + dense->bias.size();
        } else if (auto conv = dynamic_cast<const Conv2D*>(layer.get())) {
            total += conv->weights.size() + conv->bias.size();
        } else if (auto batch = dynamic_cast<const BatchNorm*>(layer.get())) {
            total += batch->gamma.size() + batch->beta.size();
        } else {
            total += layer->get_parameter_count();
        }
    }
    return total;
}

void Model::print_summary() const {
    std::cout << "Model Summary:" << std::endl;
    std::cout << "Total Parameters: " << get_parameter_count() << std::endl;
    std::cout << "Layers:" << std::endl;
    
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "  " << i << ": " << layers[i]->name 
                  << " (trainable: " << layers[i]->trainable << ")" << std::endl;
    }
}

std::size_t Model::layer_count() const {
    return layers.size();
}

std::size_t Model::parameter_count(std::size_t index) const {
    if (index >= layers.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    const auto& layer = layers[index];
    if (auto dense = dynamic_cast<const Dense*>(layer.get())) {
        return dense->weights.size() + dense->bias.size();
    }
    if (auto conv = dynamic_cast<const Conv2D*>(layer.get())) {
        return conv->weights.size() + conv->bias.size();
    }
    if (auto batch = dynamic_cast<const BatchNorm*>(layer.get())) {
        return batch->gamma.size() + batch->beta.size();
    }
    return layer->get_parameter_count();
}

const Config& Model::get_config() const {
    return config;
}

Config& Model::get_config() {
    return config;
}

const Optimizer* Model::get_optimizer() const {
    return optimizer.get();
}

void Model::set_optimizer(std::unique_ptr<Optimizer> opt) {
    optimizer = std::move(opt);
}std::pair<Matrix, Matrix> train_test_split(const Matrix& X, const Matrix& y, double test_size, std::mt19937& rng) {
    std::size_t total_samples = X.shape[0];
    std::size_t test_samples = static_cast<std::size_t>(static_cast<double>(total_samples) * test_size);
    std::size_t train_samples = total_samples - test_samples;
    
    // Create shuffled indices
    std::vector<std::size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    // Split the data
    Matrix X_train(train_samples, X.shape[1]);
    Matrix y_train(train_samples, y.shape[1]);
    Matrix X_test(test_samples, X.shape[1]);
    Matrix y_test(test_samples, y.shape[1]);
    
    for (std::size_t i = 0; i < train_samples; ++i) {
        std::size_t idx = indices[i];
        for (std::size_t j = 0; j < X.shape[1]; ++j) {
            X_train(i, j) = X(idx, j);
        }
        for (std::size_t j = 0; j < y.shape[1]; ++j) {
            y_train(i, j) = y(idx, j);
        }
    }
    
    for (std::size_t i = 0; i < test_samples; ++i) {
        std::size_t idx = indices[train_samples + i];
        for (std::size_t j = 0; j < X.shape[1]; ++j) {
            X_test(i, j) = X(idx, j);
        }
        for (std::size_t j = 0; j < y.shape[1]; ++j) {
            y_test(i, j) = y(idx, j);
        }
    }
    
    return std::make_pair(std::move(X_train), std::move(y_train));
}

} // namespace dnn
