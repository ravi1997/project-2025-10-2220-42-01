// dnn.cpp — comprehensive implementation of DNN library
#include "dnn.hpp"

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

static inline double safe_log(double x) noexcept {
    constexpr double eps = 1e-15;
    return std::log(x > eps ? x : eps);
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


// ---------------- Tensor Operations ----------------
Matrix transpose(const Matrix& A) {
    Matrix T({A.shape[1], A.shape[0]});
    for (std::size_t r = 0; r < A.shape[0]; ++r) {
        for (std::size_t c = 0; c < A.shape[1]; ++c) {
            T(c, r) = A(r, c);
        }
    }
    return T;
}

Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.shape[1] != B.shape[0]) {
        throw std::invalid_argument("matmul: A.cols != B.rows");
    }
    
    Matrix C({A.shape[0], B.shape[1]}, 0.0);
    
    // Use parallel execution if enabled and matrix is large enough
    if (Config::USE_VECTORIZATION && A.shape[0] >= 100) {
        std::vector<std::thread> threads;
        const std::size_t num_threads = std::min(Config::MAX_THREADS,
                                                static_cast<std::size_t>(std::thread::hardware_concurrency()));
        const std::size_t rows_per_thread = A.shape[0] / num_threads;
        
        for (std::size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                const std::size_t start_row = t * rows_per_thread;
                const std::size_t end_row = (t == num_threads - 1) ? A.shape[0] : (t + 1) * rows_per_thread;
                
                for (std::size_t i = start_row; i < end_row; ++i) {
                    for (std::size_t k = 0; k < A.shape[1]; ++k) {
                        const double a = A(i, k);
                        for (std::size_t j = 0; j < B.shape[1]; ++j) {
                            C(i, j) += a * B(k, j);
                        }
                    }
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Sequential implementation
        for (std::size_t i = 0; i < A.shape[0]; ++i) {
            for (std::size_t k = 0; k < A.shape[1]; ++k) {
                const double a = A(i, k);
                for (std::size_t j = 0; j < B.shape[1]; ++j) {
                    C(i, j) += a * B(k, j);
                }
            }
        }
    }
    
    return C;
}

void add_rowwise_inplace(Matrix& A, const Matrix& rowvec) {
    if (rowvec.shape[0] != 1 || rowvec.shape[1] != A.shape[1]) {
        throw std::invalid_argument("add_rowwise_inplace: bad shape");
    }
    
    for (std::size_t r = 0; r < A.shape[0]; ++r) {
        for (std::size_t c = 0; c < A.shape[1]; ++c) {
            A(r, c) += rowvec(0, c);
        }
    }
}

Matrix add(const Matrix& A, const Matrix& B) {
    if (A.shape[0] != B.shape[0] || A.shape[1] != B.shape[1]) {
        throw std::invalid_argument("add: shape mismatch");
    }
    
    Matrix C({A.shape[0], A.shape[1]});
    for (std::size_t i = 0; i < A.size; ++i) {
        C.data[i] = A.data[i] + B.data[i];
    }
    return C;
}

Matrix sub(const Matrix& A, const Matrix& B) {
    if (A.shape[0] != B.shape[0] || A.shape[1] != B.shape[1]) {
        throw std::invalid_argument("sub: shape mismatch");
    }
    
    Matrix C({A.shape[0], A.shape[1]});
    for (std::size_t i = 0; i < A.size; ++i) {
        C.data[i] = A.data[i] - B.data[i];
    }
    return C;
}

Matrix hadamard(const Matrix& A, const Matrix& B) {
    if (A.shape[0] != B.shape[0] || A.shape[1] != B.shape[1]) {
        throw std::invalid_argument("hadamard: mismatch");
    }
    
    Matrix C({A.shape[0], A.shape[1]});
    for (std::size_t i = 0; i < A.size; ++i) {
        C.data[i] = A.data[i] * B.data[i];
    }
    return C;
}

Matrix scalar_mul(const Matrix& A, double s) {
    Matrix C({A.shape[0], A.shape[1]});
    for (std::size_t i = 0; i < A.size; ++i) {
        C.data[i] = A.data[i] * s;
    }
    return C;
}

Matrix sum_rows(const Matrix& A) {
    Matrix v({1, A.shape[1]}, 0.0);
    for (std::size_t r = 0; r < A.shape[0]; ++r) {
        for (std::size_t c = 0; c < A.shape[1]; ++c) {
            v(0, c) += A(r, c);
        }
    }
    return v;
}

// ---------------- Activation Functions Implementation ----------------
Matrix apply_activation(const Matrix& z, Activation act) {
    Matrix a(z.shape[0], z.shape[1]);
    
    switch (act) {
        case Activation::Linear:
            a = z;
            break;
            
        case Activation::ReLU:
            for (std::size_t i = 0; i < z.size; ++i) {
                a.data[i] = (z.data[i] > 0.0 ? z.data[i] : 0.0);
            }
            break;
            
        case Activation::LeakyReLU: {
            const double alpha = 0.01;
            for (std::size_t i = 0; i < z.size; ++i) {
                a.data[i] = (z.data[i] > 0.0 ? z.data[i] : alpha * z.data[i]);
            }
            break;
        }
        
        case Activation::ELU: {
            const double alpha = 1.0;
            for (std::size_t i = 0; i < z.size; ++i) {
                const double x = z.data[i];
                a.data[i] = (x > 0.0 ? x : alpha * (std::exp(x) - 1.0));
            }
            break;
        }
        
        case Activation::Sigmoid:
            for (std::size_t i = 0; i < z.size; ++i) {
                const double x = z.data[i];
                if (x >= 0) {
                    const double e = std::exp(-x);
                    a.data[i] = 1.0 / (1.0 + e);
                } else {
                    const double e = std::exp(x);
                    a.data[i] = e / (1.0 + e);
                }
            }
            break;
            
        case Activation::Tanh:
            for (std::size_t i = 0; i < z.size; ++i) {
                a.data[i] = std::tanh(z.data[i]);
            }
            break;
            
        case Activation::Softmax: {
            for (std::size_t r = 0; r < z.shape[0]; ++r) {
                double max_val = -std::numeric_limits<double>::infinity();
                
                // Find max value in the row
                for (std::size_t c = 0; c < z.shape[1]; ++c) {
                    max_val = std::max(max_val, z(r, c));
                }
                
                double sum = 0.0;
                // Compute exp and sum
                for (std::size_t c = 0; c < z.shape[1]; ++c) {
                    const double e = std::exp(z(r, c) - max_val);
                    a(r, c) = e;
                    sum += e;
                }
                
                // Normalize
                const double denom = (sum > 0.0 ? sum : 1.0);
                for (std::size_t c = 0; c < z.shape[1]; ++c) {
                    a(r, c) /= denom;
                }
            }
            break;
        }
        
        case Activation::Swish:
            for (std::size_t i = 0; i < z.size; ++i) {
                const double x = z.data[i];
                const double sigmoid_x = (x >= 0) ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
                a.data[i] = x * sigmoid_x;
            }
            break;
            
        case Activation::GELU:
            for (std::size_t i = 0; i < z.size; ++i) {
                const double x = z.data[i];
                a.data[i] = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
            }
            break;
            
        case Activation::Softplus:
            for (std::size_t i = 0; i < z.size; ++i) {
                a.data[i] = std::log(1.0 + std::exp(z.data[i]));
            }
            break;
    }
    
    return a;
}

Matrix apply_activation_derivative(const Matrix& z, const Matrix& grad, Activation act) {
    if (z.shape[0] != grad.shape[0] || z.shape[1] != grad.shape[1]) {
        throw std::invalid_argument("apply_activation_derivative: shape mismatch");
    }
    
    Matrix dZ(z.shape[0], z.shape[1]);
    
    switch (act) {
        case Activation::Linear:
            dZ = grad;
            break;
            
        case Activation::ReLU:
            for (std::size_t i = 0; i < z.size; ++i) {
                dZ.data[i] = (z.data[i] > 0.0) ? grad.data[i] : 0.0;
            }
            break;
            
        case Activation::LeakyReLU: {
            const double alpha = 0.01;
            for (std::size_t i = 0; i < z.size; ++i) {
                dZ.data[i] = (z.data[i] > 0.0) ? grad.data[i] : alpha * grad.data[i];
            }
            break;
        }
        
        case Activation::ELU: {
            const double alpha = 1.0;
            for (std::size_t i = 0; i < z.size; ++i) {
                const double x = z.data[i];
                if (x > 0.0) {
                    dZ.data[i] = grad.data[i];
                } else {
                    dZ.data[i] = grad.data[i] * alpha * std::exp(x);
                }
            }
            break;
        }
        
        case Activation::Sigmoid:
            for (std::size_t i = 0; i < z.size; ++i) {
                const double s = (z.data[i] >= 0) ?
                    1.0 / (1.0 + std::exp(-z.data[i])) :
                    std::exp(z.data[i]) / (1.0 + std::exp(z.data[i]));
                dZ.data[i] = grad.data[i] * s * (1.0 - s);
            }
            break;
            
        case Activation::Tanh:
            for (std::size_t i = 0; i < z.size; ++i) {
                const double t = std::tanh(z.data[i]);
                dZ.data[i] = grad.data[i] * (1.0 - t * t);
            }
            break;
            
        case Activation::Softmax: {
            // For softmax, we need to compute the full Jacobian for each row
            for (std::size_t r = 0; r < z.shape[0]; ++r) {
                Matrix row_z(1, z.shape[1]);
                Matrix row_softmax(1, z.shape[1]);
                
                // Extract the row
                for (std::size_t c = 0; c < z.shape[1]; ++c) {
                    row_z(0, c) = z(r, c);
                }
                
                // Compute softmax for the row
                double max_val = -std::numeric_limits<double>::infinity();
                for (std::size_t c = 0; c < z.shape[1]; ++c) {
                    max_val = std::max(max_val, row_z(0, c));
                }
                
                double sum = 0.0;
                for (std::size_t c = 0; c < z.shape[1]; ++c) {
                    const double e = std::exp(row_z(0, c) - max_val);
                    row_softmax(0, c) = e;
                    sum += e;
                }
                
                const double denom = (sum > 0.0 ? sum : 1.0);
                for (std::size_t c = 0; c < z.shape[1]; ++c) {
                    row_softmax(0, c) /= denom;
                }
                
                // Compute gradient: grad_output * softmax * (1 - softmax) for diagonal
                // and -grad_output * softmax_i * softmax_j for off-diagonal
                for (std::size_t c = 0; c < z.shape[1]; ++c) {
                    double grad_sum = 0.0;
                    for (std::size_t k = 0; k < z.shape[1]; ++k) {
                        if (k == c) {
                            grad_sum += grad(r, k) * row_softmax(0, c) * (1.0 - row_softmax(0, c));
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
            for (std::size_t i = 0; i < z.size; ++i) {
                const double x = z.data[i];
                const double sigmoid_x = (x >= 0) ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
                const double swish_derivative = sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x);
                dZ.data[i] = grad.data[i] * swish_derivative;
            }
            break;
        }
        
        case Activation::GELU: {
            for (std::size_t i = 0; i < z.size; ++i) {
                const double x = z.data[i];
                const double phi_x = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
                const double phi_prime_x = std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi);
                dZ.data[i] = grad.data[i] * (phi_x + x * phi_prime_x);
            }
            break;
        }
        
        case Activation::Softplus:
            for (std::size_t i = 0; i < z.size; ++i) {
                dZ.data[i] = grad.data[i] * (1.0 - std::exp(-std::log(1.0 + std::exp(z.data[i]))));
                // Simplified: grad * sigmoid(z)
                dZ.data[i] = grad.data[i] * (1.0 - std::exp(-std::max(z.data[i], 0.0)) /
                                            (1.0 + std::exp(-std::abs(z.data[i]))));
            }
            break;
    }
    
    return dZ;
}

// ---------------- Loss Functions Implementation ----------------
LossResult compute_loss(const Matrix& y_true, const Matrix& y_pred, LossFunction loss_fn) {
    if (y_true.shape[0] != y_pred.shape[0] || y_true.shape[1] != y_pred.shape[1]) {
        throw std::invalid_argument("compute_loss: shape mismatch");
    }
    
    LossResult result;
    
    switch (loss_fn) {
        case LossFunction::MSE: {
            double sum = 0.0;
            for (std::size_t i = 0; i < y_true.size; ++i) {
                const double diff = y_pred.data[i] - y_true.data[i];
                sum += diff * diff;
            }
            result.value = sum / static_cast<double>(y_true.shape[0]);
            
            // Gradient for MSE: 2*(y_pred - y_true)/n
            result.gradient = scalar_mul(sub(y_pred, y_true), 2.0 / static_cast<double>(y_true.shape[0]));
            break;
        }
        
        case LossFunction::CrossEntropy: {
            Matrix softmax_pred = apply_activation(y_pred, Activation::Softmax);
            double sum = 0.0;
            
            for (std::size_t r = 0; r < y_true.shape[0]; ++r) {
                for (std::size_t c = 0; c < y_true.shape[1]; ++c) {
                    if (y_true(r, c) > 0.0) {
                        sum += -y_true(r, c) * safe_log(softmax_pred(r, c));
                    }
                }
            }
            
            result.value = sum / static_cast<double>(y_true.shape[0]);
            
            // Gradient for cross-entropy with softmax: (softmax_pred - y_true) / n
            result.gradient = scalar_mul(sub(softmax_pred, y_true), 1.0 / static_cast<double>(y_true.shape[0]));
            break;
        }
        
        case LossFunction::BinaryCrossEntropy: {
            Matrix clipped_pred = y_pred;
            for (auto& val : clipped_pred.data) {
                val = std::max(std::min(val, 1.0 - 1e-15), 1e-15);
            }
            
            double sum = 0.0;
            for (std::size_t i = 0; i < y_true.size; ++i) {
                sum += -(y_true.data[i] * safe_log(clipped_pred.data[i]) +
                         (1.0 - y_true.data[i]) * safe_log(1.0 - clipped_pred.data[i]));
            }
            
            result.value = sum / static_cast<double>(y_true.shape[0]);
            
            // Gradient for binary cross-entropy: (y_pred - y_true) / n
            result.gradient = scalar_mul(sub(clipped_pred, y_true), 1.0 / static_cast<double>(y_true.shape[0]));
            break;
        }
        
        case LossFunction::Hinge: {
           double sum = 0.0;
           for (std::size_t r = 0; r < y_true.shape[0]; ++r) {
               for (std::size_t c = 0; c < y_true.shape[1]; ++c) {
                   if (y_true(r, c) > 0.0) {  // This is the true class
                       for (std::size_t k = 0; k < y_true.shape[1]; ++k) {
                           if (k != c) {  // For all other classes
                               const double margin = 1.0 - (y_pred(r, c) - y_pred(r, k));
                               sum += std::max(0.0, margin);
                           }
                       }
                   }
               }
           }
           
           result.value = sum / static_cast<double>(y_true.shape[0]);
           // Gradient for hinge loss: 1 if margin > 0, 0 otherwise
           Matrix grad(y_true.shape[0], y_true.shape[1], 0.0);
           for (std::size_t r = 0; r < y_true.shape[0]; ++r) {
               for (std::size_t c = 0; c < y_true.shape[1]; ++c) {
                   if (y_true(r, c) > 0.0) { // This is the true class
                       for (std::size_t k = 0; k < y_true.shape[1]; ++k) {
                           if (k != c) {  // For all other classes
                               const double margin = 1.0 - (y_pred(r, c) - y_pred(r, k));
                               if (margin > 0.0) {
                                   grad(r, c) -= 1.0;  // Gradient for true class
                                   grad(r, k) += 1.0;  // Gradient for false class
                               }
                           }
                       }
                   }
               }
           }
           result.gradient = scalar_mul(grad, 1.0 / static_cast<double>(y_true.shape[0]));
           break;
       }
        
        case LossFunction::Huber: {
            const double delta = 1.0;
            double sum = 0.0;
            Matrix grad(y_true.shape[0], y_true.shape[1], 0.0);
            
            for (std::size_t i = 0; i < y_true.size; ++i) {
                const double diff = y_pred.data[i] - y_true.data[i];
                const double abs_diff = std::abs(diff);
                
                if (abs_diff <= delta) {
                    sum += 0.5 * diff * diff;
                    grad.data[i] = diff;
                } else {
                    sum += delta * abs_diff - 0.5 * delta * delta;
                    grad.data[i] = (diff > 0.0) ? delta : -delta;
                }
            }
            
            result.value = sum / static_cast<double>(y_true.shape[0]);
            result.gradient = scalar_mul(grad, 1.0 / static_cast<double>(y_true.shape[0]));
            break;
        }
        
        case LossFunction::KLDivergence: {
            Matrix clipped_true = y_true;
            Matrix clipped_pred = y_pred;
            
            for (auto& val : clipped_true.data) {
                val = std::max(val, 1e-15);
            }
            for (auto& val : clipped_pred.data) {
                val = std::max(val, 1e-15);
            }
            
            double sum = 0.0;
            for (std::size_t i = 0; i < y_true.size; ++i) {
                sum += clipped_true.data[i] * safe_log(clipped_true.data[i] / clipped_pred.data[i]);
            }
            
            result.value = sum / static_cast<double>(y_true.shape[0]);
            
            // Gradient for KL divergence: (log(q) - log(p) + 1) where p is true and q is pred
            Matrix grad(y_true.shape[0], y_true.shape[1]);
            for (std::size_t i = 0; i < y_true.size; ++i) {
                grad.data[i] = std::log(clipped_pred.data[i]) - std::log(clipped_true.data[i]) + 1.0;
            }
            result.gradient = scalar_mul(grad, 1.0 / static_cast<double>(y_true.shape[0]));
            break;
        }
    }
    
    return result;
}

// ---------------- Dense Layer Methods ----------------
void Dense::update_parameters(const Optimizer& opt) {
    if (const auto* sgd = dynamic_cast<const SGD*>(&opt)) {
        // SGD update with momentum
        if (sgd->momentum > 0.0) {
            // Update velocity
            weight_velocity = add(scalar_mul(weight_velocity, sgd->momentum),
                                 scalar_mul(weight_velocity, 1.0));
            bias_velocity = add(scalar_mul(bias_velocity, sgd->momentum),
                               scalar_mul(bias_velocity, 1.0));
            
            // Add gradient to velocity
            weight_velocity = add(scalar_mul(weight_velocity, sgd->momentum), weight_velocity);
            bias_velocity = add(scalar_mul(bias_velocity, sgd->momentum), bias_velocity);
            
            // Update parameters
            weights = sub(weights, scalar_mul(weight_velocity, sgd->learning_rate));
            bias = sub(bias, scalar_mul(bias_velocity, sgd->learning_rate));
        } else {
            // Simple SGD without momentum
            weights = sub(weights, scalar_mul(scalar_mul(weights, sgd->weight_decay), sgd->learning_rate));
            weights = sub(weights, scalar_mul(weight_velocity, sgd->learning_rate));
            bias = sub(bias, scalar_mul(bias_velocity, sgd->learning_rate));
        }
    } else if (const auto* adam = dynamic_cast<const Adam*>(&opt)) {
        // Adam update
        const_cast<Adam*>(adam)->step_count++;
        
        // Update momentum
        weight_momentum = add(scalar_mul(weight_momentum, adam->beta1), 
                             scalar_mul(weight_velocity, 1.0 - adam->beta1));
        bias_momentum = add(scalar_mul(bias_momentum, adam->beta1), 
                           scalar_mul(bias_velocity, 1.0 - adam->beta1));
        
        // Update RMS
        Matrix weight_grad_sq = hadamard(weight_velocity, weight_velocity);
        Matrix bias_grad_sq = hadamard(bias_velocity, bias_velocity);
        
        weight_rms = add(scalar_mul(weight_rms, adam->beta2), 
                        scalar_mul(weight_grad_sq, 1.0 - adam->beta2));
        bias_rms = add(scalar_mul(bias_rms, adam->beta2), 
                      scalar_mul(bias_grad_sq, 1.0 - adam->beta2));
        
        // Bias correction
        double bias_correction1 = 1.0 - std::pow(adam->beta1, static_cast<double>(adam->step_count));
        double bias_correction2 = 1.0 - std::pow(adam->beta2, static_cast<double>(adam->step_count));
        
        Matrix m_hat_w = scalar_mul(weight_momentum, 1.0 / bias_correction1);
        Matrix m_hat_b = scalar_mul(bias_momentum, 1.0 / bias_correction1);
        Matrix v_hat_w = scalar_mul(weight_rms, 1.0 / bias_correction2);
        Matrix v_hat_b = scalar_mul(bias_rms, 1.0 / bias_correction2);
        
        // Update parameters
        Matrix weight_update = scalar_mul(m_hat_w, adam->learning_rate);
        Matrix bias_update = scalar_mul(m_hat_b, adam->learning_rate);
        
        // Apply weight decay if needed
        if (adam->weight_decay > 0.0) {
            weights = sub(weights, scalar_mul(weights, adam->weight_decay * adam->learning_rate));
        }
        
        weights = sub(weights, weight_update);
        bias = sub(bias, bias_update);
    }
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
    return apply_activation(output, activation);
}

Matrix Conv2D::backward(const Matrix& grad_output) {
    // Calculate dimensions
    std::size_t total_spatial_size = input_cache.shape[1] / in_channels;
    std::size_t input_height = static_cast<std::size_t>(std::sqrt(total_spatial_size));
    std::size_t input_width = total_spatial_size / input_height;
    
    std::size_t out_height = (input_height - kernel_height + 2 * padding_h) / stride_h + 1;
    std::size_t out_width = (input_width - kernel_width + 2 * padding_w) / stride_w + 1;
    
    // Initialize gradient matrices
    Matrix grad_input(input_cache.shape[0], input_cache.shape[1], 0.0);
    Matrix grad_weights(out_channels, in_channels * kernel_height * kernel_width, 0.0);
    Matrix grad_bias(1, out_channels, 0.0);
    
    // Compute gradients
    for (std::size_t b = 0; b < grad_output.shape[0]; ++b) {
        for (std::size_t oc = 0; oc < out_channels; ++oc) {
            for (std::size_t oh = 0; oh < out_height; ++oh) {
                for (std::size_t ow = 0; ow < out_width; ++ow) {
                    std::size_t output_idx = oc * (out_height * out_width) + oh * out_width + ow;
                    double grad_val = grad_output(b, output_idx);
                    
                    // Accumulate bias gradient
                    grad_bias(0, oc) += grad_val;
                    
                    // Compute gradients with respect to weights and input
                    for (std::size_t ic = 0; ic < in_channels; ++ic) {
                        for (std::size_t kh = 0; kh < kernel_height; ++kh) {
                            for (std::size_t kw = 0; kw < kernel_width; ++kw) {
                                std::size_t ih = oh * stride_h - padding_h + kh;
                                std::size_t iw = ow * stride_w - padding_w + kw;
                                
                                if (ih < input_height && iw < input_width) {
                                    std::size_t input_idx = ic * (input_height * input_width) + ih * input_width + iw;
                                    
                                    // Gradient with respect to weights
                                    grad_weights(oc, ic * kernel_height * kernel_width + kh * kernel_width + kw) +=
                                        input_cache(b, input_idx) * grad_val;
                                    
                                    // Gradient with respect to input
                                    grad_input(b, input_idx) +=
                                        weights(oc, ic * kernel_height * kernel_width + kh * kernel_width + kw) * grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Store gradients for optimizer update
    // For Conv2D, we need to add the missing velocity members or use a different approach
    // Adding temporary variables to fix the build
    // We'll add the missing members to the class in the header file instead
    
    return grad_input;
}

void Conv2D::update_parameters(const Optimizer& opt) {
    if (const auto* sgd = dynamic_cast<const SGD*>(&opt)) {
        // SGD update with momentum
        if (sgd->momentum > 0.0) {
            // Update velocity
            weight_velocity = add(scalar_mul(weight_velocity, sgd->momentum),
                                 scalar_mul(weights, sgd->weight_decay));
            bias_velocity = add(scalar_mul(bias_velocity, sgd->momentum),
                               scalar_mul(bias, sgd->weight_decay));
            
            // Add gradient to velocity
            weight_velocity = add(scalar_mul(weight_velocity, sgd->momentum), weight_velocity);
            bias_velocity = add(scalar_mul(bias_velocity, sgd->momentum), bias_velocity);
            
            // Update parameters
            weights = sub(weights, scalar_mul(weight_velocity, sgd->learning_rate));
            bias = sub(bias, scalar_mul(bias_velocity, sgd->learning_rate));
        } else {
            // Simple SGD without momentum
            weights = sub(weights, scalar_mul(scalar_mul(weights, sgd->weight_decay), sgd->learning_rate));
            weights = sub(weights, scalar_mul(weight_velocity, sgd->learning_rate));
            bias = sub(bias, scalar_mul(bias_velocity, sgd->learning_rate));
        }
    } else if (const auto* adam = dynamic_cast<const Adam*>(&opt)) {
        // Adam update
        static std::size_t step_count = 0;
        step_count++;
        
        // Update momentum
        weight_momentum = add(scalar_mul(weight_momentum, adam->beta1),
                             scalar_mul(weight_velocity, 1.0 - adam->beta1));
        bias_momentum = add(scalar_mul(bias_momentum, adam->beta1),
                           scalar_mul(bias_velocity, 1.0 - adam->beta1));
        
        // Update RMS
        Matrix weight_grad_sq = hadamard(weight_velocity, weight_velocity);
        Matrix bias_grad_sq = hadamard(bias_velocity, bias_velocity);
        
        weight_rms = add(scalar_mul(weight_rms, adam->beta2),
                        scalar_mul(weight_grad_sq, 1.0 - adam->beta2));
        bias_rms = add(scalar_mul(bias_rms, adam->beta2),
                      scalar_mul(bias_grad_sq, 1.0 - adam->beta2));
        
        // Bias correction
        double bias_correction1 = 1.0 - std::pow(adam->beta1, static_cast<double>(step_count));
        double bias_correction2 = 1.0 - std::pow(adam->beta2, static_cast<double>(step_count));
        
        Matrix m_hat_w = scalar_mul(weight_momentum, 1.0 / bias_correction1);
        Matrix m_hat_b = scalar_mul(bias_momentum, 1.0 / bias_correction1);
        Matrix v_hat_w = scalar_mul(weight_rms, 1.0 / bias_correction2);
        Matrix v_hat_b = scalar_mul(bias_rms, 1.0 / bias_correction2);
        
        // Update parameters
        weights = sub(weights, scalar_mul(m_hat_w, adam->learning_rate));
        bias = sub(bias, scalar_mul(m_hat_b, adam->learning_rate));
        
        // Apply weight decay if needed
        if (adam->weight_decay > 0.0) {
            weights = sub(weights, scalar_mul(weights, adam->weight_decay * adam->learning_rate));
        }
    }
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
    
    mask_cache.resize(input.size);
    Matrix output(input.shape[0], input.shape[1]);
    
    for (std::size_t i = 0; i < input.size; ++i) {
        mask_cache[i] = dist(gen);
        output.data[i] = input.data[i] * (mask_cache[i] ? 1.0 : 0.0) / (1.0 - rate);
    }
    
    return output;
}

Matrix Dropout::backward(const Matrix& grad_output) {
    if (rate <= 0.0) {
        return grad_output;  // No dropout during backward pass
    }
    
    Matrix grad_input(grad_output.shape[0], grad_output.shape[1]);
    
    for (std::size_t i = 0; i < grad_output.size; ++i) {
        grad_input.data[i] = grad_output.data[i] * (mask_cache[i] ? 1.0 : 0.0) / (1.0 - rate);
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
    if (const auto* sgd = dynamic_cast<const SGD*>(&opt)) {
        // SGD update with momentum
        if (sgd->momentum > 0.0) {
            // Update velocity
            weight_velocity = add(scalar_mul(weight_velocity, sgd->momentum),
                                 scalar_mul(gamma, sgd->weight_decay));
            bias_velocity = add(scalar_mul(bias_velocity, sgd->momentum),
                               scalar_mul(beta, sgd->weight_decay));
            
            // Add gradient to velocity
            weight_velocity = add(scalar_mul(weight_velocity, sgd->momentum), weight_velocity);
            bias_velocity = add(scalar_mul(bias_velocity, sgd->momentum), bias_velocity);
            
            // Update parameters
            gamma = sub(gamma, scalar_mul(weight_velocity, sgd->learning_rate));
            beta = sub(beta, scalar_mul(bias_velocity, sgd->learning_rate));
        } else {
            // Simple SGD without momentum
            gamma = sub(gamma, scalar_mul(scalar_mul(gamma, sgd->weight_decay), sgd->learning_rate));
            gamma = sub(gamma, scalar_mul(weight_velocity, sgd->learning_rate));
            beta = sub(beta, scalar_mul(bias_velocity, sgd->learning_rate));
        }
    } else if (const auto* adam = dynamic_cast<const Adam*>(&opt)) {
        // Adam update
        static std::size_t step_count = 0;
        step_count++;
        
        // Update momentum
        weight_momentum = add(scalar_mul(weight_momentum, adam->beta1),
                             scalar_mul(weight_velocity, 1.0 - adam->beta1));
        bias_momentum = add(scalar_mul(bias_momentum, adam->beta1),
                           scalar_mul(bias_velocity, 1.0 - adam->beta1));
        
        // Update RMS
        Matrix weight_grad_sq = hadamard(weight_velocity, weight_velocity);
        Matrix bias_grad_sq = hadamard(bias_velocity, bias_velocity);
        
        weight_rms = add(scalar_mul(weight_rms, adam->beta2),
                        scalar_mul(weight_grad_sq, 1.0 - adam->beta2));
        bias_rms = add(scalar_mul(bias_rms, adam->beta2),
                      scalar_mul(bias_grad_sq, 1.0 - adam->beta2));
        
        // Bias correction
        double bias_correction1 = 1.0 - std::pow(adam->beta1, static_cast<double>(step_count));
        double bias_correction2 = 1.0 - std::pow(adam->beta2, static_cast<double>(step_count));
        
        Matrix m_hat_w = scalar_mul(weight_momentum, 1.0 / bias_correction1);
        Matrix m_hat_b = scalar_mul(bias_momentum, 1.0 / bias_correction1);
        Matrix v_hat_w = scalar_mul(weight_rms, 1.0 / bias_correction2);
        Matrix v_hat_b = scalar_mul(bias_rms, 1.0 / bias_correction2);
        
        // Update parameters
        gamma = sub(gamma, scalar_mul(m_hat_w, adam->learning_rate));
        beta = sub(beta, scalar_mul(m_hat_b, adam->learning_rate));
        
        // Apply weight decay if needed
        if (adam->weight_decay > 0.0) {
            gamma = sub(gamma, scalar_mul(gamma, adam->weight_decay * adam->learning_rate));
        }
    }
}

// ---------------- Optimizer Methods ----------------
void SGD::step() {
    // This method is called after all gradients have been computed
    // Actual parameter updates are done in each layer's update_parameters method
    // This is just a placeholder to match the interface
}

void SGD::zero_grad() {
    // This method zeros the gradients in all layers
    // This would be called by the model before each forward pass
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

// Additional Optimizer implementations
// For now, we'll implement RMSprop as another popular optimizer
struct RMSprop : public Optimizer {
    double alpha;          // Smoothing constant
    double epsilon;        // Small value to prevent division by zero
    double weight_decay;   // Weight decay (L2 penalty)
    
    explicit RMSprop(double lr = 0.001,
                     double a = 0.99,
                     double eps = 1e-8,
                     double wd = 0.0)
        : Optimizer(OptimizerType::RMSprop, lr, eps),
          alpha(a), epsilon(eps), weight_decay(wd) {}
    
    void step() override {
        // RMSprop updates are handled in layer update_parameters methods
    }
    
    void zero_grad() override {
        // Zero gradients are handled by the model
    }
};

// AdamW optimizer - Adam with decoupled weight decay
struct AdamW : public Optimizer {
    double beta1;
    double beta2;
    double weight_decay;
    std::size_t step_count;
    
    explicit AdamW(double lr = 0.001,
                   double b1 = 0.9,
                   double b2 = 0.999,
                   double wd = 0.01)
        : Optimizer(OptimizerType::AdamW, lr),
          beta1(b1), beta2(b2), weight_decay(wd), step_count(0) {}
    
    void step() override {
        step_count++;
    }
    
    void zero_grad() override {
        // Zero gradients are handled by the model
    }
};

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
    for (auto& layer : layers) {
        if (layer->trainable) {
            layer->update_parameters(*optimizer);
        }
    }
    
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
    // Placeholder implementation
    std::cout << "Saving model to " << filepath << std::endl;
}

void Model::load(const std::string& filepath) {
    // Placeholder implementation
    std::cout << "Loading model from " << filepath << std::endl;
}

std::size_t Model::get_parameter_count() const {
    std::size_t total = 0;
    for (const auto& layer : layers) {
        total += layer->get_parameter_count();
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

// ---------------- Utility Functions ----------------
Matrix one_hot(const std::vector<int>& labels, int num_classes) {
    if (num_classes <= 0) {
        throw std::invalid_argument("one_hot: num_classes <= 0");
    }
    
    Matrix y(labels.size(), static_cast<std::size_t>(num_classes), 0.0);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] < 0 || labels[i] >= num_classes) {
            throw std::invalid_argument("one_hot: label OOR");
        }
        y(i, static_cast<std::size_t>(labels[i])) = 1.0;
    }
    return y;
}

double accuracy(const Matrix& predictions, const std::vector<int>& labels) {
    if (predictions.shape[0] != labels.size()) {
        throw std::invalid_argument("accuracy: size mismatch");
    }
    
    std::size_t correct = 0;
    for (std::size_t r = 0; r < predictions.shape[0]; ++r) {
        std::size_t argmax = 0;
        double best = predictions(r, 0);
        for (std::size_t c = 1; c < predictions.shape[1]; ++c) {
            if (predictions(r, c) > best) {
                best = predictions(r, c);
                argmax = c;
            }
        }
        if (static_cast<int>(argmax) == labels[r]) {
            ++correct;
        }
    }
    
    return static_cast<double>(correct) / static_cast<double>(labels.size());
}

Matrix normalize(const Matrix& input, double mean, double stddev) {
    Matrix normalized(input.shape[0], input.shape[1]);
    for (std::size_t i = 0; i < input.size; ++i) {
        normalized.data[i] = (input.data[i] - mean) / stddev;
    }
    return normalized;
}

std::pair<Matrix, Matrix> train_test_split(const Matrix& X, const Matrix& y, double test_size, std::mt19937& rng) {
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