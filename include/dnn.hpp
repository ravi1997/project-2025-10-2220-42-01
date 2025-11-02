#pragma once
// dnn.hpp — stdlib-only deep neural network (C++23)
// Comprehensive DNN library with all standard features

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <concepts>
#include <coroutine>
#include <deque>
#include <execution>
#include <format>
#include <functional>
#include <future>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <stdexcept>  // For std::out_of_range
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <concepts>
#include <coroutine>
#include <deque>
#include <execution>
#include <format>
#include <functional>
#include <future>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <stdexcept>  // For std::out_of_range
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace dnn {

// ---------------- Configuration ----------------
struct Config {
    // Compile-time flags for optimization
    static constexpr bool USE_VECTORIZATION = true;
    static constexpr bool ENABLE_THREADING = true;
    static constexpr bool DEBUG_MODE = false;
    static constexpr bool ENABLE_PROFILING = false;
    static constexpr std::size_t MAX_THREADS = 16;
    static constexpr std::size_t CACHE_LINE_SIZE = 64;
    
    // Runtime configuration
    double epsilon = 1e-8;           // Small value for numerical stability
    double learning_rate = 0.001;    // Default learning rate
    std::size_t batch_size = 32;     // Default batch size
    std::size_t max_epochs = 1000;   // Maximum training epochs
    double validation_split = 0.2;   // Validation split ratio
    double dropout_rate = 0.0;       // Dropout rate (0.0 to 1.0)
    bool use_batch_norm = false;     // Whether to use batch normalization
    bool use_layer_norm = false;     // Whether to use layer normalization
};

// ---------------- Tensor ----------------
template<std::size_t NumDims = 2>
struct Tensor {
    std::array<std::size_t, NumDims> shape;
    std::vector<double> data;
    std::size_t size;
    
    Tensor() : size(0) {
        std::fill(shape.begin(), shape.end(), 0);
    }
    
    explicit Tensor(const std::array<std::size_t, NumDims>& s) : shape(s) {
        size = 1;
        for (std::size_t dim : shape) {
            size *= dim;
        }
        data.resize(size, 0.0);
    }
    
    Tensor(const std::array<std::size_t, NumDims>& s, double init_val) : shape(s) {
        size = 1;
        for (std::size_t dim : shape) {
            size *= dim;
        }
        data.resize(size, init_val);
    }
    
    // Constructor for 2D tensors with explicit dimensions
    template<size_t N = NumDims>
    requires (N == 2)
    Tensor(std::size_t rows, std::size_t cols) : shape({rows, cols}), size(rows * cols) {
        data.resize(size, 0.0);
    }
    
    // Constructor for 2D tensors with explicit dimensions and initial value
    template<size_t N = NumDims>
    requires (N == 2)
    Tensor(std::size_t rows, std::size_t cols, double init_val) : shape({rows, cols}), size(rows * cols) {
        data.resize(size, init_val);
    }
    
    // Constructor for 2D tensors using initializer list to fix ambiguity
    template<size_t N = NumDims>
    requires (N == 2)
    Tensor(std::initializer_list<std::size_t> init_list) {
        if (init_list.size() != 2) {
            throw std::invalid_argument("Initializer list for 2D tensor must have exactly 2 elements");
        }
        auto it = init_list.begin();
        shape[0] = *it++;
        shape[1] = *it;
        size = shape[0] * shape[1];
        data.resize(size, 0.0);
    }
    
    // Constructor for 2D tensors using initializer list with initial value to fix ambiguity
    template<size_t N = NumDims>
    requires (N == 2)
    Tensor(std::initializer_list<std::size_t> init_list, double init_val) {
        if (init_list.size() != 2) {
            throw std::invalid_argument("Initializer list for 2D tensor must have exactly 2 elements");
        }
        auto it = init_list.begin();
        shape[0] = *it++;
        shape[1] = *it;
        size = shape[0] * shape[1];
        data.resize(size, init_val);
    }
    
    
    template<typename... Args>
    requires (sizeof...(Args) == NumDims)
    double& operator()(Args... indices) {
        static_assert(sizeof...(Args) == NumDims, "Number of indices must match tensor dimensions");
        std::array<std::size_t, NumDims> idx = {static_cast<std::size_t>(static_cast<long unsigned int>(indices))...};
        std::size_t linear_idx = 0;
        std::size_t multiplier = 1;
        
        // Process dimensions from last to first (row-major order), avoiding signed/unsigned comparison
        for (std::size_t i = 0; i < NumDims; ++i) {
            std::size_t dim = NumDims - 1 - i;
            if (idx[dim] >= shape[dim]) {
                throw std::out_of_range("Tensor access out of bounds");
            }
            linear_idx += idx[dim] * multiplier;
            multiplier *= shape[dim];
        }
        
        return data[linear_idx];
    }
    
    template<typename... Args>
    requires (sizeof...(Args) == NumDims)
    double operator()(Args... indices) const {
        static_assert(sizeof...(Args) == NumDims, "Number of indices must match tensor dimensions");
        std::array<std::size_t, NumDims> idx = {static_cast<std::size_t>(static_cast<long unsigned int>(indices))...};
        std::size_t linear_idx = 0;
        std::size_t multiplier = 1;
        
        // Process dimensions from last to first (row-major order), avoiding signed/unsigned comparison
        for (std::size_t i = 0; i < NumDims; ++i) {
            std::size_t dim = NumDims - 1 - i;
            if (idx[dim] >= shape[dim]) {
                throw std::out_of_range("Tensor access out of bounds");
            }
            linear_idx += idx[dim] * multiplier;
            multiplier *= shape[dim];
        }
        
        return data[linear_idx];
    }
    
    double& operator[](std::size_t idx) {
        if (idx >= data.size()) {
            throw std::out_of_range("Tensor access out of bounds");
        }
        return data[idx];
    }
    
    double operator[](std::size_t idx) const {
        if (idx >= data.size()) {
            throw std::out_of_range("Tensor access out of bounds");
        }
        return data[idx];
    }
    
    // Fill all elements with a value
    
    static Tensor zeros(const std::array<std::size_t, NumDims>& s) {
        return Tensor(s, 0.0);
    }
    
    static Tensor ones(const std::array<std::size_t, NumDims>& s) {
        return Tensor(s, 1.0);
    }
    
    static Tensor filled(const std::array<std::size_t, NumDims>& s, double val) {
        return Tensor(s, val);
    }
    
    static Tensor random_normal(const std::array<std::size_t, NumDims>& s, 
                                std::mt19937& rng, 
                                double mean = 0.0, 
                                double stddev = 1.0) {
        Tensor t(s);
        std::normal_distribution<double> dist(mean, stddev);
        for (auto& val : t.data) {
            val = dist(rng);
        }
        return t;
    }
    
    static Tensor random_uniform(const std::array<std::size_t, NumDims>& s, 
                                 std::mt19937& rng, 
                                 double min_val = -1.0, 
                                 double max_val = 1.0) {
        Tensor t(s);
        std::uniform_real_distribution<double> dist(min_val, max_val);
        for (auto& val : t.data) {
            val = dist(rng);
        }
        return t;
    }
    
    void fill(double val) {
        std::fill(data.begin(), data.end(), val);
    }
    
    Tensor reshape(const std::array<std::size_t, NumDims>& new_shape) const {
        std::size_t new_size = 1;
        for (std::size_t dim : new_shape) {
            new_size *= dim;
        }
        
        if (new_size != size) {
            throw std::invalid_argument("Reshape dimensions don't match total size");
        }
        
        Tensor result(new_shape);
        result.data = data; // Copy the data
        return result;
    }
    
    Tensor flatten() const requires (NumDims > 1) {
        std::array<std::size_t, 2> new_shape = {size, 1};
        return reshape(new_shape);
    }
    
    bool any_nonfinite() const noexcept {
        for (double x : data) {
            if (!std::isfinite(x)) return true;
        }
        return false;
    }
};

// Specialize 2D tensor as Matrix for compatibility with existing code
using Matrix = Tensor<2>;

// ---------------- Activation Functions ----------------
enum class Activation {
    Linear,      // f(x) = x
    ReLU,        // f(x) = max(0, x)
    LeakyReLU,   // f(x) = x > 0 ? x : 0.01 * x
    ELU,         // f(x) = x > 0 ? x : exp(x) - 1
    Sigmoid,     // f(x) = 1 / (1 + exp(-x))
    Tanh,        // f(x) = tanh(x)
    Softmax,     // f(x_i) = exp(x_i) / sum(exp(x_j))
    Swish,       // f(x) = x * sigmoid(x)
    GELU,        // Gaussian Error Linear Unit
    Softplus     // f(x) = log(1 + exp(x))
};

// ---------------- Loss Functions ----------------
enum class LossFunction {
    MSE,                    // Mean Squared Error
    CrossEntropy,          // Cross Entropy (with softmax)
    BinaryCrossEntropy,    // Binary Cross Entropy
    Hinge,                 // Hinge loss
    Huber,                 // Huber loss
    KLDivergence          // Kullback-Leibler divergence
};

struct LossResult {
    double value;
    Matrix gradient;
};

// ---------------- Optimizer Types ----------------
enum class OptimizerType {
    SGD,        // Stochastic Gradient Descent
    Adam,       // Adaptive Moment Estimation
    RMSprop,    // Root Mean Square Propagation
    Adagrad,    // Adaptive Gradient Algorithm
    AdamW       // Adam with decoupled weight decay
};

// ---------------- Tensor Operations (Forward Declarations) ----------------
Matrix matmul(const Matrix& a, const Matrix& b);
void add_rowwise_inplace(Matrix& a, const Matrix& b);
Matrix transpose(const Matrix& m);
Matrix sum_rows(const Matrix& m);
Matrix add(const Matrix& A, const Matrix& B);
Matrix sub(const Matrix& A, const Matrix& B);
Matrix hadamard(const Matrix& A, const Matrix& B);
Matrix scalar_mul(const Matrix& A, double s);

Matrix add(const Matrix& A, const Matrix& B);
Matrix sub(const Matrix& A, const Matrix& B);
Matrix hadamard(const Matrix& A, const Matrix& B);
Matrix scalar_mul(const Matrix& A, double s);

// Function declarations for operations used in Dense layer
Matrix apply_activation(const Matrix& z, Activation act);
Matrix apply_activation_derivative(const Matrix& a, const Matrix& grad, Activation act);

// ---------------- Layer Base Class ----------------
struct Layer {
    std::string name;
    bool trainable;
    
    explicit Layer(std::string layer_name = "layer", bool is_trainable = true)
        : name(std::move(layer_name)), trainable(is_trainable) {}
    
    virtual ~Layer() = default;
    
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& grad_output) = 0;
    virtual void update_parameters(const struct Optimizer& opt) = 0;
    virtual std::size_t get_parameter_count() const = 0;
};

// ---------------- Dense Layer ----------------
struct Dense : public Layer {
    std::size_t in_features, out_features;
    Matrix weights;
    Matrix bias;
    Activation activation;
    
    // Caches for backward pass
    Matrix input_cache;
    Matrix pre_activation_cache;
    Matrix post_activation_cache;
    
    // Optimizer state
    Matrix weight_velocity;
    Matrix bias_velocity;
    Matrix weight_momentum;
    Matrix bias_momentum;
    Matrix weight_rms;
    Matrix bias_rms;
    
    explicit Dense(std::size_t in, std::size_t out,
          Activation act = Activation::ReLU,
          std::string layer_name = "dense",
          std::mt19937* rng = nullptr)
        : Layer(std::move(layer_name), true),
          in_features(in), out_features(out), weights(std::array<std::size_t, 2>{in, out}),
          bias(std::array<std::size_t, 2>{1, out}), activation(act),
          input_cache(), pre_activation_cache(), post_activation_cache(),
          weight_velocity(std::array<std::size_t, 2>{in, out}), bias_velocity(std::array<std::size_t, 2>{1, out}),
          weight_momentum(std::array<std::size_t, 2>{in, out}), bias_momentum(std::array<std::size_t, 2>{1, out}),
          weight_rms(std::array<std::size_t, 2>{in, out}), bias_rms(std::array<std::size_t, 2>{1, out}) {
        
        initialize_weights(rng);
    }
    
    void initialize_weights(std::mt19937* rng = nullptr) {
        // Xavier/Glorot initialization for better convergence
        double limit = std::sqrt(6.0 / static_cast<double>(in_features + out_features));
        std::mt19937 gen;
        if (rng) {
            gen = *rng; // Use provided random number generator
        } else {
            std::random_device rd;
            gen = std::mt19937(rd());
        }
        std::uniform_real_distribution<double> dist(-limit, limit);
        
        for (auto& w : weights.data) {
            w = dist(gen);
        }
        std::fill(bias.data.begin(), bias.data.end(), 0.0);
    }
    
    Matrix forward(const Matrix& input) override {
        if (input.shape[1] != in_features) {
            throw std::invalid_argument("Dense layer input dimension mismatch");
        }
        
        input_cache = input;
        Matrix z = matmul(input, weights);
        add_rowwise_inplace(z, bias);
        pre_activation_cache = z;
        
        Matrix a = apply_activation(z, activation);
        post_activation_cache = a;
        
        return a;
    }
    
    Matrix backward(const Matrix& grad_output) override {
        Matrix grad_act = apply_activation_derivative(post_activation_cache, grad_output, activation);
        
        // Compute gradients
        Matrix weights_t = transpose(weights);
        Matrix grad_input = matmul(grad_act, weights_t);
        
        Matrix input_t = transpose(input_cache);
        Matrix grad_weights = matmul(input_t, grad_act);
        Matrix grad_bias = sum_rows(grad_act);
        
        // Store gradients for optimizer
        weight_velocity = grad_weights;
        bias_velocity = grad_bias;
        
        return grad_input;
    }
    
    void update_parameters(const struct Optimizer& opt) override;
    Matrix apply_activation(const Matrix& z, Activation act);
    Matrix apply_activation_derivative(const Matrix& a, const Matrix& grad, Activation act);
    
    std::size_t get_parameter_count() const override {
        return weights.size + bias.size;
    }
};

// ---------------- Convolutional Layer ----------------
struct Conv2D : public Layer {
    std::size_t in_channels, out_channels;
    std::size_t kernel_height, kernel_width;
    std::size_t stride_h, stride_w;
    std::size_t padding_h, padding_w;
    Activation activation;
    
    Matrix weights;  // (out_channels, in_channels, kernel_height, kernel_width)
    Matrix bias;     // (out_channels)
    
    // Caches
    std::vector<Matrix> input_patches;  // For backward pass
    Matrix input_cache;  // Store input for backward pass
    
    // Optimizer state
    Matrix weight_velocity;
    Matrix bias_velocity;
    Matrix weight_momentum;
    Matrix bias_momentum;
    Matrix weight_rms;
    Matrix bias_rms;
    
    // Additional members for gradient clipping and regularization
    Matrix grad_weights;   // Store gradients for optimizer update
    Matrix grad_bias;      // Store bias gradients for optimizer update
    
    Conv2D(std::size_t in_ch, std::size_t out_ch,
           std::size_t kh, std::size_t kw,
           std::size_t s_h = 1, std::size_t s_w = 1,
           std::size_t p_h = 0, std::size_t p_w = 0,
           Activation act = Activation::ReLU,
           std::string layer_name = "conv2d",
           std::mt19937* rng = nullptr)
        : Layer(std::move(layer_name), true),
          in_channels(in_ch), out_channels(out_ch),
          kernel_height(kh), kernel_width(kw),
          stride_h(s_h), stride_w(s_w),
          padding_h(p_h), padding_w(p_w), activation(act),
          weights(std::array<std::size_t, 2>{out_ch, in_ch * kh * kw}),  // Flattened for efficient computation
          bias(std::array<std::size_t, 2>{1, out_ch}),
          input_cache(),
          weight_velocity(std::array<std::size_t, 2>{out_ch, in_ch * kh * kw}), bias_velocity(std::array<std::size_t, 2>{1, out_ch}),
          weight_momentum(std::array<std::size_t, 2>{out_ch, in_ch * kh * kw}), bias_momentum(std::array<std::size_t, 2>{1, out_ch}),
          weight_rms(std::array<std::size_t, 2>{out_ch, in_ch * kh * kw}), bias_rms(std::array<std::size_t, 2>{1, out_ch}) {
        
        initialize_weights(rng);
    }
    
    void initialize_weights(std::mt19937* rng = nullptr) {
        // Initialize weights with He initialization for ReLU networks
        double stddev = std::sqrt(2.0 / static_cast<double>(in_channels * kernel_height * kernel_width));
        std::mt19937 gen;
        if (rng) {
            gen = *rng; // Use provided random number generator
        } else {
            std::random_device rd;
            gen = std::mt19937(rd());
        }
        std::normal_distribution<double> dist(0.0, stddev);
        
        for (auto& w : weights.data) {
            w = dist(gen);
        }
        std::fill(bias.data.begin(), bias.data.end(), 0.0);
    }
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update_parameters(const struct Optimizer& opt) override;
    std::size_t get_parameter_count() const override {
        return weights.size + bias.size;
    }
};

// ---------------- Pooling Layer ----------------
struct MaxPool2D : public Layer {
    std::size_t pool_height, pool_width;
    std::size_t stride_h, stride_w;
    
    // Caches
    Matrix input_cache;
    Matrix mask_cache;  // To store where max values came from
    
    MaxPool2D(std::size_t ph, std::size_t pw,
              std::size_t s_h = 1, std::size_t s_w = 1,
              std::string layer_name = "maxpool2d")
        : Layer(std::move(layer_name), false),
          pool_height(ph), pool_width(pw),
          stride_h(s_h), stride_w(s_w) {}
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update_parameters(const struct Optimizer& /*opt*/) override {}
    std::size_t get_parameter_count() const override { return 0; }
};

// ---------------- Dropout Layer ----------------
struct Dropout : public Layer {
    double rate;
    std::vector<bool> mask_cache;
    
    explicit Dropout(double dropout_rate = 0.5, std::string layer_name = "dropout")
        : Layer(std::move(layer_name), false), rate(dropout_rate) {}
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update_parameters(const struct Optimizer& /*opt*/) override {}
    std::size_t get_parameter_count() const override { return 0; }
};

// ---------------- Batch Normalization Layer ----------------
struct BatchNorm : public Layer {
    std::size_t features;
    double momentum;
    double epsilon;
    
    Matrix gamma;    // Scale parameter
    Matrix beta;     // Shift parameter
    Matrix running_mean;
    Matrix running_var;
    
    // For backward pass
    Matrix input_cache;
    Matrix x_norm_cache;
    Matrix x_centered_cache;
    Matrix inv_std_cache;
    
    // Optimizer state
    Matrix weight_velocity;  // gamma gradient
    Matrix bias_velocity;    // beta gradient
    Matrix weight_momentum;  // gamma momentum
    Matrix bias_momentum;    // beta momentum
    Matrix weight_rms;       // gamma rms
    Matrix bias_rms;         // beta rms
    
    BatchNorm(std::size_t feat, 
              double mom = 0.1, 
              double eps = 1e-5,
              std::string layer_name = "batchnorm")
        : Layer(std::move(layer_name), true),
          features(feat), momentum(mom), epsilon(eps),
          gamma(std::array<std::size_t, 2>{1, feat}), beta(std::array<std::size_t, 2>{1, feat}),
          running_mean(std::array<std::size_t, 2>{1, feat}), running_var(std::array<std::size_t, 2>{1, feat}) {
        
        std::fill(gamma.data.begin(), gamma.data.end(), 1.0);
        std::fill(beta.data.begin(), beta.data.end(), 0.0);
        std::fill(running_mean.data.begin(), running_mean.data.end(), 0.0);
        std::fill(running_var.data.begin(), running_var.data.end(), 1.0);
    }
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update_parameters(const struct Optimizer& opt) override;
    std::size_t get_parameter_count() const override {
        return gamma.size + beta.size;
    }
};

// ---------------- Optimizer Base Class ----------------
struct Optimizer {
    OptimizerType type;
    double learning_rate;
    double epsilon;
    
    // Regularization parameters
    double l1_lambda;
    double l2_lambda;
    
    // Gradient clipping parameters
    bool use_gradient_clipping;
    double max_gradient_norm;
    double clip_value;
    
    // Learning rate scheduler
    std::unique_ptr<struct LRScheduler> lr_scheduler;
    
    explicit Optimizer(OptimizerType opt_type, double lr = 0.001, double eps = 1e-8,
                      double l1_reg = 0.0, double l2_reg = 0.0)
        : type(opt_type), learning_rate(lr), epsilon(eps),
          l1_lambda(l1_reg), l2_lambda(l2_reg),
          use_gradient_clipping(false), max_gradient_norm(0.0), clip_value(0.0) {}
    
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    
    // Gradient clipping methods
    void enable_gradient_clipping(double max_norm);
    void enable_gradient_clipping_by_value(double clip_val);
    void disable_gradient_clipping();
    
    // Regularization methods
    void set_regularization(double l1_reg, double l2_reg);
    
    // Learning rate scheduler methods
    void set_lr_scheduler(std::unique_ptr<LRScheduler> scheduler);
    void update_learning_rate();
};

// ---------------- Learning Rate Scheduler Base Class ----------------
struct LRScheduler {
    Optimizer& optimizer;
    double initial_lr;
    std::size_t last_epoch;
    
    explicit LRScheduler(Optimizer& opt)
        : optimizer(opt), initial_lr(opt.learning_rate), last_epoch(0) {}
    
    virtual ~LRScheduler() = default;
    virtual void step() = 0;
    virtual void step(double metric) = 0;  // For schedulers that depend on a metric
};

// ---------------- StepLR Scheduler ----------------
struct StepLR : public LRScheduler {
    std::size_t step_size;
    double gamma;
    
    explicit StepLR(Optimizer& opt, std::size_t step_sz, double gamma_val = 0.1)
        : LRScheduler(opt), step_size(step_sz), gamma(gamma_val) {}
    
    void step() override;
    void step(double /*metric*/) override { step(); }  // Not used for StepLR
};

// ---------------- ExponentialLR Scheduler ----------------
struct ExponentialLR : public LRScheduler {
    double gamma;
    
    explicit ExponentialLR(Optimizer& opt, double gamma_val)
        : LRScheduler(opt), gamma(gamma_val) {}
    
    void step() override;
    void step(double /*metric*/) override { step(); }  // Not used for ExponentialLR
};

// ---------------- PolynomialLR Scheduler ----------------
struct PolynomialLR : public LRScheduler {
    std::size_t max_epochs;
    double end_lr;
    double power;
    
    explicit PolynomialLR(Optimizer& opt, std::size_t max_ep, double end_lr_val = 0.0001, double power_val = 1.0)
        : LRScheduler(opt), max_epochs(max_ep), end_lr(end_lr_val), power(power_val) {}
    
    void step() override;
    void step(double /*metric*/) override { step(); }  // Not used for PolynomialLR
};

// ---------------- CosineAnnealingLR Scheduler ----------------
struct CosineAnnealingLR : public LRScheduler {
    std::size_t t_max;
    double eta_min;
    
    explicit CosineAnnealingLR(Optimizer& opt, std::size_t t_max_val, double eta_min_val = 0.0)
        : LRScheduler(opt), t_max(t_max_val), eta_min(eta_min_val) {}
    
    void step() override;
    void step(double /*metric*/) override { step(); }  // Not used for CosineAnnealingLR
};

// ---------------- ReduceLROnPlateau Scheduler ----------------
struct ReduceLROnPlateau : public LRScheduler {
    double mode_min;
    double factor;
    std::size_t patience;
    double threshold;
    std::size_t threshold_mode;
    std::size_t cooldown;
    std::size_t num_bad_epochs;
    bool in_cooldown;
    double best;
    
    explicit ReduceLROnPlateau(Optimizer& opt, double mode_min_val = 1.0, double factor_val = 0.1,
                              std::size_t patience_val = 10, double threshold_val = 1e-4,
                              std::size_t threshold_mode_val = 0, std::size_t cooldown_val = 0)
        : LRScheduler(opt), mode_min(mode_min_val), factor(factor_val),
          patience(patience_val), threshold(threshold_val),
          threshold_mode(threshold_mode_val), cooldown(cooldown_val),
          num_bad_epochs(0), in_cooldown(false),
          best(mode_min_val > 0 ? std::numeric_limits<double>::max() : std::numeric_limits<double>::lowest()) {}
    
    void step() override;  // Uses internal metric
    void step(double metric) override;  // Uses provided metric
};

// ---------------- SGD Optimizer ----------------
struct SGD : public Optimizer {
    double momentum;
    double weight_decay;
    double dampening;
    bool nesterov;
    
    explicit SGD(double lr = 0.01, 
                 double mom = 0.0, 
                 double wd = 0.0, 
                 double damp = 0.0, 
                 bool nest = false)
        : Optimizer(OptimizerType::SGD, lr), 
          momentum(mom), weight_decay(wd), dampening(damp), nesterov(nest) {}
    
    void step() override;
    void zero_grad() override;
};

// ---------------- Adam Optimizer ----------------
struct Adam : public Optimizer {
    double beta1;
    double beta2;
    double weight_decay;
    std::size_t step_count;
    
    explicit Adam(double lr = 0.001, 
                  double b1 = 0.9, 
                  double b2 = 0.999, 
                  double wd = 0.0)
        : Optimizer(OptimizerType::Adam, lr), 
          beta1(b1), beta2(b2), weight_decay(wd), step_count(0) {}
    
    void step() override;
    void zero_grad() override;
};

// ---------------- Neural Network Model ----------------
class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Optimizer> optimizer;
    Config config;
    
public:
    explicit Model(const Config& cfg = Config{});
    
    void add(std::unique_ptr<Layer> layer);
    
    Matrix forward(const Matrix& input);
    double compute_loss(const Matrix& predictions, const Matrix& targets, LossFunction loss_fn);
    void backward(const Matrix& loss_gradient);
    void train_step(const Matrix& inputs, const Matrix& targets, LossFunction loss_fn);
    
    void compile(std::unique_ptr<Optimizer> opt);
    void fit(const Matrix& X, const Matrix& y, 
             int epochs, LossFunction loss_fn,
             std::mt19937& rng,
             double validation_split = 0.0,
             bool verbose = true);
    
    Matrix predict(const Matrix& input);
    double evaluate(const Matrix& X, const Matrix& y, LossFunction loss_fn);
    
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
    
    std::size_t get_parameter_count() const;
    
    void print_summary() const;
};


// ---------------- Loss Functions Implementation ----------------
LossResult compute_loss(const Matrix& y_true, const Matrix& y_pred, LossFunction loss_fn);

// ---------------- Utility Functions ----------------
Matrix one_hot(const std::vector<int>& labels, int num_classes);
double accuracy(const Matrix& predictions, const std::vector<int>& labels);
Matrix normalize(const Matrix& input, double mean = 0.0, double stddev = 1.0);
std::pair<Matrix, Matrix> train_test_split(const Matrix& X, const Matrix& y, double test_size, std::mt19937& rng);

} // namespace dnn