#ifndef DNN_IMPROVED_HPP
#define DNN_IMPROVED_HPP

// dnn_improved.hpp — stdlib-only deep neural network (C++23)
// Comprehensive DNN library with all standard features and industrial-grade improvements

#include "tensor_improved.hpp"
#include "tensor_ops.hpp"
#include "layers_improved.hpp"
#include "optimizers_improved.hpp"
#include "model_serializer.hpp"

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
    float epsilon = 1e-8f;           // Small value for numerical stability
    float learning_rate = 0.001f;    // Default learning rate
    std::size_t batch_size = 32;     // Default batch size
    std::size_t max_epochs = 1000;   // Maximum training epochs
    float validation_split = 0.2f;   // Validation split ratio
    float dropout_rate = 0.0f;       // Dropout rate (0.0 to 1.0)
    bool use_batch_norm = false;     // Whether to use batch normalization
    bool use_layer_norm = false;     // Whether to use layer normalization
};

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
    float value;
    TensorF gradient;
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
TensorF transpose(const TensorF& A);
TensorF matmul(const TensorF& A, const TensorF& B);
void add_rowwise_inplace(TensorF& A, const TensorF& rowvec);
TensorF add(const TensorF& A, const TensorF& B);
TensorF sub(const TensorF& A, const TensorF& B);
TensorF hadamard(const TensorF& A, const TensorF& B);
TensorF scalar_mul(const TensorF& A, float s);
TensorF sum_rows(const TensorF& A);

// Function declarations for operations used in Dense layer
TensorF apply_activation(const TensorF& z, Activation act);
TensorF apply_activation_derivative(const TensorF& a, const TensorF& grad, Activation act);

// ---------------- Layer Base Class ----------------
struct Layer {
    std::string name;
    bool trainable;
    
    explicit Layer(std::string layer_name = "layer", bool is_trainable = true)
        : name(std::move(layer_name)), trainable(is_trainable) {}
    
    virtual ~Layer() = default;
    
    virtual TensorF forward(const TensorF& input) = 0;
    virtual TensorF backward(const TensorF& grad_output) = 0;
    virtual void update_parameters(const struct Optimizer& opt) = 0;
    virtual std::size_t get_parameter_count() const = 0;
};

// ---------------- Dense Layer ----------------
struct Dense : public Layer {
    std::size_t in_features, out_features;
    TensorF weights;
    TensorF bias;
    Activation activation;
    
    // Caches for backward pass
    TensorF input_cache;
    TensorF pre_activation_cache;
    TensorF post_activation_cache;
    
    // Optimizer state
    TensorF weight_velocity;
    TensorF bias_velocity;
    TensorF weight_momentum;
    TensorF bias_momentum;
    TensorF weight_rms;
    TensorF bias_rms;
    
    explicit Dense(std::size_t in, std::size_t out,
          Activation act = Activation::ReLU,
          std::string layer_name = "dense",
          std::mt19937* rng = nullptr)
        : Layer(std::move(layer_name), true),
          in_features(in), out_features(out), weights(std::vector<size_t>{in, out}),
          bias(std::vector<size_t>{1, out}), activation(act),
          input_cache(), pre_activation_cache(), post_activation_cache(),
          weight_velocity(std::vector<size_t>{in, out}), bias_velocity(std::vector<size_t>{1, out}),
          weight_momentum(std::vector<size_t>{in, out}), bias_momentum(std::vector<size_t>{1, out}),
          weight_rms(std::vector<size_t>{in, out}), bias_rms(std::vector<size_t>{1, out}) {
        initialize_weights(rng);
    }
    
    void initialize_weights(std::mt19937* rng = nullptr) {
        // Xavier/Glorot initialization for better convergence
        float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
        std::mt19937 gen;
        if (rng) {
            gen = *rng; // Use provided random number generator
        } else {
            std::random_device rd;
            gen = std::mt19937(rd());
        }
        std::uniform_real_distribution<float> dist(-limit, limit);
        
        for (auto& w : weights.data()) {
            w = dist(gen);
        }
        std::fill(bias.data(), bias.data() + bias.size(), 0.0f);
    }
    
    TensorF forward(const TensorF& input) override {
        if (input.shape()[1] != in_features) {
            throw DimensionMismatchException("Dense layer input dimension mismatch");
        }
        
        input_cache = input;
        TensorF z = matmul(input, weights);
        add_rowwise_inplace(z, bias);
        pre_activation_cache = z;
        
        TensorF a = apply_activation(z, activation);
        post_activation_cache = a;
        
        return a;
    }
    
    TensorF backward(const TensorF& grad_output) override {
        TensorF grad_act = apply_activation_derivative(post_activation_cache, grad_output, activation);
        
        // Compute gradients
        TensorF weights_t = transpose(weights);
        TensorF grad_input = matmul(grad_act, weights_t);
        
        TensorF input_t = transpose(input_cache);
        TensorF grad_weights = matmul(input_t, grad_act);
        TensorF grad_bias = sum_rows(grad_act);
        
        // Store gradients for optimizer
        weight_velocity = grad_weights;
        bias_velocity = grad_bias;
        
        return grad_input;
    }
    
    void update_parameters(const struct Optimizer& opt) override;
    TensorF apply_activation(const TensorF& z, Activation act);
    TensorF apply_activation_derivative(const TensorF& a, const TensorF& grad, Activation act);
    
    std::size_t get_parameter_count() const override {
        return weights.size() + bias.size();
    }
};

// ---------------- Convolutional Layer ----------------
struct Conv2D : public Layer {
    std::size_t in_channels, out_channels;
    std::size_t kernel_height, kernel_width;
    std::size_t stride_h, stride_w;
    std::size_t padding_h, padding_w;
    Activation activation;
    
    TensorF weights;  // (out_channels, in_channels, kernel_height, kernel_width)
    TensorF bias;     // (out_channels)
    
    // Caches
    std::vector<TensorF> input_patches;  // For backward pass
    TensorF input_cache;  // Store input for backward pass
    TensorF pre_activation_cache;
    TensorF post_activation_cache;
    std::size_t last_input_height{0};
    std::size_t last_input_width{0};
    
    // Optimizer state
    TensorF weight_velocity;
    TensorF bias_velocity;
    TensorF weight_momentum;
    TensorF bias_momentum;
    TensorF weight_rms;
    TensorF bias_rms;
    
    // Additional members for gradient clipping and regularization
    TensorF grad_weights;   // Store gradients for optimizer update
    TensorF grad_bias;      // Store bias gradients for optimizer update
    
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
          weights(std::vector<size_t>{out_ch, in_ch * kh * kw}),  // Flattened for efficient computation
          bias(std::vector<size_t>{1, out_ch}),
          input_cache(),
          weight_velocity(std::vector<size_t>{out_ch, in_ch * kh * kw}), bias_velocity(std::vector<size_t>{1, out_ch}),
          weight_momentum(std::vector<size_t>{out_ch, in_ch * kh * kw}), bias_momentum(std::vector<size_t>{1, out_ch}),
          weight_rms(std::vector<size_t>{out_ch, in_ch * kh * kw}), bias_rms(std::vector<size_t>{1, out_ch}) {
        
        initialize_weights(rng);
    }
    
    void initialize_weights(std::mt19937* rng = nullptr) {
        // Initialize weights with He initialization for ReLU networks
        float stddev = std::sqrt(2.0f / static_cast<float>(in_channels * kernel_height * kernel_width));
        std::mt19937 gen;
        if (rng) {
            gen = *rng; // Use provided random number generator
        } else {
            std::random_device rd;
            gen = std::mt19937(rd());
        }
        std::normal_distribution<float> dist(0.0f, stddev);
        
        for (auto& w : weights.data()) {
            w = dist(gen);
        }
        std::fill(bias.data(), bias.data() + bias.size(), 0.0f);
    }
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(const struct Optimizer& opt) override;
    std::size_t get_parameter_count() const override {
        return weights.size() + bias.size();
    }
};

// ---------------- Pooling Layer ----------------
struct MaxPool2D : public Layer {
    std::size_t pool_height, pool_width;
    std::size_t stride_h, stride_w;
    
    // Caches
    TensorF input_cache;
    TensorF mask_cache;  // To store where max values came from
    
    MaxPool2D(std::size_t ph, std::size_t pw,
              std::size_t s_h = 1, std::size_t s_w = 1,
              std::string layer_name = "maxpool2d")
        : Layer(std::move(layer_name), false),
          pool_height(ph), pool_width(pw),
          stride_h(s_h), stride_w(s_w) {}
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(const struct Optimizer& /*opt*/) override {}
    std::size_t get_parameter_count() const override { return 0; }
};

// ---------------- Dropout Layer ----------------
struct Dropout : public Layer {
    float rate;
    std::vector<bool> mask_cache;
    
    explicit Dropout(float dropout_rate = 0.5f, std::string layer_name = "dropout")
        : Layer(std::move(layer_name), false), rate(dropout_rate) {}
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(const struct Optimizer& /*opt*/) override {}
    std::size_t get_parameter_count() const override { return 0; }
    void initialize_parameters(std::mt19937& rng) override;
};

// ---------------- Batch Normalization Layer ----------------
struct BatchNorm : public Layer {
    std::size_t features;
    float momentum;
    float epsilon;
    
    TensorF gamma;    // Scale parameter
    TensorF beta;     // Shift parameter
    TensorF running_mean;
    TensorF running_var;
    
    // For backward pass
    TensorF input_cache;
    TensorF x_norm_cache;    // Normalized input
    TensorF x_centered_cache; // Centered input
    TensorF inv_std_cache;   // Inverse standard deviation
    
    // Optimizer state
    TensorF weight_velocity;  // gamma gradient
    TensorF bias_velocity;    // beta gradient
    TensorF weight_momentum;  // gamma momentum
    TensorF bias_momentum;    // beta momentum
    TensorF weight_rms;       // gamma rms
    TensorF bias_rms;         // beta rms
    
    BatchNorm(std::size_t feat, 
              float mom = 0.1f, 
              float eps = 1e-5f,
              std::string layer_name = "batchnorm")
        : Layer(std::move(layer_name), true),
          features(feat), momentum(mom), epsilon(eps),
          gamma(std::vector<size_t>{1, feat}), beta(std::vector<size_t>{1, feat}),
          running_mean(std::vector<size_t>{1, feat}), running_var(std::vector<size_t>{1, feat}) {
        
        std::fill(gamma.data(), gamma.data() + gamma.size(), 1.0f);
        std::fill(beta.data(), beta.data() + beta.size(), 0.0f);
        std::fill(running_mean.data(), running_mean.data() + running_mean.size(), 0.0f);
        std::fill(running_var.data(), running_var.data() + running_var.size(), 1.0f);
    }
    
    TensorF forward(const TensorF& input) override;
    TensorF backward(const TensorF& grad_output) override;
    void update_parameters(const struct Optimizer& opt) override;
    std::size_t get_parameter_count() const override {
        return gamma.size() + beta.size();
    }
};

// ---------------- Optimizer Base Class ----------------
struct Optimizer {
    OptimizerType type;
    float learning_rate;
    float epsilon;
    
    // Regularization parameters
    float l1_lambda;
    float l2_lambda;
    
    // Gradient clipping parameters
    bool use_gradient_clipping;
    float max_gradient_norm;
    float clip_value;
    
    // Learning rate scheduler
    std::unique_ptr<struct LRScheduler> lr_scheduler;
    
    explicit Optimizer(OptimizerType opt_type, float lr = 0.001f, float eps = 1e-8f,
                      float l1_reg = 0.0f, float l2_reg = 0.0f)
        : type(opt_type), learning_rate(lr), epsilon(eps),
          l1_lambda(l1_reg), l2_lambda(l2_reg),
          use_gradient_clipping(false), max_gradient_norm(0.0f), clip_value(0.0f) {}
    
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    
    // Gradient clipping methods
    void enable_gradient_clipping(float max_norm);
    void enable_gradient_clipping_by_value(float clip_val);
    void disable_gradient_clipping();
    
    // Regularization methods
    void set_regularization(float l1_reg, float l2_reg);
    
    // Learning rate scheduler methods
    void set_lr_scheduler(std::unique_ptr<LRScheduler> scheduler);
    void update_learning_rate();
};

// ---------------- Learning Rate Scheduler Base Class ----------------
struct LRScheduler {
    Optimizer& optimizer;
    float initial_lr;
    std::size_t last_epoch;
    
    explicit LRScheduler(Optimizer& opt)
        : optimizer(opt), initial_lr(opt.learning_rate), last_epoch(0) {}
    
    virtual ~LRScheduler() = default;
    virtual void step() = 0;
    virtual void step(float metric) = 0;  // For schedulers that depend on a metric
};

// ---------------- StepLR Scheduler ----------------
struct StepLR : public LRScheduler {
    std::size_t step_size;
    float gamma;
    
    explicit StepLR(Optimizer& opt, std::size_t step_sz, float gamma_val = 0.1f)
        : LRScheduler(opt), step_size(step_sz), gamma(gamma_val) {}
    
    void step() override;
    void step(float /*metric*/) override { step(); }  // Not used for StepLR
};

// ---------------- ExponentialLR Scheduler ----------------
struct ExponentialLR : public LRScheduler {
    float gamma;
    
    explicit ExponentialLR(Optimizer& opt, float gamma_val)
        : LRScheduler(opt), gamma(gamma_val) {}
    
    void step() override;
    void step(float /*metric*/) override { step(); }  // Not used for ExponentialLR
};

// ---------------- PolynomialLR Scheduler ----------------
struct PolynomialLR : public LRScheduler {
    std::size_t max_epochs;
    float end_lr;
    float power;
    
    explicit PolynomialLR(Optimizer& opt, std::size_t max_ep, float end_lr_val = 0.0001f, float power_val = 1.0f)
        : LRScheduler(opt), max_epochs(max_ep), end_lr(end_lr_val), power(power_val) {}
    
    void step() override;
    void step(float /*metric*/) override { step(); }  // Not used for PolynomialLR
};

// ---------------- CosineAnnealingLR Scheduler ----------------
struct CosineAnnealingLR : public LRScheduler {
    std::size_t t_max;
    float eta_min;
    
    explicit CosineAnnealingLR(Optimizer& opt, std::size_t t_max_val, float eta_min_val = 0.0f)
        : LRScheduler(opt), t_max(t_max_val), eta_min(eta_min_val) {}
    
    void step() override;
    void step(float /*metric*/) override { step(); }  // Not used for CosineAnnealingLR
};

// ---------------- ReduceLROnPlateau Scheduler ----------------
struct ReduceLROnPlateau : public LRScheduler {
    float mode_min;
    float factor;
    std::size_t patience;
    float threshold;
    std::size_t threshold_mode;
    std::size_t cooldown;
    std::size_t num_bad_epochs;
    bool in_cooldown;
    float best;
    
    explicit ReduceLROnPlateau(Optimizer& opt, float mode_min_val = 1.0f, float factor_val = 0.1f,
                              std::size_t patience_val = 10, float threshold_val = 1e-4f,
                              std::size_t threshold_mode_val = 0, std::size_t cooldown_val = 0)
        : LRScheduler(opt), mode_min(mode_min_val), factor(factor_val),
          patience(patience_val), threshold(threshold_val),
          threshold_mode(threshold_mode_val), cooldown(cooldown_val),
          num_bad_epochs(0), in_cooldown(false),
          best(mode_min_val > 0 ? std::numeric_limits<float>::max() : std::numeric_limits<float>::lowest()) {}
    
    void step() override;  // Uses internal metric
    void step(float metric) override;  // Uses provided metric
};

// ---------------- SGD Optimizer ----------------
struct SGD : public Optimizer {
    float momentum;
    float weight_decay;
    float dampening;
    bool nesterov;
    
    explicit SGD(float lr = 0.01f, 
                 float mom = 0.0f, 
                 float wd = 0.0f, 
                 float damp = 0.0f, 
                 bool nest = false)
        : Optimizer(OptimizerType::SGD, lr), 
          momentum(mom), weight_decay(wd), dampening(damp), nesterov(nest) {}
    
    void step() override;
    void zero_grad() override;
};

// ---------------- Adam Optimizer ----------------
struct Adam : public Optimizer {
    float beta1;
    float beta2;
    float weight_decay;
    std::size_t step_count;
    
    explicit Adam(float lr = 0.001f, 
                  float b1 = 0.9f, 
                  float b2 = 0.999f, 
                  float wd = 0.0f)
        : Optimizer(OptimizerType::Adam, lr), 
          beta1(b1), beta2(b2), weight_decay(wd), step_count(0) {}
    
    void step() override;
    void zero_grad() override;
};

// ---------------- RMSprop Optimizer ----------------
struct RMSprop : public Optimizer {
    float alpha;
    float epsilon;
    float weight_decay;
    
    explicit RMSprop(float lr = 0.001f,
                     float a = 0.99f,
                     float eps = 1e-8f,
                     float wd = 0.0f)
        : Optimizer(OptimizerType::RMSprop, lr, eps),
          alpha(a), epsilon(eps), weight_decay(wd) {}
    
    void step() override {}
    void zero_grad() override {}
};

// ---------------- AdamW Optimizer ----------------
struct AdamW : public Optimizer {
    float beta1;
    float beta2;
    float weight_decay;
    std::size_t step_count;
    
    explicit AdamW(float lr = 0.001f,
                   float b1 = 0.9f,
                   float b2 = 0.999f,
                   float wd = 0.01f)
        : Optimizer(OptimizerType::AdamW, lr),
          beta1(b1), beta2(b2), weight_decay(wd), step_count(0) {}
    
    void step() override { ++step_count; }
    void zero_grad() override {}
};

// ---------------- Neural Network Model ----------------
class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Optimizer> optimizer;
    Config config;
    
public:
    friend class ModelSerializer;
    
    explicit Model(const Config& cfg = Config{});
    
    void add(std::unique_ptr<Layer> layer);
    
    TensorF forward(const TensorF& input);
    float compute_loss(const TensorF& predictions, const TensorF& targets, LossFunction loss_fn);
    void backward(const TensorF& loss_gradient);
    void train_step(const TensorF& inputs, const TensorF& targets, LossFunction loss_fn);
    
    void compile(std::unique_ptr<Optimizer> opt);
    void fit(const TensorF& X, const TensorF& y, 
             int epochs, LossFunction loss_fn,
             std::mt19937& rng,
             float validation_split = 0.0f,
             bool verbose = true);
    
    TensorF predict(const TensorF& input);
    float evaluate(const TensorF& X, const TensorF& y, LossFunction loss_fn);
    
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
    
    std::size_t get_parameter_count() const;
    
    void print_summary() const;
    
    std::size_t layer_count() const;
    std::size_t parameter_count(std::size_t index) const;
    const Config& get_config() const;
    Config& get_config();
    const Optimizer* get_optimizer() const;
    void set_optimizer(std::unique_ptr<Optimizer> opt);
};

// ---------------- Loss Functions Implementation ----------------
LossResult compute_loss(const TensorF& y_true, const TensorF& y_pred, LossFunction loss_fn);

// ---------------- Utility Functions ----------------
TensorF one_hot(const std::vector<int>& labels, int num_classes);
float accuracy(const TensorF& predictions, const std::vector<int>& labels);
TensorF normalize(const TensorF& input, float mean = 0.0f, float stddev = 1.0f);
std::pair<TensorF, TensorF> train_test_split(const TensorF& X, const TensorF& y, float test_size, std::mt19937& rng);

} // namespace dnn

#endif // DNN_IMPROVED_HPP