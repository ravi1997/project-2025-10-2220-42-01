# API Reference

## Overview
This document provides a comprehensive reference for all public APIs in the DNN library. Each API is documented with its purpose, parameters, return values, and usage examples.

## Namespace: dnn

### Config Struct
Configuration settings for the neural network library.

```cpp
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
```

### Tensor Template
Multi-dimensional tensor implementation.

```cpp
template<std::size_t NumDims = 2>
struct Tensor {
    std::array<std::size_t, NumDims> shape;
    std::vector<double> data;
    std::size_t size;
    
    // Constructors
    Tensor();  // Default constructor
    explicit Tensor(const std::array<std::size_t, NumDims>& s);  // Shape constructor
    Tensor(const std::array<std::size_t, NumDims>& s, double init_val);  // Shape + value constructor
    Tensor(std::size_t rows, std::size_t cols);  // 2D constructor
    Tensor(std::size_t rows, std::size_t cols, double init_val);  // 2D + value constructor
    
    // Element access
    template<typename... Args>
    double& operator()(Args... indices);  // Access by indices
    template<typename... Args>
    double operator()(Args... indices) const;  // Const access by indices
    double& operator[](std::size_t idx);  // Linear access
    double operator[](std::size_t idx) const;  // Const linear access
    
    // Utility methods
    static Tensor zeros(const std::array<std::size_t, NumDims>& s);
    static Tensor ones(const std::array<std::size_t, NumDims>& s);
    static Tensor filled(const std::array<std::size_t, NumDims>& s, double val);
    static Tensor random_normal(const std::array<std::size_t, NumDims>& s, 
                                std::mt19937& rng, 
                                double mean = 0.0, 
                                double stddev = 1.0);
    static Tensor random_uniform(const std::array<std::size_t, NumDims>& s, 
                                 std::mt19937& rng, 
                                 double min_val = -1.0, 
                                 double max_val = 1.0);
    void fill(double val);
    Tensor reshape(const std::array<std::size_t, NumDims>& new_shape) const;
    Tensor flatten() const requires (NumDims > 1);
    bool any_nonfinite() const noexcept;
};
```

### Type Aliases
```cpp
using Matrix = Tensor<2>;  // 2D tensor alias
```

### Enumerations

#### Activation Functions
```cpp
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
```

#### Loss Functions
```cpp
enum class LossFunction {
    MSE,                    // Mean Squared Error
    CrossEntropy,          // Cross Entropy (with softmax)
    BinaryCrossEntropy,    // Binary Cross Entropy
    Hinge,                 // Hinge loss
    Huber,                 // Huber loss
    KLDivergence          // Kullback-Leibler divergence
};
```

#### Optimizer Types
```cpp
enum class OptimizerType {
    SGD,        // Stochastic Gradient Descent
    Adam,       // Adaptive Moment Estimation
    RMSprop,    // Root Mean Square Propagation
    Adagrad,    // Adaptive Gradient Algorithm
    AdamW       // Adam with decoupled weight decay
};
```

### LossResult Struct
```cpp
struct LossResult {
    double value;      // Loss value
    Matrix gradient;   // Gradient of loss w.r.t. predictions
};
```

### Layer Base Class
```cpp
struct Layer {
    std::string name;
    bool trainable;
    
    explicit Layer(std::string layer_name = "layer", bool is_trainable = true);
    virtual ~Layer() = default;
    
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& grad_output) = 0;
    virtual void update_parameters(const Optimizer& opt) = 0;
    virtual std::size_t get_parameter_count() const = 0;
};
```

### Dense Layer
```cpp
struct Dense : public Layer {
    std::size_t in_features, out_features;
    Matrix weights;
    Matrix bias;
    Activation activation;
    
    Dense(std::size_t in, std::size_t out,
          Activation act = Activation::ReLU,
          std::string layer_name = "dense",
          std::mt19937* rng = nullptr);
    
    void initialize_weights(std::mt19937* rng = nullptr);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update_parameters(const Optimizer& opt) override;
    std::size_t get_parameter_count() const override;
};
```

### Conv2D Layer
```cpp
struct Conv2D : public Layer {
    std::size_t in_channels, out_channels;
    std::size_t kernel_height, kernel_width;
    std::size_t stride_h, stride_w;
    std::size_t padding_h, padding_w;
    Activation activation;
    
    Conv2D(std::size_t in_ch, std::size_t out_ch,
           std::size_t kh, std::size_t kw,
           std::size_t s_h = 1, std::size_t s_w = 1,
           std::size_t p_h = 0, std::size_t p_w = 0,
           Activation act = Activation::ReLU,
           std::string layer_name = "conv2d",
           std::mt19937* rng = nullptr);
    
    void initialize_weights(std::mt19937* rng = nullptr);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update_parameters(const Optimizer& opt) override;
    std::size_t get_parameter_count() const override;
};
```

### MaxPool2D Layer
```cpp
struct MaxPool2D : public Layer {
    std::size_t pool_height, pool_width;
    std::size_t stride_h, stride_w;
    
    MaxPool2D(std::size_t ph, std::size_t pw,
              std::size_t s_h = 1, std::size_t s_w = 1,
              std::string layer_name = "maxpool2d");
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update_parameters(const Optimizer& opt) override;
    std::size_t get_parameter_count() const override;
};
```

### Dropout Layer
```cpp
struct Dropout : public Layer {
    double rate;
    
    explicit Dropout(double dropout_rate = 0.5, std::string layer_name = "dropout");
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update_parameters(const Optimizer& opt) override;
    std::size_t get_parameter_count() const override;
};
```

### BatchNorm Layer
```cpp
struct BatchNorm : public Layer {
    std::size_t features;
    double momentum;
    double epsilon;
    
    BatchNorm(std::size_t feat, 
              double mom = 0.1, 
              double eps = 1e-5,
              std::string layer_name = "batchnorm");
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update_parameters(const Optimizer& opt) override;
    std::size_t get_parameter_count() const override;
};
```

### Optimizer Base Class
```cpp
struct Optimizer {
    OptimizerType type;
    double learning_rate;
    double epsilon;
    
    explicit Optimizer(OptimizerType opt_type, double lr = 0.001, double eps = 1e-8);
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
};
```

### SGD Optimizer
```cpp
struct SGD : public Optimizer {
    double momentum;
    double weight_decay;
    double dampening;
    bool nesterov;
    
    explicit SGD(double lr = 0.01, 
                 double mom = 0.0, 
                 double wd = 0.0, 
                 double damp = 0.0, 
                 bool nest = false);
    
    void step() override;
    void zero_grad() override;
};
```

### Adam Optimizer
```cpp
struct Adam : public Optimizer {
    double beta1;
    double beta2;
    double weight_decay;
    std::size_t step_count;
    
    explicit Adam(double lr = 0.001, 
                  double b1 = 0.9, 
                  double b2 = 0.99, 
                  double wd = 0.0);
    
    void step() override;
    void zero_grad() override;
};
```

### Model Class
```cpp
class Model {
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
```

## Utility Functions

### Tensor Operations
```cpp
Matrix matmul(const Matrix& a, const Matrix& b);           // Matrix multiplication
void add_rowwise_inplace(Matrix& a, const Matrix& b);      // Add row vector to matrix
Matrix transpose(const Matrix& m);                         // Matrix transpose
Matrix sum_rows(const Matrix& m);                          // Sum along rows
Matrix add(const Matrix& A, const Matrix& B);              // Matrix addition
Matrix sub(const Matrix& A, const Matrix& B);              // Matrix subtraction
Matrix hadamard(const Matrix& A, const Matrix& B);         // Element-wise multiplication
Matrix scalar_mul(const Matrix& A, double s);              // Scalar multiplication
```

### Activation Functions
```cpp
Matrix apply_activation(const Matrix& z, Activation act);  // Apply activation function
Matrix apply_activation_derivative(const Matrix& a, const Matrix& grad, Activation act);  // Activation derivative
```

### Loss Functions
```cpp
LossResult compute_loss(const Matrix& y_true, const Matrix& y_pred, LossFunction loss_fn);  // Compute loss and gradient
```

### Utility Functions
```cpp
Matrix one_hot(const std::vector<int>& labels, int num_classes);  // Convert labels to one-hot encoding
double accuracy(const Matrix& predictions, const std::vector<int>& labels);  // Calculate accuracy
Matrix normalize(const Matrix& input, double mean = 0.0, double stddev = 1.0);  // Normalize input
std::pair<Matrix, Matrix> train_test_split(const Matrix& X, const Matrix& y, double test_size, std::mt19937& rng);  // Split data