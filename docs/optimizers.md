# Optimizers Documentation

## Overview
Optimizers in the DNN library are responsible for updating model parameters during training based on computed gradients. The library implements several popular optimization algorithms with consistent interfaces.

## Optimizer Hierarchy

### Base Optimizer Class
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

The base `Optimizer` class defines the common interface:
- `step()`: Performs parameter updates (called after gradients are computed)
- `zero_grad()`: Resets gradients (called before each forward pass)

## Implemented Optimizers

### SGD (Stochastic Gradient Descent)
The `SGD` optimizer implements stochastic gradient descent with optional momentum.

#### Constructor
```cpp
SGD(double lr = 0.01, 
    double mom = 0.0, 
    double wd = 0.0, 
    double damp = 0.0, 
    bool nest = false);
```

#### Parameters
- `lr`: Learning rate (default: 0.01)
- `mom`: Momentum factor (default: 0.0, no momentum)
- `wd`: Weight decay (L2 regularization, default: 0.0)
- `damp`: Damping for momentum (default: 0.0)
- `nest`: Enable Nesterov momentum (default: false)

#### Features
- Simple gradient descent with momentum support
- Weight decay for regularization
- Nesterov momentum for faster convergence
- Configurable parameters for fine-tuning

#### Usage Example
```cpp
// Basic SGD
auto sgd_basic = std::make_unique<dnn::SGD>(0.01);

// SGD with momentum
auto sgd_momentum = std::make_unique<dnn::SGD>(0.01, 0.9);

// SGD with momentum and weight decay
auto sgd_full = std::make_unique<dnn::SGD>(0.01, 0.9, 1e-4);
```

### Adam (Adaptive Moment Estimation)
The `Adam` optimizer implements the Adam optimization algorithm with adaptive learning rates.

#### Constructor
```cpp
Adam(double lr = 0.001, 
     double b1 = 0.9, 
     double b2 = 0.999, 
     double wd = 0.0);
```

#### Parameters
- `lr`: Learning rate (default: 0.001)
- `b1`: Exponential decay rate for first moment (default: 0.9)
- `b2`: Exponential decay rate for second moment (default: 0.999)
- `wd`: Weight decay (default: 0.0)

#### Features
- Adaptive learning rates for each parameter
- Bias correction for early iterations
- First and second moment estimation
- Default hyperparameters that work well in practice

#### Usage Example
```cpp
// Basic Adam
auto adam_basic = std::make_unique<dnn::Adam>(0.001);

// Adam with custom parameters
auto adam_custom = std::make_unique<dnn::Adam>(0.001, 0.95, 0.999, 1e-4);
```

## Optimizer Integration

### With Model
Optimizers are integrated with the model through the compile method:

```cpp
dnn::Model model;
// ... add layers ...

// Compile with optimizer
auto optimizer = std::make_unique<dnn::Adam>(0.001);
model.compile(std::move(optimizer));

// Train the model
model.fit(X, y, 100, dnn::LossFunction::CrossEntropy, rng);
```

### With Layers
Each trainable layer maintains its own optimizer state:

- **Dense Layer**: Maintains momentum and RMS for weights and biases
- **Conv2D Layer**: Maintains momentum and RMS for weights and biases
- **BatchNorm Layer**: Maintains momentum and RMS for gamma and beta

## Algorithm Details

### SGD Update Rule
For parameters θ and gradients g:
- With momentum: `v = momentum * v + g`
- Parameter update: `θ = θ - lr * v`

### Adam Update Rule
For parameters θ and gradients g:
- First moment: `m = β1 * m + (1 - β1) * g`
- Second moment: `v = β2 * v + (1 - β2) * g²`
- Bias correction: `m_hat = m / (1 - β1^t)`, `v_hat = v / (1 - β2^t)`
- Parameter update: `θ = θ - lr * m_hat / (sqrt(v_hat) + ε)`

## Performance Considerations

### Memory Usage
- SGD: Minimal additional memory (only velocity if using momentum)
- Adam: Higher memory usage (maintains first and second moments)

### Convergence
- SGD: Slower but steady convergence, good for fine-tuning
- Adam: Faster initial convergence, may not reach global optimum

### Hyperparameter Sensitivity
- SGD: More sensitive to learning rate selection
- Adam: More robust to hyperparameter choices

## Best Practices

### Choosing an Optimizer
- **SGD**: Use for fine-tuning or when precise control over learning is needed
- **Adam**: Use as default for most problems, especially with sparse gradients
- Consider the trade-off between convergence speed and final accuracy

### Hyperparameter Tuning
1. Start with default hyperparameters for Adam (lr=0.001)
2. For SGD, experiment with learning rates between 0.01 and 0.1
3. Adjust momentum parameters based on problem characteristics
4. Use learning rate scheduling for better convergence

### Monitoring
- Track loss curves to identify optimizer performance
- Monitor gradient magnitudes to detect optimization issues
- Adjust hyperparameters if training is unstable or slow