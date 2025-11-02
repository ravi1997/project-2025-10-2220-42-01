# Activation Functions Documentation

## Overview
Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. The DNN library implements a comprehensive set of activation functions with efficient forward and backward computation.

## Activation Function Types

### Enum Definition
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

## Implemented Activation Functions

### Linear
The linear activation function is the identity function.

#### Formula
```
f(x) = x
f'(x) = 1
```

#### Use Cases
- Output layers for regression
- Linear transformations
- When no non-linearity is desired

#### Implementation Details
- No computation needed (identity function)
- Gradient is always 1

### ReLU (Rectified Linear Unit)
ReLU sets all negative values to zero while preserving positive values.

#### Formula
```
f(x) = max(0, x)
f'(x) = { 1 if x > 0
        { 0 if x ≤ 0
```

#### Use Cases
- Hidden layers in most deep networks
- Default choice for many architectures
- Good for sparse representations

#### Implementation Details
- Efficient computation with simple thresholding
- Gradient is 0 for negative inputs
- Potential "dying ReLU" problem for negative inputs

### Leaky ReLU
Leaky ReLU allows small negative values to pass through instead of setting them to zero.

#### Formula
```
f(x) = { x      if x > 0
       { 0.01*x if x ≤ 0
f'(x) = { 1   if x > 0
        { 0.01 if x ≤ 0
```

#### Use Cases
- Addresses dying ReLU problem
- When ReLU performs poorly
- Similar benefits to ReLU with less risk of dead neurons

#### Implementation Details
- Fixed small slope (0.01) for negative values
- Prevents complete gradient flow stoppage
- More robust than standard ReLU

### ELU (Exponential Linear Unit)
ELU uses an exponential function for negative values to reduce bias shift.

#### Formula
```
f(x) = { x       if x > 0
       { α(exp(x) - 1) if x ≤ 0
f'(x) = { 1           if x > 0
        { f(x) + α    if x ≤ 0
```
where α = 1.0

#### Use Cases
- Deep networks where bias shift is a concern
- When smooth gradients are beneficial
- Alternative to ReLU with better properties

#### Implementation Details
- Exponential computation for negative values
- Smooth transition at zero
- Reduces mean activation shift toward zero

### Sigmoid
Sigmoid maps any real value to the range (0, 1).

#### Formula
```
f(x) = 1 / (1 + exp(-x))
f'(x) = f(x) * (1 - f(x))
```

#### Use Cases
- Output layers for binary classification
- Probability outputs
- When bounded outputs are needed

#### Implementation Details
- Numerically stable implementation with overflow protection
- Outputs always between 0 and 1
- Suffers from vanishing gradient for extreme inputs
- Utilises a shared `stable_sigmoid` helper to avoid overflow for large |x|

### Tanh (Hyperbolic Tangent)
Tanh maps any real value to the range (-1, 1).

#### Formula
```
f(x) = tanh(x)
f'(x) = 1 - f(x)²
```

#### Use Cases
- Hidden layers (less common now)
- When zero-centered outputs are beneficial
- Similar to sigmoid but zero-centered

#### Implementation Details
- Zero-centered output (-1 to 1)
- Steeper gradient than sigmoid
- Still suffers from vanishing gradients

### Softmax
Softmax converts a vector of values to a probability distribution.

#### Formula
```
f(x_i) = exp(x_i - max) / Σ(exp(x_j - max))
f'(x_i) = softmax(x_i) * (1 - softmax(x_i)) for diagonal
         = -softmax(x_i) * softmax(x_j) for off-diagonal
```

#### Use Cases
- Output layer for multi-class classification
- Probability distributions over classes
- When outputs need to sum to 1

#### Implementation Details
- Max-subtraction for numerical stability
- Full Jacobian computation for gradients
- Proper normalization to ensure probabilities sum to 1

### Swish
Swish is a self-gated activation function.

#### Formula
```
f(x) = x * sigmoid(x)
f'(x) = f(x) + sigmoid(x) * (1 - f(x))
```

#### Use Cases
- Modern deep networks
- Empirically performs well
- Smooth, non-monotonic function

#### Implementation Details
- Combination of linear and sigmoid
- Smooth, non-monotonic behavior
- Requires sigmoid computation
- Leverages the same `stable_sigmoid` core used by Sigmoid/Softplus for numerical safety

### GELU (Gaussian Error Linear Unit)
GELU uses the Gaussian CDF for smooth approximation.

#### Formula
```
f(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
f'(x) = 0.5 * (1 + erf(x / sqrt(2))) + x * phi(x)
```
where φ is the standard normal PDF

#### Use Cases
- Transformer architectures
- Modern neural networks
- Smooth approximation to ReLU

#### Implementation Details
- Uses error function for computation
- Smooth, non-monotonic behavior
- Well-performing in practice

### Softplus
Softplus is a smooth approximation to ReLU.

#### Formula
```
f(x) = log(1 + exp(x))
f'(x) = sigmoid(x)
```

#### Use Cases
- Smooth approximation to ReLU
- When smooth gradients are needed
- Probabilistic models

#### Implementation Details
- Smooth, differentiable everywhere
- Computationally more expensive than ReLU
- Always positive output
- Evaluated via `log1p`/`exp` combination to remain stable for large |x|

## Usage in Layers

### With Dense Layers
```cpp
// Create dense layers with different activations
auto hidden_layer = std::make_unique<dnn::Dense>(784, 128, dnn::Activation::ReLU);
auto output_layer = std::make_unique<dnn::Dense>(128, 10, dnn::Activation::Softmax);
```

### Direct Computation
```cpp
// Apply activation to a matrix
dnn::Matrix activated = dnn::apply_activation(input, dnn::Activation::ReLU);

// Compute activation derivative for backpropagation
dnn::Matrix grad = dnn::apply_activation_derivative(output, gradient, dnn::Activation::ReLU);
```

## Performance Considerations

### Computational Complexity
- **Fastest**: Linear, ReLU, Leaky ReLU
- **Moderate**: Sigmoid, Tanh, Softplus
- **Slowest**: ELU, Swish, GELU (due to transcendental functions)

### Memory Efficiency
- Activation values cached for backward pass
- Derivative computation uses cached values
- Memory usage is O(n) where n is tensor size

### Numerical Stability
- Overflow protection in exponential functions via `stable_sigmoid`, `log1p`, and max-subtraction.
- Probabilities are clamped to `[ε, 1-ε]` (`ε = 1e-12`) before log/division operations to avoid NaNs.
- Softplus, softmax, sigmoid, and swish all reuse the shared numerical guard utilities introduced in Phase 1.

## Choosing the Right Activation Function

### For Hidden Layers
- **ReLU**: Default choice for most networks
- **Leaky ReLU**: When dying ReLU is a concern
- **GELU**: For transformer-style architectures
- **ELU**: When smooth gradients are important

### For Output Layers
- **Linear**: Regression problems
- **Sigmoid**: Binary classification
- **Softmax**: Multi-class classification

### For Special Cases
- **Tanh**: When zero-centered outputs are needed
- **Softplus**: When smooth ReLU approximation is required
- **Swish**: When empirical performance is prioritized

## Best Practices

1. Start with ReLU for hidden layers unless there's a specific reason not to
2. Use appropriate activations for your output layer based on the task
3. Monitor for vanishing gradients with sigmoid/tanh in deep networks
4. Consider computational cost vs. performance trade-offs
5. Use activation functions that match the desired output range
6. Be aware of the gradient behavior of your chosen activation functions
