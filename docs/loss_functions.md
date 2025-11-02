# Loss Functions Documentation

## Overview
Loss functions measure the discrepancy between predicted and actual values during training. They provide the gradient signal needed for backpropagation and parameter updates. The DNN library implements several commonly used loss functions with efficient gradient computation.

## Loss Function Types

### Enum Definition
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

## Implemented Loss Functions

### Mean Squared Error (MSE)
MSE measures the average squared difference between predicted and actual values.

#### Formula
```
MSE = (1/n) * Σ(y_pred - y_true)²
Gradient = 2*(y_pred - y_true)/n
```

#### Use Cases
- Regression problems
- When errors are normally distributed
- Problems where large errors should be heavily penalized

#### Implementation Details
- Efficient computation with vectorized operations
- Proper gradient scaling by batch size
- Numerical stability for small differences

### Cross Entropy Loss
Cross entropy measures the performance of classification models where the output is a probability distribution.

#### Formula
```
CrossEntropy = -(1/n) * Σ(y_true * log(softmax(y_pred)))
Gradient = (softmax(y_pred) - y_true) / n
```

#### Use Cases
- Multi-class classification
- When outputs represent probability distributions
- Problems requiring probabilistic outputs

#### Implementation Details
- Combined softmax and cross-entropy for numerical stability
- Gradient computation with proper scaling
- Softmax outputs are clamped to `[ε, 1-ε]` (`ε = 1e-12`) before any log/division operations
- Protection against log(0) with epsilon clipping

### Binary Cross Entropy Loss
Binary cross entropy measures performance for binary classification problems.

#### Formula
```
BCE = -(1/n) * Σ(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
Gradient = (y_pred - y_true) / n
```

#### Use Cases
- Binary classification
- Multi-label classification
- Problems with independent binary outputs

#### Implementation Details
- Clipping to prevent log(0)
- Stable gradient computation
- Proper handling of edge cases
- Shared probability clamp ensures `log(1 - y_pred)` remains finite

### Hinge Loss
Hinge loss is used for training classifiers, particularly SVMs.

#### Formula
```
Hinge = max(0, 1 - y_true * y_pred)
Gradient = -y_true if y_true * y_pred < 1, else 0
```

#### Use Cases
- Binary classification
- Max-margin classification
- Support Vector Machine-like problems

#### Implementation Details
- Multi-class extension for neural networks
- Proper gradient computation for backpropagation
- Margin-based learning

### Huber Loss
Huber loss combines the benefits of MSE and MAE, being less sensitive to outliers.

#### Formula
```
Huber = { 0.5 * |error|²                 if |error| ≤ δ
        { δ * |error| - 0.5 * δ²        otherwise
```

#### Use Cases
- Regression with outliers
- Problems sensitive to extreme values
- Robust regression tasks

#### Implementation Details
- Configurable delta parameter (currently fixed at 1.0)
- Smooth transition between quadratic and linear
- Robust gradient computation

### Kullback-Leibler Divergence
KL divergence measures how one probability distribution differs from another.

#### Formula
```
KL(P||Q) = Σ(P(x) * log(P(x)/Q(x)))
Gradient = log(Q) - log(P) + 1
```

#### Use Cases
- Variational autoencoders
- Generative models
- Distribution matching problems

#### Implementation Details
- Proper handling of log(0) with epsilon
- Numerical stability for probability distributions
- Symmetric gradient computation
- Both P and Q distributions are clamped to `[ε, 1-ε]` prior to ratio/log evaluation

## Loss Result Structure
```cpp
struct LossResult {
    double value;      // Scalar loss value
    Matrix gradient;   // Gradient w.r.t. predictions
};
```

The `LossResult` structure contains both the loss value and the gradient needed for backpropagation.

## Usage in Training

### With Model
```cpp
// Train with MSE loss
model.fit(X, y, 100, dnn::LossFunction::MSE, rng);

// Evaluate with Cross Entropy
double loss = model.evaluate(X_val, y_val, dnn::LossFunction::CrossEntropy);
```

### Direct Computation
```cpp
dnn::LossResult result = dnn::compute_loss(y_true, y_pred, dnn::LossFunction::CrossEntropy);
double loss_value = result.value;
dnn::Matrix gradient = result.gradient;
```

## Performance Considerations

### Numerical Stability
- Consistent epsilon (`ε = 1e-12`) applied to all probability-based operations
- Stable softmax computation with max-subtraction and probability clamping
- Binary/softmax/KL losses reuse shared helpers to avoid division-by-zero and NaN propagation

### Memory Efficiency
- In-place operations where possible
- Efficient gradient computation without unnecessary copies
- Proper memory management for large tensors

### Computational Complexity
- Optimized implementations for common loss functions
- Vectorized operations for efficiency
- Parallel computation where beneficial

## Choosing the Right Loss Function

### For Regression
- **MSE**: Default choice for most regression problems
- **Huber**: When data contains outliers

### For Classification
- **Cross Entropy**: Multi-class classification with softmax
- **Binary Cross Entropy**: Binary or multi-label classification

### For Specialized Tasks
- **KLDivergence**: Generative models, variational inference
- **Hinge**: Max-margin classification

## Best Practices

1. Match the loss function to your problem type (regression vs classification)
2. Consider the data distribution when choosing loss functions
3. Monitor loss values during training to detect issues
4. Use appropriate activation functions with each loss type
5. Be aware of gradient magnitudes and potential vanishing/exploding gradients
