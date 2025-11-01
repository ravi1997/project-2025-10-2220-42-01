# Data Flow Diagrams and Process Charts

## High-Level Data Flow

```mermaid
graph TD
    A[Input Data] --> B[Tensor Creation]
    B --> C[Model Forward Pass]
    C --> D[Layer 1 Forward]
    D --> E[Layer 2 Forward]
    E --> F[...]
    F --> G[Layer N Forward]
    G --> H[Output Prediction]
    H --> I[Loss Calculation]
    I --> J[Backward Pass]
    J --> K[Layer N Backward]
    K --> L[Layer N-1 Backward]
    L --> M[...]
    M --> N[Layer 1 Backward]
    N --> O[Parameter Update]
    O --> P[Next Batch]
    P --> C
```

## Forward Pass Data Flow

During the forward pass, data flows sequentially through each layer:

1. **Input Layer**: Raw data enters the network
2. **Processing**: Each layer applies its transformation
   - Dense: Matrix multiplication + bias + activation
   - Conv2D: Convolution operation + activation
   - Pooling: Down-sampling operation
   - Dropout: Random masking (training mode)
3. **Output Layer**: Final predictions are generated

## Backward Pass Data Flow

The backward pass implements backpropagation:

1. **Loss Gradient**: Computed from loss function derivative
2. **Layer-wise Backpropagation**: Gradients flow backwards through each layer
   - Each layer computes gradients w.r.t. its inputs and parameters
   - Uses chain rule to propagate gradients
3. **Parameter Updates**: Optimizer updates weights based on gradients

## Training Process Flow

```mermaid
flowchart LR
    Start([Start Training]) --> Init{Initialize Model}
    Init --> Load[Load Training Data]
    Load --> EpochLoop{For each epoch}
    EpochLoop --> BatchLoop{For each batch}
    BatchLoop --> Forward[Forward Pass]
    Forward --> Loss[Compute Loss]
    Loss --> Backward[Backward Pass]
    Backward --> Update[Update Parameters]
    Update --> CheckBatch{Batch Complete?}
    CheckBatch -->|No| BatchLoop
    CheckBatch -->|Yes| CheckEpoch{Epoch Complete?}
    CheckEpoch -->|No| EpochLoop
    CheckEpoch -->|Yes| Validation[Validation Check]
    Validation --> End([Training Complete])
```

## Memory Data Flow

### Tensor Memory Layout
- Data stored in contiguous `std::vector<double>`
- Row-major ordering for multi-dimensional access
- Shape information stored separately for dimension calculations

### Caching Mechanism
- Each layer caches inputs for backward pass
- Activation values cached for gradient computation
- Gradient accumulations stored for optimizer updates

## Optimizer Data Flow

### SGD with Momentum
1. Compute gradients from backward pass
2. Update velocity: `velocity = momentum * velocity + gradient`
3. Update parameters: `params = params - learning_rate * velocity`

### Adam Optimizer
1. Compute gradients from backward pass
2. Update momentum: `momentum = beta1 * momentum + (1-beta1) * gradient`
3. Update RMS: `rms = beta2 * rms + (1-beta2) * gradient^2`
4. Apply bias correction and update parameters

## Parallel Processing Flow

When enabled, matrix operations can use parallel execution:
- Large matrix multiplications are divided among threads
- Each thread processes a subset of rows
- Results are combined in the final matrix