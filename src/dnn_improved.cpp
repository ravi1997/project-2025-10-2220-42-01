#include "dnn_improved.hpp"

namespace dnn {

// Model implementation
Model::Model(const Config& cfg) : config(cfg) {}

void Model::add(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

TensorF Model::forward(const TensorF& input) {
    TensorF output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

float Model::compute_loss(const TensorF& predictions, const TensorF& targets, LossFunction loss_fn) {
    auto loss_result = compute_loss(targets, predictions, loss_fn);
    return loss_result.value;
}

void Model::backward(const TensorF& loss_gradient) {
    TensorF grad = loss_gradient;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
}

void Model::train_step(const TensorF& inputs, const TensorF& targets, LossFunction loss_fn) {
    // Forward pass
    TensorF predictions = forward(inputs);
    
    // Compute loss
    auto loss_result = compute_loss(targets, predictions, loss_fn);
    
    // Backward pass
    backward(loss_result.gradient);
    
    // Update parameters
    if (optimizer) {
        optimizer->step();
    }
}

void Model::compile(std::unique_ptr<Optimizer> opt) {
    optimizer = std::move(opt);
}

void Model::fit(const TensorF& X, const TensorF& y, 
                int epochs, LossFunction loss_fn,
                std::mt19937& rng,
                float validation_split, bool verbose) {
    if (validation_split > 0.0f) {
        auto [X_train, X_val, y_train, y_val] = train_test_split_with_validation(X, y, validation_split, rng);
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            train_step(X_train, y_train, loss_fn);
            
            if (verbose && epoch % 10 == 0) {
                float val_loss = evaluate(X_val, y_val, loss_fn);
                std::cout << "Epoch " << epoch << ", Validation Loss: " << val_loss << std::endl;
            }
        }
    } else {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            train_step(X, y, loss_fn);
            
            if (verbose && epoch % 10 == 0) {
                float train_loss = evaluate(X, y, loss_fn);
                std::cout << "Epoch " << epoch << ", Training Loss: " << train_loss << std::endl;
            }
        }
    }
}

TensorF Model::predict(const TensorF& input) {
    return forward(input);
}

float Model::evaluate(const TensorF& X, const TensorF& y, LossFunction loss_fn) {
    TensorF predictions = forward(X);
    return compute_loss(predictions, y, loss_fn);
}

void Model::save(const std::string& filepath) const {
    ModelSerializer serializer;
    serializer.save_model(*this, filepath);
}

void Model::load(const std::string& filepath) {
    ModelSerializer serializer;
    serializer.load_model(*this, filepath);
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

std::size_t Model::layer_count() const {
    return layers.size();
}

std::size_t Model::parameter_count(std::size_t index) const {
    if (index < layers.size()) {
        return layers[index]->get_parameter_count();
    }
    return 0;
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
}

// Optimizer methods
void Optimizer::enable_gradient_clipping(float max_norm) {
    use_gradient_clipping = true;
    max_gradient_norm = max_norm;
    clip_value = 0.0f; // Use norm-based clipping
}

void Optimizer::enable_gradient_clipping_by_value(float clip_val) {
    use_gradient_clipping = true;
    clip_value = clip_val;
    max_gradient_norm = 0.0f; // Use value-based clipping
}

void Optimizer::disable_gradient_clipping() {
    use_gradient_clipping = false;
    max_gradient_norm = 0.0f;
    clip_value = 0.0f;
}

void Optimizer::set_regularization(float l1_reg, float l2_reg) {
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

// SGD implementation
void SGD::step() {
    // Implementation would go here
}

void SGD::zero_grad() {
    // Implementation would go here
}

// Adam implementation
void Adam::step() {
    ++step_count;
    // Implementation would go here
}

void Adam::zero_grad() {
    // Implementation would go here
}

// Loss function implementations
LossResult compute_loss(const TensorF& y_true, const TensorF& y_pred, LossFunction loss_fn) {
    switch (loss_fn) {
        case LossFunction::MSE: {
            // MSE = mean((y_true - y_pred)^2)
            TensorF diff = sub(y_true, y_pred);
            TensorF squared_diff = hadamard(diff, diff);
            float mse = 0.0f;
            for (float val : squared_diff.data()) {
                mse += val;
            }
            mse /= squared_diff.size();
            
            // Gradient for MSE: 2*(y_pred - y_true)/n
            TensorF grad = scalar_mul(sub(y_pred, y_true), 2.0f / static_cast<float>(y_pred.size()));
            return {mse, grad};
        }
        case LossFunction::CrossEntropy: {
            // Cross entropy with softmax
            // Softmax is applied internally
            TensorF softmax_pred = apply_activation(y_pred, Activation::Softmax);
            
            // Cross entropy: -sum(y_true * log(softmax_pred + epsilon))
            float ce = 0.0f;
            for (size_t i = 0; i < y_true.size(); ++i) {
                ce -= y_true.data()[i] * std::log(softmax_pred.data()[i] + 1e-8f);
            }
            ce /= y_true.shape()[0]; // Average over batch
            
            // Gradient for cross entropy with softmax: softmax_pred - y_true
            TensorF grad = sub(softmax_pred, y_true);
            return {ce, grad};
        }
        default:
            throw std::runtime_error("Unsupported loss function");
    }
}

// Utility function implementations
TensorF one_hot(const std::vector<int>& labels, int num_classes) {
    std::size_t batch_size = labels.size();
    TensorF result(std::vector<size_t>{batch_size, static_cast<size_t>(num_classes)});
    
    for (std::size_t i = 0; i < batch_size; ++i) {
        if (labels[i] < 0 || labels[i] >= num_classes) {
            throw std::out_of_range("Label index out of range for one-hot encoding");
        }
        result[{i, static_cast<size_t>(labels[i])}] = 1.0f;
    }
    
    return result;
}

float accuracy(const TensorF& predictions, const std::vector<int>& labels) {
    if (predictions.shape()[0] != labels.size()) {
        throw std::invalid_argument("Prediction and label batch sizes do not match");
    }
    
    std::size_t correct = 0;
    std::size_t batch_size = predictions.shape()[0];
    std::size_t num_classes = predictions.shape()[1];
    
    for (std::size_t i = 0; i < batch_size; ++i) {
        // Find the index of the maximum value in the prediction
        std::size_t predicted_class = 0;
        float max_val = predictions[{i, 0}];
        for (std::size_t j = 1; j < num_classes; ++j) {
            if (predictions[{i, j}] > max_val) {
                max_val = predictions[{i, j}];
                predicted_class = j;
            }
        }
        
        if (static_cast<int>(predicted_class) == labels[i]) {
            ++correct;
        }
    }
    
    return static_cast<float>(correct) / static_cast<float>(batch_size);
}

TensorF normalize(const TensorF& input, float mean, float stddev) {
    TensorF result = input;
    for (auto& val : result.data()) {
        val = (val - mean) / stddev;
    }
    return result;
}

std::pair<TensorF, TensorF> train_test_split(const TensorF& X, const TensorF& y, float test_size, std::mt19937& rng) {
    if (X.shape()[0] != y.shape()[0]) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    std::size_t total_samples = X.shape()[0];
    std::size_t test_samples = static_cast<std::size_t>(total_samples * test_size);
    std::size_t train_samples = total_samples - test_samples;
    
    // Create shuffled indices
    std::vector<std::size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    // Create result tensors
    auto x_shape = X.shape();
    auto y_shape = y.shape();
    x_shape[0] = train_samples;
    y_shape[0] = train_samples;
    TensorF X_train(x_shape);
    TensorF y_train(y_shape);
    
    x_shape[0] = test_samples;
    y_shape[0] = test_samples;
    TensorF X_test(x_shape);
    TensorF y_test(y_shape);
    
    // Split the data
    for (std::size_t i = 0; i < train_samples; ++i) {
        std::size_t idx = indices[i];
        for (std::size_t j = 0; j < X.shape()[1]; ++j) {
            X_train[{i, j}] = X[{idx, j}];
        }
        for (std::size_t j = 0; j < y.shape()[1]; ++j) {
            y_train[{i, j}] = y[{idx, j}];
        }
    }
    
    for (std::size_t i = 0; i < test_samples; ++i) {
        std::size_t idx = indices[train_samples + i];
        for (std::size_t j = 0; j < X.shape()[1]; ++j) {
            X_test[{i, j}] = X[{idx, j}];
        }
        for (std::size_t j = 0; j < y.shape()[1]; ++j) {
            y_test[{i, j}] = y[{idx, j}];
        }
    }
    
    return std::make_pair(std::move(X_train), std::move(y_train));
}

// Scheduler implementations
void StepLR::step() {
    if (++last_epoch % step_size == 0) {
        optimizer.learning_rate *= gamma;
    }
}

void ExponentialLR::step() {
    optimizer.learning_rate *= gamma;
    ++last_epoch;
}

void PolynomialLR::step() {
    ++last_epoch;
    float progress = static_cast<float>(last_epoch) / static_cast<float>(max_epochs);
    optimizer.learning_rate = initial_lr - (initial_lr - end_lr) * std::pow(progress, power);
}

void CosineAnnealingLR::step() {
    ++last_epoch;
    float cosine = std::cos(static_cast<float>(last_epoch) / static_cast<float>(t_max) * std::numbers::pi_v<float>);
    optimizer.learning_rate = eta_min + 0.5f * (initial_lr - eta_min) * (1.0f + cosine);
}

void ReduceLROnPlateau::step() {
    // Implementation would go here
}

void ReduceLROnPlateau::step(float metric) {
    ++last_epoch;
    
    bool is_better = false;
    if (mode_min > 0) {  // Minimizing mode
        is_better = metric < (best - threshold);
    } else {  // Maximizing mode
        is_better = metric > (best + threshold);
    }
    
    if (is_better) {
        best = metric;
        num_bad_epochs = 0;
        in_cooldown = false;
    } else {
        num_bad_epochs++;
        if (in_cooldown) {
            num_bad_epochs = 0;
        }
        
        if (num_bad_epochs >= patience) {
            optimizer.learning_rate *= factor;
            num_bad_epochs = 0;
            in_cooldown = true;
        }
    }
}

} // namespace dnn