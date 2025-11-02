// model_serializer.cpp - Binary serialization for model persistence
// Implements complete save/load functionality for DNN models

#include "../include/model_serializer.hpp"
#include "../include/dnn.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cstdint>

namespace dnn {

// Write a string to binary stream
void ModelSerializer::write_string(std::ofstream& file, const std::string& str) {
    size_t size = str.size();
    write_value(file, size);
    if (!str.empty()) {
        file.write(str.c_str(), size);
    }
}

// Read a string from binary stream
void ModelSerializer::read_string(std::ifstream& file, std::string& str) {
    size_t size;
    read_value(file, size);
    str.resize(size);
    if (size > 0) {
        file.read(&str[0], size);
    }
}

void ModelSerializer::write_matrix(std::ofstream& file, const Matrix& matrix) {
    write_value(file, matrix.shape[0]);
    write_value(file, matrix.shape[1]);
    write_vector(file, matrix.data);
}

void ModelSerializer::read_matrix(std::ifstream& file, Matrix& matrix) {
    std::size_t rows{0}, cols{0};
    read_value(file, rows);
    read_value(file, cols);
    if (matrix.shape[0] != rows || matrix.shape[1] != cols) {
        matrix = Matrix(rows, cols);
    }
    read_vector(file, matrix.data);
    sync_matrix_metadata(matrix);
}

void ModelSerializer::sync_matrix_metadata(Matrix& matrix) {
    matrix.size = matrix.data.size();
}

void ModelSerializer::save_optimizer(std::ofstream& file, const Optimizer* optimizer) {
    if (!optimizer) {
        throw std::runtime_error("Attempted to serialize null optimizer");
    }
    
    write_value(file, static_cast<uint32_t>(optimizer->type));
    write_value(file, optimizer->learning_rate);
    write_value(file, optimizer->epsilon);
    write_value(file, optimizer->l1_lambda);
    write_value(file, optimizer->l2_lambda);
    write_value(file, optimizer->use_gradient_clipping);
    write_value(file, optimizer->max_gradient_norm);
    write_value(file, optimizer->clip_value);
    
    const bool has_scheduler = optimizer->lr_scheduler != nullptr;
    write_value(file, has_scheduler);
    if (has_scheduler) {
        throw std::runtime_error("Serializing learning rate schedulers is not supported");
    }
    
    switch (optimizer->type) {
        case OptimizerType::SGD: {
            const auto* sgd = dynamic_cast<const SGD*>(optimizer);
            if (!sgd) {
                throw std::runtime_error("Optimizer type mismatch during serialization (SGD)");
            }
            write_value(file, sgd->momentum);
            write_value(file, sgd->weight_decay);
            write_value(file, sgd->dampening);
            write_value(file, sgd->nesterov);
            break;
        }
        case OptimizerType::Adam: {
            const auto* adam = dynamic_cast<const Adam*>(optimizer);
            if (!adam) {
                throw std::runtime_error("Optimizer type mismatch during serialization (Adam)");
            }
            write_value(file, adam->beta1);
            write_value(file, adam->beta2);
            write_value(file, adam->weight_decay);
            write_value(file, static_cast<std::uint64_t>(adam->step_count));
            break;
        }
        case OptimizerType::RMSprop: {
            const auto* rms = dynamic_cast<const RMSprop*>(optimizer);
            if (!rms) {
                throw std::runtime_error("Optimizer type mismatch during serialization (RMSprop)");
            }
            write_value(file, rms->alpha);
            write_value(file, rms->epsilon);
            write_value(file, rms->weight_decay);
            break;
        }
        case OptimizerType::AdamW: {
            const auto* adamw = dynamic_cast<const AdamW*>(optimizer);
            if (!adamw) {
                throw std::runtime_error("Optimizer type mismatch during serialization (AdamW)");
            }
            write_value(file, adamw->beta1);
            write_value(file, adamw->beta2);
            write_value(file, adamw->weight_decay);
            write_value(file, static_cast<std::uint64_t>(adamw->step_count));
            break;
        }
        default:
            throw std::runtime_error("Unsupported optimizer type for serialization");
    }
}

std::unique_ptr<Optimizer> ModelSerializer::load_optimizer(std::ifstream& file) {
    uint32_t type_raw{0};
    read_value(file, type_raw);
    OptimizerType type = static_cast<OptimizerType>(type_raw);
    
    double learning_rate{0.0};
    double epsilon{0.0};
    double l1_lambda{0.0};
    double l2_lambda{0.0};
    bool use_clip{false};
    double max_norm{0.0};
    double clip_value{0.0};
    bool has_scheduler{false};
    
    read_value(file, learning_rate);
    read_value(file, epsilon);
    read_value(file, l1_lambda);
    read_value(file, l2_lambda);
    read_value(file, use_clip);
    read_value(file, max_norm);
    read_value(file, clip_value);
    read_value(file, has_scheduler);
    
    if (has_scheduler) {
        throw std::runtime_error("Deserializing optimizers with schedulers is not supported");
    }
    
    std::unique_ptr<Optimizer> optimizer;
    
    switch (type) {
        case OptimizerType::SGD: {
            double momentum{0.0};
            double weight_decay{0.0};
            double dampening{0.0};
            bool nesterov{false};
            read_value(file, momentum);
            read_value(file, weight_decay);
            read_value(file, dampening);
            read_value(file, nesterov);
            optimizer = std::make_unique<SGD>(learning_rate, momentum, weight_decay, dampening, nesterov);
            break;
        }
        case OptimizerType::Adam: {
            double beta1{0.9};
            double beta2{0.999};
            double weight_decay{0.0};
            std::uint64_t step_count{0};
            read_value(file, beta1);
            read_value(file, beta2);
            read_value(file, weight_decay);
            read_value(file, step_count);
            auto adam = std::make_unique<Adam>(learning_rate, beta1, beta2, weight_decay);
            adam->step_count = static_cast<std::size_t>(step_count);
            optimizer = std::move(adam);
            break;
        }
        case OptimizerType::RMSprop: {
            double alpha{0.99};
            double eps{epsilon};
            double weight_decay{0.0};
            read_value(file, alpha);
            read_value(file, eps);
            read_value(file, weight_decay);
            auto rms = std::make_unique<RMSprop>(learning_rate, alpha, eps, weight_decay);
            optimizer = std::move(rms);
            break;
        }
        case OptimizerType::AdamW: {
            double beta1{0.9};
            double beta2{0.999};
            double weight_decay{0.0};
            std::uint64_t step_count{0};
            read_value(file, beta1);
            read_value(file, beta2);
            read_value(file, weight_decay);
            read_value(file, step_count);
            auto adamw = std::make_unique<AdamW>(learning_rate, beta1, beta2, weight_decay);
            adamw->step_count = static_cast<std::size_t>(step_count);
            optimizer = std::move(adamw);
            break;
        }
        default:
            throw std::runtime_error("Unsupported optimizer type in model file");
    }
    
    optimizer->learning_rate = learning_rate;
    optimizer->epsilon = epsilon;
    optimizer->l1_lambda = l1_lambda;
    optimizer->l2_lambda = l2_lambda;
    optimizer->use_gradient_clipping = use_clip;
    optimizer->max_gradient_norm = max_norm;
    optimizer->clip_value = clip_value;
    
    return optimizer;
}

// Save model to binary file
bool ModelSerializer::save_model(const Model& model, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    try {
        // Write magic number and version
        const uint32_t magic = 0x444E4E4D; // 'DNNM'
        const uint32_t version = 1;
        write_value(file, magic);
        write_value(file, version);
        
        // Save configuration
        const auto& config = model.config;
        write_value(file, config.epsilon);
        write_value(file, config.learning_rate);
        write_value(file, config.batch_size);
        write_value(file, config.max_epochs);
        write_value(file, config.validation_split);
        write_value(file, config.dropout_rate);
        write_value(file, config.use_batch_norm);
        write_value(file, config.use_layer_norm);
        
        // Save optimizer state if available
        const Optimizer* optimizer = model.get_optimizer();
        bool has_optimizer = (optimizer != nullptr);
        write_value(file, has_optimizer);
        if (has_optimizer) {
            save_optimizer(file, optimizer);
        }
        
        // Save number of layers
        const auto& layers = model.layers;
        size_t num_layers = layers.size();
        write_value(file, num_layers);
        
        // Save each layer
        for (size_t i = 0; i < num_layers; ++i) {
            save_layer(file, layers[i].get());
        }
        
        file.close();
        return true;
    } catch (...) {
        file.close();
        return false;
    }
}

// Save layer information
void ModelSerializer::save_layer(std::ofstream& file, const Layer* layer) {
    // Write layer name
    write_string(file, layer->name);
    
    // Write trainable flag
    write_value(file, layer->trainable);
    
    // Determine layer type and save accordingly
    if (const auto* dense = dynamic_cast<const Dense*>(layer)) {
        // Layer type: Dense = 0
        uint32_t layer_type = 0;
        write_value(file, layer_type);
        
        // Save Dense-specific properties
        write_value(file, dense->in_features);
        write_value(file, dense->out_features);
        
        // Save activation function
        int activation = static_cast<int>(dense->activation);
        write_value(file, activation);
        
        // Save weights and bias
        write_matrix(file, dense->weights);
        write_matrix(file, dense->bias);
        
        // Save optimizer state
        write_matrix(file, dense->weight_velocity);
        write_matrix(file, dense->bias_velocity);
        write_matrix(file, dense->weight_momentum);
        write_matrix(file, dense->bias_momentum);
        write_matrix(file, dense->weight_rms);
        write_matrix(file, dense->bias_rms);
    }
    else if (const auto* conv2d = dynamic_cast<const Conv2D*>(layer)) {
        // Layer type: Conv2D = 1
        uint32_t layer_type = 1;
        write_value(file, layer_type);
        
        // Save Conv2D-specific properties
        write_value(file, conv2d->in_channels);
        write_value(file, conv2d->out_channels);
        write_value(file, conv2d->kernel_height);
        write_value(file, conv2d->kernel_width);
        write_value(file, conv2d->stride_h);
        write_value(file, conv2d->stride_w);
        write_value(file, conv2d->padding_h);
        write_value(file, conv2d->padding_w);
        
        // Save activation function
        int activation = static_cast<int>(conv2d->activation);
        write_value(file, activation);
        
        // Save weights and bias
        write_matrix(file, conv2d->weights);
        write_matrix(file, conv2d->bias);
        
        // Save optimizer state
        write_matrix(file, conv2d->weight_velocity);
        write_matrix(file, conv2d->bias_velocity);
        write_matrix(file, conv2d->weight_momentum);
        write_matrix(file, conv2d->bias_momentum);
        write_matrix(file, conv2d->weight_rms);
        write_matrix(file, conv2d->bias_rms);
    }
    else if (const auto* maxpool = dynamic_cast<const MaxPool2D*>(layer)) {
        // Layer type: MaxPool2D = 2
        uint32_t layer_type = 2;
        write_value(file, layer_type);
        
        // Save MaxPool2D-specific properties
        write_value(file, maxpool->pool_height);
        write_value(file, maxpool->pool_width);
        write_value(file, maxpool->stride_h);
        write_value(file, maxpool->stride_w);
    }
    else if (const auto* dropout = dynamic_cast<const Dropout*>(layer)) {
        // Layer type: Dropout = 3
        uint32_t layer_type = 3;
        write_value(file, layer_type);
        
        // Save Dropout-specific properties
        write_value(file, dropout->rate);
    }
    else if (const auto* batchnorm = dynamic_cast<const BatchNorm*>(layer)) {
        // Layer type: BatchNorm = 4
        uint32_t layer_type = 4;
        write_value(file, layer_type);
        
        // Save BatchNorm-specific properties
        write_value(file, batchnorm->features);
        write_value(file, batchnorm->momentum);
        write_value(file, batchnorm->epsilon);
        
        // Save parameters
        write_matrix(file, batchnorm->gamma);
        write_matrix(file, batchnorm->beta);
        write_matrix(file, batchnorm->running_mean);
        write_matrix(file, batchnorm->running_var);
        
        // Save optimizer state
        write_matrix(file, batchnorm->weight_velocity);
        write_matrix(file, batchnorm->bias_velocity);
        write_matrix(file, batchnorm->weight_momentum);
        write_matrix(file, batchnorm->bias_momentum);
        write_matrix(file, batchnorm->weight_rms);
        write_matrix(file, batchnorm->bias_rms);
    }
    else {
        // Unknown layer type - save as generic layer
        uint32_t layer_type = 99; // Unknown
        write_value(file, layer_type);
    }
}

// Load layer information
std::unique_ptr<Layer> ModelSerializer::load_layer(std::ifstream& file) {
    std::string layer_name;
    bool trainable;
    read_string(file, layer_name);
    read_value(file, trainable);
    
    uint32_t layer_type;
    read_value(file, layer_type);
    
    std::unique_ptr<Layer> layer = nullptr;
    
    switch (layer_type) {
        case 0: { // Dense
            size_t in_features, out_features;
            read_value(file, in_features);
            read_value(file, out_features);
            
            int activation_int;
            read_value(file, activation_int);
            Activation activation = static_cast<Activation>(activation_int);
            
            // Create layer
            layer = std::make_unique<Dense>(in_features, out_features, activation, layer_name);
            auto* dense = static_cast<Dense*>(layer.get());
            
            // Load weights and bias
            read_matrix(file, dense->weights);
            read_matrix(file, dense->bias);
            
            // Load optimizer state
            read_matrix(file, dense->weight_velocity);
            read_matrix(file, dense->bias_velocity);
            read_matrix(file, dense->weight_momentum);
            read_matrix(file, dense->bias_momentum);
            read_matrix(file, dense->weight_rms);
            read_matrix(file, dense->bias_rms);
            break;
        }
        case 1: { // Conv2D
            size_t in_channels, out_channels, kernel_height, kernel_width;
            size_t stride_h, stride_w, padding_h, padding_w;
            
            read_value(file, in_channels);
            read_value(file, out_channels);
            read_value(file, kernel_height);
            read_value(file, kernel_width);
            read_value(file, stride_h);
            read_value(file, stride_w);
            read_value(file, padding_h);
            read_value(file, padding_w);
            
            int activation_int;
            read_value(file, activation_int);
            Activation activation = static_cast<Activation>(activation_int);
            
            // Create layer
            layer = std::make_unique<Conv2D>(in_channels, out_channels, kernel_height, kernel_width,
                                           stride_h, stride_w, padding_h, padding_w, activation, layer_name);
            auto* conv2d = static_cast<Conv2D*>(layer.get());
            
            // Load weights and bias
            read_matrix(file, conv2d->weights);
            read_matrix(file, conv2d->bias);
            
            // Load optimizer state
            read_matrix(file, conv2d->weight_velocity);
            read_matrix(file, conv2d->bias_velocity);
            read_matrix(file, conv2d->weight_momentum);
            read_matrix(file, conv2d->bias_momentum);
            read_matrix(file, conv2d->weight_rms);
            read_matrix(file, conv2d->bias_rms);
            break;
        }
        case 2: { // MaxPool2D
            size_t pool_height, pool_width, stride_h, stride_w;
            
            read_value(file, pool_height);
            read_value(file, pool_width);
            read_value(file, stride_h);
            read_value(file, stride_w);
            
            layer = std::make_unique<MaxPool2D>(pool_height, pool_width, stride_h, stride_w, layer_name);
            break;
        }
        case 3: { // Dropout
            double rate;
            read_value(file, rate);
            
            layer = std::make_unique<Dropout>(rate, layer_name);
            break;
        }
        case 4: { // BatchNorm
            size_t features;
            double momentum, epsilon;
            
            read_value(file, features);
            read_value(file, momentum);
            read_value(file, epsilon);
            
            layer = std::make_unique<BatchNorm>(features, momentum, epsilon, layer_name);
            auto* batchnorm = static_cast<BatchNorm*>(layer.get());
            
            // Load parameters
            read_matrix(file, batchnorm->gamma);
            read_matrix(file, batchnorm->beta);
            read_matrix(file, batchnorm->running_mean);
            read_matrix(file, batchnorm->running_var);
            
            // Load optimizer state
            read_matrix(file, batchnorm->weight_velocity);
            read_matrix(file, batchnorm->bias_velocity);
            read_matrix(file, batchnorm->weight_momentum);
            read_matrix(file, batchnorm->bias_momentum);
            read_matrix(file, batchnorm->weight_rms);
            read_matrix(file, batchnorm->bias_rms);
            break;
        }
        default:
            // Unknown layer type
            break;
    }
    
    if (layer) {
        layer->trainable = trainable;
    }
    
    return layer;
}

// Load model from binary file
bool ModelSerializer::load_model(Model& model, const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    try {
        // Read and verify magic number and version
        uint32_t magic, version;
        read_value(file, magic);
        read_value(file, version);
        
        if (magic != 0x444E4E4D) {
            std::cerr << "Invalid file format" << std::endl;
            file.close();
            return false;
        }
        
        if (version != 1) {
            std::cerr << "Unsupported version: " << version << std::endl;
            file.close();
            return false;
        }
        
        // Load configuration
        auto& config = model.config;
        read_value(file, config.epsilon);
        read_value(file, config.learning_rate);
        read_value(file, config.batch_size);
        read_value(file, config.max_epochs);
        read_value(file, config.validation_split);
        read_value(file, config.dropout_rate);
        read_value(file, config.use_batch_norm);
        read_value(file, config.use_layer_norm);
        
        bool has_optimizer = false;
        read_value(file, has_optimizer);
        std::unique_ptr<Optimizer> optimizer_ptr;
        if (has_optimizer) {
            optimizer_ptr = load_optimizer(file);
        }
        
        // Clear existing layers
        model.layers.clear();
        
        // Load number of layers
        size_t num_layers;
        read_value(file, num_layers);
        
        // Load each layer
        for (size_t i = 0; i < num_layers; ++i) {
            auto layer = load_layer(file);
            if (layer) {
                model.layers.push_back(std::move(layer));
            } else {
                std::cerr << "Failed to load layer " << i << std::endl;
                file.close();
                return false;
            }
        }
        
        if (optimizer_ptr) {
            model.set_optimizer(std::move(optimizer_ptr));
        } else {
            model.set_optimizer(std::unique_ptr<Optimizer>{});
        }
        
        file.close();
        return true;
    } catch (...) {
        file.close();
        return false;
    }
}

} // namespace dnn
