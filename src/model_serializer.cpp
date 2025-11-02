// model_serializer.cpp - Binary serialization for model persistence
// Implements complete save/load functionality for DNN models

#include "../include/model_serializer.hpp"
#include "../include/dnn.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

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

// Save model to binary file
bool ModelSerializer::save_model(const Model& model, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    try {
        // Write magic number and version
        const uint32_t magic = 0xDNNM; // DNN Model
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
        write_vector(file, dense->weights.data);
        write_vector(file, dense->bias.data);
        
        // Save optimizer state
        write_vector(file, dense->weight_velocity.data);
        write_vector(file, dense->bias_velocity.data);
        write_vector(file, dense->weight_momentum.data);
        write_vector(file, dense->bias_momentum.data);
        write_vector(file, dense->weight_rms.data);
        write_vector(file, dense->bias_rms.data);
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
        write_vector(file, conv2d->weights.data);
        write_vector(file, conv2d->bias.data);
        
        // Save optimizer state
        write_vector(file, conv2d->weight_velocity.data);
        write_vector(file, conv2d->bias_velocity.data);
        write_vector(file, conv2d->weight_momentum.data);
        write_vector(file, conv2d->bias_momentum.data);
        write_vector(file, conv2d->weight_rms.data);
        write_vector(file, conv2d->bias_rms.data);
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
        write_vector(file, batchnorm->gamma.data);
        write_vector(file, batchnorm->beta.data);
        write_vector(file, batchnorm->running_mean.data);
        write_vector(file, batchnorm->running_var.data);
        
        // Save optimizer state
        write_vector(file, batchnorm->weight_velocity.data);
        write_vector(file, batchnorm->bias_velocity.data);
        write_vector(file, batchnorm->weight_momentum.data);
        write_vector(file, batchnorm->bias_momentum.data);
        write_vector(file, batchnorm->weight_rms.data);
        write_vector(file, batchnorm->bias_rms.data);
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
            read_vector(file, dense->weights.data);
            read_vector(file, dense->bias.data);
            
            // Load optimizer state
            read_vector(file, dense->weight_velocity.data);
            read_vector(file, dense->bias_velocity.data);
            read_vector(file, dense->weight_momentum.data);
            read_vector(file, dense->bias_momentum.data);
            read_vector(file, dense->weight_rms.data);
            read_vector(file, dense->bias_rms.data);
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
            read_vector(file, conv2d->weights.data);
            read_vector(file, conv2d->bias.data);
            
            // Load optimizer state
            read_vector(file, conv2d->weight_velocity.data);
            read_vector(file, conv2d->bias_velocity.data);
            read_vector(file, conv2d->weight_momentum.data);
            read_vector(file, conv2d->bias_momentum.data);
            read_vector(file, conv2d->weight_rms.data);
            read_vector(file, conv2d->bias_rms.data);
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
            read_vector(file, batchnorm->gamma.data);
            read_vector(file, batchnorm->beta.data);
            read_vector(file, batchnorm->running_mean.data);
            read_vector(file, batchnorm->running_var.data);
            
            // Load optimizer state
            read_vector(file, batchnorm->weight_velocity.data);
            read_vector(file, batchnorm->bias_velocity.data);
            read_vector(file, batchnorm->weight_momentum.data);
            read_vector(file, batchnorm->bias_momentum.data);
            read_vector(file, batchnorm->weight_rms.data);
            read_vector(file, batchnorm->bias_rms.data);
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
        
        if (magic != 0xDNNM) {
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
        
        file.close();
        return true;
    } catch (...) {
        file.close();
        return false;
    }
}

} // namespace dnn