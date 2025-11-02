#pragma once
// model_serializer.hpp - Binary serialization for model persistence
// Implements complete save/load functionality for DNN models

#include <string>
#include <memory>
#include <fstream>
#include <vector>

// Forward declarations to avoid circular dependency
namespace dnn {
    class Model;
    class Layer;
}

namespace dnn {

// Helper functions for binary serialization
class ModelSerializer {
public:
    // Save model to binary file
    static bool save_model(const Model& model, const std::string& filepath);
    
    // Load model from binary file
    static bool load_model(Model& model, const std::string& filepath);
    
private:
    // Write a value to binary stream
    template<typename T>
    static void write_value(std::ofstream& file, const T& value) {
        file.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }
    
    // Write a string to binary stream
    static void write_string(std::ofstream& file, const std::string& str);
    
    // Write a vector to binary stream
    template<typename T>
    static void write_vector(std::ofstream& file, const std::vector<T>& vec);
    
    // Read a value from binary stream
    template<typename T>
    static void read_value(std::ifstream& file, T& value) {
        file.read(reinterpret_cast<char*>(&value), sizeof(T));
    }
    
    // Read a string from binary stream
    static void read_string(std::ifstream& file, std::string& str);
    
    // Read a vector from binary stream
    template<typename T>
    static void read_vector(std::ifstream& file, std::vector<T>& vec);
    
    // Save layer information
    static void save_layer(std::ofstream& file, const Layer* layer);
    
    // Load layer information
    static std::unique_ptr<Layer> load_layer(std::ifstream& file);
};

// Implementation of template methods
template<typename T>
void ModelSerializer::write_vector(std::ofstream& file, const std::vector<T>& vec) {
    size_t size = vec.size();
    write_value(file, size);
    if (!vec.empty()) {
        file.write(reinterpret_cast<const char*>(vec.data()), sizeof(T) * size);
    }
}

template<typename T>
void ModelSerializer::read_vector(std::ifstream& file, std::vector<T>& vec) {
    size_t size;
    read_value(file, size);
    vec.resize(size);
    if (size > 0) {
        file.read(reinterpret_cast<char*>(vec.data()), sizeof(T) * size);
    }
}

} // namespace dnn