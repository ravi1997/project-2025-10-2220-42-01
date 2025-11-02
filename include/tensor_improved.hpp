#ifndef TENSOR_IMPROVED_HPP
#define TENSOR_IMPROVED_HPP

#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <numeric>
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <cmath>
#include <limits>
#include <type_traits>

namespace dnn {

// Exception classes for error handling
class TensorException : public std::runtime_error {
public:
    explicit TensorException(const std::string& msg) : std::runtime_error("Tensor Exception: " + msg) {}
};

class DimensionMismatchException : public TensorException {
public:
    explicit DimensionMismatchException(const std::string& msg) : TensorException("Dimension Mismatch: " + msg) {}
};

class IndexOutOfBoundsException : public TensorException {
public:
    explicit IndexOutOfBoundsException(const std::string& msg) : TensorException("Index Out of Bounds: " + msg) {}
};

class MemoryAllocationException : public TensorException {
public:
    explicit MemoryAllocationException(const std::string& msg) : TensorException("Memory Allocation Failed: " + msg) {}
};

class InvalidOperation : public TensorException {
public:
    explicit InvalidOperation(const std::string& msg) : TensorException("Invalid Operation: " + msg) {}
};

class NumericalStabilityException : public TensorException {
public:
    explicit NumericalStabilityException(const std::string& msg) : TensorException("Numerical Stability Error: " + msg) {}
};

// Memory layout options
enum class MemoryLayout {
    ROW_MAJOR,
    COLUMN_MAJOR
};

// Data type trait for template specialization
template<typename T>
struct DataTypeTrait {
    static const char* name;
};

template<> struct DataTypeTrait<float> { static constexpr const char* name = "float"; };
template<> struct DataTypeTrait<double> { static constexpr const char* name = "double"; };
template<> struct DataTypeTrait<int> { static constexpr const char* name = "int"; };
template<> struct DataTypeTrait<long> { static constexpr const char* name = "long"; };
template<> struct DataTypeTrait<bool> { static constexpr const char* name = "bool"; };

// Numerical constants for stability
template<typename T>
struct NumericalLimits {
    static constexpr T epsilon() { 
        if constexpr (std::is_same_v<T, float>) {
            return static_cast<T>(1e-7);
        } else if constexpr (std::is_same_v<T, double>) {
            return static_cast<T>(1e-15);
        } else {
            return static_cast<T>(1e-7);
        }
    }
    
    static constexpr T max_value() {
        return std::numeric_limits<T>::max();
    }
    
    static constexpr T min_value() {
        return std::numeric_limits<T>::min();
    }
};

// Internal data storage with reference counting and memory pooling
template<typename T>
class TensorData {
private:
    std::shared_ptr<T[]> data_;
    size_t size_;
    size_t capacity_;
    bool is_pooled_;

public:
    explicit TensorData(size_t size) : size_(size), capacity_(size), is_pooled_(false) {
        try {
            data_ = std::shared_ptr<T[]>(new T[size]);
        } catch (const std::bad_alloc& e) {
            throw MemoryAllocationException("Failed to allocate memory for tensor data: " + std::string(e.what()));
        }
    }

    TensorData(size_t size, const T& value) : size_(size), capacity_(size), is_pooled_(false) {
        try {
            data_ = std::shared_ptr<T[]>(new T[size]);
            std::fill(data_.get(), data_.get() + size, value);
        } catch (const std::bad_alloc& e) {
            throw MemoryAllocationException("Failed to allocate memory for tensor data: " + std::string(e.what()));
        }
    }

    TensorData(size_t size, const T* source_data) : size_(size), capacity_(size), is_pooled_(false) {
        try {
            data_ = std::shared_ptr<T[]>(new T[size]);
            std::copy(source_data, source_data + size, data_.get());
        } catch (const std::bad_alloc& e) {
            throw MemoryAllocationException("Failed to allocate memory for tensor data: " + std::string(e.what()));
        }
    }

    T* get() const { return data_.get(); }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    
    void fill(const T& value) {
        std::fill(data_.get(), data_.get() + size_, value);
    }
    
    // Copy the data to ensure unique ownership
    void ensure_unique() {
        if (data_.use_count() > 1) {
            auto new_data = std::shared_ptr<T[]>(new T[size_]);
            std::copy(data_.get(), data_.get() + size_, new_data.get());
            data_ = new_data;
        }
    }
    
    // Resize the tensor data if needed
    void resize(size_t new_size) {
        if (new_size <= capacity_) {
            size_ = new_size;
            return;
        }
        
        // Allocate new memory
        auto new_data = std::shared_ptr<T[]>(new T[new_size]);
        if (data_) {
            // Copy existing data
            std::copy(data_.get(), data_.get() + size_, new_data.get());
        }
        
        data_ = new_data;
        size_ = new_size;
        capacity_ = new_size;
    }
};

// Numerical stability utilities
template<typename T>
class NumericalStability {
public:
    // Stable softmax calculation
    static void softmax_inplace(T* data, size_t size) {
        if (size == 0) return;
        
        // Find max value for numerical stability
        T max_val = data[0];
        for (size_t i = 1; i < size; ++i) {
            if (data[i] > max_val) max_val = data[i];
        }
        
        // Subtract max and exponentiate
        T sum = T(0);
        for (size_t i = 0; i < size; ++i) {
            data[i] = std::exp(data[i] - max_val);
            sum += data[i];
        }
        
        // Normalize
        if (sum > NumericalLimits<T>::epsilon()) {
            for (size_t i = 0; i < size; ++i) {
                data[i] /= sum;
            }
        } else {
            // Handle case where all values are very small
            T uniform_val = T(1) / static_cast<T>(size);
            for (size_t i = 0; i < size; ++i) {
                data[i] = uniform_val;
            }
        }
    }
    
    // Stable log-sum-exp calculation
    static T log_sum_exp(const T* data, size_t size) {
        if (size == 0) return T(0);
        
        // Find max value
        T max_val = data[0];
        for (size_t i = 1; i < size; ++i) {
            if (data[i] > max_val) max_val = data[i];
        }
        
        // Calculate sum of exponentials
        T sum = T(0);
        for (size_t i = 0; i < size; ++i) {
            sum += std::exp(data[i] - max_val);
        }
        
        return max_val + std::log(sum);
    }
    
    // Clamp values to prevent overflow/underflow
    static T clamp(T value, T min_val, T max_val) {
        if (value < min_val) return min_val;
        if (value > max_val) return max_val;
        return value;
    }
    
    // Check for NaN or infinity
    static bool is_valid(T value) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::isfinite(value);
        } else {
            return true; // For integer types, all values are valid
        }
    }
    
    // Check if tensor contains NaN or infinity
    static bool is_valid_tensor(const T* data, size_t size) {
        if constexpr (std::is_floating_point_v<T>) {
            for (size_t i = 0; i < size; ++i) {
                if (!std::isfinite(data[i])) {
                    return false;
                }
            }
            return true;
        } else {
            return true; // For integer types, all values are valid
        }
    }
};

template<typename T>
class Tensor {
private:
    std::shared_ptr<TensorData<T>> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
    MemoryLayout layout_;

    // Calculate strides based on memory layout
    void calculate_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) {
            return;
        }

        if (layout_ == MemoryLayout::ROW_MAJOR) {
            strides_[shape_.size() - 1] = 1;
            for (int i = shape_.size() - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        } else { // COLUMN_MAJOR
            strides_[0] = 1;
            for (size_t i = 1; i < shape_.size(); ++i) {
                strides_[i] = strides_[i - 1] * shape_[i - 1];
            }
        }
    }

    // Calculate total size from shape
    size_t calculate_size() const {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
    }

    // Calculate linear index from multi-dimensional indices
    size_t calculate_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw DimensionMismatchException("Number of indices does not match tensor dimensions");
        }
        
        size_t linear_index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw IndexOutOfBoundsException("Index " + std::to_string(indices[i]) + 
                                              " is out of bounds for dimension " + std::to_string(i) + 
                                              " with size " + std::to_string(shape_[i]));
            }
            linear_index += indices[i] * strides_[i];
        }
        return linear_index;
    }

public:
    // Default constructor - creates empty tensor
    Tensor() : size_(0), layout_(MemoryLayout::ROW_MAJOR) {}

    // Constructor with shape
    explicit Tensor(const std::vector<size_t>& shape, MemoryLayout layout = MemoryLayout::ROW_MAJOR) 
        : shape_(shape), layout_(layout) {
        size_ = calculate_size();
        data_ = std::make_shared<TensorData<T>>(size_);
        calculate_strides();
    }

    // Constructor with shape and initial value
    Tensor(const std::vector<size_t>& shape, const T& initial_value, MemoryLayout layout = MemoryLayout::ROW_MAJOR) 
        : shape_(shape), layout_(layout) {
        size_ = calculate_size();
        data_ = std::make_shared<TensorData<T>>(size_, initial_value);
        calculate_strides();
    }

    // Constructor with shape and data
    Tensor(const std::vector<size_t>& shape, const std::vector<T>& init_data, MemoryLayout layout = MemoryLayout::ROW_MAJOR)
        : shape_(shape), layout_(layout) {
        size_ = calculate_size();
        if (init_data.size() != size_) {
            throw DimensionMismatchException("Initial data size does not match tensor size");
        }
        // Handle special case for std::vector<bool> which doesn't have .data() method
        if constexpr (std::is_same_v<T, bool>) {
            data_ = std::make_shared<TensorData<T>>(size_);
            bool* data_ptr = data_->get();
            for (size_t i = 0; i < init_data.size(); ++i) {
                data_ptr[i] = init_data[i];
            }
        } else {
            data_ = std::make_shared<TensorData<T>>(size_, init_data.data());
        }
        calculate_strides();
    }

    // Constructor with initializer list (for scalar)
    Tensor(const T& value) : shape_({1}), layout_(MemoryLayout::ROW_MAJOR), size_(1) {
        data_ = std::make_shared<TensorData<T>>(1, value);
        strides_ = {1};
    }

    // Copy constructor
    Tensor(const Tensor& other) 
        : data_(other.data_), 
          shape_(other.shape_), 
          strides_(other.strides_), 
          size_(other.size_),
          layout_(other.layout_) {}

    // Move constructor
    Tensor(Tensor&& other) noexcept 
        : data_(std::move(other.data_)), 
          shape_(std::move(other.shape_)), 
          strides_(std::move(other.strides_)), 
          size_(other.size_),
          layout_(other.layout_) {
        other.size_ = 0;
    }

    // Copy assignment operator
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            data_ = other.data_;
            shape_ = other.shape_;
            strides_ = other.strides_;
            size_ = other.size_;
            layout_ = other.layout_;
        }
        return *this;
    }

    // Move assignment operator
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            size_ = other.size_;
            layout_ = other.layout_;
            other.size_ = 0;
        }
        return *this;
    }

    // Access operators
    T& operator[](size_t idx) {
        if (idx >= size_) {
            throw IndexOutOfBoundsException("Index " + std::to_string(idx) + " is out of bounds for tensor size " + std::to_string(size_));
        }
        return data_->get()[idx];
    }

    const T& operator[](size_t idx) const {
        if (idx >= size_) {
            throw IndexOutOfBoundsException("Index " + std::to_string(idx) + " is out of bounds for tensor size " + std::to_string(size_));
        }
        return data_->get()[idx];
    }

    // Multi-dimensional access
    T& operator()(const std::vector<size_t>& indices) {
        size_t linear_index = calculate_index(indices);
        return data_->get()[linear_index];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        size_t linear_index = calculate_index(indices);
        return data_->get()[linear_index];
    }

    // Operator() with variadic arguments for convenience
    template<typename... Args>
    T& operator()(Args... args) {
        std::vector<size_t> indices = {static_cast<size_t>(args)...};
        return operator()(indices);
    }

    template<typename... Args>
    const T& operator()(Args... args) const {
        std::vector<size_t> indices = {static_cast<size_t>(args)...};
        return operator()(indices);
    }

    // Getters
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    MemoryLayout layout() const { return layout_; }
    
    // Data access
    T* data() { 
        data_->ensure_unique(); // Ensure unique ownership before returning mutable pointer
        return data_->get(); 
    }
    
    const T* data() const { return data_->get(); }

    // Fill tensor with a value
    void fill(const T& value) {
        data_->ensure_unique();
        data_->fill(value);
    }

    // Reshape tensor (returns a new tensor with the same data but different shape)
    Tensor reshape(const std::vector<size_t>& new_shape) const {
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
        if (new_size != size_) {
            throw DimensionMismatchException("New shape size does not match current tensor size");
        }
        
        Tensor result = *this;  // Copy the tensor (shares data)
        result.shape_ = new_shape;
        result.calculate_strides();
        return result;
    }

    // Transpose - for 2D tensors
    Tensor transpose() const {
        if (shape_.size() != 2) {
            throw InvalidOperation("Transpose is only implemented for 2D tensors");
        }
        
        std::vector<size_t> transposed_shape = {shape_[1], shape_[0]};
        Tensor result(transposed_shape, layout_);
        
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        
        return result;
    }

    // Matrix multiplication
    Tensor matmul(const Tensor& other) const {
        if (ndim() != 2 || other.ndim() != 2) {
            throw InvalidOperation("Matrix multiplication is only implemented for 2D tensors");
        }
        
        if (shape_[1] != other.shape_[0]) {
            throw DimensionMismatchException("Cannot multiply matrices with incompatible dimensions: (" +
                                           std::to_string(shape_[0]) + "x" + std::to_string(shape_[1]) +
                                           ") and (" + std::to_string(other.shape_[0]) + "x" +
                                           std::to_string(other.shape_[1]) + ")");
        }
        
        std::vector<size_t> result_shape = {shape_[0], other.shape_[1]};
        Tensor result(result_shape);
        
        T* result_data = result.data();
        const T* this_data = data_->get();
        const T* other_data = other.data_->get();
        
        // Perform matrix multiplication with numerical stability checks
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < other.shape_[1]; ++j) {
                T sum = T(0);
                bool overflow_detected = false;
                
                for (size_t k = 0; k < shape_[1]; ++k) {
                    T product = this_data[i * shape_[1] + k] * other_data[k * other.shape_[1] + j];
                    
                    // Check for potential overflow before addition
                    if constexpr (std::is_floating_point_v<T>) {
                        if (!std::isfinite(product)) {
                            throw NumericalStabilityException("Matrix multiplication resulted in non-finite value");
                        }
                    }
                    
                    sum += product;
                    
                    // Check for overflow
                    if constexpr (std::is_floating_point_v<T>) {
                        if (!std::isfinite(sum)) {
                            throw NumericalStabilityException("Matrix multiplication overflow detected");
                        }
                    } else {
                        // For integer types, check for overflow
                        if (sum < T(0) && product > T(0)) {
                            overflow_detected = true;
                            break;
                        }
                    }
                }
                
                if (overflow_detected) {
                    throw NumericalStabilityException("Matrix multiplication integer overflow detected");
                }
                
                result_data[i * other.shape_[1] + j] = sum;
            }
        }
        
        return result;
    }

    // Basic arithmetic operations
    Tensor operator+(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw DimensionMismatchException("Cannot add tensors with different shapes");
        }
        
        Tensor result(shape_, layout_);
        T* result_data = result.data();
        const T* this_data = data_->get();
        const T* other_data = other.data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T sum = this_data[i] + other_data[i];
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(sum)) {
                    throw NumericalStabilityException("Addition resulted in non-finite value");
                }
            }
            
            result_data[i] = sum;
        }
        
        return result;
    }
    
    Tensor operator-(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw DimensionMismatchException("Cannot subtract tensors with different shapes");
        }
        
        Tensor result(shape_, layout_);
        T* result_data = result.data();
        const T* this_data = data_->get();
        const T* other_data = other.data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T diff = this_data[i] - other_data[i];
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(diff)) {
                    throw NumericalStabilityException("Subtraction resulted in non-finite value");
                }
            }
            
            result_data[i] = diff;
        }
        
        return result;
    }
    
    Tensor operator*(const Tensor& other) const {
        if (shape_ != other.shape_) {
            // Try broadcasting if shapes don't match
            auto broadcasted = broadcast_to(compute_broadcast_shape(shape_, other.shape_));
            auto other_broadcasted = other.broadcast_to(compute_broadcast_shape(shape_, other.shape_));
            
            Tensor result(broadcasted.shape_, layout_);
            T* result_data = result.data();
            const T* this_data = broadcasted.data_->get();
            const T* other_data = other_broadcasted.data_->get();
            
            for (size_t i = 0; i < result.size_; ++i) {
                T product = this_data[i] * other_data[i];
                
                // Check for numerical stability
                if constexpr (std::is_floating_point_v<T>) {
                    if (!std::isfinite(product)) {
                        throw NumericalStabilityException("Multiplication resulted in non-finite value");
                    }
                }
                
                result_data[i] = product;
            }
            
            return result;
        }
        
        Tensor result(shape_, layout_);
        T* result_data = result.data();
        const T* this_data = data_->get();
        const T* other_data = other.data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T product = this_data[i] * other_data[i];
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(product)) {
                    throw NumericalStabilityException("Multiplication resulted in non-finite value");
                }
            }
            
            result_data[i] = product;
        }
        
        return result;
    }
    
    Tensor operator/(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw DimensionMismatchException("Cannot divide tensors with different shapes without broadcasting");
        }
        
        Tensor result(shape_, layout_);
        T* result_data = result.data();
        const T* this_data = data_->get();
        const T* other_data = other.data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            if (other_data[i] == T(0)) {
                throw InvalidOperation("Division by zero");
            }
            
            T quotient = this_data[i] / other_data[i];
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(quotient)) {
                    throw NumericalStabilityException("Division resulted in non-finite value");
                }
            }
            
            result_data[i] = quotient;
        }
        
        return result;
    }
    
    // In-place operations
    Tensor& operator+=(const Tensor& other) {
        if (shape_ != other.shape_) {
            throw DimensionMismatchException("Cannot add tensors with different shapes");
        }
        
        data_->ensure_unique();
        T* this_data = data_->get();
        const T* other_data = other.data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T sum = this_data[i] + other_data[i];
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(sum)) {
                    throw NumericalStabilityException("In-place addition resulted in non-finite value");
                }
            }
            
            this_data[i] = sum;
        }
        
        return *this;
    }
    
    Tensor& operator-=(const Tensor& other) {
        if (shape_ != other.shape_) {
            throw DimensionMismatchException("Cannot subtract tensors with different shapes");
        }
        
        data_->ensure_unique();
        T* this_data = data_->get();
        const T* other_data = other.data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T diff = this_data[i] - other_data[i];
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(diff)) {
                    throw NumericalStabilityException("In-place subtraction resulted in non-finite value");
                }
            }
            
            this_data[i] = diff;
        }
        
        return *this;
    }
    
    Tensor& operator*=(const Tensor& other) {
        if (shape_ != other.shape_) {
            throw DimensionMismatchException("Cannot multiply tensors with different shapes");
        }
        
        data_->ensure_unique();
        T* this_data = data_->get();
        const T* other_data = other.data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T product = this_data[i] * other_data[i];
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(product)) {
                    throw NumericalStabilityException("In-place multiplication resulted in non-finite value");
                }
            }
            
            this_data[i] = product;
        }
        
        return *this;
    }
    
    Tensor& operator/=(const Tensor& other) {
        if (shape_ != other.shape_) {
            throw DimensionMismatchException("Cannot divide tensors with different shapes");
        }
        
        data_->ensure_unique();
        T* this_data = data_->get();
        const T* other_data = other.data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            if (other_data[i] == T(0)) {
                throw InvalidOperation("Division by zero");
            }
            
            T quotient = this_data[i] / other_data[i];
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(quotient)) {
                    throw NumericalStabilityException("In-place division resulted in non-finite value");
                }
            }
            
            this_data[i] = quotient;
        }
        
        return *this;
    }

    // Scalar operations
    Tensor operator+(const T& scalar) const {
        Tensor result(shape_, layout_);
        T* result_data = result.data();
        const T* this_data = data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T sum = this_data[i] + scalar;
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(sum)) {
                    throw NumericalStabilityException("Scalar addition resulted in non-finite value");
                }
            }
            
            result_data[i] = sum;
        }
        
        return result;
    }
    
    Tensor operator-(const T& scalar) const {
        Tensor result(shape_, layout_);
        T* result_data = result.data();
        const T* this_data = data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T diff = this_data[i] - scalar;
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(diff)) {
                    throw NumericalStabilityException("Scalar subtraction resulted in non-finite value");
                }
            }
            
            result_data[i] = diff;
        }
        
        return result;
    }
    
    Tensor operator*(const T& scalar) const {
        Tensor result(shape_, layout_);
        T* result_data = result.data();
        const T* this_data = data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T product = this_data[i] * scalar;
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(product)) {
                    throw NumericalStabilityException("Scalar multiplication resulted in non-finite value");
                }
            }
            
            result_data[i] = product;
        }
        
        return result;
    }
    
    Tensor operator/(const T& scalar) const {
        if (scalar == T(0)) {
            throw InvalidOperation("Division by zero");
        }
        
        Tensor result(shape_, layout_);
        T* result_data = result.data();
        const T* this_data = data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T quotient = this_data[i] / scalar;
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(quotient)) {
                    throw NumericalStabilityException("Scalar division resulted in non-finite value");
                }
            }
            
            result_data[i] = quotient;
        }
        
        return result;
    }
    
    // In-place scalar operations
    Tensor& operator+=(const T& scalar) {
        data_->ensure_unique();
        T* this_data = data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T sum = this_data[i] + scalar;
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(sum)) {
                    throw NumericalStabilityException("In-place scalar addition resulted in non-finite value");
                }
            }
            
            this_data[i] = sum;
        }
        
        return *this;
    }
    
    Tensor& operator-=(const T& scalar) {
        data_->ensure_unique();
        T* this_data = data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T diff = this_data[i] - scalar;
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(diff)) {
                    throw NumericalStabilityException("In-place scalar subtraction resulted in non-finite value");
                }
            }
            
            this_data[i] = diff;
        }
        
        return *this;
    }
    
    Tensor& operator*=(const T& scalar) {
        data_->ensure_unique();
        T* this_data = data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T product = this_data[i] * scalar;
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(product)) {
                    throw NumericalStabilityException("In-place scalar multiplication resulted in non-finite value");
                }
            }
            
            this_data[i] = product;
        }
        
        return *this;
    }
    
    Tensor& operator/=(const T& scalar) {
        if (scalar == T(0)) {
            throw InvalidOperation("Division by zero");
        }
        
        data_->ensure_unique();
        T* this_data = data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T quotient = this_data[i] / scalar;
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(quotient)) {
                    throw NumericalStabilityException("In-place scalar division resulted in non-finite value");
                }
            }
            
            this_data[i] = quotient;
        }
        
        return *this;
    }

    // Reduction operations
    Tensor sum(int axis = -1) const {
        if (axis == -1) {
            // Sum all elements
            T total = T(0);
            const T* this_data = data_->get();
            
            for (size_t i = 0; i < size_; ++i) {
                total += this_data[i];
                
                // Check for numerical stability
                if constexpr (std::is_floating_point_v<T>) {
                    if (!std::isfinite(total)) {
                        throw NumericalStabilityException("Sum operation resulted in non-finite value");
                    }
                }
            }
            
            return Tensor<T>(total);
        }
        
        if (axis < 0 || static_cast<size_t>(axis) >= shape_.size()) {
            throw IndexOutOfBoundsException("Axis " + std::to_string(axis) + " is out of bounds for tensor with " + std::to_string(shape_.size()) + " dimensions");
        }
        
        // Create result shape by removing the specified axis
        std::vector<size_t> result_shape;
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i != static_cast<size_t>(axis)) {
                result_shape.push_back(shape_[i]);
            }
        }
        
        if (result_shape.empty()) {
            // If summing along the last remaining axis, return scalar
            T total = T(0);
            const T* this_data = data_->get();
            
            for (size_t i = 0; i < size_; ++i) {
                total += this_data[i];
                
                // Check for numerical stability
                if constexpr (std::is_floating_point_v<T>) {
                    if (!std::isfinite(total)) {
                        throw NumericalStabilityException("Sum operation resulted in non-finite value");
                    }
                }
            }
            
            return Tensor<T>(total);
        }
        
        Tensor result(result_shape, layout_);
        T* result_data = result.data();
        
        // Calculate the size of the axis being summed
        size_t axis_size = shape_[axis];
        size_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape_[i];
        }
        size_t inner_size = 1;
        for (size_t i = axis + 1; i < shape_.size(); ++i) {
            inner_size *= shape_[i];
        }
        
        // Perform the sum along the specified axis
        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                T sum_val = T(0);
                for (size_t ax = 0; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    sum_val += data_->get()[idx];
                    
                    // Check for numerical stability
                    if constexpr (std::is_floating_point_v<T>) {
                        if (!std::isfinite(sum_val)) {
                            throw NumericalStabilityException("Sum operation resulted in non-finite value");
                        }
                    }
                }
                result_data[outer * inner_size + inner] = sum_val;
            }
        }
        
        return result;
    }
    
    Tensor mean(int axis = -1) const {
        if (axis == -1) {
            // Mean of all elements
            T total = T(0);
            const T* this_data = data_->get();
            
            for (size_t i = 0; i < size_; ++i) {
                total += this_data[i];
                
                // Check for numerical stability
                if constexpr (std::is_floating_point_v<T>) {
                    if (!std::isfinite(total)) {
                        throw NumericalStabilityException("Mean operation resulted in non-finite value");
                    }
                }
            }
            
            T mean_val = total / static_cast<T>(size_);
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(mean_val)) {
                    throw NumericalStabilityException("Mean operation resulted in non-finite value");
                }
            }
            
            return Tensor<T>(mean_val);
        }
        
        // For specific axis, we can reuse sum and divide by axis size
        Tensor sum_result = sum(axis);
        if (axis >= 0 && static_cast<size_t>(axis) < shape_.size()) {
            T divisor = static_cast<T>(shape_[axis]);
            if (divisor == T(0)) {
                throw InvalidOperation("Cannot compute mean: axis size is zero");
            }
            
            return sum_result / divisor;
        }
        
        if (size_ == 0) {
            throw InvalidOperation("Cannot compute mean of empty tensor");
        }
        
        return sum_result / static_cast<T>(size_);
    }
    
    Tensor max(int axis = -1) const {
        if (axis == -1) {
            // Max of all elements
            if (size_ == 0) {
                throw InvalidOperation("Cannot find max of empty tensor");
            }
            
            T max_val = data_->get()[0];
            const T* this_data = data_->get();
            
            for (size_t i = 1; i < size_; ++i) {
                if (this_data[i] > max_val) {
                    max_val = this_data[i];
                }
            }
            
            return Tensor<T>(max_val);
        }
        
        if (axis < 0 || static_cast<size_t>(axis) >= shape_.size()) {
            throw IndexOutOfBoundsException("Axis " + std::to_string(axis) + " is out of bounds for tensor with " + std::to_string(shape_.size()) + " dimensions");
        }
        
        // Create result shape by removing the specified axis
        std::vector<size_t> result_shape;
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i != static_cast<size_t>(axis)) {
                result_shape.push_back(shape_[i]);
            }
        }
        
        if (result_shape.empty()) {
            // If max along the last remaining axis, return scalar
            if (size_ == 0) {
                throw InvalidOperation("Cannot find max of empty tensor");
            }
            
            T max_val = data_->get()[0];
            const T* this_data = data_->get();
            
            for (size_t i = 1; i < size_; ++i) {
                if (this_data[i] > max_val) {
                    max_val = this_data[i];
                }
            }
            
            return Tensor<T>(max_val);
        }
        
        Tensor result(result_shape, layout_);
        T* result_data = result.data();
        
        // Calculate the size of the axis being reduced
        size_t axis_size = shape_[axis];
        size_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape_[i];
        }
        size_t inner_size = 1;
        for (size_t i = axis + 1; i < shape_.size(); ++i) {
            inner_size *= shape_[i];
        }
        
        // Find the max along the specified axis
        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                T max_val = data_->get()[outer * axis_size * inner_size + 0 * inner_size + inner];
                for (size_t ax = 1; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    if (data_->get()[idx] > max_val) {
                        max_val = data_->get()[idx];
                    }
                }
                result_data[outer * inner_size + inner] = max_val;
            }
        }
        
        return result;
    }
    
    Tensor min(int axis = -1) const {
        if (axis == -1) {
            // Min of all elements
            if (size_ == 0) {
                throw InvalidOperation("Cannot find min of empty tensor");
            }
            
            T min_val = data_->get()[0];
            const T* this_data = data_->get();
            
            for (size_t i = 1; i < size_; ++i) {
                if (this_data[i] < min_val) {
                    min_val = this_data[i];
                }
            }
            
            return Tensor<T>(min_val);
        }
        
        if (axis < 0 || static_cast<size_t>(axis) >= shape_.size()) {
            throw IndexOutOfBoundsException("Axis " + std::to_string(axis) + " is out of bounds for tensor with " + std::to_string(shape_.size()) + " dimensions");
        }
        
        // Create result shape by removing the specified axis
        std::vector<size_t> result_shape;
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i != static_cast<size_t>(axis)) {
                result_shape.push_back(shape_[i]);
            }
        }
        
        if (result_shape.empty()) {
            // If min along the last remaining axis, return scalar
            if (size_ == 0) {
                throw InvalidOperation("Cannot find min of empty tensor");
            }
            
            T min_val = data_->get()[0];
            const T* this_data = data_->get();
            
            for (size_t i = 1; i < size_; ++i) {
                if (this_data[i] < min_val) {
                    min_val = this_data[i];
                }
            }
            
            return Tensor<T>(min_val);
        }
        
        Tensor result(result_shape, layout_);
        T* result_data = result.data();
        
        // Calculate the size of the axis being reduced
        size_t axis_size = shape_[axis];
        size_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape_[i];
        }
        size_t inner_size = 1;
        for (size_t i = axis + 1; i < shape_.size(); ++i) {
            inner_size *= shape_[i];
        }
        
        // Find the min along the specified axis
        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                T min_val = data_->get()[outer * axis_size * inner_size + 0 * inner_size + inner];
                for (size_t ax = 1; ax < axis_size; ++ax) {
                    size_t idx = outer * axis_size * inner_size + ax * inner_size + inner;
                    if (data_->get()[idx] < min_val) {
                        min_val = data_->get()[idx];
                    }
                }
                result_data[outer * inner_size + inner] = min_val;
            }
        }
        
        return result;
    }

    // Element-wise operations
    Tensor pow(T exponent) const {
        Tensor result(shape_, layout_);
        T* result_data = result.data();
        const T* this_data = data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            T power_val = std::pow(this_data[i], exponent);
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(power_val)) {
                    throw NumericalStabilityException("Power operation resulted in non-finite value");
                }
            }
            
            result_data[i] = power_val;
        }
        
        return result;
    }
    
    Tensor sqrt() const {
        Tensor result(shape_, layout_);
        T* result_data = result.data();
        const T* this_data = data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            if (this_data[i] < T(0)) {
                throw InvalidOperation("Cannot take square root of negative number");
            }
            T sqrt_val = std::sqrt(this_data[i]);
            
            // Check for numerical stability
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(sqrt_val)) {
                    throw NumericalStabilityException("Square root operation resulted in non-finite value");
                }
            }
            
            result_data[i] = sqrt_val;
        }
        
        return result;
    }
    
    Tensor abs() const {
        Tensor result(shape_, layout_);
        T* result_data = result.data();
        const T* this_data = data_->get();
        
        for (size_t i = 0; i < size_; ++i) {
            result_data[i] = (this_data[i] < T(0)) ? -this_data[i] : this_data[i];
        }
        
        return result;
    }
    
    // Numerically stable softmax
    Tensor softmax(int axis = -1) const {
        if (axis == -1) {
            // Apply softmax to all elements as a single vector
            Tensor result = *this; // Copy tensor
            T* result_data = result.data();
            
            NumericalStability<T>::softmax_inplace(result_data, size_);
            return result;
        } else {
            // Apply softmax along specified axis
            if (axis < 0 || static_cast<size_t>(axis) >= shape_.size()) {
                throw IndexOutOfBoundsException("Axis " + std::to_string(axis) + " is out of bounds for tensor with " + std::to_string(shape_.size()) + " dimensions");
            }
            
            // Calculate dimensions for the softmax operation
            size_t axis_size = shape_[axis];
            size_t outer_size = 1;
            for (int i = 0; i < axis; ++i) {
                outer_size *= shape_[i];
            }
            size_t inner_size = 1;
            for (size_t i = axis + 1; i < shape_.size(); ++i) {
                inner_size *= shape_[i];
            }
            
            Tensor result = *this; // Copy tensor
            T* result_data = result.data();
            
            // Apply softmax along the specified axis
            for (size_t outer = 0; outer < outer_size; ++outer) {
                for (size_t inner = 0; inner < inner_size; ++inner) {
                    // Create a temporary array for this specific slice along the axis
                    std::vector<T> temp_slice(axis_size);
                    for (size_t ax = 0; ax < axis_size; ++ax) {
                        temp_slice[ax] = result_data[outer * axis_size * inner_size + ax * inner_size + inner];
                    }
                    
                    // Apply softmax to the temporary slice
                    NumericalStability<T>::softmax_inplace(temp_slice.data(), axis_size);
                    
                    // Copy back to the result
                    for (size_t ax = 0; ax < axis_size; ++ax) {
                        result_data[outer * axis_size * inner_size + ax * inner_size + inner] = temp_slice[ax];
                    }
                }
            }
            
            return result;
        }
    }
    
    // Broadcasting operations (simplified version)
    Tensor broadcast_to(const std::vector<size_t>& target_shape) const {
        // Check if broadcasting is possible
        if (target_shape.size() < shape_.size()) {
            throw DimensionMismatchException("Target shape must have at least as many dimensions as current shape");
        }
        
        size_t ndim_diff = target_shape.size() - shape_.size();
        
        for (size_t i = 0; i < shape_.size(); ++i) {
            size_t target_idx = ndim_diff + i;
            if (shape_[i] != target_shape[target_idx] && shape_[i] != 1) {
                throw DimensionMismatchException("Cannot broadcast dimension " + std::to_string(shape_[i]) +
                                              " to " + std::to_string(target_shape[target_idx]));
            }
        }
        
        // Create a new tensor with the target shape
        Tensor result(target_shape, layout_);
        
        // For simplicity, implement a basic broadcasting by repeating values
        // A more efficient implementation would use strides to avoid copying
        std::vector<size_t> result_indices(target_shape.size(), 0);
        std::vector<size_t> source_indices(shape_.size(), 0);
        
        for (size_t linear_idx = 0; linear_idx < result.size(); ++linear_idx) {
            // Calculate multi-dimensional indices for result
            size_t temp_idx = linear_idx;
            for (int dim = result.ndim() - 1; dim >= 0; --dim) {
                result_indices[dim] = temp_idx % target_shape[dim];
                temp_idx /= target_shape[dim];
            }
            
            // Map result indices to source indices
            for (size_t dim = 0; dim < shape_.size(); ++dim) {
                size_t src_dim = result_indices[ndim_diff + dim];
                if (shape_[dim] == 1) {
                    source_indices[dim] = 0;  // Broadcast dimension, always use index 0
                } else {
                    source_indices[dim] = src_dim;
                }
            }
            
            result.data()[linear_idx] = (*this)(source_indices);
        }
        
        return result;
    }

    // Tensor view and slicing functionality
    class TensorView {
    private:
        const Tensor<T>& tensor_;
        std::vector<size_t> offsets_;
        std::vector<size_t> view_shape_;
        
    public:
        TensorView(const Tensor<T>& tensor, const std::vector<size_t>& offsets, const std::vector<size_t>& view_shape)
            : tensor_(tensor), offsets_(offsets), view_shape_(view_shape) {
            if (offsets.size() != tensor.ndim() || view_shape.size() != tensor.ndim()) {
                throw InvalidOperation("View dimensions must match tensor dimensions");
            }
            
            for (size_t i = 0; i < offsets.size(); ++i) {
                if (offsets[i] + view_shape[i] > tensor.shape()[i]) {
                    throw IndexOutOfBoundsException("View range exceeds tensor bounds at dimension " + std::to_string(i));
                }
            }
        }
        
        T& operator()(const std::vector<size_t>& indices) {
            if (indices.size() != view_shape_.size()) {
                throw DimensionMismatchException("Number of indices does not match view dimensions");
            }
            
            std::vector<size_t> tensor_indices;
            for (size_t i = 0; i < indices.size(); ++i) {
                if (indices[i] >= view_shape_[i]) {
                    throw IndexOutOfBoundsException("Index " + std::to_string(indices[i]) +
                                                  " is out of bounds for view dimension " + std::to_string(i));
                }
                tensor_indices.push_back(offsets_[i] + indices[i]);
            }
            
            return const_cast<Tensor<T>&>(tensor_)(tensor_indices);
        }
        
        const T& operator()(const std::vector<size_t>& indices) const {
            if (indices.size() != view_shape_.size()) {
                throw DimensionMismatchException("Number of indices does not match view dimensions");
            }
            
            std::vector<size_t> tensor_indices;
            for (size_t i = 0; i < indices.size(); ++i) {
                if (indices[i] >= view_shape_[i]) {
                    throw IndexOutOfBoundsException("Index " + std::to_string(indices[i]) +
                                                  " is out of bounds for view dimension " + std::to_string(i));
                }
                tensor_indices.push_back(offsets_[i] + indices[i]);
            }
            
            return tensor_(tensor_indices);
        }
        
        template<typename... Args>
        T& operator()(Args... args) {
            std::vector<size_t> indices = {static_cast<size_t>(args)...};
            return operator()(indices);
        }
        
        template<typename... Args>
        const T& operator()(Args... args) const {
            std::vector<size_t> indices = {static_cast<size_t>(args)...};
            return operator()(indices);
        }
        
        const std::vector<size_t>& shape() const { return view_shape_; }
        size_t size() const { return std::accumulate(view_shape_.begin(), view_shape_.end(), 1ULL, std::multiplies<size_t>()); }
        size_t ndim() const { return view_shape_.size(); }
    };
    
    // Helper function to copy data from view to tensor recursively
    void copy_view_to_tensor(const TensorView& view, Tensor<T>& result, std::vector<size_t> indices) const {
        if (indices.size() == view.ndim()) {
            result(indices) = view(indices);
            return;
        }
        
        for (size_t i = 0; i < view.shape()[indices.size()]; ++i) {
            std::vector<size_t> new_indices = indices;
            new_indices.push_back(i);
            copy_view_to_tensor(view, result, new_indices);
        }
    }
    
    // Create a view of a portion of the tensor
    TensorView view(const std::vector<std::pair<size_t, size_t>>& ranges) const {
        if (ranges.size() != shape_.size()) {
            throw DimensionMismatchException("Number of ranges must match tensor dimensions");
        }
        
        std::vector<size_t> offsets, view_shape;
        for (size_t i = 0; i < ranges.size(); ++i) {
            if (ranges[i].second <= ranges[i].first || ranges[i].first >= shape_[i]) {
                throw IndexOutOfBoundsException("Invalid range [" + std::to_string(ranges[i].first) +
                                              ", " + std::to_string(ranges[i].second) + ") for dimension " +
                                              std::to_string(i) + " with size " + std::to_string(shape_[i]));
            }
            if (ranges[i].second > shape_[i]) {
                throw IndexOutOfBoundsException("Range end " + std::to_string(ranges[i].second) +
                                              " exceeds dimension " + std::to_string(i) + " size " +
                                              std::to_string(shape_[i]));
            }
            
            offsets.push_back(ranges[i].first);
            view_shape.push_back(ranges[i].second - ranges[i].first);
        }
        
        return TensorView(*this, offsets, view_shape);
    }
    
    // Slice operator - creates a new tensor from a slice
    Tensor slice(const std::vector<std::pair<size_t, size_t>>& ranges) const {
        auto view_obj = view(ranges);
        Tensor result(view_obj.shape());
        
        // Copy data from view to new tensor
        copy_view_to_tensor(view_obj, result, std::vector<size_t>());
        
        return result;
    }

    // Print tensor
    void print() const {
        print_recursive(0, std::vector<size_t>());
    }
    
    // Check if tensor contains any invalid values (NaN or infinity)
    bool is_valid() const {
        return NumericalStability<T>::is_valid_tensor(data_->get(), size_);
    }
    
    // Clamp tensor values to a range
    Tensor clamp(T min_val, T max_val) const {
        Tensor result = *this;
        T* result_data = result.data();
        
        for (size_t i = 0; i < size_; ++i) {
            result_data[i] = NumericalStability<T>::clamp(result_data[i], min_val, max_val);
        }
        
        return result;
    }

private:
    void print_recursive(size_t dim, const std::vector<size_t>& indices) const {
        if (dim == shape_.size()) {
            std::cout << data_->get()[calculate_index(indices)] << " ";
            return;
        }
        
        std::cout << "[";
        for (size_t i = 0; i < shape_[dim]; ++i) {
            std::vector<size_t> new_indices = indices;
            new_indices.push_back(i);
            
            if (dim + 1 < shape_.size()) {
                print_recursive(dim + 1, new_indices);
            } else {
                std::cout << data_->get()[calculate_index(new_indices)];
                if (i < shape_[dim] - 1) std::cout << ", ";
            }
        }
        std::cout << "]";
        if (dim == 0) std::cout << std::endl;
    }
};

// Define type aliases for common tensor types
using TensorF = Tensor<float>;
using TensorD = Tensor<double>;
using TensorI = Tensor<int>;
using TensorL = Tensor<long>;
using TensorB = Tensor<bool>;

} // namespace dnn

#endif // TENSOR_IMPROVED_HPP