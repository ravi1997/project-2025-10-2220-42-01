#include <array>
#include <stdexcept>
#include <vector>

// Minimal reproduction of the Tensor class with only the relevant parts
template<std::size_t NumDims = 2>
struct Tensor {
    std::array<std::size_t, NumDims> shape;
    std::vector<double> data;
    std::size_t size;
    
    Tensor() : size(0) {
        std::fill(shape.begin(), shape.end(), 0);
    }
    
    explicit Tensor(const std::array<std::size_t, NumDims>& s) : shape(s) {
        size = 1;
        for (std::size_t dim : shape) {
            size *= dim;
        }
        data.resize(size, 0.0);
    }
    
    template<typename... Args>
    requires (sizeof...(Args) == NumDims)
    double& operator()(Args... indices) {
        static_assert(sizeof...(Args) == NumDims, "Number of indices must match tensor dimensions");
        std::array<std::size_t, NumDims> idx = {static_cast<std::size_t>(static_cast<long unsigned int>(indices))...};
        std::size_t linear_idx = 0;
        std::size_t multiplier = 1;
        
        // Process dimensions from last to first (row-major order), avoiding signed/unsigned comparison
        for (std::size_t i = 0; i < NumDims; ++i) {
            std::size_t dim = NumDims - 1 - i;
            if (idx[dim] >= shape[dim]) {
                throw std::out_of_range("Tensor access out of bounds");
            }
            linear_idx += idx[dim] * multiplier;
            multiplier *= shape[dim];
        }
        
        return data[linear_idx];
    }
    
    template<typename... Args>
    requires (sizeof...(Args) == NumDims)
    double operator()(Args... indices) const {
        static_assert(sizeof...(Args) == NumDims, "Number of indices must match tensor dimensions");
        std::array<std::size_t, NumDims> idx = {static_cast<std::size_t>(static_cast<long unsigned int>(indices))...};
        std::size_t linear_idx = 0;
        std::size_t multiplier = 1;
        
        // Process dimensions from last to first (row-major order), avoiding signed/unsigned comparison
        for (std::size_t i = 0; i < NumDims; ++i) {
            std::size_t dim = NumDims - 1 - i;
            if (idx[dim] >= shape[dim]) {
                throw std::out_of_range("Tensor access out of bounds");
            }
            linear_idx += idx[dim] * multiplier;
            multiplier *= shape[dim];
        }
        
        return data[linear_idx];
    }
};

using Matrix = Tensor<2>;

int main() {
    // Test tensor access with integer indices (this should no longer cause warnings)
    Matrix A({3, 4});  // 3x4 matrix
    
    int r = 1, c = 2;
    A(c, r) = 5.0;  // This was causing the sign conversion warning before
    
    // Test with another access pattern
    Matrix B({2, 3});  // 2x3 matrix
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            B(i, j) = i * 3 + j;  // Fill with values
        }
    }
    
    return 0;
}