#include "include/dnn.hpp"
#include <iostream>

int main() {
    // Create a 2D tensor
    dnn::Matrix A({3, 4}, 1.0);  // 3x4 matrix filled with 1.0
    
    // Test tensor access with integer indices (this should no longer cause warnings)
    int r = 1, c = 2;
    A(c, r) = 5.0;  // This was causing the sign conversion warning before
    
    std::cout << "Tensor access with int indices works: A(2,1) = " << A(c, r) << std::endl;
    
    // Test with another access pattern
    dnn::Matrix B(2, 3);  // 2x3 matrix
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            B(i, j) = i * 3 + j;  // Fill with values
        }
    }
    
    std::cout << "Filled matrix B:" << std::endl;
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            std::cout << B(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}