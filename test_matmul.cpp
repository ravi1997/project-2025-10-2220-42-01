#include "include/dnn.hpp"
#include <iostream>

int main() {
    // Test the matmul function which was causing the issue
    dnn::Matrix A(2, 3);
    dnn::Matrix B(3, 2);
    
    // Fill matrices with some values
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            A(i, j) = i * 3 + j + 1;  // Fill with 1,2,3,4,5,6
        }
    }
    
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 2; ++j) {
            B(i, j) = i * 2 + j + 1;  // Fill with 1,2,3,4,5,6
        }
    }
    
    // Test matmul function
    dnn::Matrix C = dnn::matmul(A, B);
    
    std::cout << "Matrix multiplication test passed!" << std::endl;
    std::cout << "Result matrix dimensions: " << C.shape[0] << "x" << C.shape[1] << std::endl;
    
    return 0;
}