#include "include/tensor.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace dnn;

void test_basic_operations() {
    std::cout << "Testing basic tensor operations..." << std::endl;
    
    // Test 1: Create tensors
    TensorF t1({2, 3}, 1.0f);  // 2x3 tensor filled with 1.0
    TensorF t2({2, 3}, 2.0f);  // 2x3 tensor filled with 2.0
    
    std::cout << "Created tensors t1 and t2" << std::endl;
    
    // Test 2: Addition
    TensorF t3 = t1 + t2;
    assert(t3(0, 0) == 3.0f);
    assert(t3(1, 2) == 3.0f);
    std::cout << "Addition test passed" << std::endl;
    
    // Test 3: Subtraction
    TensorF t4 = t2 - t1;
    assert(t4(0, 0) == 1.0f);
    std::cout << "Subtraction test passed" << std::endl;
    
    // Test 4: Multiplication
    TensorF t5 = t1 * t2;
    assert(t5(0, 0) == 2.0f);
    std::cout << "Multiplication test passed" << std::endl;
    
    // Test 5: Division
    TensorF t6 = t2 / t1;
    assert(t6(0, 0) == 2.0f);
    std::cout << "Division test passed" << std::endl;
    
    // Test 6: Scalar operations
    TensorF t7 = t1 + 5.0f;
    assert(t7(0, 0) == 6.0f);
    std::cout << "Scalar addition test passed" << std::endl;
    
    // Test 7: In-place operations
    TensorF t8 = t1;
    t8 += t2;
    assert(t8(0, 0) == 3.0f);
    std::cout << "In-place addition test passed" << std::endl;
}

void test_reductions() {
    std::cout << "Testing reduction operations..." << std::endl;
    
    // Create a 2x3 tensor with values 1,2,3,4,5,6
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    TensorF t({2, 3}, data);
    
    // Test sum of all elements
    TensorF sum_all = t.sum();
    assert(sum_all(0) == 21.0f);  // 1+2+3+4+5+6 = 21
    std::cout << "Sum all elements test passed" << std::endl;
    
    // Test sum along axis 0 (rows)
    TensorF sum_axis0 = t.sum(0);  // Should be [5, 7, 9] (1+4, 2+5, 3+6)
    assert(sum_axis0(0) == 5.0f);
    assert(sum_axis0(1) == 7.0f);
    assert(sum_axis0(2) == 9.0f);
    std::cout << "Sum along axis 0 test passed" << std::endl;
    
    // Test sum along axis 1 (cols)
    TensorF sum_axis1 = t.sum(1);  // Should be [6, 15] (1+2+3, 4+5+6)
    assert(sum_axis1(0) == 6.0f);
    assert(sum_axis1(1) == 15.0f);
    std::cout << "Sum along axis 1 test passed" << std::endl;
    
    // Test mean
    TensorF mean_all = t.mean();
    assert(mean_all(0) == 3.5f);  // 21/6 = 3.5
    std::cout << "Mean test passed" << std::endl;
    
    // Test max
    TensorF max_all = t.max();
    assert(max_all(0) == 6.0f);
    std::cout << "Max test passed" << std::endl;
    
    // Test min
    TensorF min_all = t.min();
    assert(min_all(0) == 1.0f);
    std::cout << "Min test passed" << std::endl;
}

void test_matrix_operations() {
    std::cout << "Testing matrix operations..." << std::endl;
    
    // Create 2x3 matrix
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    TensorF t1({2, 3}, data1);  // 2x3 matrix
    
    // Create 3x2 matrix 
    std::vector<float> data2 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    TensorF t2({3, 2}, data2);  // 3x2 matrix
    
    // Matrix multiplication: (2x3) * (3x2) = (2x2)
    TensorF result = t1.matmul(t2);
    
    // Expected result:
    // [1*1+2*3+3*5, 1*2+2*4+3*6]   [1+6+15, 2+8+18]   [22, 28]
    // [4*1+5*3+6*5, 4*2+5*4+6*6] = [4+15+30, 8+20+36] = [49, 64]
    assert(result(0, 0) == 22.0f);
    assert(result(0, 1) == 28.0f);
    assert(result(1, 0) == 49.0f);
    assert(result(1, 1) == 64.0f);
    std::cout << "Matrix multiplication test passed" << std::endl;
    
    // Test transpose
    TensorF t3({2, 3}, data1);
    TensorF transposed = t3.transpose();
    assert(transposed.shape()[0] == 3);
    assert(transposed.shape()[1] == 2);
    assert(transposed(0, 0) == t3(0, 0));  // [0,0] stays [0,0]
    assert(transposed(0, 1) == t3(1, 0));  // [1,0] becomes [0,1]
    std::cout << "Transpose test passed" << std::endl;
}

void test_views_and_slicing() {
    std::cout << "Testing views and slicing..." << std::endl;
    
    // Create a 3x4 tensor
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    TensorF t({3, 4}, data);
    
    // Create a view of the first 2 rows and columns 1-3
    std::vector<std::pair<size_t, size_t>> ranges = {{0, 2}, {1, 4}};  // rows [0,2), cols [1,4)
    auto view = t.view(ranges);
    
    assert(view.shape()[0] == 2);
    assert(view.shape()[1] == 3);
    assert(view(0, 0) == 2);  // t(0,1)
    assert(view(0, 1) == 3);  // t(0,2)
    assert(view(1, 2) == 8);  // t(1,3)
    std::cout << "View test passed" << std::endl;
    
    // Create a slice (copy) of the same region
    TensorF slice = t.slice(ranges);
    assert(slice.shape()[0] == 2);
    assert(slice.shape()[1] == 3);
    assert(slice(0, 0) == 2);  // t(0,1)
    assert(slice(0, 1) == 3);  // t(0,2)
    assert(slice(1, 2) == 8);  // t(1,3)
    std::cout << "Slice test passed" << std::endl;
}

void test_error_handling() {
    std::cout << "Testing error handling..." << std::endl;
    
    TensorF t1({2, 3});
    TensorF t2({3, 2});
    
    // Test dimension mismatch for addition
    try {
        TensorF result = t1 + t2;
        assert(false);  // Should not reach here
    } catch (const DimensionMismatchException& e) {
        std::cout << "Dimension mismatch caught correctly: " << e.what() << std::endl;
    }
    
    // Test index out of bounds
    try {
        float val = t1(5, 5);  // Out of bounds
        assert(false);  // Should not reach here
    } catch (const IndexOutOfBoundsException& e) {
        std::cout << "Index out of bounds caught correctly: " << e.what() << std::endl;
    }
    // Test invalid operation (matmul with incompatible dimensions)
    try {
        TensorF result = t1.matmul(t1);  // 2x3 * 2x3 is invalid
        assert(false);  // Should not reach here
    } catch (const InvalidOperation& e) {
        std::cout << "Invalid operation caught correctly: " << e.what() << std::endl;
    } catch (const DimensionMismatchException& e) {
        std::cout << "Dimension mismatch caught correctly: " << e.what() << std::endl;
    }
    
    std::cout << "Error handling tests passed" << std::endl;
}


void test_broadcasting() {
    std::cout << "Testing broadcasting..." << std::endl;
    
    // Create a 2x3 tensor
    TensorF t1({2, 3}, 2.0f);
    
    // Create a 1x3 tensor (will be broadcasted)
    TensorF t2({1, 3}, 3.0f);
    
    // This should broadcast t2 to match t1's shape
    TensorF result = t1 * t2;
    assert(result(0, 0) == 6.0f);  // 2 * 3
    assert(result(1, 2) == 6.0f);  // 2 * 3
    std::cout << "Broadcasting test passed" << std::endl;
}

int main() {
    std::cout << "Starting Tensor System Tests..." << std::endl;
    
    test_basic_operations();
    test_reductions();
    test_matrix_operations();
    test_views_and_slicing();
    test_error_handling();
    test_broadcasting();
    
    std::cout << "All tests passed successfully!" << std::endl;
    
    return 0;
}