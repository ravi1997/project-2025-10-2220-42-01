#include "../include/tensor.hpp"

namespace dnn {

// Define trait names - these are specializations of template struct
template<>
const char* DataTypeTrait<float>::name = "float";

template<>
const char* DataTypeTrait<double>::name = "double";

template<>
const char* DataTypeTrait<int>::name = "int";

template<>
const char* DataTypeTrait<long>::name = "long";

template<>
const char* DataTypeTrait<bool>::name = "bool";

} // namespace dnn

// Explicit template instantiations to ensure the linker can find the implementations
// These are already defined in layers.cpp, so we don't need to instantiate them here
// template class Tensor<float>;
// template class Tensor<double>;
// template class Tensor<int>;
// template class Tensor<long>;
// template class Tensor<bool>;