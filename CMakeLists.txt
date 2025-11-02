cmake_minimum_required(VERSION 3.10)
project(DNNLibrary)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/tensor.cpp
    src/dnn.cpp
    src/layers.cpp
    src/optimizers.cpp
    src/model_serializer.cpp
)

# Header files
set(HEADERS
    include/tensor.hpp
    include/dnn.hpp
    include/layers.hpp
    include/utils.hpp
    include/optimizers.hpp
)

# Create the library
add_library(dnn SHARED ${SOURCES} ${HEADERS})
target_link_libraries(dnn ${CMAKE_THREAD_LIBS_INIT})

# Create a static library as well
add_library(dnn_static STATIC ${SOURCES} ${HEADERS})
target_link_libraries(dnn_static ${CMAKE_THREAD_LIBS_INIT})

add_executable(xor_example examples/xor_example.cpp)
# Example applications
target_link_libraries(xor_example dnn)

add_executable(mnist_example examples/mnist_example.cpp)
target_link_libraries(mnist_example dnn)

# Note: layer_example has compilation issues, so we'll skip it for now
# add_executable(layer_example examples/layer_example.cpp)
# target_link_libraries(layer_example dnn)

# Main application
add_executable(main apps/main.cpp)
target_link_libraries(main dnn)

# Test application
add_executable(test_model_persistence test_model_persistence.cpp)
target_link_libraries(test_model_persistence dnn)


# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(dnn PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(dnn_static PRIVATE -O3 -march=native -ffast-math)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(dnn PRIVATE /O2 /arch:AVX2)
    target_compile_options(dnn_static PRIVATE /O2 /arch:AVX2)
endif()

# Install targets
install(TARGETS dnn dnn_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")