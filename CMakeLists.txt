cmake_minimum_required(VERSION 3.21)

# Project declaration with version and description
project(
    dnn_stdlib
    VERSION 1.0.0
    DESCRIPTION "A stdlib-only deep neural network library"
    LANGUAGES CXX
)

# Set C++ standard requirement
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable IPO (LTO) if the toolchain supports it
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
if(ipo_ok)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
else()
    message(WARNING "IPO/LTO not supported: ${ipo_msg}")
endif()

# Use proper toolchain-specific settings for LTO
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_program(GCC_AR NAMES gcc-ar)
    find_program(GCC_RANLIB NAMES gcc-ranlib)
    if(GCC_AR AND GCC_RANLIB)
        set(CMAKE_AR "${GCC_AR}")
        set(CMAKE_RANLIB "${GCC_RANLIB}")
    else()
        message(WARNING "gcc-ar/gcc-ranlib not found; disabling IPO to avoid LTO archive issues")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # For Clang, use llvm-ar and llvm-ranlib if available
    find_program(LLVM_AR NAMES llvm-ar)
    find_program(LLVM_RANLIB NAMES llvm-ranlib)
    if(LLVM_AR AND LLVM_RANLIB)
        set(CMAKE_AR "${LLVM_AR}")
        set(CMAKE_RANLIB "${LLVM_RANLIB}")
    endif()
endif()

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Create the main library with proper target structure
add_library(dnn src/dnn.cpp)

# Set library properties
set_target_properties(dnn PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Define interface include directories
target_include_directories(dnn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Define compile features instead of compiler-specific flags
target_compile_features(dnn
    PUBLIC
        cxx_std_23
)

# Add compiler-specific options with better cross-platform support
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(dnn
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>
            $<$<COMPILE_LANGUAGE:CXX>:-Wmisleading-indentation>
            $<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>
            $<$<COMPILE_LANGUAGE:CXX>:-Wold-style-cast>
            $<$<COMPILE_LANGUAGE:CXX>:-Wsign-conversion>
    )
elseif(MSVC)
    target_compile_options(dnn
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    )
endif()

# Create executable targets
add_executable(dnn_app apps/main.cpp)
set_target_properties(dnn_app PROPERTIES
    OUTPUT_NAME dnn
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

# Link the library to the executable
target_link_libraries(dnn_app PRIVATE dnn)

# Add example executables
add_executable(xor_example examples/xor_example.cpp)
set_target_properties(xor_example PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
target_link_libraries(xor_example PRIVATE dnn)

add_executable(mnist_example examples/mnist_example.cpp)
set_target_properties(mnist_example PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
target_link_libraries(mnist_example PRIVATE dnn)

# Create an interface library for common usage
add_library(dnn::dnn ALIAS dnn)

# Install rules (optional but good practice)
include(GNUInstallDirs)
install(TARGETS dnn
    EXPORT dnnTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Export targets for find_package compatibility
install(EXPORT dnnTargets
    FILE dnnConfig.cmake
    NAMESPACE dnn::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dnn
)

# Create a package config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/dnnConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dnnConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dnn
)

# Add testing support if tests exist
if(BUILD_TESTING)
    enable_testing()
    # Add test executables here if they exist
endif()
