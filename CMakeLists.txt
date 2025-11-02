cmake_minimum_required(VERSION 3.20)

# Set project name and language
project(DNNLibrary 
    VERSION 1.0.0
    DESCRIPTION "Industrial-grade Deep Neural Network Library"
    HOMEPAGE_URL "https://github.com/your-organization/dnn-library"
    LANGUAGES CXX)

# Include CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Include project settings and options
include(StandardProjectSettings)
include(Options)
include(CompilerWarnings)
include(Sanitizers)
include(Dependencies)

# Setup dependencies
setup_dependencies()

# Define library targets
add_library(dnn INTERFACE)
add_library(dnn::dnn ALIAS dnn)

# Define the static and shared libraries
set(SOURCES
    src/tensor_improved.cpp
    src/tensor_ops.cpp
    src/dnn.cpp
    src/layers.cpp
    src/optimizers.cpp
    src/model_serializer.cpp
    src/dnn_improved.cpp
)

# Static library
add_library(dnn_static STATIC ${SOURCES})
set_target_properties(dnn_static PROPERTIES
    OUTPUT_NAME "dnn"
    POSITION_INDEPENDENT_CODE ON
)

# Shared library
add_library(dnn_shared SHARED ${SOURCES})
set_target_properties(dnn_shared PROPERTIES
    OUTPUT_NAME "dnn"
    SOVERSION 1
    VERSION ${PROJECT_VERSION}
    POSITION_INDEPENDENT_CODE ON
)

# Set up common properties for both libraries
set_target_properties(dnn_static dnn_shared PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Add include directories
target_include_directories(dnn INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(dnn_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(dnn_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link libraries
target_link_libraries(dnn_static PUBLIC Threads::Threads)
target_link_libraries(dnn_shared PUBLIC Threads::Threads)

# Apply compiler warnings
set_project_warnings(dnn_static)
set_project_warnings(dnn_shared)

# Apply sanitizers if enabled
enable_sanitizers(dnn_static)
enable_sanitizers(dnn_shared)

# Apply LTO if enabled for Release builds
if(DNN_ENABLE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(dnn_static dnn_shared PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION ON
    )
endif()

# Add compiler-specific flags for optimization
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(dnn_static PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(dnn_shared PRIVATE -O3 -march=native -ffast-math)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(dnn_static PRIVATE /O2 /arch:AVX2)
    target_compile_options(dnn_shared PRIVATE /O2 /arch:AVX2)
endif()

# Add examples if enabled
if(DNN_BUILD_EXAMPLES)
    add_executable(xor_example examples/xor_example.cpp)
    target_link_libraries(xor_example PRIVATE dnn_shared)

    add_executable(mnist_example examples/mnist_example.cpp)
    target_link_libraries(mnist_example PRIVATE dnn_shared)

    add_executable(layer_example examples/layer_example.cpp)
    target_link_libraries(layer_example PRIVATE dnn_shared)

    # Main application
    add_executable(main apps/main.cpp)
    target_link_libraries(main PRIVATE dnn_shared)
endif()

# Add test executable if enabled
if(DNN_BUILD_TESTS)
    enable_testing()
    
    add_executable(test_model_persistence test_model_persistence.cpp)
    target_link_libraries(test_model_persistence PRIVATE dnn_shared)
    
    # Add test
    add_test(NAME model_persistence_test COMMAND test_model_persistence)
endif()

# Add benchmarks if enabled
if(DNN_BUILD_BENCHMARKS)
    add_executable(benchmark_dnn examples/xor_example.cpp)
    target_link_libraries(benchmark_dnn PRIVATE dnn_shared)
    set_target_properties(benchmark_dnn PROPERTIES COMPILE_DEFINITIONS "DNN_BENCHMARK_MODE")
endif()

# Code coverage setup
if(DNN_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(dnn_static PRIVATE --coverage)
        target_link_options(dnn_static PRIVATE --coverage)
        target_compile_options(dnn_shared PRIVATE --coverage)
        target_link_options(dnn_shared PRIVATE --coverage)
    endif()
endif()

# Clang-tidy setup
if(DNN_USE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES "clang-tidy" REQUIRED)
    if(CLANG_TIDY_EXE)
        set(CLANG_TIDY_CHECKS "-*,performance-*,bugprone-*,readability-*,portability-*,modernize-*,-modernize-use-trailing-return-type")
        set_target_properties(dnn_static dnn_shared PROPERTIES
            CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=${CLANG_TIDY_CHECKS};-header-filter=.*"
        )
    endif()
endif()

# Cppcheck setup
if(DNN_USE_CPPCHECK)
    find_program(CPPCHECK_EXE NAMES "cppcheck" REQUIRED)
    if(CPPCHECK_EXE)
        set(CPPCHECK_OPTIONS 
            --enable=warning,performance,portability,information
            --std=c++20
            --verbose
            --check-library
            --inline-suppr
        )
        set_target_properties(dnn_static dnn_shared PROPERTIES
            CXX_CPPCHECK "${CPPCHECK_EXE};${CPPCHECK_OPTIONS}"
        )
    endif()
endif()

# Install targets
include(GNUInstallDirs)

# Install library files
install(TARGETS dnn_static dnn_shared
    EXPORT DNNLibraryTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install header files
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dnn
    FILES_MATCHING PATTERN "*.hpp"
)

# Install CMake package files
install(EXPORT DNNLibraryTargets
    FILE DNNLibraryTargets.cmake
    NAMESPACE DNN::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DNNLibrary
)

# Create package config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DNNLibraryConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install package config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/DNNLibraryConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DNNLibrary
)

# Create and install pkg-config file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DNNLibrary.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/DNNLibrary.pc"
    @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DNNLibrary.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# Export targets for build tree
export(EXPORT DNNLibraryTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/DNNLibraryTargets.cmake"
    NAMESPACE DNN::
)

# CPack support for packaging
set(CPACK_PACKAGE_NAME "DNNLibrary")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Industrial-grade Deep Neural Network Library")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
include(CPack)