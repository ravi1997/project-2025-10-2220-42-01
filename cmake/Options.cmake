# Compiler options for the DNN Library

# Define project options
option(DNN_BUILD_TESTS "Build unit tests" ON)
option(DNN_BUILD_EXAMPLES "Build examples" ON)
option(DNN_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(DNN_ENABLE_COVERAGE "Enable code coverage" OFF)
option(DNN_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(DNN_ENABLE_LTO "Enable Link Time Optimization" ON)
option(DNN_USE_CLANG_TIDY "Enable clang-tidy static analysis" OFF)
option(DNN_USE_CPPCHECK "Enable Cppcheck static analysis" OFF)
option(DNN_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
option(DNN_ENABLE_SANITIZER_UNDEFINED "Enable undefined behavior sanitizer" OFF)