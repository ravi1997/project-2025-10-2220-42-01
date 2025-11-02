# CMake Configuration for DNN Library

This document outlines the comprehensive CMake build system for the DNN library, designed to follow industrial-grade standards and best practices.

## Features

- Modern CMake (3.20+) with proper target-based configuration
- Support for both static and shared library builds
- Comprehensive testing infrastructure
- Package export and installation support
- Cross-platform compatibility
- Multiple build configurations (Debug, Release, RelWithDebInfo, MinSizeRel)
- Dependency management
- Code quality tools integration

## Structure

```
cmake/
├── README.md
├── CompilerWarnings.cmake
├── Conan.cmake
├── Dependencies.cmake
├── Doxygen.cmake
├── Options.cmake
├── Sanitizers.cmake
├── StandardProjectSettings.cmake
└── Utils.cmake
```

## Configuration Options

The build system supports various options that can be configured via CMAKE variables:

- `DNN_BUILD_TESTS`: Build unit tests (default: ON)
- `DNN_BUILD_EXAMPLES`: Build examples (default: ON)
- `DNN_BUILD_BENCHMARKS`: Build benchmarks (default: OFF)
- `DNN_ENABLE_COVERAGE`: Enable code coverage (default: OFF)
- `DNN_WARNINGS_AS_ERRORS`: Treat warnings as errors (default: OFF)
- `DNN_ENABLE_LTO`: Enable Link Time Optimization (default: ON for Release builds)
- `DNN_USE_CLANG_TIDY`: Enable clang-tidy static analysis (default: OFF)
- `DNN_USE_CPPCHECK`: Enable Cppcheck static analysis (default: OFF)