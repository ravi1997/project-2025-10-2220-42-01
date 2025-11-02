# Standard project settings for the DNN Library

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set minimum CMake version
cmake_minimum_required(VERSION 3.20)

# Make sure that global CMAKE_* variables are not used in the project
set(CMAKE_CXX_FLAGS_INIT "" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_INIT "" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_INIT "" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_INIT "" CACHE STRING "" FORCE)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Create a compile features property for C++20 with specific features
set(DNN_COMPILE_FEATURES
    cxx_std_20
    cxx_constexpr
    cxx_lambdas
    cxx_generic_lambdas
    cxx_variable_templates
    cxx_fold_expressions
    cxx_if_constexpr
    cxx_extended_constexpr
    cxx_relaxed_constexpr
)

# Define warning levels for different compilers
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(DNN_WARNING_FLAGS
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wsign-conversion
      -Wcast-align
      -Wcast-qual
      -Wctor-dtor-privacy
      -Wdisabled-optimization
      -Wformat=2
      -Winit-self
      -Wlogical-op
      -Wmissing-declarations
      -Wmissing-include-dirs
      -Wnoexcept
      -Wold-style-cast
      -Woverloaded-virtual
      -Wredundant-decls
      -Wshadow
      -Wstrict-aliasing
      -Wstrict-overflow=5
      -Wswitch-default
      -Wundef
      -Wno-unused
  )
elseif(MSVC)
  set(DNN_WARNING_FLAGS
      /W4
      /permissive-
  )
endif()

# Set warning flags as a cache variable that can be modified by users
set(DNN_WARNING_FLAGS
    ${DNN_WARNING_FLAGS}
    CACHE STRING "Warning flags used by the project")