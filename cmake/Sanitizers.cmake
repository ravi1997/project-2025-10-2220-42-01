# Sanitizers.cmake - Configure sanitizer options

function(enable_sanitizers target_name)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(SANITIZERS "")

    if(DNN_ENABLE_SANITIZER_ADDRESS)
      list(APPEND SANITIZERS "address")
    endif()

    if(DNN_ENABLE_SANITIZER_LEAK)
      list(APPEND SANITIZERS "leak")
    endif()

    if(DNN_ENABLE_SANITIZER_UNDEFINED)
      list(APPEND SANITIZERS "undefined")
    endif()

    if(DNN_ENABLE_SANITIZER_THREAD)
      if("address" IN_LIST SANITIZERS OR "leak" IN_LIST SANITIZERS)
        message(WARNING "Thread sanitizer is not compatible with Address and Leak sanitizers")
      else()
        list(APPEND SANITIZERS "thread")
      endif()
    endif()

    if(DNN_ENABLE_SANITIZER_MEMORY AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      if("address" IN_LIST SANITIZERS OR "leak" IN_LIST SANITIZERS OR "thread" IN_LIST SANITIZERS)
        message(WARNING "Memory sanitizer is not compatible with Address, Leak, or Thread sanitizers")
      else()
        list(APPEND SANITIZERS "memory")
      endif()
    endif()

    list(JOIN SANITIZERS "," LIST_OF_SANITIZERS)

  endif()

  if(LIST_OF_SANITIZERS)
    if(NOT "${LIST_OF_SANITIZERS}" STREQUAL "")
      target_compile_options(${target_name} PRIVATE -fsanitize=${LIST_OF_SANITIZERS})
      target_link_options(${target_name} PRIVATE -fsanitize=${LIST_OF_SANITIZERS})
    endif()
  endif()

endfunction()