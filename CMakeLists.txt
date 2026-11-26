cmake_minimum_required(VERSION 3.20)
project(keel_comm VERSION 0.1.0 LANGUAGES CXX)

option(KEEL_COMM_NO_EXCEPTIONS_RTTI "Compile keel_comm without C++ exceptions and RTTI" OFF)
option(KEEL_COMM_BUILD_TESTS "Build keel_comm unit tests and header purity checks" ${PROJECT_IS_TOP_LEVEL})

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
endif()
if(NOT TARGET keel::keel)
    include(FetchContent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../keel/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL "${CMAKE_CURRENT_SOURCE_DIR}/../keel" CACHE PATH "Local keel checkout")
    endif()
    if(KEEL_COMM_NO_EXCEPTIONS_RTTI)
        set(KEEL_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
        keel
        GIT_REPOSITORY https://github.com/keelcpp/keel.git
        GIT_TAG        4b37783925fe7af7a64daaff90f268df267fbbf3
    )
    FetchContent_MakeAvailable(keel)
endif()

if(NOT TARGET keel::fft)
    find_package(keel_fft QUIET CONFIG)
endif()
if(NOT TARGET keel::fft)
    include(FetchContent)
    set(KEEL_FFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(KEEL_COMM_NO_EXCEPTIONS_RTTI)
        set(KEEL_FFT_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../fft/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_FFT)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_FFT "${CMAKE_CURRENT_SOURCE_DIR}/../fft" CACHE PATH "Local keel_fft checkout")
    endif()
    FetchContent_Declare(
        keel_fft
        GIT_REPOSITORY https://github.com/keelcpp/fft.git
        GIT_TAG        739550655dfea621d6f666d189ce3bc0ed49efa5
    )
    FetchContent_MakeAvailable(keel_fft)
endif()

if(NOT TARGET keel::signal)
    find_package(keel_signal QUIET CONFIG)
endif()
if(NOT TARGET keel::signal)
    include(FetchContent)
    set(KEEL_SIGNAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    if(KEEL_COMM_NO_EXCEPTIONS_RTTI)
        set(KEEL_SIGNAL_NO_EXCEPTIONS_RTTI ON CACHE BOOL "" FORCE)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../signal/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL_SIGNAL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL_SIGNAL "${CMAKE_CURRENT_SOURCE_DIR}/../signal" CACHE PATH "Local keel_signal checkout")
    endif()
    FetchContent_Declare(
        keel_signal
        GIT_REPOSITORY https://github.com/keelcpp/signal.git
        GIT_TAG        [main 1f09ab4] fix(signal): align quadratic chirp phase sign and bandstop ellipord with reference engine  3 files changed, 33 insertions(+), 10 deletions(-) 1f09ab4b3b668c10c9ed7ec94adc514c91f0e354
    )
    FetchContent_MakeAvailable(keel_signal)
endif()

add_library(keel_comm STATIC
    src/modulation.cpp
    src/pulse_compand.cpp
    src/scrambler_errors.cpp
    src/block_coding.cpp
    src/conv_coding.cpp
    src/source_coding.cpp
    src/fsk_ofdm_apsk.cpp
    src/mil188_channel.cpp
)
add_library(keel::comm ALIAS keel_comm)

target_link_libraries(keel_comm PUBLIC keel::keel PRIVATE keel::fft keel::signal)

target_compile_features(keel_comm PUBLIC cxx_std_20)
set_target_properties(keel_comm PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(keel_comm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(keel_comm PRIVATE /W4 /utf-8)
    if(KEEL_COMM_NO_EXCEPTIONS_RTTI)
        string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        target_compile_options(keel_comm PUBLIC /EHs-c- /GR-)
        target_compile_definitions(keel_comm PUBLIC _HAS_EXCEPTIONS=0)
    endif()
else()
    target_compile_options(keel_comm PRIVATE -Wall -Wextra -Wpedantic)
    if(KEEL_COMM_NO_EXCEPTIONS_RTTI)
        target_compile_options(keel_comm PUBLIC -fno-exceptions -fno-rtti)
    endif()
endif()

if(COMMAND keel_enforce_guidelines)
    keel_enforce_guidelines(keel_comm
        SCOPE ALL
        BAN_RAW_INTS ALL
        BAN_RAW_FLOATS ALL
        HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include"
        STANDALONE_HEADERS ${KEEL_COMM_BUILD_TESTS}
        TEST_NAME test_guidelines_headers
        WARNINGS OFF
    )
endif()

include(CMakePackageConfigHelpers)
install(TARGETS keel_comm EXPORT keel_commTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/keel DESTINATION include)
install(EXPORT keel_commTargets
    FILE keel_commTargets.cmake
    NAMESPACE keel::
    DESTINATION lib/cmake/keel_comm
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/keel_commConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_commConfig.cmake"
    INSTALL_DESTINATION lib/cmake/keel_comm
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/keel_commConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/keel_commConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_commConfigVersion.cmake"
    DESTINATION lib/cmake/keel_comm
)

if(KEEL_COMM_BUILD_TESTS)
    enable_testing()

    add_executable(test_comm tests/test_comm.cpp)
    target_link_libraries(test_comm PRIVATE keel::comm)
    if(MSVC)
        target_compile_options(test_comm PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_comm PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if(COMMAND keel_enforce_guidelines)
        keel_enforce_guidelines(test_comm
            SCOPE ALL
            BAN_RAW_INTS ALL
            BAN_RAW_FLOATS ALL
            WARNINGS OFF
        )
    endif()
    add_test(NAME test_comm COMMAND test_comm)

    get_target_property(KEEL_CORE_INC_DIRS keel::keel INTERFACE_INCLUDE_DIRECTORIES)
    set(KEEL_CORE_INC_CLEAN "")
    foreach(IDIR IN LISTS KEEL_CORE_INC_DIRS)
        string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" IDIR_STRIPPED "${IDIR}")
        if(NOT IDIR_STRIPPED MATCHES "\\$<INSTALL_INTERFACE:" AND EXISTS "${IDIR_STRIPPED}")
            list(APPEND KEEL_CORE_INC_CLEAN "${IDIR_STRIPPED}")
        endif()
    endforeach()
    string(REPLACE ";" "|" KEEL_DEP_INC_JOINED "${KEEL_CORE_INC_CLEAN}")
    string(REPLACE ";" "|" KEEL_STD_INC_JOINED "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    add_test(
        NAME test_header_purity
        COMMAND ${CMAKE_COMMAND}
            "-DKEEL_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-DKEEL_DEP_INC_DIRS=${KEEL_DEP_INC_JOINED}"
            "-DCOMPILER_PATH=${CMAKE_CXX_COMPILER}"
            "-DCOMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
            "-DSTD_INC_DIRS=${KEEL_STD_INC_JOINED}"
            "-DCMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckHeaderPurity.cmake"
    )
endif()
