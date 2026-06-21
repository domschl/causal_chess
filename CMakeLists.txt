cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(causal-chess-cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure LibTorch search path (Homebrew location for pytorch on Apple Silicon)
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/Cellar/pytorch/2.12.1/libexec/lib/python3.14/site-packages/torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Executable targets
add_executable(causal-chess-cpp
    src/main.cpp
    src/encoding.cpp
    src/model.cpp
    src/search.cpp
    src/play.cpp
)

target_include_directories(causal-chess-cpp PRIVATE 
    src
    src/third_party
)

target_link_libraries(causal-chess-cpp PRIVATE "${TORCH_LIBRARIES}")

# Test executable target
add_executable(causal-chess-tests
    tests/test_all.cpp
    src/encoding.cpp
    src/model.cpp
    src/search.cpp
    src/play.cpp
)

target_include_directories(causal-chess-tests PRIVATE 
    src
    src/third_party
)

target_link_libraries(causal-chess-tests PRIVATE "${TORCH_LIBRARIES}")
