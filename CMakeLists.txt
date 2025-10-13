cmake_minimum_required(VERSION 3.15)

project(CUDA_Image_Processing_Tutorial
        LANGUAGES C CXX
        VERSION 0.0.1
)

enable_language(CUDA)


# 配置构建树
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )

include(ConfigBuildTree)
include(ConfigCompiler)
include(ConfigOpenCV)
include(ConfigCUDA)

add_subdirectory(src)

message(STATUS "")
include(PrintConfig)