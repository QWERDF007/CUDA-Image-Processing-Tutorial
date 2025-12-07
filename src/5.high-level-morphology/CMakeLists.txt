set(TARGET_NAME "high-level-morphology")

# 获取所有源文件的相对路径
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp *.cuh)


add_executable(${TARGET_NAME}
    ${SOURCES}
    ${HEADERS}
)

target_link_libraries(${TARGET_NAME} 
PUBLIC
    CUDA::cudart
    opencv_core
)