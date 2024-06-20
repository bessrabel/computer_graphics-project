
cmake_minimum_required(VERSION 3.0)
set(OpenCV_DIR C:/opencv-4.5.2-mingw/)
find_package( OpenCV REQUIRED )

# 빌드 디렉토리 설정
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include/GLFW
    ${CMAKE_SOURCE_DIR}/include/GLAD
    ${CMAKE_SOURCE_DIR}/include/GLM
    ${CMAKE_SOURCE_DIR}/include/Eigen   
    ${CMAKE_SOURCE_DIR}/include 
    )

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
project(${ProjectId})

set (CMAKE_CXX_STANDARD 11)

# file(GLOB SOURCES  *.cpp)

add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/main.cpp ${CMAKE_SOURCE_DIR}/include/GLAD/glad.c )

# OpenGL 라이브러리 링크
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME}  OpenGL::GL)

# GLFW 라이브러리 링크
target_link_libraries(${PROJECT_NAME}  ${CMAKE_SOURCE_DIR}/lib/libglfw3dll.a)


target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})

# 추가 리소스 파일 복사
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/src/input.PNG
        ${CMAKE_SOURCE_DIR}/lib/glfw3.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_core452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_dnn452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_features2d452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_flann452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_gapi452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_highgui452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_imgcodecs452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_imgproc452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_ml452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_objdetect452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_photo452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_stitching452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_video452.dll
        ${CMAKE_SOURCE_DIR}/lib/libopencv_videoio452.dll
        ${CMAKE_SOURCE_DIR}/lib/opencv_videoio_ffmpeg452_64.dll
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# 빌드 결과물 설정
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
