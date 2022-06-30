cmake_minimum_required(VERSION 3.17)
project(StreamUnwrappedImages)

set(CMAKE_CXX_STANDARD 14)

# 设置jsoncpp路径
set(Jsoncpp_INCLUDE_PATH "/usr/include/jsoncpp")
set(Jsoncpp_LIBRARY "/usr/lib/x86_64-linux-gnu/libjsoncpp.so")

find_package(OpenCV 3.2 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(imageWrapper src/ImageWrapper.cpp)
target_link_libraries(imageWrapper ${OpenCV_LIBS})

add_executable(streamCameraVideo ${PROJECT_SOURCE_DIR}/src/streamCameraVideo.cpp)
target_link_libraries(streamCameraVideo ${OpenCV_LIBS})

add_executable(StreamUnwrappedImages ${PROJECT_SOURCE_DIR}/src/StreamUnwarppedImages.cpp)
target_link_libraries(StreamUnwrappedImages ${Jsoncpp_LIBRARY} ${OpenCV_LIBS} imageWrapper)
