cmake_minimum_required(VERSION 3.0)

set(ENV{TritonClientBuild_DIR} "/trt-client")

project(image_client)
set(CMAKE_CXX_STANDARD 17)

set(TritonClient_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonClient")
set(TritonCommon_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonCommon")
message(STATUS ${TritonClient_DIR})

find_package(OpenCV REQUIRED)
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/model_executables/densenet_client.cpp)
target_include_directories(
    ${PROJECT_NAME} 
    PRIVATE ${OpenCV_INCLUDE_DIRS} $ENV{TritonClientBuild_DIR}/include 
  )
target_link_directories(${PROJECT_NAME} PRIVATE $ENV{TritonClientBuild_DIR}/lib)
target_link_libraries(${PROJECT_NAME} 
  PRIVATE
  grpcclient
  httpclient
  ${OpenCV_LIBS}
)

