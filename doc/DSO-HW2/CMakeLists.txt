SET(PROJECT_NAME DSO_Course_2)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# CMAKE_BUILD_TYPE
set(CMAKE_BUILD_TYPE Debug)

# setting path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(Sophus_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus/Install/lib/cmake/Sophus/)

# libraries
find_package(Eigen3 REQUIRED)
find_package(OpenCV QUIET)
find_package(Sophus REQUIRED)

message("Sophus :" ${Sophus_INCLUDE_DIR})

# gcc flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -std=c++11 -march=native -fopenmp")

if( CMAKE_BUILD_TYPE MATCHES "Debug" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -o0 -g")
else( CMAKE_BUILD_TYPE MATCHES "Release" )
# release模式下，最高优化；
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -O3")
endif( CMAKE_BUILD_TYPE MATCHES "Debug" )

# include
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories( ${OpenCV_INCLUDE_DIRS} 
                     ${EIGEN3_INCLUDE_DIR}
                     ${Sophus_INCLUDE_DIR})

set(PROJECT_SRC 
   ${PROJECT_SOURCE_DIR}/src/main.cpp
   ${PROJECT_SOURCE_DIR}/src/track.cpp
   ${PROJECT_SOURCE_DIR}/src/base.cpp)
   
# src                     
add_executable(${PROJECT_NAME} ${PROJECT_SRC} )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})