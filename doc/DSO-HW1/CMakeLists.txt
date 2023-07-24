SET(PROJECT_NAME DSO_Course_1)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# CMAKE_BUILD_TYPE
set(BUILD_TYPE Debug)

# setting path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# libraries
find_package(Eigen3 REQUIRED)
find_package(OpenCV QUIET)

# gcc flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O0 -g -std=c++11 -march=native"
)

# include
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories( ${OpenCV_INCLUDE_DIRS} 
                     ${EIGEN3_INCLUDE_DIR})
   
# src 
set(PROJECT_SRC 
   ${PROJECT_SOURCE_DIR}/src/main.cpp
   ${PROJECT_SOURCE_DIR}/src/geometry.cpp
   ${PROJECT_SOURCE_DIR}/src/photometric.cpp)
                    
add_executable(${PROJECT_NAME} ${PROJECT_SRC} )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})