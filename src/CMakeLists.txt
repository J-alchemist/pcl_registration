cmake_minimum_required(VERSION 3.5)

add_compile_options(-std=c++11)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories( ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/inc )
link_directories( ${PCL_LIBRARY_DIRS} ${EIGEN3_LIBRARIES} )

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
	
message(STATUS "binary dir: " ${PROJECT_BINARY_DIR})
message(STATUS "source dir: " ${PROJECT_SOURCE_DIR})

set(SRC_LIST   registration_points.cpp)
add_executable(registration_points   ${SRC_LIST})
target_link_libraries(registration_points   ${PCL_LIBRARIES}  ${EIGEN3_LIBRARIES})

