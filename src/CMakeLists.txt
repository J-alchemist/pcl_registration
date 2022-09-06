cmake_minimum_required(VERSION 3.5)

add_compile_options(-std=c++11)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread) 

include_directories( ${PCL_INCLUDE_DIRS} 
			${EIGEN3_INCLUDE_DIR}
  			${Boost_INCLUDE_DIR}
 			${PROJECT_SOURCE_DIR}/inc )

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/lib)

message(STATUS "binary dir: " ${PROJECT_BINARY_DIR})
message(STATUS "source dir: " ${PROJECT_SOURCE_DIR})

add_library(voxel_filter STATIC
    voxel_filter.cpp
)
target_include_directories(voxel_filter  PUBLIC
    ${PROJECT_SOURCE_DIR}/inc
)
target_link_libraries(voxel_filter
 ${PCL_LIBRARIES}  
 ${EIGEN3_LIBRARIES} 
 ${Boost_LIBRARIES} 
)

set(SRC_LIST   registration_points.cpp)
add_executable(registration_points   ${SRC_LIST})
target_link_libraries(registration_points  
			voxel_filter )

add_executable(fpfh   fpfh.cpp)
target_link_libraries(fpfh  
			voxel_filter)




