project(mcl_gpu) # required cmake version 
cmake_minimum_required(VERSION 2.8) # packages 

find_package(Eigen3 REQUIRED)
find_package(kindr)
add_definitions('--std=c++11')

set(CUDA_NVCC_FLAGS -O3;-G;-g)  
#include_directories(${EIGEN3_INCLUDE_DIRS})
file(GLOB_RECURSE CURRENT_HEADERS *.h *.hpp) 
file(GLOB CURRENT_SOURCES *.cpp) 
source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 
cuda_add_library(mcl_gpu SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})

