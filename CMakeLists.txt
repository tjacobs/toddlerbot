cmake_minimum_required(VERSION 3.15)
project(toddlerbot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug or Release)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# ==================== Include Directories ====================
include_directories(${CMAKE_SOURCE_DIR}/toddlerbot/actuation/include)
include_directories(${CMAKE_SOURCE_DIR}/toddlerbot/sim/include)

# ==================== Dynamixel SDK ====================
file(GLOB SDK_SRCS ${CMAKE_SOURCE_DIR}/toddlerbot/actuation/src/dynamixel_sdk/*.cpp)

add_library(dynamixel_sdk STATIC ${SDK_SRCS})
set_target_properties(dynamixel_sdk PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(dynamixel_sdk PUBLIC ${CMAKE_SOURCE_DIR}/toddlerbot/actuation/include/dynamixel_sdk)

# ==================== Dynamixel Control ====================
file(GLOB CTRL_SRCS ${CMAKE_SOURCE_DIR}/toddlerbot/actuation/src/dynamixel_control/*.cpp)

add_library(dynamixel_control STATIC ${CTRL_SRCS})
set_target_properties(dynamixel_control PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(dynamixel_control PUBLIC ${CMAKE_SOURCE_DIR}/toddlerbot/actuation/include/dynamixel_control)
target_link_libraries(dynamixel_control PUBLIC dynamixel_sdk)

# ==================== Python Binding Module ====================
find_package(pybind11 REQUIRED)

pybind11_add_module(dynamixel_cpp
  toddlerbot/actuation/src/dynamixel_mch.cpp
)
target_link_libraries(dynamixel_cpp PRIVATE dynamixel_control)
set_target_properties(dynamixel_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/toddlerbot/actuation
)