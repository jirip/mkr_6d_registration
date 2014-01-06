cmake_minimum_required(VERSION 2.8)

project(dift)

# libraries
find_package(OpenCV 2 REQUIRED)
find_package(PCL 1.3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(dift main.cpp)

target_link_libraries(dift ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS dift RUNTIME DESTINATION bin)
