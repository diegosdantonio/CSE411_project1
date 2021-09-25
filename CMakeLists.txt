cmake_minimum_required (VERSION 2.8)
project("project1")

set(CPM_DOWNLOAD_VERSION 0.27.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})



add_executable(project1 gauss.cc)
find_package(TBB)
target_link_libraries(project1 TBB::tbb)

#target_link_libraries(project1 matplotlib-cpp)

find_package(PythonLibs 2.7)
target_include_directories(project1 PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(project1 ${PYTHON_LIBRARIES})

#find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/matplotlibcpp.h")

#target_include_directories(matplotlib-cpp/matplotlibcpp.h)

#

