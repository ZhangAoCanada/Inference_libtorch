cmake_minimum_required(VERSION 3.0)
list(APPEND CMAKE_PREFIX_PATH "/mnt/f/DeepLearningPractice/libtorch")

project(nanodet_libtorch)
#set(INCLUDE_DIRECTORY ${PROJECT_SOURCE_DIR}/include)
set(SOURCE_DIRECTORY ${PROJECT_SOURCE_DIR}/src)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wall -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB source_files 
	${SOURCE_DIRECTORY}/nanodet.cpp
	${SOURCE_DIRECTORY}/nanodet.h
	${SOURCE_DIRECTORY}/draw.cpp
	${SOURCE_DIRECTORY}/draw.h
	${SOURCE_DIRECTORY}/timer.cpp
	${SOURCE_DIRECTORY}/timer.h
	${SOURCE_DIRECTORY}/main.cpp
	)

add_executable(nanodet_libtorch ${source_files})
target_link_libraries(nanodet_libtorch ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_property(TARGET nanodet_libtorch PROPERTY CXX_STANDARD 14)

