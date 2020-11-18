cmake_minimum_required(VERSION 3.10)
project(guided_filtering)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(Halide_DIR "/home/rakhunzy/workspace/Halide-10.0.0/Halide-install/lib/cmake/Halide/")

set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
include("macro.cmake")

find_package(Halide REQUIRED)


link_libtiff()
find_package(ZLIB REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

set(src_files)
set(header_files)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR}/genfiles)

add_executable(guided_image_filter_generator src/guided_image_filter_generator.cpp)
target_link_libraries(guided_image_filter_generator PRIVATE Halide::Generator)
add_halide_library(guided_image_filter
    # GENERATOR_DEPS  # We don't have any yet
    FROM guided_image_filter_generator
    # GENERATOR_ARGS  # We don't have any yet
    FUNCTION_NAME guided_image_filter
    # HALIDE_TARGET ${HALIDE_TARGET}  # TODO: add option with custom HALIDE_TARGET
    # HALIDE_TARGET_FEATURES ${HALIDE_TARGET_FEATURES}  # TODO: add option with custom HALIDE_TARGET
    # EXTRA_OUTPUTS "stmt;html;schedule") # uncomment for extra output
)

add_executable(smoothing bin/smoothing.cpp ${src_files})
add_dependencies(smoothing guided_image_filter)
target_link_libraries(smoothing guided_image_filter Halide::Halide Halide::Tools Halide::ImageIO png ${TIFF_LIBRARIES} ${TIFFXX_LIBRARY})
