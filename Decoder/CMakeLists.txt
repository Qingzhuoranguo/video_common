cmake_minimum_required(VERSION 3.16)

project(decoders LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
    gstreamer-gl-1.0
    gstreamer-gl-x11-1.0
)

find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)

add_library(decoders STATIC
    decoders.cpp

)

target_include_directories(decoders
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GST_INCLUDE_DIRS}
)

target_compile_options(decoders PRIVATE
    ${GST_CFLAGS_OTHER}
)

target_link_directories(decoders PRIVATE
    ${GST_LIBRARY_DIRS}
)

target_link_libraries(decoders PUBLIC
    ${GST_LIBRARIES}
    OpenGL::GL
    X11
)

target_compile_definitions(decoders PRIVATE
    GST_USE_UNSTABLE_API
)