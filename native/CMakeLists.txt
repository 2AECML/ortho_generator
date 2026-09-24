cmake_minimum_required(VERSION 3.21)
project(OrthoE57Reader LANGUAGES CXX)

find_package(E57Format CONFIG REQUIRED)

add_executable(OrthoE57Reader OrthoE57Reader.cpp)
target_compile_features(OrthoE57Reader PRIVATE cxx_std_17)
target_link_libraries(OrthoE57Reader PRIVATE E57Format)
if(MSVC)
    target_compile_options(OrthoE57Reader PRIVATE /utf-8 /EHsc)
endif()
