cmake_minimum_required(VERSION 3.31)
project(tarea1)

set(CMAKE_CXX_STANDARD 20)

add_executable(tarea1
        main.cpp
        Tensor.cpp
        Tensor.h
        TensorTransform.h
        Activations.h
        Activations.cpp
)