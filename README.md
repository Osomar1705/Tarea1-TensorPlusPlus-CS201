Librería de tensores en C++ implementada mediante el uso de punteros crudos para la gestión de memoria dinámica, inspirada en NumPy y PyTorch.

## Archivos incluidos
* `Tensor.h` / `Tensor.cpp`: Implementación principal de la clase Tensor y manejo de memoria.
* `TensorTransform.h`: Interfaz abstracta para aplicar polimorfismo.
* `Activations.h` / `Activations.cpp`: Implementación de funciones ReLU y Sigmoid.
* `main.cpp`: Implementación de la red neuronal de prueba.
* `CMakeLists.txt`: Archivo de configuración para compilar.

## Cómo compilar y ejecutar

Se requiere **CMake** (versión 3.31 o superior) y un compilador compatible con **C++20**.

**1. Compilar usando la terminal:**
Desde la carpeta principal del proyecto, ejecuta los siguientes comandos:

mkdir build
cd build
cmake ..
cmake --build .

**2. Ejecutar:**
En Windows:
./tarea1.exe

En Linux/Mac:
./tarea1
