#include "Activations.h"
#include <cmath>
#include <algorithm>

Tensor ReLU::apply(const Tensor& t) const {
    Tensor result = t; 
    
    double* data = result.get_data();
    size_t size = result.get_total_size();
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::max(0.0, data[i]);
    }
    
    return result;
}

Tensor Sigmoid::apply(const Tensor& t) const {
    Tensor result = t;
    
    double* data = result.get_data();
    size_t size = result.get_total_size();
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = 1.0 / (1.0 + std::exp(-data[i]));
    }
    
    return result;
}
