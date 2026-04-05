#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "TensorTransform.h"
#include "Tensor.h"

class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

#endif