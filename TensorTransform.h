#ifndef TENSORTRANSFORM_H
#define TENSORTRANSFORM_H

class Tensor;

class TensorTransform {
public:
    virtual ~TensorTransform() = default;
    virtual Tensor apply(const Tensor& t) const = 0;
};

#endif