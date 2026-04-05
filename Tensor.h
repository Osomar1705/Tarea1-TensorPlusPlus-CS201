#ifndef TENSOR_H
#define TENSOR_H

#include <initializer_list>
#include <stdexcept>
#include "TensorTransform.h"

class Tensor {
private:
    double* data;
    size_t* shape;
    size_t num_dims;
    size_t total_size;

    void allocate(const size_t* shape_arr, size_t dims);
    void copy_data(const double* values, size_t values_size);

public:
    Tensor(std::initializer_list<size_t> shape_list, std::initializer_list<double> values_list = {});
    Tensor(const size_t* shape_arr, size_t dims);

    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();
    double* get_data() { return data; }
    const double* get_data() const { return data; }
    size_t get_total_size() const { return total_size; }

    static Tensor zeros(std::initializer_list<size_t> shape_list);
    static Tensor ones(std::initializer_list<size_t> shape_list);
    static Tensor random(std::initializer_list<size_t> shape_list, double min, double max);
    static Tensor arange(double start, double end);
    static Tensor concat(std::initializer_list<Tensor> tensors, size_t dim);

    Tensor view(std::initializer_list<size_t> new_shape) const;
    Tensor unsqueeze(size_t dim) const;
    Tensor apply(const TensorTransform& transform) const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);

    const size_t* get_shape() const { return shape; }
    size_t get_num_dims() const { return num_dims; }
};

#endif