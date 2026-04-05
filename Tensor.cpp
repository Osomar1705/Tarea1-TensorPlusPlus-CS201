#include "Tensor.h"
#include <random>

void Tensor::allocate(const size_t* shape_arr, size_t dims) {
    if (dims == 0 || dims > 3) throw std::invalid_argument("Invalid dimensions");
    num_dims = dims;
    shape = new size_t[num_dims];
    total_size = 1;
    for (size_t i = 0; i < num_dims; ++i) {
        shape[i] = shape_arr[i];
        total_size *= shape[i];
    }
    data = new double[total_size]();
}

void Tensor::copy_data(const double* values, size_t values_size) {
    for (size_t i = 0; i < total_size; ++i) {
        data[i] = (i < values_size) ? values[i] : 0.0;
    }
}

Tensor::Tensor(std::initializer_list<size_t> shape_list, std::initializer_list<double> values_list) {
    allocate(shape_list.begin(), shape_list.size());
    copy_data(values_list.begin(), values_list.size());
}

Tensor::Tensor(const size_t* shape_arr, size_t dims) {
    allocate(shape_arr, dims);
}

Tensor::Tensor(const Tensor& other) {
    allocate(other.shape, other.num_dims);
    copy_data(other.data, other.total_size);
}

Tensor::Tensor(Tensor&& other) noexcept : data(other.data), shape(other.shape), num_dims(other.num_dims), total_size(other.total_size) {
    other.data = nullptr;
    other.shape = nullptr;
    other.num_dims = 0;
    other.total_size = 0;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        delete[] data;
        delete[] shape;
        allocate(other.shape, other.num_dims);
        copy_data(other.data, other.total_size);
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        delete[] data;
        delete[] shape;
        data = other.data;
        shape = other.shape;
        num_dims = other.num_dims;
        total_size = other.total_size;
        other.data = nullptr;
        other.shape = nullptr;
        other.num_dims = 0;
        other.total_size = 0;
    }
    return *this;
}

Tensor::~Tensor() {
    delete[] data;
    delete[] shape;
}

Tensor Tensor::zeros(std::initializer_list<size_t> shape_list) {
    return Tensor(shape_list);
}

Tensor Tensor::ones(std::initializer_list<size_t> shape_list) {
    Tensor t(shape_list);
    for (size_t i = 0; i < t.total_size; ++i) t.data[i] = 1.0;
    return t;
}

Tensor Tensor::random(std::initializer_list<size_t> shape_list, double min, double max) {
    Tensor t(shape_list);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    for (size_t i = 0; i < t.total_size; ++i) t.data[i] = dis(gen);
    return t;
}

Tensor Tensor::arange(double start, double end) {
    size_t size = (end > start) ? static_cast<size_t>(end - start) : 0;
    Tensor t({size});
    for (size_t i = 0; i < size; ++i) t.data[i] = start + i;
    return t;
}

Tensor Tensor::view(std::initializer_list<size_t> new_shape) const {
    size_t new_total = 1;
    for (auto dim : new_shape) new_total *= dim;
    if (new_total != total_size) throw std::invalid_argument("Size mismatch in view");

    Tensor result(new_shape.begin(), new_shape.size());
    for (size_t i = 0; i < total_size; ++i) result.data[i] = data[i];
    return result;
}

Tensor Tensor::unsqueeze(size_t dim) const {
    if (num_dims >= 3) throw std::invalid_argument("Max 3 dimensions");
    if (dim > num_dims) throw std::invalid_argument("Invalid dimension index");

    size_t new_shape[3];
    for (size_t i = 0, j = 0; i <= num_dims; ++i) {
        if (i == dim) new_shape[i] = 1;
        else new_shape[i] = shape[j++];
    }

    Tensor result(new_shape, num_dims + 1);
    for (size_t i = 0; i < total_size; ++i) result.data[i] = data[i];
    return result;
}

Tensor Tensor::apply(const TensorTransform& transform) const {
    Tensor result(shape, num_dims);
    return transform.apply(*this);
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (total_size == other.total_size) {
        Tensor result(shape, num_dims);
        for (size_t i = 0; i < total_size; ++i) result.data[i] = data[i] + other.data[i];
        return std::move(result);
    }

    if (num_dims == 2 && other.num_dims == 2 && shape[1] == other.shape[1] && other.shape[0] == 1) {
        Tensor result(shape, num_dims);
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result.data[i * shape[1] + j] = data[i * shape[1] + j] + other.data[j];
            }
        }
        return std::move(result);
    }
    throw std::invalid_argument("Incompatible dimensions");
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (total_size != other.total_size) throw std::invalid_argument("Incompatible dimensions");
    Tensor result(shape, num_dims);
    for (size_t i = 0; i < total_size; ++i) result.data[i] = data[i] - other.data[i];
    return std::move(result);
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (total_size != other.total_size) throw std::invalid_argument("Incompatible dimensions");
    Tensor result(shape, num_dims);
    for (size_t i = 0; i < total_size; ++i) result.data[i] = data[i] * other.data[i];
    return std::move(result);
}

Tensor Tensor::operator*(double scalar) const {
    Tensor result(shape, num_dims);
    for (size_t i = 0; i < total_size; ++i) result.data[i] = data[i] * scalar;
    return std::move(result);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.num_dims != 2 || b.num_dims != 2 || a.shape[1] != b.shape[0]) {
        throw std::invalid_argument("Incompatible dimensions for matmul");
    }

    Tensor result({a.shape[0], b.shape[1]});
    for (size_t i = 0; i < a.shape[0]; ++i) {
        for (size_t j = 0; j < b.shape[1]; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < a.shape[1]; ++k) {
                sum += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
            }
            result.data[i * b.shape[1] + j] = sum;
        }
    }
    return std::move(result);
}

Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.total_size != b.total_size) throw std::invalid_argument("Incompatible dimensions for dot");
    double sum = 0.0;
    for (size_t i = 0; i < a.total_size; ++i) sum += a.data[i] * b.data[i];
    return Tensor({1}, {sum});
}

Tensor Tensor::concat(std::initializer_list<Tensor> tensors, size_t dim) {
    throw std::runtime_error("Concat fully implemented logic would go here depending on memory offsets.");
}