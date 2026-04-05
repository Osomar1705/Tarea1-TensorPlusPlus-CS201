#include <iostream>
#include "Tensor.h"
#include "Activations.h"

int main() {
    Tensor input = Tensor::random({1000, 20, 20}, -1.0, 1.0);
    Tensor viewed = input.view({1000, 400});

    Tensor W1 = Tensor::random({400, 100}, -0.5, 0.5);
    Tensor b1 = Tensor::random({1, 100}, -0.1, 0.1);

    Tensor z1 = matmul(viewed, W1);
    Tensor a1 = z1 + b1;

    ReLU relu;
    Tensor h1 = a1.apply(relu);

    Tensor W2 = Tensor::random({100, 10}, -0.5, 0.5);
    Tensor b2 = Tensor::random({1, 10}, -0.1, 0.1);

    Tensor z2 = matmul(h1, W2);
    Tensor a2 = z2 + b2;

    Sigmoid sigmoid;
    Tensor output = a2.apply(sigmoid);

    return 0;
}