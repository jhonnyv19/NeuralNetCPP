// activations.hpp

#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "matrix.hpp"

Matrix softmax(const Matrix& m);
Matrix relu(const Matrix& m);
Matrix softmax_prime(const Matrix& m);
Matrix relu_prime(const Matrix& m);
Matrix sigmoid_prime(const Matrix& m);
Matrix sigmoid(const Matrix& m);
double cross_entropy_loss(const Matrix& m1, const Matrix& m2);
Matrix cross_entropy_loss_prime(const Matrix& m1, const Matrix& m2);

#endif // ACTIVATIONS_HPP
