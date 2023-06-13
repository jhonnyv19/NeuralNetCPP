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
Matrix cross_entropy_loss(const Matrix& y, const Matrix& y_hat);
Matrix cross_entropy_loss_prime(const Matrix& y, const Matrix& y_hat);

#endif // ACTIVATIONS_HPP
