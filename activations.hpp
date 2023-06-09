#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cmath> // for std::exp
#include "matrix.hpp"

Matrix sigmoid(const Matrix& mat) {
    // Apply the sigmoid function to each element of the matrix
    MatrixData result = mat.getData();
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            result[i][j] = 1 / (1 + std::exp(-mat.getData()[i][j]));
        }
    }

    return Matrix(result);
}

Matrix relu(const Matrix& mat) {
    // Apply the relu function to each element of the matrix
    MatrixData result = mat.getData();
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            result[i][j] = std::max(0.0, mat.getData()[i][j]);
        }
    }

    return Matrix(result);
}

// More activation functions as needed...

#endif // ACTIVATIONS_H
