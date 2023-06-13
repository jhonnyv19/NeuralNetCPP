#include "activations.hpp"
#include <cmath> // for std::exp
#include <algorithm> // for std::max
#include <iostream>

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

// Derivative of the sigmoid function with respect to the input
Matrix sigmoid_prime(const Matrix& mat) {
    // Apply the derivative of the sigmoid function to each element of the matrix
    MatrixData result = mat.getData();
    Matrix sig = sigmoid(mat);
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++)
            result[i][j] = sig.getData()[i][j] * (1 - sig.getData()[i][j]);
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

// Derivative of the relu function
Matrix relu_prime(const Matrix& mat) {
    // Apply the derivative of the relu function to each element of the matrix
    MatrixData result = mat.getData();
    Matrix rel = relu(mat);
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++)
            result[i][j] = rel.getData()[i][j] > 0 ? 1 : 0;
    }

    return Matrix(result);
}

// Softmax activation function 
Matrix softmax(const Matrix& mat) {
    // Apply the softmax function to each row of the matrix
    MatrixData result(mat.rows, std::vector<double>(mat.cols, 0.0));

    for (int i = 0; i < mat.rows; i++) {
        // Compute the sum of exp of every element in the row
        double sum = 0.0;
        for (int j = 0; j < mat.cols; j++) {
            result[i][j] = std::exp(mat.getData()[i][j]);
            sum += result[i][j];
        }

        // Divide every element in the row by the sum
        for (int j = 0; j < mat.cols; j++) {
            result[i][j] /= sum;
        }
    }

    return Matrix(result);
}


// Derivative of the softmax function
Matrix softmax_prime(const Matrix& mat) {
    // Apply the derivative of the softmax function to each element of the matrix
    MatrixData result = mat.getData();
    Matrix soft = softmax(mat);
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++)
            result[i][j] = soft.getData()[i][j] * (1 - soft.getData()[i][j]);
    }

    return Matrix(result);
}

// Multi class cross entropy loss function
Matrix cross_entropy_loss(const Matrix& y, const Matrix& y_hat) {
    // Check if dimensions match
    if (y.rows != y_hat.rows || y.cols != y_hat.cols) {
        throw std::invalid_argument("Dimensions of y and y_hat do not match.");
    }

    // Calculate the loss for each example in the batch
    MatrixData loss_data(y.rows, std::vector<double>(1, 0.0));
    for (int i = 0; i < y.rows; i++) {
        for (int j = 0; j < y.cols; j++) {
            loss_data[i][0] -= y.getData()[i][j] * std::log(y_hat.getData()[i][j]);
        }
    }

    return Matrix(loss_data);
}

// Derivative of the cross entropy loss function, assuming softmax is applied
Matrix cross_entropy_loss_prime(const Matrix& y, const Matrix& y_hat) {
    // Calculate the derivative of the loss function
    MatrixData result = y.getData();

    for (int i = 0; i < y.rows; i++) {
        for (int j = 0; j < y.cols; j++)
            result[i][j] = y_hat.getData()[i][j] - y.getData()[i][j];
    }

    return Matrix(result);

}

