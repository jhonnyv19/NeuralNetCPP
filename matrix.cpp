// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the declaration of the Matrix class, which represents a matrix.

#include <matrix.hpp>
#include <random>

bool dimensionsMatchElementWise(const Matrix &m1, const Matrix &m2) {
    return m1.rows == m2.rows && m1.cols == m2.cols;
}

bool dimensionsMatchDot(const Matrix &m1, const Matrix &m2) {
    return m1.cols == m2.rows;
}

Matrix::Matrix(int rows, int cols) {
    // Initialize the data
    data = MatrixData(rows, Vector(cols, 0.0));

    // Initialize the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // Fill the matrix with random values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            data[i][j] = dis(gen);
    }

    // Set the data
    this->data = data;

    // Set the dimensions
    this->rows = rows;
    this->cols = cols;
}

Matrix::Matrix(MatrixData data) {
    // Set the data
    this->data = data;

    // Set the dimensions
    this->rows = data.size();
    this->cols = data[0].size();
}

Matrix::Matrix(const Matrix &m) {
    // Set the data
    this->data = m.data;

    // Set the dimensions
    this->rows = m.rows;
    this->cols = m.cols;
}

Matrix::MatrixData Matrix::getData() const {
    return data;
}

Matrix Matrix::add(const Matrix &m1, const Matrix &m2) {
    // Check if the dimensions match
    if (!dimensionsMatchElementWise(m1, m2))
        throw std::invalid_argument("The dimensions of the matrices do not match.");

    // Initialize the data
    MatrixData data(m1.rows, Vector(m1.cols, 0.0));

    // Add the matrices
    for(int i = 0; i < m1.rows; i++) {
        for(int j = 0; j < m1.cols; j++)
            data[i][j] = m1.data[i][j] + m2.data[i][j];
    }

    return Matrix(data);
}

Matrix Matrix::subtract(const Matrix &m1, const Matrix &m2) {
    // Check if the dimensions match
    if (!dimensionsMatchElementWise(m1, m2))
        throw std::invalid_argument("The dimensions of the matrices do not match.");

    // Initialize the data
    MatrixData data(m1.rows, Vector(m1.cols, 0.0));

    // Subtract the matrices
    for(int i = 0; i < m1.rows; i++) {
        for(int j = 0; j < m1.cols; j++)
            data[i][j] = m1.data[i][j] - m2.data[i][j];
    }

    return Matrix(data);
}


Matrix Matrix::multiply(const Matrix &m1, const Matrix &m2) {
    // Check if the dimensions match
    if (!dimensionsMatchElementWise(m1, m2))
        throw std::invalid_argument("The dimensions of the matrices do not match.");

    // Initialize the data
    MatrixData data(m1.rows, Vector(m1.cols, 0.0));

    // Multiply the matrices
    for(int i = 0; i < m1.rows; i++) {
        for(int j = 0; j < m1.cols; j++)
            data[i][j] = m1.data[i][j] * m2.data[i][j];
    }

    return Matrix(data);
}

Matrix Matrix::dot(const Matrix &m1, const Matrix &m2) {
    // Check if the dimensions match
    if (!dimensionsMatchDot(m1, m2))
        throw std::invalid_argument("The dimensions of the matrices do not match.");

    // Initialize the data
    MatrixData data(m1.rows, Vector(m2.cols, 0.0));

    // Multiply the matrices, using the naive algorithm
    for(int i = 0; i < m1.rows; i++) {
        for(int j = 0; j < m2.cols; j++) {
            for(int k = 0; i < m1.cols; k++) {
                // Compute dot product between row i of m1 and column j of m2
                // Use the += operator to enable cascaded matrix multiplication
                data[i][j] += m1.data[i][k] * m2.data[k][j];
            }
        }
    }

    return Matrix(data);
}

Matrix Matrix::multiply(double scalar, const Matrix &m) {
    // Initialize the data
    MatrixData data(m.rows, Vector(m.cols, 0.0));

    // Multiply the matrix by the scalar
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++)
            data[i][j] = scalar * m.data[i][j];
    }

    return Matrix(data);
}

Matrix Matrix::transpose(const Matrix &m) {
    // Initialize the data
    MatrixData data(m.cols, Vector(m.rows, 0.0));

    // Transpose the matrix i.e rows become columns and columns become rows
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++)
            data[j][i] = m.data[i][j];
    }

    return Matrix(data);
}

double Matrix::sum(const Matrix &m) {
    // Initialize the sum
    double sum = 0.0;

    // Sum the matrix
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++)
            sum += m.data[i][j];
    }

    return sum;
}
            






