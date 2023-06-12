// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the declaration of the Matrix class, which represents a matrix.

#include "matrix.hpp"
#include <random>
#include <iostream>

bool dimensionsMatchElementWise(const Matrix &m1, const Matrix &m2) {
    return m1.rows == m2.rows && m1.cols == m2.cols;
}

bool dimensionsMatchDot(const Matrix &m1, const Matrix &m2) {
    return m1.cols == m2.rows;
}

Matrix::Matrix() {
    // Default constructor for a matrix
    this->rows = 0;
    this->cols = 0;
}

Matrix::Matrix(int rows, int cols) {
    // Initialize the data
    MatrixData data = MatrixData(rows, vector<double>(cols, 0.0));

    // Initialize the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

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

std::string Matrix::shape() const {
    // Return the shape of the matrix
    return "[" + std::to_string(this->rows) + "x" + std::to_string(this->cols) + "]";
}

// Implement the << operator for printing
std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    // Print the matrix
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++)
            os << m.data[i][j] << " ";
        os << std::endl;
    }

    return os;
}

MatrixData Matrix::getData() const {
    return data;
}

Matrix Matrix::add(const Matrix &m1, const Matrix &m2) {
    // Check if the dimensions match
    if (!(m1.rows == m2.rows && m1.cols == m2.cols)) {
        throw std::invalid_argument("Error: Dimensions of matrices do not match for element-wise addition. "
                                    "Attempted operation on matrices of size [" + std::to_string(m1.rows) + "x" + std::to_string(m1.cols) + "] and [" + std::to_string(m2.rows) + "x" + std::to_string(m2.cols) + "].");
    }

    // Initialize the data
    MatrixData data(m1.rows, vector<double>(m1.cols, 0.0));

    // Add the matrices
    for(int i = 0; i < m1.rows; i++) {
        for(int j = 0; j < m1.cols; j++)
            data[i][j] = m1.data[i][j] + m2.data[i][j];
    }

    return Matrix(data);
}

Matrix Matrix::subtract(const Matrix &m1, const Matrix &m2) {
    // Check if the dimensions match
    if (!(m1.rows == m2.rows && m1.cols == m2.cols)) {
        throw std::invalid_argument("Error: Dimensions of matrices do not match for element-wise subtraction. "
                                    "Attempted operation on matrices of size [" + std::to_string(m1.rows) + "x" + std::to_string(m1.cols) + "] and [" + std::to_string(m2.rows) + "x" + std::to_string(m2.cols) + "].");
    }

    // Initialize the data
    MatrixData data(m1.rows, vector<double>(m1.cols, 0.0));

    // Subtract the matrices
    for(int i = 0; i < m1.rows; i++) {
        for(int j = 0; j < m1.cols; j++)
            data[i][j] = m1.data[i][j] - m2.data[i][j];
    }

    return Matrix(data);
}


Matrix Matrix::multiply(const Matrix &m1, const Matrix &m2) {
    // Check if the dimensions match
    if (!(m1.rows == m2.rows && m1.cols == m2.cols)) {
        throw std::invalid_argument("Error: Dimensions of matrices do not match for element-wise multiplication. "
                                    "Attempted operation on matrices of size [" + std::to_string(m1.rows) + "x" + std::to_string(m1.cols) + "] and [" + std::to_string(m2.rows) + "x" + std::to_string(m2.cols) + "].");
    }

    // Initialize the data
    MatrixData data(m1.rows, vector<double>(m1.cols, 0.0));

    // Multiply the matrices
    for(int i = 0; i < m1.rows; i++) {
        for(int j = 0; j < m1.cols; j++)
            data[i][j] = m1.data[i][j] * m2.data[i][j];
    }

    return Matrix(data);
}

Matrix Matrix::divide(const Matrix &m1, double scalar) {

    if (scalar == 0.0) {
        throw std::invalid_argument("Cannot divide by zero!");
    }

    // Initialize the data
    MatrixData data(m1.rows, vector<double>(m1.cols, 0.0));

    // Divide the matrix
    for(int i = 0; i < m1.rows; i++) {
        for(int j = 0; j < m1.cols; j++)
            data[i][j] = m1.data[i][j] / scalar;
    }

    return Matrix(data);
}

Matrix Matrix::dot(const Matrix &m1, const Matrix &m2) {
    // Check if the dimensions match
    if (!(m1.cols == m2.rows)) {
        throw std::invalid_argument("Error: Dimensions of matrices do not match for matrix multiplication. "
                                    "Attempted operation on matrices of size [" + std::to_string(m1.rows) + "x" + std::to_string(m1.cols) + "] and [" + std::to_string(m2.rows) + "x" + std::to_string(m2.cols) + "].");
    }


    // Initialize the data
    MatrixData data(m1.rows, vector<double>(m2.cols, 0.0));

    // Multiply the matrices, using the naive algorithm
    for(int i = 0; i < m1.rows; i++) {
        for(int j = 0; j < m2.cols; j++) {
            for(int k = 0; k < m1.cols; k++) {
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
    MatrixData data(m.rows, vector<double>(m.cols, 0.0));

    // Multiply the matrix by the scalar
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++)
            data[i][j] = scalar * m.data[i][j];
    }

    return Matrix(data);
}

Matrix Matrix::transpose(const Matrix &m) {
    // Initialize the data
    MatrixData data(m.cols, vector<double>(m.rows, 0.0));

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

Matrix Matrix::mean(const Matrix& m, int axis) {
    // Initialize the data
    MatrixData data;

    // Check the axis
    if(axis == 0) { // Mean of each column
        data = MatrixData(1, vector<double>(m.cols, 0.0));

        for(int i = 0; i < m.rows; i++) {
            for(int j = 0; j < m.cols; j++) {
                data[0][j] += m.data[i][j];
            }
        }

        for(int j = 0; j < m.cols; j++) {
            data[0][j] /= m.rows;
        }

    } else if(axis == 1) { // Mean of each row
        data = MatrixData(m.rows, vector<double>(1, 0.0));

        for(int i = 0; i < m.rows; i++) {
            for(int j = 0; j < m.cols; j++) {
                data[i][0] += m.data[i][j];
            }
            data[i][0] /= m.cols;
        }
    } else {
        throw std::invalid_argument("Error: Invalid axis for mean. Only 0 and 1 are accepted.");
    }

    return Matrix(data);
}


Matrix Matrix::slice(int start_row, int end_row) const {
    // Check if start_row and end_row are within the range
    if (start_row < 0 || start_row >= rows || end_row < 0 || end_row > rows || start_row > end_row) {
        throw std::invalid_argument("Error: Invalid slice indices.");
    }
    
    // Create a new MatrixData to hold the sliced data
    MatrixData sliced_data(end_row - start_row, std::vector<double>(cols, 0.0));
    
    // Copy the data from the specified rows into the new MatrixData
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < cols; ++j) {
            sliced_data[i - start_row][j] = data[i][j];
        }
    }
    
    // Return a new Matrix created from the sliced data
    return Matrix(sliced_data);
}

            
Matrix Matrix::randomInitialize(Matrix m) {
    // Get the data from the matrix
    MatrixData data = m.getData();
    

    // Initialize the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // Fill the matrix with random values
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++)
            data[i][j] = dis(gen);
    }

    return Matrix(data);
}


