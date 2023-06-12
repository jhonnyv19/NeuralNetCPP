// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the declaration of the Matrix class, which represents a matrix.

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include "types.hpp"

using std::vector;

class Matrix {
    public:
        // Constructor
        Matrix();
        Matrix(int rows, int cols);
        Matrix(MatrixData data);
        Matrix(const Matrix &m);

        // Dimensions of the matrix
        int rows;
        int cols;

        // Getter method for the data
        MatrixData getData() const;

        // Getter method to print the shape of the matrix
        std::string shape() const;

        // Matrix operations
        static Matrix add(const Matrix &m1, const Matrix &m2);
        static Matrix subtract(const Matrix &m1, const Matrix &m2);
        static Matrix multiply(const Matrix &m1, const Matrix &m2);
        static Matrix dot(const Matrix &m1, const Matrix &m2);
        static Matrix multiply(double scalar, const Matrix &m);
        static Matrix divide(const Matrix &m, double scalar);
        static Matrix transpose(const Matrix &m);
        static double sum(const Matrix &m);
        static Matrix mean(const Matrix& m, int axis);
        Matrix slice(int start_row, int end_row) const;
        
        // Random intialization for a bias vector
        static Matrix randomInitialize(Matrix m);

        // Implement the << operator for printing
        friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

    private:
        MatrixData data;

};

#endif
