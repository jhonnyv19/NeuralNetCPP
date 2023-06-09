// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the declaration of the Matrix class, which represents a matrix.

#include <vector>


class Matrix {
    public:
        // Type aliases
        using Vector = std::vector<double>; // a vector of doubles
        using MatrixData = std::vector<Vector>; // a vector of vectors of doubles

        // Constructor
        Matrix(int rows, int cols);
        Matrix(MatrixData data);
        Matrix(const Matrix &m);

        // Dimensions of the matrix
        int rows;
        int cols;

        // Getter method for the data
        MatrixData getData() const;

        // Matrix operations
        static Matrix add(const Matrix &m1, const Matrix &m2);
        static Matrix subtract(const Matrix &m1, const Matrix &m2);
        static Matrix multiply(const Matrix &m1, const Matrix &m2);
        static Matrix dot(const Matrix &m1, const Matrix &m2);
        static Matrix multiply(double scalar, const Matrix &m);
        static Matrix divide(const Matrix &m, double scalar);
        static Matrix transpose(const Matrix &m);
        static double sum(const Matrix &m);

    private:
        MatrixData data;

};
