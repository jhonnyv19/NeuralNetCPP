// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the implementation of the class Layer, which represents a layer of neurons in a neural network.

#include "layer.hpp"

#include <iostream>

Layer::Layer() {
    // Default constructor for a layer
    this->input_size = 0;
    this->output_size = 0;
}

Layer::Layer(int input_size, int output_size, int batch_size, std::function<Matrix(const Matrix&)> activation, std::function<Matrix(const Matrix&)> activation_prime, bool is_softmax) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->softmax = is_softmax;

    // Verify that dimensions are valid
    if (input_size <= 0 || output_size <= 0) {
        throw std::invalid_argument("Invalid dimensions for layer");
    }

    // Initialize the weights and biases
    weights = Matrix(input_size, output_size);
    biases = Matrix(output_size, 1);

    // Initialize the activation function and its derivative
    this->activation = activation;
    this->activation_prime = activation_prime;
    
}

Layer::~Layer() {
    // Destructor

}

Matrix Layer::forward(const Matrix& input) {

    // Forward propagation
    // z = X * W + b
    // X has shape (batch_size, input_size)
    // W has shape (input_size, output_size)
    // b has shape (output_size, batch_size)
    Matrix z = Matrix::dot(input, weights);  // Matrix multiplication with input and weights

    MatrixData z_data = z.getData();

    // Add the biases through broadcasting
    for (int i = 0; i < z.rows; i++) {
        for (int j = 0; j < z.cols; j++) {
            z_data[i][j] += biases.getData()[j][0];
        }
    }

    // Apply the activation function and save the activations
    z = Matrix(z_data);
    Matrix a = activation(Matrix(z_data));

    // Save the activations and z values
    this->a = a;
    this->z = z;

    return a;

}

void Layer::updateWeights(double learning_rate, const Matrix& d_weights) {
    // Update the weights and biases
    weights = Matrix::subtract(weights, Matrix::multiply(learning_rate, d_weights));
}

void Layer::updateBiases(double learning_rate, const Matrix& d_biases) {
    // Update the weights and biases
    biases = Matrix::subtract(biases, Matrix::multiply(learning_rate, d_biases));
}

Matrix Layer::getWeights() const {
    return weights;
}

Matrix Layer::getBiases() const {
    return biases;
}

Matrix Layer::getActivations() const {
    return a;
}

Matrix Layer::getZ() const {
    return z;
}

bool Layer::isSoftmax() const {
    return softmax;
}
