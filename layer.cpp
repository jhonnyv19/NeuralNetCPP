// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the implementation of the class Layer, which represents a layer of neurons in a neural network.

#include "layer.hpp"

Layer::Layer() {
    // Default constructor for a layer
    this->input_size = 0;
    this->output_size = 0;
}

Layer::Layer(int input_size, int output_size, std::function<Matrix(const Matrix&)> activation) {
    this->input_size = input_size;
    this->output_size = output_size;

    // Verify that dimensions are valid
    if (input_size <= 0 || output_size <= 0) {
        throw std::invalid_argument("Invalid dimensions for layer");
    }

    // Initialize the weights and biases
    weights = Matrix(input_size, output_size);
    biases = Matrix(output_size, 0.0);

    // Randomly initialize the biases
    Matrix::randomInitialize(biases);

    // Initialize the activation function
    this->activation = activation;

}

Layer::~Layer() {
    // Destructor

}

Matrix Layer::forward(Matrix input) const {
    // Forward propagation
    Matrix z = Matrix::add(Matrix::dot(input, weights), biases);
    return activation(z);
}

Matrix Layer::getWeights() const {
    return weights;
}

Matrix Layer::getBiases() const {
    return biases;
}
