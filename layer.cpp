// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the implementation of the class Layer, which represents a layer of neurons in a neural network.

#include "layer.hpp"

Layer::Layer() {
    // Default constructor for a layer
    this->input_size = 0;
    this->output_size = 0;
}

Layer::Layer(int input_size, int output_size, std::function<Matrix(const Matrix&)> activation, std::function<Matrix(const Matrix&)> activation_prime) {
    this->input_size = input_size;
    this->output_size = output_size;

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

    // Initialize the activations
    a = Matrix(output_size, 1);
    z = Matrix(output_size, 1);

}

Layer::~Layer() {
    // Destructor

}

Matrix Layer::forward(Matrix input) {
    // Forward propagation
    Matrix z = Matrix::add(Matrix::dot(Matrix::transpose(weights), input), biases);

    // Apply the activation function and save the activations
    Matrix a = activation(z);

    this->a = a;
    this->z = z;

    return a;
}

void Layer::updateWeights(double learning_rate, Matrix d_weights) {
    // Update the weights and biases
    weights = Matrix::subtract(weights, Matrix::multiply(learning_rate, d_weights));
}

void Layer::updateBiases(double learning_rate, Matrix d_biases) {
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