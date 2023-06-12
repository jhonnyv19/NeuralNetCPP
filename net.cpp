// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the implementation of the class Net, which represents a neural network.

#include "net.hpp"
#include "activations.hpp"
#include <iostream>

Net::Net(vector<int> sizes, double learning_rate, int batch_size, vector<std::function<Matrix(const Matrix&)>> activations, vector<std::function<Matrix(const Matrix&)>> activation_primes) {
    // Default constructor for a neural network
    this->sizes = sizes;
    this->num_layers = sizes.size() - 1;
    this->learning_rate = learning_rate;

    // Verify that dimensions are valid
    if (num_layers <= 0) {
        throw std::invalid_argument("Invalid dimensions for network");
    }

    if (activations.size() != num_layers && activation_primes.size() != num_layers) {
        // std::cout << activations.size() << std::endl;
        // std::cout << num_layers << std::endl;
        throw std::invalid_argument("Invalid number of activation functions");
    }

    // Initialize the layers
    for (int i = 1; i <= num_layers; i++) {
        layers.push_back(Layer(sizes[i - 1], sizes[i], batch_size, activations[i - 1], activation_primes[i - 1]));
    }

}

Net::~Net() {
    // Destructor

}

Matrix Net::forward(Matrix& input) {
    // Forward propagation

    // Print shape of input
    // std::cout << "Input shape: " << input.shape() << std::endl;

    Matrix a = Matrix(input);

    for (int i = 0; i < num_layers; i++) {
        // std::cout << "Layer " << i << std::endl;
        a = layers[i].forward(a);
    }

    return a;
}

void Net::backward(const Matrix& y_hat, const Matrix& y, const Matrix& input) {
    // Calculate derivative of the loss function w.r.t. a
    Matrix dL_da = cross_entropy_loss_prime(y_hat, y);

    // Print shape of dL_da
    // std::cout << "dL_da shape: " << dL_da.shape() << std::endl;

    int batch_size = y_hat.rows;

    for (int i = num_layers - 1; i >= 0; i--) {

        // Calculate dL_dz = dL_da * da_dz
        Matrix da_dz = layers[i].activation_prime(layers[i].getZ());

        // Print shape of da_dz and dL_da
        // std::cout << "da_dz shape: " << da_dz.shape() << std::endl;
        // std::cout << "dL_da shape: " << dL_da.shape() << std::endl;

        Matrix dL_dz = Matrix::multiply(dL_da, da_dz);

        // If not the input layer, we need to calculate dL/da for the next layer

        // Print shape of dL_dz and weights of the current layer
        // std::cout << "dL_dz shape: " << dL_dz.shape() << std::endl;
        // std::cout << "Weights shape: " << layers[i].getWeights().shape() << std::endl;

        Matrix dL_da_next_layer;
        if (i > 0) {
            dL_da_next_layer = Matrix::dot(dL_dz, Matrix::transpose(layers[i].getWeights()));
        }

        // Compute the Gradients for the Layer's Parameters
        Matrix d_weights;
        if (i > 0) {
            d_weights = Matrix::dot(Matrix::transpose(layers[i - 1].getActivations()), dL_dz);
        } else {
            d_weights = Matrix::dot(Matrix::transpose(input), dL_dz);
        }

        // Shape of d_weights
        // std::cout << "d_weights shape: " << d_weights.shape() << std::endl;

        // Average the gradients
        d_weights = Matrix::divide(d_weights, (double)batch_size);

        // Calculate the gradient of the bias, which is just the mean of dL_dz across the batch dimension
        Matrix d_bias = Matrix::transpose(Matrix::mean(dL_dz, 0));  // Assuming axis argument represents the axis along which to compute the mean


        // Shape of d_bias
        // std::cout << "d_bias shape: " << d_bias.shape() << std::endl;

        // Update the Parameters
        layers[i].updateWeights(learning_rate, d_weights);
        layers[i].updateBiases(learning_rate, d_bias);

        // Prepare for the next iteration
        if (i > 0) {
            dL_da = dL_da_next_layer;
        }
    }
}


vector<Matrix> Net::getWeights() const {
    // Return the weights of the network
    vector<Matrix> weights;

    for (int i = 0; i < num_layers; i++) {
        weights.push_back(layers[i].getWeights());
    }

    return weights;
}

vector<Matrix> Net::getBiases() const {
    // Return the biases of the network
    vector<Matrix> biases;

    for (int i = 0; i < num_layers; i++) {
        biases.push_back(layers[i].getBiases());
    }

    return biases;
}


