// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the implementation of the class Net, which represents a neural network.

#include "net.hpp"
#include "activations.hpp"
#include <iostream>

Net::Net(vector<int> sizes, double learning_rate, vector<std::function<Matrix(const Matrix&)>> activations, vector<std::function<Matrix(const Matrix&)>> activation_primes) {
    // Default constructor for a neural network
    this->sizes = sizes;
    this->num_layers = sizes.size() - 1;
    this->learning_rate = learning_rate;

    // Verify that dimensions are valid
    if (num_layers <= 0) {
        throw std::invalid_argument("Invalid dimensions for network");
    }

    if (activations.size() != num_layers && activation_primes.size() != num_layers) {
        std::cout << activations.size() << std::endl;
        std::cout << num_layers << std::endl;
        throw std::invalid_argument("Invalid number of activation functions");
    }

    // Initialize the layers
    for (int i = 1; i <= num_layers; i++) {
        layers.push_back(Layer(sizes[i - 1], sizes[i], activations[i - 1], activation_primes[i - 1]));
    }

}

Net::~Net() {
    // Destructor

}

Matrix Net::forward(Matrix input) {
    // Forward propagation
    for (int i = 0; i < num_layers; i++) {
        input = layers[i].forward(input);
    }

    return input;
}

void Net::backward(Matrix y_hat, Matrix y, Matrix input) {
    // Calculate derivative of the loss function w.r.t. a
    Matrix dL_da = cross_entropy_loss_prime(y_hat, y);

    for (int i = num_layers - 1; i >= 0; i--) {
        // Debugging
        // std::cout << "Layer: " << i << std::endl;

        // Calculate dL_dz = dL_da * da_dz
        Matrix dL_dz = Matrix::multiply(dL_da, layers[i].activation_prime(layers[i].getZ()));

        // Print shape of dL_dz
        // std::cout << "dL_dz: " << dL_dz.shape() << std::endl;

        // If not the input layer, we need to calculate dL/da for the next layer
        // dL/da = dL/da * da/dz * dz/da
        Matrix dL_da_next_layer;
        if (i > 0) {
            dL_da_next_layer = Matrix::dot(layers[i].getWeights(), dL_dz);
        }

        // Compute the Gradients for the Layer's Parameters
        Matrix d_weights;
        if (i > 0) {
            d_weights = Matrix::dot(dL_dz, Matrix::transpose(layers[i - 1].getActivations()));
        } else {
            d_weights = Matrix::dot(dL_dz, Matrix::transpose(input));
        }

        MatrixData d_bias = std::vector<std::vector<double>>(dL_dz.rows, std::vector<double>(1));
        for (int j = 0; j < dL_dz.rows; j++) {
            double sum = 0;
            for (int k = 0; k < dL_dz.cols; k++) {
                sum += dL_dz.getData()[j][k];
            }
            d_bias[j][0] = sum;
        }


        // Once you have the gradients, use them to update the weights and biases (Step 3)
        layers[i].updateWeights(learning_rate, Matrix::transpose(d_weights));
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


