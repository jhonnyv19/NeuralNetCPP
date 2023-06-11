// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the implementation of the class Net, which represents a neural network.

#include "net.hpp"
#include "activations.hpp"
#include <iostream>

Net::Net(vector<int> sizes, vector<std::function<Matrix(const Matrix&)>> activations, vector<std::function<Matrix(const Matrix&)>> activation_primes) {
    // Default constructor for a neural network
    this->sizes = sizes;
    this->num_layers = sizes.size() - 1;

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
    // Calculate derivative of the loss function with respect to the output
    Matrix d_loss = cross_entropy_loss_prime(y_hat, y);

    for (int i = num_layers - 1; i >= 0; i--) {
        // Debugging
        std::cout << "Layer: " << i << std::endl;

        d_loss = Matrix::multiply(d_loss, layers[i].activation_prime(layers[i].getActivations()));
        
        // Print shape of d_loss
        std::cout << "d_loss: " << d_loss.shape() << std::endl;

        // If not the input layer, we need to calculate d_loss for the next layer
        Matrix d_loss_next_layer;
        if (i > 0) {
            std::cout << "d_loss: " << d_loss.shape() << std::endl;
            d_loss_next_layer = Matrix::dot(layers[i].getWeights(), d_loss);

        }

        // Step 2b: Compute the Gradients for the Layer's Parameters
        Matrix d_weights;
        if (i > 0) {;
            d_weights = Matrix::dot(Matrix::transpose(layers[i - 1].getActivations()), d_loss);
        } else {
            d_weights = Matrix::dot(Matrix::transpose(input), d_loss);
        }

        // Matrix d_bias = d_loss.sum(0);  // Assume sum function that operates over rows is available
        MatrixData d_bias = std::vector<std::vector<double>>(d_loss.rows, std::vector<double>(1));
        for (int j = 0; j < d_loss.rows; j++) {
            double sum = 0;
            for (int k = 0; k < d_loss.cols; k++) {
                sum += d_loss.getData()[j][k];
            }
            d_bias[j][0] = sum;
        }

        // Step 3: Update the Parameters
        layers[i].updateWeights(learning_rate, d_weights);
        layers[i].updateBiases(learning_rate, d_bias);

        // Prepare for the next iteration
        if (i > 0) {
            d_loss = d_loss_next_layer;
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


