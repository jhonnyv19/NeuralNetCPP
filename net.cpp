// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the implementation of the class Net, which represents a neural network.

#include "net.hpp"

Net::Net(vector<int> sizes) {
    // Default constructor for a neural network
    this->sizes = sizes;
    this->num_layers = sizes.size();

    // Verify that dimensions are valid
    if (num_layers <= 0) {
        throw std::invalid_argument("Invalid dimensions for network");
    }

    if (activations.size() != num_layers - 1) {
        throw std::invalid_argument("Invalid number of activation functions");
    }

    // Initialize the layers
    for (int i = 1; i < num_layers; i++) {
        layers.push_back(Layer(sizes[i - 1], sizes[i], activations[i - 1]));
    }

}

Net::~Net() {
    // Destructor

}

Matrix Net::forward(Matrix input) const {
    // Forward propagation
    for (int i = 0; i < num_layers - 1; i++) {
        input = layers[i].forward(input);
    }

    return input;
}

vector<Matrix> Net::getWeights() const {
    // Return the weights of the network
    vector<Matrix> weights;

    for (int i = 0; i < num_layers - 1; i++) {
        weights.push_back(layers[i].getWeights());
    }

    return weights;
}

vector<Matrix> Net::getBiases() const {
    // Return the biases of the network
    vector<Matrix> biases;

    for (int i = 0; i < num_layers - 1; i++) {
        biases.push_back(layers[i].getBiases());
    }

    return biases;
}


