// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: The main function for the training of a neural network.

#include <iostream>
#include "activations.hpp"
#include "net.hpp"

int main(int argc, char** argv) {
    // Create the network, first and last arguments are the input and output sizes, respectively
    vector<int> sizes = {2, 3, 2};

    // Apply relu activation to the first layer and sigmoid activation to the second layer
    vector<std::function<Matrix(const Matrix&)>> activations = {relu, sigmoid};

    Net net(sizes, activations);

    // Print the number of layers
    std::cout << "Number of layers: " << net.getWeights().size() << std::endl;

    // Create the training data
    vector<Matrix> inputs;
    for(int i = 0; i < 5; i++) {
        inputs.push_back(Matrix(2, 1));
    }

    // Only test the forward propagation
    for (int i = 0; i < inputs.size(); i++) {
        Matrix out = net.forward(inputs[i]);
        std::cout << "Input: " << inputs[i] << std::endl;
        std::cout << "Output: " << out << std::endl;
    }

    return 0;
}