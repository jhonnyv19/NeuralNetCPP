// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: The main function for the training of a neural network.

#include <iostream>
#include "activations.hpp"
#include "net.hpp"

int main(int argc, char** argv) {
    // Create the network, first and last arguments are the input and output sizes, respectively
    vector<int> sizes = {2, 2, 1};

    // Apply relu activation to the first layer and sigmoid activation to the second layer
    vector<std::function<Matrix(const Matrix&)>> activations = {relu, softmax};
    vector<std::function<Matrix(const Matrix&)>> activation_primes = {relu_prime, softmax_prime};

    Net net(sizes, activations, activation_primes);

    // Create the training data
    vector<Matrix> inputs;
    for(int i = 0; i < 5; i++) {
        inputs.push_back(Matrix(2, 1));
    }

    // Print the weights and biases of the network before running the forward propagation
    std::cout << "Weights: " << std::endl;
    for (int i = 0; i < net.getWeights().size(); i++) {
        std::cout << net.getWeights()[i] << std::endl;
    }

    std::cout << "Biases: " << std::endl;
    for (int i = 0; i < net.getBiases().size(); i++) {
        std::cout << net.getBiases()[i] << std::endl;
    }

    // Only test the forward propagation
    for (int i = 0; i < 1; i++) {
        // Calculate the output of the network
        Matrix out = net.forward(inputs[i]);
        std::cout << "Input: \n" << inputs[i] << std::endl;
        std::cout << "Output: \n" << out << std::endl;
        
        // Print dimensions of the output matrix
        std::cout << "Dimensions: " << out.rows << "x" << out.cols << std::endl;

        // Calculate the loss
        double loss = cross_entropy_loss(out, Matrix(2, 1));
        std::cout << "Loss: " << loss << std::endl;

        net.backward(out, Matrix(2, 1), inputs[i]);


    }

    return 0;
}