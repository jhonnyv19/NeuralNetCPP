// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: The main function for the training of a neural network.

#include <iostream>
#include <net.hpp>

int main(int argc, char** argv) {
    // Create the network
    vector<int> sizes = {2, 3, 1};
    Net net(sizes);

    // Create the training data
    vector<Matrix> inputs = {Matrix({{0, 0}}), Matrix({{0, 1}}), Matrix({{1, 0}}), Matrix({{1, 1}})};
    vector<Matrix> targets = {Matrix({{0}}), Matrix({{1}}), Matrix({{1}}), Matrix({{0}})};

    // Only test the forward propagation
    for (int i = 0; i < inputs.size(); i++) {
        Matrix out = net.forward(inputs[i]);
        std::cout << "Input: " << inputs[i] << std::endl;
        std::cout << "Output: " << out << std::endl;
    }

    return 0;
}