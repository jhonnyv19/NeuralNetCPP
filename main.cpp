// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: The main function for the training of a neural network.

#include <iostream>
#include <fstream>
#include <sstream>
#include "activations.hpp"
#include "net.hpp"

std::vector<double> to_one_hot(int value, int num_classes) {
    std::vector<double> one_hot(num_classes, 0.0);
    one_hot[value] = 1.0;
    return one_hot;
}

std::pair<std::vector<int>, std::vector<std::vector<double>>> load_csv(const std::string &path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<int> labels;
    std::vector<std::vector<double>> values;
    std::vector<double> val_line;
    bool is_first_line = true;

    while (std::getline(indata, line)) {
        if (is_first_line) {
            is_first_line = false;
            continue;
        }

        std::stringstream lineStream(line);
        std::string cell;

        std::getline(lineStream, cell, ',');
        labels.push_back(std::stoi(cell));

        while (std::getline(lineStream, cell, ',')) {
            val_line.push_back(std::stod(cell) / 255.0);
        }

        values.push_back(val_line);
        val_line.clear();
    }

    return {labels, values};
}

int main(int argc, char** argv) {
    // Create the network, first and last arguments are the input and output sizes, respectively
    int input_size = 28 * 28;
    vector<int> sizes = {input_size, input_size * 2, input_size * 2, 10};
    double lr = 0.001;

    // Apply relu activation to the first layer and sigmoid activation to the second layer
    vector<std::function<Matrix(const Matrix&)>> activations = {relu, relu, softmax};
    vector<std::function<Matrix(const Matrix&)>> activation_primes = {relu_prime, relu_prime, softmax_prime};
    
    Net net(sizes, lr, activations, activation_primes);

    auto [labels, data] = load_csv("data/mnist_train.csv");
    Matrix mnist_data(data);
    std::vector<std::vector<double>> one_hot_labels;
    
    for (const auto& label : labels) {
        one_hot_labels.push_back(to_one_hot(label, 10));  // Assuming 10 classes for MNIST
    }

    // Numbers of epochs
    int epochs = 500;

    // Only test the forward propagation
    for (int i = 0; i < epochs; i++) {
        for(int j = 0; j < mnist_data.rows; j++) {
            // Convert a row of the mnist_data matrix to another Matrix object with the same dimensions
            // i.e embed the vector into another matrix
            std::vector<std::vector<double>> inputs = {mnist_data.getData()[j]};
            Matrix input(inputs);

            // Transpose the matrix
            input = Matrix::transpose(input);

            // Print shape of the input matrix
            // std::cout << "Input shape: " << input.shape() << std::endl;

            // Do the same for the label
            std::vector<std::vector<double>> labels = {one_hot_labels[j]};
            Matrix label(labels);
            label = Matrix::transpose(label);

            // Calculate the output of the network
            // std::cout << "Forward propagation" << std::endl;
            Matrix out = net.forward(input);
            // std::cout << "Input: \n" << input << std::endl;
            // std::cout << "Output: \n" << out << std::endl;
            
            // Print shape of output and label
            // std::cout << "Output shape: " << out.shape() << std::endl;
            // std::cout << "Label shape: " << label.shape() << std::endl;

            // Calculate the loss
            double loss = cross_entropy_loss(label, out);
            std::cout << "Loss: " << loss << std::endl;

            net.backward(out, label, input);
        }
    }

    return 0;
}