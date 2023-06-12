// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: The main function for the training of a neural network.

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>    // for std::shuffle
#include <random>       // for std::default_random_engine
#include <chrono>       // for std::chrono::system_clock

#include "activations.hpp"
#include "net.hpp"

std::pair<std::vector<int>, std::vector<std::vector<double>>> shuffle(const std::vector<int>& labels, const std::vector<std::vector<double>>& data) {
    assert(labels.size() == data.size());  // make sure we have the same amount of data and labels
    
    // Create a vector with indices
    std::vector<size_t> indices(labels.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Use a random engine
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine(seed);
    std::shuffle(indices.begin(), indices.end(), engine);

    // Use the shuffled indices to reorder the labels and data
    std::vector<int> shuffled_labels(labels.size());
    std::vector<std::vector<double>> shuffled_data(data.size());
    for (size_t i = 0; i < indices.size(); i++) {
        shuffled_labels[i] = labels[indices[i]];
        shuffled_data[i] = data[indices[i]];
    }

    return {shuffled_labels, shuffled_data};
}


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

    // Batch size
    int batch_size = 16;

    // Apply relu activation to the first layer and sigmoid activation to the second layer
    vector<std::function<Matrix(const Matrix&)>> activations = {relu, relu, softmax};
    vector<std::function<Matrix(const Matrix&)>> activation_primes = {relu_prime, relu_prime, softmax_prime};
    
    Net net(sizes, lr, batch_size, activations, activation_primes);

    auto [labels, data] = load_csv("data/mnist_train.csv");
    Matrix mnist_data(data);
    std::vector<std::vector<double>> one_hot_labels;
    
    for (const auto& label : labels) {
        one_hot_labels.push_back(to_one_hot(label, 10));  // Assuming 10 classes for MNIST
    }

    // Numbers of epochs
    int epochs = 500;



    // Batch gradient descent
    for (int i = 0; i < epochs; i++) {
        // Shuffle training data and labels before each epoch
        // Implement the shuffle function as per your requirement
        auto [shuffled_labels, shuffled_data] = shuffle(labels, data);
        Matrix shuffled_mnist_data(shuffled_data);
        std::vector<std::vector<double>> shuffled_one_hot_labels;
    
        for (const auto& label : shuffled_labels) {
            shuffled_one_hot_labels.push_back(to_one_hot(label, 10));  // Assuming 10 classes for MNIST
        }

        int num_batches = shuffled_mnist_data.rows / batch_size;

        // Print epoch number
        std::cout << "Epoch: " << i + 1 << std::endl;
        
        for (int j = 0; j < num_batches; j++) {
            int start = j * batch_size;
            int end = std::min(start + batch_size, shuffled_mnist_data.rows);
            Matrix batch_inputs = shuffled_mnist_data.slice(start, end);
            std::vector<std::vector<double>> batch_labels(shuffled_one_hot_labels.begin() + start, shuffled_one_hot_labels.begin() + end);

            // Transpose the batch_inputs and batch_labels matrices
            // batch_inputs = Matrix::transpose(batch_inputs);
            Matrix batch_label(batch_labels);
            // batch_label = Matrix::transpose(batch_label);

            // Print out the shape of the batch_inputs and batch_labels matrices
            // std::cout << "Batch inputs shape: " << batch_inputs.shape() << std::endl;
            // std::cout << "Batch labels shape: " << batch_label.shape() << std::endl;

            // Forward and backward propagation
            Matrix out = net.forward(batch_inputs);

            // std::cout << "Beginning backward propagation" << std::endl;
            net.backward(out, batch_label, batch_inputs);

            // Compute loss for the batch and print it
            Matrix loss = cross_entropy_loss(batch_label, out);

            std::cout << "Shape of loss: " << loss.shape() << std::endl;

            double avg_loss = 0.0;

            // Calculate average loss over each batch
            for (int k = 0; k < loss.rows; k++) {
                avg_loss += loss.getData()[k][0];
            }

            avg_loss /= loss.rows;

            // double average_loss = Matrix::mean(loss, 0).getData()[0][0];

            std::cout << "Batch: " << j << " Loss: " << avg_loss << std::endl;
        }

    }

    return 0;
}