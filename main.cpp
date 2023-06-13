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

// Shuffles the data and labels in the same order
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

// Converts a value to a one-hot vector
std::vector<double> to_one_hot(int value, int num_classes) {
    std::vector<double> one_hot(num_classes, 0.0);
    one_hot[value] = 1.0;
    return one_hot;
}

// Loads the MNIST dataset
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
    // Define the hyperparameters
    int input_size = 28 * 28;
    vector<int> sizes = {input_size, 500, 10};
    double lr = 0.001;
    int batch_size = 32;

    // Apply relu activation to the first layer and sigmoid activation to the second layer
    vector<std::function<Matrix(const Matrix&)>> activations = {relu, softmax};
    vector<std::function<Matrix(const Matrix&)>> activation_primes = {relu_prime, softmax_prime};
    
    // Create the neural network
    Net net(sizes, lr, batch_size, activations, activation_primes, true);

    // Load the MNIST dataset
    auto [labels, data] = load_csv("data/mnist_train.csv");
    Matrix mnist_data(data);

    // Numbers of epochs
    int epochs = 1;

    // Print every certain number of batches
    int print_every = 100;

    // Batch gradient descent
    for (int i = 0; i < epochs; i++) {
        // Shuffle training data and labels before each epoch
        auto [shuffled_labels, shuffled_data] = shuffle(labels, data);
        Matrix shuffled_mnist_data(shuffled_data);
        std::vector<std::vector<double>> shuffled_one_hot_labels;
    
        // Convert the labels to one-hot vectors
        for (const auto& label : shuffled_labels) {
            shuffled_one_hot_labels.push_back(to_one_hot(label, 10));  // Assuming 10 classes for MNIST
        }

        // Calculate the number of batches
        int num_batches = shuffled_mnist_data.rows / batch_size;

        // Store the loss for each batch in a vector
        std::vector<double> losses;

        // Print epoch number
        std::cout << "\nEpoch " << i + 1 << std::endl;
        
        for (int j = 0; j < num_batches; j++) {
            int start = j * batch_size;
            int end = std::min(start + batch_size, shuffled_mnist_data.rows);

            // Get the batch inputs and batch labels
            Matrix batch_inputs = shuffled_mnist_data.slice(start, end);
            std::vector<std::vector<double>> batch_labels(shuffled_one_hot_labels.begin() + start, shuffled_one_hot_labels.begin() + end);
            Matrix batch_label(batch_labels);

            // Forward pass through the network
            Matrix out = net.forward(batch_inputs);
            
            // Compute loss for the batch
            Matrix loss = cross_entropy_loss(batch_label, out);

            // Calculate average loss over each batch
            double avg_loss = 0.0;
            for (int k = 0; k < loss.rows; k++) {
                avg_loss += loss.getData()[k][0];
            }

            avg_loss /= loss.rows;

            // Add the average loss to the losses vector
            losses.push_back(avg_loss);

            // Only print every certain number batches
            if (j % print_every == 0) {
                std::cout << "Batch " << j << " Loss: " << avg_loss << std::endl;
            }

            // Backpropagate the loss and update the weights and biases
            net.backward(out, batch_label, batch_inputs);

        }

        // Compute the average loss over the epoch
        double avg_loss_epoch = 0.0;
        for (const auto& loss : losses) {
            avg_loss_epoch += loss;
        }

        avg_loss_epoch /= losses.size();

        // Print epoch number and average loss
        std::cout << "Epoch " << i + 1 << " Average loss: " << avg_loss_epoch << std::endl;

    }

    // Print beginning of test phase
    std::cout << "\nBeginning model testing on data/mnist_test.csv" << std::endl;

    // Load the test data and evaluate the model accuracy
    auto [test_labels, test_data] = load_csv("data/mnist_test.csv");
    Matrix test_mnist_data(test_data);

    // Convert the labels to one-hot vectors
    std::vector<std::vector<double>> test_one_hot_labels;
    for (const auto& label : test_labels) {
        test_one_hot_labels.push_back(to_one_hot(label, 10));  // Assuming 10 classes for MNIST
    }

    // Calculate the number of batches
    int num_test_batches = test_mnist_data.rows / batch_size;

    // Collect the total number of correct predictions
    int total_correct = 0;

    // Collect accuracy for each batch to calculate the average accuracy
    std::vector<double> accuracies;

    // Collect losses for each batch to calculate the average loss
    std::vector<double> test_losses;

    // Iterate over each batch
    for (int i = 0; i < num_test_batches; i++) {
        
        // Store the number of correct predictions for each batch
        int correct = 0;

        // Get the batch inputs and batch labels
        int start = i * batch_size;
        int end = std::min(start + batch_size, test_mnist_data.rows);
        Matrix batch_inputs = test_mnist_data.slice(start, end);

        std::vector<std::vector<double>> batch_labels(test_one_hot_labels.begin() + start, test_one_hot_labels.begin() + end);
        Matrix batch_label(batch_labels);

        // Forward pass through the network
        Matrix out = net.forward(batch_inputs);

        // Iterate over each row of the output matrix to find the index of the maximum value
        for (int j = 0; j < out.rows; j++) {

            // Find the index of the maximum value in the row
            int max_index = 0;
            double max_val = out.getData()[j][0];

            for (int k = 0; k < out.cols; k++) {
                if (out.getData()[j][k] > max_val) {
                    // Update the max value and index if a new max value is found
                    max_val = out.getData()[j][k];
                    max_index = k;
                }
            }

            // If the index of the maximum value is equal to the index of the one-hot label, increment correct
            if (batch_label.getData()[j][max_index] == 1.0) {
                correct++;
                total_correct++;
            }
        }

        // Add the accuracy to the accuracies vector
        double batch_accuracy = (double)correct / batch_size;
        accuracies.push_back(batch_accuracy);

        // Calculate loss for the batch
        Matrix batch_loss = cross_entropy_loss(batch_label, out);

        // Calculate average loss for the batch
        double avg_loss = 0.0;
        for (int k = 0; k < batch_loss.rows; k++) {
            avg_loss += batch_loss.getData()[k][0];
        }

        avg_loss /= batch_loss.rows;

        // Add the average loss to the losses vector
        test_losses.push_back(avg_loss);

        // Print the average accuracy and loss for the batch
        std::cout << "Batch " << i << " Accuracy: " << (double)correct / batch_size << " Loss: " << avg_loss << std::endl;

    }

    // Calculate the average accuracy over the test set
    double avg_accuracy = 0.0;
    for (const auto& accuracy : accuracies) {
        avg_accuracy += accuracy;
    }

    avg_accuracy /= accuracies.size();

    // Calculate the average loss over the test set
    double avg_loss = 0.0;
    for (const auto& loss : test_losses) {
        avg_loss += loss;
    }

    avg_loss /= test_losses.size();

    // Print the average accuracy and loss over the test set
    std::cout << "\nAverage accuracy over test set: " << avg_accuracy << std::endl;
    std::cout << "Average loss over test set: " << avg_loss << std::endl;

    // Calculate and print total accuracy
    double total_accuracy = (double)total_correct / test_mnist_data.rows;
    std::cout << "Total accuracy: " << total_accuracy << std::endl;


    return 0;
}