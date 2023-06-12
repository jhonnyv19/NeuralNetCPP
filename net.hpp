// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the implementation of the class NeuralNetwork, which represents a neural network.

#ifndef NET_HPP
#define NET_HPP

#include "layer.hpp"

class Net {
    public:
        Net(vector<int> sizes, double learning_rate, int batch_size, vector<std::function<Matrix(const Matrix&)>> activations, vector<std::function<Matrix(const Matrix&)>> activation_primes);
        ~Net();

        // Dimensions of the network
        int num_layers;
        vector<int> sizes;

        // Activation functions for each layer
        vector<std::function<Matrix(const Matrix&)>> activations;
        
        // Getter methods for the weights and biases
        vector<Matrix> getWeights() const;
        vector<Matrix> getBiases() const;

        // Forward propagation
        Matrix forward(Matrix& input);

        // Backward propagation
        void backward(const Matrix& y_hat, const Matrix& y, const Matrix& input);

        // Update the weights and biases
        void updateWeights(double learning_rate);

        // Learning rate
        double learning_rate;

    private:
        // Layers of the network
        vector<Layer> layers;

};

#endif