// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the declaration of the class Layer, which represents a layer of neurons in a neural network.

#ifndef LAYER_HPP
#define LAYER_HPP

#include "matrix.hpp"
#include <vector>
#include <functional>

using std::vector;

class Layer {
    public:
        // Constructors
        Layer();
        Layer(int input_size, int output_size, std::function<Matrix(const Matrix&)> activation, std::function<Matrix(const Matrix&)> activation_prime);
        ~Layer();

        // Dimensions of the layer
        int input_size;
        int output_size;

        // Forward propagation
        Matrix forward(Matrix input);

        // Backward propagation
        void backward(Matrix y_hat, Matrix y, Matrix input); 

        // Getter methods for the weights and biases
        Matrix getWeights() const;
        Matrix getBiases() const;
        Matrix getActivations() const;

        // Update the weights and biases
        void updateWeights(double learning_rate, Matrix d_weights);
        void updateBiases(double learning_rate, Matrix d_biases);

        void updateActivations(Matrix activations);

        // Activation function for this layer
        std::function<Matrix(const Matrix&)> activation;
        std::function<Matrix(const Matrix&)> activation_prime;


    private:

        // Activations of the neurons in this layer
        Matrix activations;
        
        // Each layer L has a matrix of weights W that represent the connections between layer L -1 and layer L
        Matrix weights;

        // Each layer L has a vector of biases b that represent the biases of the neurons in layer L
        Matrix biases;

};

#endif