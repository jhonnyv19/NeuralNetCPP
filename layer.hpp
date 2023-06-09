// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the declaration of the class Layer, which represents a layer of neurons in a neural network.

#include "matrix.hpp"
#include <vector>
#include <functional>

using std::vector;

class Layer {
    public:
        // Constructors
        Layer();
        Layer(int input_size, int output_size, std::function<Matrix(const Matrix&)> activation);
        ~Layer();

        // Dimensions of the layer
        int input_size;
        int output_size;

        // Forward propagation
        Matrix forward(Matrix input) const;

        // Backward propagation
        void backward(Matrix input, Matrix output, Matrix target);

        // Getter methods for the weights and biases
        Matrix getWeights() const;
        Matrix getBiases() const;


    private:

        // Each layer L has a matrix of weights W that represent the connections between layer L -1 and layer L
        Matrix weights;

        // Each layer L has a vector of biases b that represent the biases of the neurons in layer L
        Matrix biases;

        // Activation function for this layer
        std::function<Matrix(const Matrix&)> activation;
};