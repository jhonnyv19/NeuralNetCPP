// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the declaration of the class Layer, which represents a layer of neurons in a neural network.

#include "matrix.hpp"
#include <vector>

using std::vector;

class Layer {
    public:
        Layer(int input_size, int output_size);
        ~Layer();

    private:
        // Dimensions of the layer
        int input_size;
        int output_size;

        // Weights and biases
        Matrix weights;
        vector<double> biases;
};