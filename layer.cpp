// Author: Jhonny Velasquez
// Last modified: 06/09/2023
// Description: This file contains the implementation of the class Layer, which represents a layer of neurons in a neural network.

#include "layer.hpp"

Layer::Layer(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;
}

Layer::~Layer() {
    // Destructor
}