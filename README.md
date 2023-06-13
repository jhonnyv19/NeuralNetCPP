# NeuralNetCPP
A vanilla neural net implementation from scratch in C++, implementing the forward and backward passes to train the network.
I decided to start this project as a way for me to gain a deeper intution of the backpropogration algorithm, and to gain a little more appreciation for all the magic that goes on under the hood of the popular ML libraries we take for granted sometimes.

Before running the project, make sure you have the MNIST dataset downloaded as a CSV file. I recommend using the MNIST dataset from Kaggle, which can be found here: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

To compile you can use the following command:
``` clang++ -O3 -std=c++17 -o net layer.cpp matrix.cpp net.cpp main.cpp activations.cpp ```

I plan on adding a makefile in the future, but for now this will do.

This will create an executable called net, which you can run with the following command:
``` ./net ```

Hyperparameters can be changed in the main.cpp file for now, but I plan to add command line arguments for ease of use in the future.

