/*
this is an example of one of the 
most important theme in the book,
backpropagation. It is the backbone of machine learning,
and it is the main reason why "neural networks"
do not work like brains do.
the number of connections within a brain is exponentially bigger
than the ones on a neural network,
and they are different since the "nodes"
transfer information linearly instead of
using all the factors that define a neuron
(which we do not know of all!)
*/
#include <iostream>
#include <cmath>

// Define the Neuron structure
struct Neuron {
    double value;       // Local input
    double bias;
    double gradient;
    double out;         // Output value
};

// Define the NeuralNetwork class
class NeuralNetwork {
private:
    double learningRate;

public:
    NeuralNetwork(double rate) : learningRate(rate) {}

    // Sigmoid activation function
    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    // Derivative of the sigmoid function
    double sigmoidDerivative(double x) {
        double sig = sigmoid(x);
        return sig * (1 - sig);
    }

    // Perform forward propagation
    void forwardPropagation(Neuron& neuron, double input) {
        neuron.value = input;
        neuron.out = sigmoid(neuron.value);
    }

    // Perform backward propagation
    void backwardPropagation(Neuron& neuron, double target) {
        neuron.gradient = (target - neuron.out) * sigmoidDerivative(neuron.value);
    }

    // Update the weights of the neuron
    void updateWeights(Neuron& neuron, double input) {
        neuron.bias += learningRate * neuron.gradient;
        neuron.value += learningRate * neuron.gradient * input;
    }
};

int main() {
    // Create a neural network with a learning rate of 0.1
    NeuralNetwork network(0.1);

    // Create a neuron
    Neuron neuron;

    // Input and target values
    double input = 0.5;
    double target = 0.8;

    // Perform forward propagation
    network.forwardPropagation(neuron, input);

    // Perform backward propagation
    network.backwardPropagation(neuron, target);

    // Update the weights
    network.updateWeights(neuron, input);

    // Output the updated neuron's value
    std::cout << "Updated neuron value: " << neuron.out << std::endl;

    return 0;
}