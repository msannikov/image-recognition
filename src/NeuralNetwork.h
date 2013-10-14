#include "Layer.h"

class NeuralNetwork
{
public:
    vector<Layer> layers;
    void Backpropagate(vector<double> input, vector<double> &output);
    void Calculate(vector<double> input, vector<double> &output);
};