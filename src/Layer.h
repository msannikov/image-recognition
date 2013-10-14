#include "Neuron.h"

class Layer
{
public:
    Layer* prevLayer;
    vector<Neuron> neurons;
    vector<double> weights;
    void Calculate();
    //void Backpropagate(vector<double> )
};