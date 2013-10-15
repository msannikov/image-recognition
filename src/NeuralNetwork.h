#include <vector>
using namespace std;

class Neuron
{
public:
    float output;
    vector<Neuron> connections;
    //void AddConnection(Connection &connection);
};

class Layer
{
public:
    Layer* prevLayer;
    vector<Neuron> neurons;
    vector<float> weights;
    void Calculate();
    //void Backpropagate(vector<double> )
};

class NeuralNetwork
{
public:
    vector<Layer> layers;
    void Backpropagate(vector<float> input, vector<float> &output);
    void Calculate(vector<float> input, vector<float> &output);
};