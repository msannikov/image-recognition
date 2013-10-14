#include "Connection.h"
#include <vector>

using namespace std;

class Neuron
{
public:
    double output;
    vector<Connection> connections;
    void AddConnection(Connection &connection);
};