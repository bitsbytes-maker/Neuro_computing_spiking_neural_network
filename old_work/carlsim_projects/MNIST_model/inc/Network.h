//
// Created by patrick on 8/10/20.
//

#ifndef MNISTMODEL_NETWORK_H
#define MNISTMODEL_NETWORK_H

#include <carlsim.h>

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <tuple>
#include <map>
#include <fstream>



//todo little helper struct
// static method able to generate a key name from a group name and neuron num



// contains the CARLsim groups, spike monitors, connection monitors for a single connected network. This class does not
// create any groups or connection, only holds IDs and pointers for created networks
class Network {

    // layer information for a CARLsim neuron group
    struct Layer {
        Layer(int _id, const std::string &_name, SpikeMonitor* _monitor = nullptr) {
            id = _id;
            name = _name;
            monitor = _monitor;
        }
        int id;  // unique neuron group id from CARLsim
        std::string name;  // user assigned group name, not unique
        SpikeMonitor* monitor;  // spike monitor, optional
    };

    // connection information fo a CARLsim synaptic connection between two neuron groups
    struct Connection {
        Connection(short int _id, Layer* _pre, Layer* _post, const std::string &connType, ConnectionMonitor* _monitor = nullptr) {
            id = _id;
            preLayer = _pre;
            postLayer = _post;
            connectionType = connType;
            monitor = _monitor;
        }
        short int id;  // unique connection id from CARLsim
        Layer* preLayer;  // unique group id of pre-synaptic group
        Layer* postLayer;  // unique group id of the post-synaptic group
        std::string connectionType;  // string describing the connection. eg full, random, full uniform random
        ConnectionMonitor* monitor;  // connection monitor, optional
    };

    struct GraphEdge {
        GraphEdge(std::string _nodeName, float _edgeWeight) : nodeName(_nodeName), edgeWeight(_edgeWeight) {}

        static std::string makeName(int layerID, int neuronID) {
            return (std::to_string(layerID) + "_" + std::to_string(neuronID));
        }

        std::string nodeName;
        float edgeWeight;
    };

public:
    // adds a new network layer (neuron group); layers are accessed by name
    void addLayer(int groupId, const std::string &name, SpikeMonitor* monitor = nullptr);

    // sets the spike monitor for an existing layer
    void setSpikeMonitor(const std::string &name, SpikeMonitor* monitor);

    // sets the spike monitor for an existing layer
    void setSpikeMonitor(int layerId, SpikeMonitor* monitor);

    // adds a connection between two existing network layers. This stores the record, but does not create the CARLsim connection
    void addConnection(int connectionId, const std::string &preGroup, const std::string &postGroup, const std::string &connType, ConnectionMonitor* monitor = nullptr);

    // adds a connection between two existing network layers. This stores the record, but does not create the CARLsim connection
    void addConnection(int connectionId, int preId, int postId, const std::string &connType, ConnectionMonitor* monitor = nullptr);

    // builds connection monitors for all connections, output file to NULL
    // and disables summary terminal output
    // if genDataFile is true a default monitor file will be generated for every conn monitor
    // WARNING: names are based on the group name, NOT ID. If multiple networks run in parallel or multiple sims
    // with identical layer names WILL conflict with each other
    void buildConnectionMonitors(CARLsim* sim, bool genDataFile = false);

    // builds spike monitors for all layers, output file to NULL
    // if genDataFile is true a default monitor file will be generated for every spike monitor
    // WARNING: names are based on the group name, NOT ID. If multiple networks run in parallel or multiple sims
    // with identical layer names WILL conflict with each other
    void buildSpikeMonitors(CARLsim* sim, bool genDataFile = false);

    // adds a connection monitor to an existing connection in the network
    void setConnectionMonitor(const std::string &preGroup, const std::string &postGroup, ConnectionMonitor* monitor);

    // adds a connection monitor to an existing connection in the network
    void setConnectionMonitor(int preGroup, int postGroup, ConnectionMonitor* monitor);

    // performs null check: only monitors that exist will be set
    void startSpikeRecording();

    // performs null check: only monitors that exist will be set
    void stopSpikeRecording();

    // returns the CARLsim group id for the layer
    int getLayerId(const std::string &name);

    // returns a pointer to the layer spike monitor
    SpikeMonitor* getSpikeMonitor(const std::string &name);

    // returns the CARLsim connection id between the layer names sent as arguments
    int getConnectionID(const std::string &pre, const std::string &post);

    // returns all connection ids in the network
    std::vector<int> getConnIds();

    // returns the pre- and post-synapse CARLsim group names for the given connection id
    std::pair<int,int> getConnectionGroupIds(int connId);

    // returns the CARLsim connection monitor for the given connection id
    ConnectionMonitor* getConnectionMonitor(int connId);

    // returns the spike monitor for the pre-synapse neuron group
    SpikeMonitor* getPreSynapseMonitor(int connId);

    // updates vertices and edges, clears all graph contents before building
    // function must be called after simulation has been run, CARL state is RUN
    void buildGraph();

    // writes network to file as adjacency list with format [preneuron, postneuron, edge (fire rate [dot] synapse weight)];
    // resets and builds graph
    void writeGraph(const std::string &file, bool rebuildGraph = true);

    // writes architecture and simulation log for the network
    // conductance mode can not be queried from CARLsim so must be passed as arg
    void writeLog(const std::string &file, CARLsim* sim, bool conductanceMode);

private:
    // writes layer name, group id, num nerons, neuron type
    // fstream must already be opened
    void writeLayers(std::fstream &fout, CARLsim* sim);

    // writes connection pre and post neuron groups id, names, connection type, num synapses,
    // fstream must already be opened
    void writeConnections(std::fstream &fout, CARLsim* sim);

    std::vector<Layer> layers_;
    std::vector<Connection> connections_;
    std::map<std::string, std::vector<GraphEdge> > graph_;
};





#endif //MNISTMODEL_NETWORK_H
