//
// Created by patrick on 8/18/20.
//

#ifndef MNISTMODEL_SYNTHETICSIMULATION_H
#define MNISTMODEL_SYNTHETICSIMULATION_H

#include "Simulation.h"
#include <string>
#include <vector>


/*
 * workflow
 * create synthetic network
 * add networks
 * setup sim
 * apply input patterm
 * run network
 * write graphs
 */

// synthetic 3 layer feedfoward network from "mapping spiking neural networks to neuromorphic hardware" paper
class SyntheticSimulation : public Simulation {
public:
    SyntheticSimulation(SimOptions::Verbosity verbosity, LoggerMode loggerMode, const std::string &simName="Synthetic");
    ~SyntheticSimulation();

    // arch: 3 layers
    // layer1 -> all to all -> layer2 & layer3
    // layer2 -> all to all -> layer3
    // layer sizes are the number of neuron in each layer
    // weights are randomized between weight arg and weight/2
    void addNetwork(int layerSize1, int layerSize2, int layerSize3, float weight1_2, float weight2_3);

    // applies a constant current to every neuron in layer1
    void applyConstantPoisson(float poissonRate);

    // runs the network, exposing layer1 to the given input pattern
    // single exposure, no loop
    void run(int exposureSeconds, int exposureMilliseconds, bool printSummary = false);

private:
    // layer names
    const std::string LAYER_1 = "layer1";
    const std::string LAYER_2 = "layer2";
    const std::string LAYER_3 = "layer3";

    // behavior generators - needed by CARLsim to kept around until the simulation uses
    std::vector<PoissonRate*> poissonRates_;  // holds the poisson rate objects
    std::vector<ConnectionGenerator*> connectionGenerators_; // connection generators to randomize weights
};

#endif //MNISTMODEL_SYNTHETICSIMULATION_H
