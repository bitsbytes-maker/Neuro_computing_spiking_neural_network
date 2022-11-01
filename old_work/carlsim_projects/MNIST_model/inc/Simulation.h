//
// Created by patrick on 7/29/20.
//

#ifndef MNISTMODEL_SIMULATION_H
#define MNISTMODEL_SIMULATION_H

#include <carlsim.h>
#include <visual_stimulus.h>
#include "../inc/StimulusWrapper.h"//todo these dont need the ../inc/
#include "../inc/mnistUtils.h"
#include "../inc/Network.h"


//todo which are needed?
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iterator>
#include <stdexcept>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <map>

//todo monitors for other than output should be optional, require debug mode


/* todo outdated
 * runner workflow:
 * construct network
 * config sim
 * add models
 * setup sim -> sets up monitors
 * runTrain
 * runTest
 * get accuracies
 *
 */


namespace SimOptions {  // simulation verbosity options
    enum Verbosity {
        VERBOSE,
        DEBUG,
        SILENT
    };
}

class Simulation {
public:
    Simulation(SimOptions::Verbosity verbosity, LoggerMode loggerMode, const std::string &simName="sim");
    ~Simulation();

    void setNumClasses(int numClasses) {numClasses_ = numClasses;};

    // default arguments are from CARLsim
    // except for update interval, which is lowered from INTERVAL_1000MS
    void configSim(bool conductanceMode = false, UpdateInterval interval = INTERVAL_100MS, bool enableWeightDecay = true, float weightChangeRatio = 0.9);

    // moves sim into SETUP and creates simulation monitors
    virtual void setupSim();

    // all network should have same dimensions + setup
    // this method MUST be overridden in child classes
    virtual void addNetwork() {};
    // this method MUST be overridden in child classes
    // this method is a stub
    virtual void runTrain() {};
    // this method MUST be overridden in child classes
    // this method is a stub
    virtual void runTest() {};

    // returns accuracy for network at index
    // throws runtime error if confusion matrix does not exist at index
    float getAccuracy(int networkIndex);

    // returns error rate for network at index
    // error rate is 1 - accuracy
    // confusion matrices must be populated to report error rate
    float getErrorRate(int networkIndex);

    // accuracy averaged across all networks in the simulation
    float getAverageAccuracy();

    // loads a saved simulation from file
    // calls setupSim() internally
    void loadAndSetupSim(const char* simFile = "output/trainingState.dat");

    // loads neurons class assignments from file
    void loadNeuronClasses(int networkIndex = 0, const std::string &classFile = "output/classLabels.txt");//todo test
    // writes neuron class assignments to file
    void writeNeuronClasses(int networkIndex = 0, const std::string &classFile = "output/classLabels.txt");
    // saves CARLsim simulation state to file
    void saveSim(const std::string &simFile = "output/trainingState.dat");//todo finish

    void writeGraph(const std::string &file, int networkIndex = 0);

    void writeNetworkLog(const std::string &file, int networkIndex = 0);

protected:
    // resets class assignments for neurons for all models
    void resetClassAssignments();
    // resets confusion matrices for all models
    void resetConfusionMatrices();



    //constants for MNIST Diehl and Cook model
    const char FILE_DELIMITER = ',';
    const char END_OF_LINE = '\n';

    int numClasses_;
    bool conductanceMode_;  // CUBA (false) or COBA (true) for CARLsim conductance mode

    // simulation options
    SimOptions::Verbosity verbosity_;

    // simulation variables
    CARLsim* sim_;  //CARLsim lacks a default constructor
    int numNetworks_;  // number of models in simulation network
    std::string simName_;  // name of the CARLsim simulation

    // networks - contain layer ids, spike + connection monitors
    std::vector<Network> networks_;

    // neuron groups
//    std::vector<int> inputGroups_;
//    std::vector<int> outputGroups_;
//    std::vector<int> inhibGroups_;
//
//    // simulation monitors
//    std::vector<SpikeMonitor*> inputMonitors_;
//    std::vector<SpikeMonitor*> outputMonitors_;
//    std::vector<SpikeMonitor*> inhibMonitors_;
//    std::vector<ConnectionMonitor*> connectionMonitor_;  // monitor between input and output groups
//    std::vector<std::vector<short int> > connectionIDs_;

    // results data
    std::vector<ConfusionMatrix> confusionMatrices_;
    std::vector<ClassLabel> classAssignments_;




};





#endif //MNISTMODEL_SIMULATION_H
