//
// Created by patrick on 8/1/20.
//

#ifndef MNISTMODEL_SIMMNIST_H
#define MNISTMODEL_SIMMNIST_H

#include "Simulation.h"
#include <random>
#include <chrono>
#include <stopwatch.h>
#include <iomanip>

class SimMNIST : public Simulation {
public:
    // if buildAllMonitors is set then all spike and connection monitors will be built
    // DEBUG mode auto-sets buildAllMonitors to true
    // building all monitors requires additional memory and compute time
    // all monitors must be built when outputting graphs
    SimMNIST(SimOptions::Verbosity verbosity, LoggerMode loggerMode, const std::string &simName="SimMNIST",
             bool buildAllMonitors = false);
    ~SimMNIST();

    // sets data and ground truth csvs to read in for train and test
    void setInputFiles(const std::string &trainData, const std::string &testData, const std::string &blankData,
                       const std::string &trainGroundTruth, const std::string &testGroundTruth);

    // moves sim into SETUP and creates simulation monitors
    // overrides Simulation method allowing creation of only some monitors to reduce computation overhead
    void setupSim();

    // adds a model to the simulation. Each model must have the same dimensions but can have different hyper-parameters
    void addNetwork(ExpCurve curve, HomeostasisParams homeostasis, std::string connType, float connProb, std::vector<RangeWeight> weights,
                    bool randomizePlasticWeights = true, bool fastSpikingInhib = false, bool setSTDP = true);

    // executes model training and testing; record will record all spikes for output to graph
    void runTrain(int numStim, int presentationTime, int restTime, bool earlyAbort = false, int abortIndex = 1000);

    // test network predictions against a testing dataset
    // classes must be assigned for all neurons
    // populates a confusion matrix for each network in the simulation
    void runTest(int numStim, int presentationTime, int restTime);

    // reports if a valid trained network exists in the simulation. Used to check for mode collapse
    // a network is invalid if all neurons are assigned to a single class
    // this is a common form of learning failure
    bool hasValidNetwork();

private:
    // assigns neurons in each network to classes.
    // returns true if at least one network is valid (not all neurons assigned to one class)
    // resets classAssignments_ before assigning classes
    bool assignClasses();

    // clears records from previous training runs
    void resetSpikeRecords();

    // displays class assignment information: num neurons per class, average firing rates for neurons assigned to a class,
    // standard deviations, average firing rates of all neurons actually in each class
    void printTrainResults();

    // displays grids of relative spike rates of the input, excitatory, and inhibitory neuron layers
    // predictedClasses is used during testing debug
    int printDebugASCII(int stepSkip, int label, int stimIndex, int totalNumStims, std::vector<int> *predictedClasses = nullptr);

    // determines if all monitors are built for networks, or only the output spike monitor
    // all monitors must be built when outputting graphs, or when debugging
    bool buildAllMonitors_;

    // layer names
    const std::string INPUT_LAYER = "input";
    const std::string EXCIT_LAYER = "excitatory";
    const std::string INHIB_LAYER = "inhibitory";


    std::string trainDataFile_;
    std::string testDataFile_;
    std::string trainGroundTruthFile_;
    std::string testGroundTruthFile_;
    std::string blankDataFile_;

    const int INPUT_DIM = 28;  // input layer is 28 x 28 grid
    const int EXT_DIM = 10;  // num neurons on a side ext grid
    const int INHIB_DIM = 10;  // num neurons on a side inhib grid
    const int NUM_EXT_NEURONS = 100;  // EXT_DIM ** 2

    // stores RangeWeights for plastic connections
    std::vector<std::vector<RangeWeight> > plasticRangeWeights_;
    std::vector<ConnectionGenerator*> connectionGenerators_;

    // results data
    std::vector<std::vector<ClassFiringRateRecord> > spikeRecords_;


};


#endif //MNISTMODEL_SIMMNIST_H
