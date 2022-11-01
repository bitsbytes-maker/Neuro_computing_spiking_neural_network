//
// Created by patrick on 7/28/20.
//

#ifndef MNISTMODEL_MNISTUTILS_H
#define MNISTMODEL_MNISTUTILS_H

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



struct HomeostasisParams {
    // default values taken from homeostasis tutorial
    HomeostasisParams(bool enable_arg=true, float alpha_arg=0.1, float T_arg=10.0, float R_target_arg=10.0) {
        enable = enable_arg;
        alpha = alpha_arg;
        T = T_arg;
        R_target = R_target_arg;
    }

    bool enable;
    float alpha;
    float T;
    float R_target;
};

// struct to calculate and hold a confusion matrix
// actual x predicted
class ConfusionMatrix {
public:
    // size is same for row and column
    ConfusionMatrix(int size) {
        numClasses = size;
        matrix = std::vector<std::vector<int> >(numClasses, std::vector<int>(numClasses,0));
    }

    // returns reference to value at (column, row)
    int& at(int actual, int predicted) {
        return matrix.at(actual).at(predicted);
    }

    // clears confusion matrix
    void reset() {
        matrix = std::vector<std::vector<int> >(numClasses, std::vector<int>(numClasses,0));
    }

    // returns accuracy of the matrix, interpreted as true positives
    float getAccuracy() {
        int positives = 0;
        int popCount = 0;
        int hits;

        for (int actual = 0; actual < numClasses; ++actual) {
            for (int predicted = 0; predicted < numClasses; ++predicted) {
                hits = matrix.at(actual).at(predicted);
                if (actual == predicted)
                    positives += hits;
                popCount += hits;
            }
        }

        float accuracy = 0.0;
        if (popCount != 0)
            accuracy = (float) positives / popCount;
        return accuracy;
    }

    // returns error rate = 1 - accuracy
    float getErrorRate() {
        return (1 - getAccuracy());
    }

    // prints confusion matrix to terminal
    void print() {
        std::cout << "CONFUSION MATRIX" << std::endl;

        std::cout << "pred: \t";
        for (int predicted = 0; predicted < numClasses; ++predicted)
            std::cout << predicted << "\t";
        std::cout << "ac_acc" << std::endl << "actual:" << std::endl;

        int numClassPresentations;
        std::vector<int> sumClassPredictions(numClasses);

        for (int actual = 0; actual < numClasses; ++actual) {  // line / row
            numClassPresentations = 0;

            std::cout << actual << ":\t";
            for (int predicted = 0; predicted < numClasses; ++predicted) {  // columns
                numClassPresentations += matrix.at(actual).at(predicted);
                sumClassPredictions.at(predicted) += matrix.at(actual).at(predicted);

                std::cout << matrix.at(actual).at(predicted);
                if (actual == predicted)  // visual notation for correct predictions
                    std::cout << "*";
                std::cout << "\t";
            }
            // prints accuracy for actual: true positives / num presentations
            std::cout << std::setprecision(2) << (float) matrix.at(actual).at(actual) / numClassPresentations;
            std::cout << std::endl;
        }

        // prints accuracy for predictions
        std::cout << "pr_acc:\t";
        for (int predicted = 0; predicted < numClasses; predicted++) {
            std::cout << std::setprecision(2) << (float) matrix.at(predicted).at(predicted) / sumClassPredictions.at(predicted) << "\t";
        }

        std::cout << std::endl;
        std::cout << "Total accuracy: " << getAccuracy() * 100 << "%"  <<  std::endl;
    }

private:
    std::vector<std::vector<int> > matrix;  // 2d confusion matrix
    int numClasses;  // dimensions of rows and columns
};



// holds spike rate records for a class label
// for mnist there are 10 classes, and each model run in the simulation will have 10 class records
class ClassFiringRateRecord {
public:
    ClassFiringRateRecord(int label) {
        label_ = label;
        mean_ = 0.0;
    }

    // adds a firing rate vector, outputted from a SpikeMonitor->getAllFiringRates() to the class record
    void add(const std::vector<float> &rates) {
        firingRateSamples_.push_back(rates);
    }

    // returns a vector of average firing rates by neuron
    std::vector<float> getNeuronAverageRates() {
        int numSamples = firingRateSamples_.size();
        if (numSamples == 0)
            throw std::runtime_error("ClassFiringRateRecord::getNeuronAverageRates has no samples to average by neuron");

        int numNeurons = firingRateSamples_.at(0).size();
        if (numNeurons == 0)
            throw std::runtime_error("ClassFiringRateRecord::getNeuronAverageRates firing rate vector(s) is empty");

        std::vector<float> neuronAverages = std::vector<float>(numNeurons, 0.0);

        // sums fire rates by neuron
        for (int i = 0; i < numSamples; ++i) {
            for (int j = 0; j < numNeurons; ++j) {
                neuronAverages.at(j) += firingRateSamples_.at(i).at(j);
            }
        }

        // averages fire rates
        for (auto it = neuronAverages.begin(); it != neuronAverages.end(); it++) {
            *it = *it / numSamples;
        }

        return neuronAverages;
    }

    // returns the average firing rate across every neuron in the class
    float getAverageRate() {
        int numSamples = firingRateSamples_.size();
        if (numSamples == 0)
            throw std::runtime_error("ClassFiringRateRecord::getAverageRate sample vector is empty");

        int numNeurons = firingRateSamples_.at(0).size();
        if (numNeurons == 0)
            throw std::runtime_error("ClassFiringRateRecord::getAverageRate firing rate vector(s) is empty");

        float sum = 0;
        float mean;
        // sums fire rates for all neurons
        for (auto sample = firingRateSamples_.begin(); sample != firingRateSamples_.end(); sample++) {
            for (auto neuron = sample->begin(); neuron != sample->end(); neuron++) {
                sum += *neuron;
            }
        }

        mean = sum / (numNeurons * numSamples);
        return mean;
    }

    // given a vector of neuron IDs, returns the average firing rate for those neurons across all samples
    // neurons are considered to be assigned to this class. The mean is saved for later use
    float getAverageRatesForAssigned(const std::vector<int> &neuronIDs) {
        int numNeuronIDs = neuronIDs.size();
        int numSamples = firingRateSamples_.size();
        if (numSamples == 0)
            throw std::runtime_error("ClassFiringRateRecord::getAverageRatesForAssigned sample vector is empty");
        if (numNeuronIDs == 0)
            return 0;  // label was assigned no neurons

        float sum = 0;
        // sums fire rates for given neurons
        for (auto sample = firingRateSamples_.begin(); sample != firingRateSamples_.end(); sample++) {
            for (auto neuron = neuronIDs.begin(); neuron != neuronIDs.end(); neuron++) {
                sum += sample->at(*neuron);
            }
        }

        mean_ = sum / (numNeuronIDs * numSamples);
        return mean_;
    }

    // given a vector of neuron IDs, returns the standard deviation of the firing rate for those neurons across all samples
    // neurons are considered to be assigned to this class
    // equation: std = sqrt( (sum(x_i - mean)**2) / (n - 1) )
    float getStandardDeviationForAssigned(const std::vector<int> &neurons) {
        int numNeurons = neurons.size();
        int numSamples = firingRateSamples_.size();

        if (numSamples == 0)
            throw std::runtime_error("ClassFiringRateRecord::getStandardDeviationForAssigned sample vector is empty");
        if (numNeurons == 0)  // label was assigned no neurons
            return 0;

        if (mean_ == 0)  // updates mean if equate to default
            getAverageRatesForAssigned(neurons);

        float sum = 0;
        // sums fire rates
        for (auto sample = firingRateSamples_.begin(); sample != firingRateSamples_.end(); sample++) {
            for (auto neuron = neurons.begin(); neuron != neurons.end(); neuron++) {
                sum += pow((sample->at(*neuron) - mean_), 2);
            }
        }

        float stddev = sqrt(sum / ((numNeurons * numSamples) - 1));
        return stddev;
    }

    // returns label assigned to this class
    int getLabel() {return label_;};

private:
    int label_;  // label for this class, e.g. '1'
    std::vector<std::vector<float> > firingRateSamples_;  // samples from SpikeMonitor->getAllFiringRates()
    float mean_;  // average neuron firing rate for neurons assigned to this class
};


// record of neurons assigned to a class
// can be accessed by neuron or by label
class ClassLabel {
public:
    ClassLabel(int numClasses) {
        numClasses_ = numClasses;
        labels_ = std::vector<std::vector<int> >(numClasses_);
    }

    // a class assignment is considered malformed (mode collapse) if all neurons are assigned to one class
    bool isValid() {
        for (auto it = labels_.begin(); it != labels_.end(); it++) {
            if (it->size() == neurons_.size())
                return false;
        }
        return true;
    }

    // assumes neurons are inserted in ascending order starting at neuron id 0
    void add(int label, int neuron) {
        neurons_.push_back(label);
        labels_.at(label).push_back(neuron);
    }

    // returns label of neuron ID
    int getNeuronLabel(int neuron) {
        if (neuron < 0 or neuron >= neurons_.size())
            throw std::runtime_error("classlabel::getNeuronLabel neuron id not in range");
        return neurons_.at(neuron);
    }

    std::vector<int> getAllNeuronLabels() {
        return neurons_;
    }

    // returns all neurons for a given label
    std::vector<int> getNeuronsAtLabel(int label) {
        return labels_.at(label);
    }

    std::vector<std::vector<int> > getAllNeuronsByLabel() {
        return labels_;
    }

    // returns number of neurons assigned to a given label
    int getLabelCount(int label) {
        return labels_.at(label).size();
    }

    std::vector<int> getAllLabelCounts() {
        std::vector<int> counts;
        for (auto it = labels_.begin(); it != labels_.end(); it++)
            counts.push_back(labels_.size());

        return counts;
    }

    int getNumClasses() {
        return numClasses_;
    }

private:
    int numClasses_;  // number of classes labels
    std::vector<int> neurons_;  // vector of labels for neurons. Index is used as neuron ID
    std::vector<std::vector<int> > labels_;  // neuron IDs for labels. labels are indices of first vector dimension
};


// connection generator to randomize initial starting weights according to a normal distribution
// std of init weight is initWeight / 2
class RandomNormalConnectionGen : public ConnectionGenerator {
public:
    RandomNormalConnectionGen(double initWeight, double maxWeight, float connProb) : maxWeight_(maxWeight), connProbInverted_(1.0 - connProb) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        gen_ = std::default_random_engine(seed);

        float stdValue = initWeight / 2;
        normalGen_ = std::normal_distribution<float>(initWeight, stdValue);
        uniformGen_ = std::uniform_real_distribution<float>(0, 1.0);
    };
    ~RandomNormalConnectionGen() {};

    void connect(CARLsim* s, int srcGrpId, int i, int destGrpId, int j, float& weight, float& maxWt, float& delay, bool& connected) {
        float initialWeight;
        // sets initial weight between 0 and max weight according to a normal distribution
        do {
            initialWeight = normalGen_(gen_);
        } while (initialWeight < 0.0 or initialWeight >= maxWeight_);

        weight = initialWeight;
        maxWt = maxWeight_;
        delay = 1.0;

        if (uniformGen_(gen_) >= connProbInverted_)
            connected = true;
        else
            connected = false;
    }
private:
    float maxWeight_;  // maximum connection weight
    float connProbInverted_;  // 1 - connection probability; used to determine if connection is made between synapses
    std::default_random_engine gen_;  // random number generator used to create random distributions
    std::normal_distribution<float> normalGen_;  // normal distribution generator to determine init weight
    std::uniform_real_distribution<float> uniformGen_;  // used to determine if a connection is made
};

// implements full-no-direct connections between different groups with the same dimensions / num neurons
// CANNOT randomize starting weights
// must connection groups with same dimensions
class FullNoDirect : public ConnectionGenerator {
public:
    FullNoDirect(double initWeight, double maxWeight, float delay) {
        initWeight_ = initWeight;
        maxWeight_ = maxWeight;
        delay_ = delay;
    }
    void connect(CARLsim* s, int srcGrpId, int i, int destGrpId, int j, float& weight, float& maxWt, float& delay, bool& connected) {
        if (s->getGroupNumNeurons(srcGrpId) != s->getGroupNumNeurons(destGrpId))
            throw std::runtime_error("FullNoDirect::connect neurons groups must have same number of neurons");

        weight = initWeight_;
        maxWt = maxWeight_;
        delay = delay_;

        if (i != j)
            connected = true;
        else
            connected = false;
    }

private:
    float initWeight_;
    float maxWeight_;
    float delay_;
};


// random weight connection generator according to uniform distribution
// delay is always set to 1.0
class RandomUniformConnectionGen : public ConnectionGenerator {
public:
    RandomUniformConnectionGen(double maxWeight, float connProb) : maxWeight_(maxWeight), connProbInverted_(1.0 - connProb) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        gen_ = std::default_random_engine(seed);
        uniformGen_ = std::uniform_real_distribution<float>(0, 1.0);
    };
    ~RandomUniformConnectionGen() {};

    void connect(CARLsim* s, int srcGrpId, int i, int destGrpId, int j, float& weight, float& maxWt, float& delay, bool& connected) {
        // weight is in uniform distribution between 0 and max weight
        weight = uniformGen_(gen_) * maxWeight_;
        maxWt = maxWeight_;
        delay = 1.0;

        if (uniformGen_(gen_) >= connProbInverted_)
            connected = true;
        else
            connected = false;
    }
private:
    float maxWeight_;  // maximum connection weight
    float connProbInverted_;  // 1 - connection probability; used to determine if connection is made between synapses
    std::default_random_engine gen_;  // random number generator used to create random distributions
    std::uniform_real_distribution<float> uniformGen_;  // used to determine if a connection is made & init weight
};




// allows stepping through simulation via user input
// if user inputs an integer returns the value
int step();

// to generate newline call func with no args - used to terminate progress bar
void progressBar(int cur = 1, int max = 1);

// displays two spike frequency grids side by side, square grids only
// dim1 must be larger or equal to dim2
void displayASCIIGrids(SpikeMonitor* grid1, int dim1, SpikeMonitor* grid2 = nullptr, int dim2 = 0, SpikeMonitor* grid3 = nullptr, int dim3 = 0);

//square grids only
void displaySpikeGrid(SpikeMonitor* grid, int dim, std::string name);






#endif //MNISTMODEL_MNISTUTILS_H
