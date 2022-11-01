//
// Created by patrick on 7/31/20.
//

#ifndef MNISTMODEL_MODELEVALUATIONS_H
#define MNISTMODEL_MODELEVALUATIONS_H

#include "../inc/Simulation.h"
#include "../inc/SimMNIST.h"

#include <parameters.hpp>
#include <bayesopt.hpp>


#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>

// for polling thread
#include <pthread.h>
// https://stackoverflow.com/questions/40903215/how-can-i-interrupt-a-loop-with-a-key-press-in-c-on-os-x
static volatile bool continueSearch = true;

// input polling function for quiting random searches
// called in a different thread from main program
// user must input 'q' then enter to stop search. only polled at end of generation, stop may take time
static void* userInput_thread(void*) {
    while (continueSearch) {
        if (std::cin.get() == 'q')
            continueSearch = false;
        sleep(2);
    }
}




//// random search hyper parameter optimizers

void randomSearchHyperParams(int population = 10);

void randomSearchHyperParamsMinimal(int population = 10);


//// bayes optimization runners for evaluating hyper parameters

void bayesFullSearch(bool loadFile, bool saveData, const std::string &resultsFile = "output/bayesfull.dat");

void bayesTestRunner();

void bayesSearchHyperParams(bool loadFile = false, bool saveData = false,
                            const std::string &resultsFile = "output/bayesopt.dat");

void bayesSearchHyperParamsMinimal(bool loadFile = false, bool saveData = false,
                                   const std::string &resultsFile = "output/bayesoptminimal.dat");

void cobaTrivialSearchBayes();

void cobaMnistSearch();



//// derived classes for implementing bayes optimization

// 14 hyper parameters, random connections
class DiehlAndCook_bayesopt : public bayesopt::ContinuousModel {
public:
    DiehlAndCook_bayesopt(bayesopt::Parameters param) : bayesopt::ContinuousModel(14,param) {};
    double evaluateSample(const boost::numeric::ublas::vector<double> &query);
};


// 5 hyper parameters, random connections
class DiehlAndCook_bayesoptMinimal : public bayesopt::ContinuousModel {
public:
    DiehlAndCook_bayesoptMinimal(bayesopt::Parameters param) : bayesopt::ContinuousModel(5,param) {};
    double evaluateSample(const boost::numeric::ublas::vector<double> &query);
};


// test class, for debugging only
class BayesTest : public bayesopt::ContinuousModel {
public:
    BayesTest(bayesopt::Parameters param) : bayesopt::ContinuousModel(1,param) {};
    double evaluateSample(const boost::numeric::ublas::vector<double> &query);
};


// 7 hyper parameter optimization for full synapse connection
class BayesFull : public bayesopt::ContinuousModel {
public:
    BayesFull(bayesopt::Parameters param) : bayesopt::ContinuousModel(7,param) {};
    double evaluateSample(const boost::numeric::ublas::vector<double> &query);
};

// 4 hyper parameter optimization for trivial coba
class cobaTrivialBayes : public bayesopt::ContinuousModel {
public:
    cobaTrivialBayes(bayesopt::Parameters param) : bayesopt::ContinuousModel(4,param) {};
    double evaluateSample(const boost::numeric::ublas::vector<double> &query);
};

// 7 hyper parameter optimization for full synapse connection
class mnistCoba : public bayesopt::ContinuousModel {
public:
    mnistCoba(bayesopt::Parameters param) : bayesopt::ContinuousModel(4,param) {};
    double evaluateSample(const boost::numeric::ublas::vector<double> &query);
};





#endif //MNISTMODEL_MODELEVALUATIONS_H












