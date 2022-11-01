project(mnistModel)
cmake_minimum_required(VERSION 3.0)
#set(carlsim_DIR "/media/patrick/HDD/VLSI_VOC/CARLsim4-4.0/cmake")
#list(APPEND CMAKE_PREFIX_PATH "/media/patrick/HDD/VLSI_VOC/CARLsim4-4.0/cmake/")
#set(CMAKE_PREFIX_PATH "/media/patrick/HDD/VLSI_VOC/CARLsim4-4.0/cmake")

#include_directories("/media/patrick/HDD/VLSI_VOC/CARLsim4-4.0/")

#add_subdirectory("/media/patrick/HDD/VLSI_VOC/CARLsim4-4.0" "carlbuild")

include_directories("../../CARLlibraries/include/")
include_directories("/usr/local/include/bayesopt/")

#include_directories("../../CARLsim4/external/googletest/include/gtest")

# Dependencies
#find_package(carlsim REQUIRED)

# Targets
add_executable(MNIST_model src/main_MNIST_model.cpp inc/StimulusWrapper.h src/StimulusWrapper.cpp
        inc/mnistUtils.h src/mnistUtils.cpp src/Simulation.cpp inc/Simulation.h inc/modelEvaluations.h
        src/modelEvaluations.cpp src/SimMNIST.cpp inc/SimMNIST.h testsuite/simFeat_test.cpp src/prototypes.cpp src/basicNetworks.cpp src/Network.cpp inc/Network.h inc/basicNetworks.h inc/prototypes.h testsuite/validateSim.cpp src/SyntheticSimulation.cpp inc/SyntheticSimulation.h)

# Linking
#target_link_libraries(MNIST_model PUBLIC carlsim)
target_link_libraries(MNIST_model)
