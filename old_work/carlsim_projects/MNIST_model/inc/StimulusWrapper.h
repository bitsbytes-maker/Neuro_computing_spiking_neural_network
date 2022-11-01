//
// Created by patrick on 7/28/20.
//

#ifndef MNISTMODEL_STIMULUSWRAPPER_H
#define MNISTMODEL_STIMULUSWRAPPER_H

#include <carlsim.h>
#include <visual_stimulus.h>

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <iostream>

class StimulusWrapper {
public:
    StimulusWrapper();
    StimulusWrapper(const std::string &presentationFile, const std::string &blankFile, const std::string &csvFile,
            float maxRate=50.0f, float minRate=0.0f);

    void importGroundTruthCSV(const std::string &csvFile);
    void importPresentation(const std::string &presentationFile);
    void importBlank(const std::string &blankFile);
    PoissonRate* getNextFrame();
    PoissonRate* getBlankFrame();
    void resetFrameIndex();

    //if no frame has been read, current frame is -1, and 0 will be returned
    int getLabel();  //returns current label
    std::vector<int> getDim();  //returns width, height, channels
    int getLength();

private:
    // visual stimulus has no default constructor, used pointer instead as work around to enable
    // wrapper class member usage
    VisualStimulus* presentation_;
    VisualStimulus* baseline_;
    float maxRate_;
    float minRate_;
    std::vector<int> groundTruthLabels_;
    int frameIndex_;
};

#endif //MNISTMODEL_STIMULUSWRAPPER_H
