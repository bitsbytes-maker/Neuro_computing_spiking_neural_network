/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*
* 3. The names of its contributors may not be used to endorse or promote
*    products derived from this software without specific prior written
*    permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* *********************************************************************************************** *
* CARLsim
* created by: (MDR) Micah Richert, (JN) Jayram M. Nageswaran
* maintained by:
* (MA) Mike Avery <averym@uci.edu>
* (MB) Michael Beyeler <mbeyeler@uci.edu>,
* (KDC) Kristofor Carlson <kdcarlso@uci.edu>
* (TSC) Ting-Shuo Chou <tingshuc@uci.edu>
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/

#include <carlsim.h>
#include <cmath>
#include <vector>
#include <iostream>
using std::vector;
using std::isnan;
using std::cout;
using std::endl;
using std::abs;

const int maxFireRate = 200;


//functions can overflow if input is over max fire rate
bool outFreqHigherThanTarget(float input, float output) {
	if (abs(input - maxFireRate) < output)
		return true;
	return false;
}

bool outFreqLowerThanTarget(float input, float output) {
	if (abs(input - maxFireRate) > output)
		return true;
	return false;
}

void printRateDifferenceFromInversion(SpikeMonitor* inputMon, SpikeMonitor* outputMon) {
	float sum = 0.0f;
	float diff = 0.0f;

	for (int i = 0; i < 9; ++i) {
		diff = abs(inputMon->getNeuronMeanFiringRate(i) - maxFireRate) - outputMon->getNeuronMeanFiringRate(i);
		cout << diff << " ";
		sum += abs(diff);
	}
		cout << " avg: " << sum / 9 << endl;
}

void printRateGrid(SpikeMonitor* monitor) {
	int index = 0;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			index = (i * 3) + j;
			cout << monitor->getNeuronMeanFiringRate(index) << " ";
		}
		cout << endl;
	}
}

int main() {

	const float weightStep = 0.5f;
	vector<float> inputRates{
		50.0f,125.0f,100.0f,
		75.0f,175.0f,75.0f,
		100.0f,125.0f,50.0f};
	

	Grid3D gridDim(3, 3, 1);

	CARLsim sim("invert", GPU_MODE, USER);

	int gIn = sim.createSpikeGeneratorGroup("input", gridDim, EXCITATORY_NEURON);
	int gExt = sim.createGroup("ext", gridDim, EXCITATORY_NEURON);
	int gOut = sim.createGroup("output", gridDim, EXCITATORY_NEURON);

	//regular spiking neuron settings
	sim.setNeuronParameters(gExt, 0.02f, 0.2f, -65.0f, 8.0f);  
	sim.setNeuronParameters(gOut, 0.02f, 0.2f, -65.0f, 8.0f);

	int connInExt = sim.connect(gIn, gExt, "one-to-one", RangeWeight(300.0f), 1.0f, RangeDelay(1));

	int connExtOut = sim.connect(gExt, gOut, "one-to-one", RangeWeight(0.0f,0.0f,200.0f), 1.0f, RangeDelay(1), RadiusRF(0,0,0), SYN_PLASTIC);
	int connExtFeedback = sim.connect(gExt, gExt, "one-to-one", RangeWeight(0.0f,0.0f,200.0f), 1.0f, RangeDelay(1), RadiusRF(0,0,0), SYN_PLASTIC);


	sim.setConductances(false);  //COBA true, CUBA false

	sim.setupNetwork();

	PoissonRate rates(9, true);
	rates.setRates(inputRates);
	sim.setSpikeRate(gIn, &rates);

	SpikeMonitor* spkMonIn = sim.setSpikeMonitor(gIn,"DEFAULT");
	SpikeMonitor* spkMonOut = sim.setSpikeMonitor(gOut,"DEFAULT");
	SpikeMonitor* spkMonExt = sim.setSpikeMonitor(gExt,"DEFAULT");

	ConnectionMonitor* connMonExt = sim.setConnectionMonitor(gExt, gOut, "default");

	//disables periodic storing of snapshots to binary
	connMonExt->setUpdateTimeIntervalSec(-1);  

	//initial print of connection states
	cout<< endl;
	connMonExt->print();  //feedback connection are same as excititory 

	cout << endl;
	cout << "weight differences through time steps:" << endl;

	for (int i = 0; i < 500; ++i) {
		spkMonIn->startRecording();
		spkMonOut->startRecording();
		spkMonExt->startRecording();
		sim.runNetwork(1,0, false);  //true false sets if connection snapshot printed to terminal
		spkMonIn->stopRecording();
		spkMonOut->stopRecording();
		spkMonExt->stopRecording();

		vector<vector<float> > weightSnapshotExt = connMonExt->takeSnapshot();

		if (!(i % 10))
			printRateDifferenceFromInversion(spkMonIn, spkMonOut);

		for (int j = 0; j < 9; ++j) {//loops over pre-synaptic and post, are one to one
			vector<float> extConns = weightSnapshotExt[j];

			if (outFreqHigherThanTarget(spkMonIn->getNeuronMeanFiringRate(j), spkMonOut->getNeuronMeanFiringRate(j))) {
				if (!isnan(extConns[j]) and extConns[j] - weightStep > 0)  {//can't set negative weight
						sim.setWeight(connExtOut, j, j, extConns[j] - weightStep, false);
						sim.setWeight(connExtFeedback, j, j, extConns[j] - weightStep, false);
					}
			}
			else if (outFreqLowerThanTarget(spkMonIn->getNeuronMeanFiringRate(j), spkMonOut->getNeuronMeanFiringRate(j))) {
				if (!isnan(extConns[j])) {  
					sim.setWeight(connExtOut, j, j, extConns[j] + weightStep, false);
					sim.setWeight(connExtFeedback, j, j, extConns[j] + weightStep, false);
				}
			}
		}
	}

	cout << endl << "input rate grid:" << endl;
	printRateGrid(spkMonIn);
	cout << endl << "output rate grid:" << endl;
	printRateGrid(spkMonOut);
	cout << endl;

	//final print of connection states
	connMonExt->print();
	//connMonInhib->print();

	return 0;
}
