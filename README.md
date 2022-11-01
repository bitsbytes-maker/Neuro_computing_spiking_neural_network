# Overview
The purpose of this tool is to create directed graphs of Spiking Neural Networks from Convolutional Neural Networks built and trained in Keras. The workflow consists of Keras (Tensorflow api), SNN_Toolbox, and this graph creation tool. 

First a dataset is loaded and prepared for Keras. Formatted copies of the dataset are saved for later simulation by SNN_Toolbox. Next a Keras CNN is created. Currently, graph creation only supports Dense (fully connected), 2d Convolutional, Average Pooling, Flatten, Dropout, and Batch Normalization Keras layers. Other restrictions to CNN architecture are described later. Once the CNN is defined, it is trained and evaluated in Keras. At this point the network is a trained CNN and implemented in Tensorflow.

Once a CNN is created and trained, SNN_Toolbox translates the CNN into a SNN in two steps. First, the CNN is parsed into an intermediate CNN where Dropout and Batch Normalization layers are removed / incorporated into adjacent layers. The weights for each layer are normalized at this stage. Second, the parsed CNN is converted into a SNN model. SNN Toolbox supports [several](https://snntoolbox.readthedocs.io/en/latest/guide/intro.html#simulating) simulator backends, but currently the built-in INI backend is used with a [temporal mean rate approximation](https://snntoolbox.readthedocs.io/en/latest/guide/configuration.html#conversion). No training is performed once the network is converted into a SNN.

The trained SNN is then simulated against test data. Spike events for each neuron and the classification accuracy is recorded for graph creation. Simulation input can be either constant current or poisson spike trains.

Finally, the graph creation tool uses the SNN architecture and spike event data to translate the CNN connectivity between layers into equivalent synaptic connections. The edges are the dot product of the pre-synaptic neuron fire rate and the weight of the connection.


## Work flow
Base model (Keras):
- The ANN is created with the Keras machine learning package
- The ANN is built, compiled, and trained with Keras

Parsing (SNN Toolbox):
- The trained base model is parsed into a modified Tensorflow model before conversion into a SNN
- Weight normalization is performed

Conversion (SNN Toolbox):
- The parsed model is used to produce an equivalent SNN

Simulation (SNN Toolbox):
- The converted SNN model is simulated in the INI simulator to generate spike trains
- spike trains are used to calculate fire rates

Graph creation:
- The weights from the parsed model and fire rates from the SNN simulation are used to construct a directed graph


## Install
The conda environment is already installed globally on the lab machine.


Create Conda environment: 
```
conda create --name snntoolbox python=3.8
conda activate snntoolbox
pip install tensorflow-gpu==2.2 tensorflow==2.2
conda install cudnn==7.6
pip install snntoolbox tabulate pytest sklearn matplotlib imgaug
```

Activate Conda environment: `conda activate snntoolbox`

## no conda install
I recommend using a Python virtual environment with Python versionnn 3.8. The package is untested for other versions. `cudnn` may not be installable through pip, so GPU may not be useable without system packages (untested).

```bash
pip install -r requirements.txt
``` 


## Running snnbuilder
The `snnbuilder` tool is setup to be run as a python package. First activate the Conda environment with `conda activate snntoolbox`. Then the following code can either be run from the python interactive interpreter, or from a python script.
```python
from snnbuilder.models import mnist

# optionally send kwargs to constructor
# Default output path is SNN_Toolbox/outputs/
x = mnist.CNN_Mnist(samples=100, epochs=5, output_path='../path/to/output/dir')
x.train()  # train the tensorflow model
x.parse()  # parse the trained tensorflow model
x.sim()  # convert the parsed model into a SNN, simulate
x.graph()  # graph the simulation outputs
```

## Running from terminal
- activate conda environment: `conda activate snntoolbox`
- run `python run.py MODEL_NAME` to run a specific model
- output is written to `PACKAGE_DIR/outputs/` by default
- change output directory via `--path` arg: `python run.py MODEL_NAME --path outputs`
- run `python run.py --help` for list of current models


## Package layout
```
examples/
outputs/
	- by default snn cache, models, and results are written here
snnbuilder/
	bin/
		- functional modules for snnbuilder
	models/
		- this is where snn models are defined
		- any new models should be placed here
```

## snnbuilder outputs
snnbuilder uses a cache and a simulation output dir to store the results of a run. Training and parsing are output inside a cache `0_cache/` in the ouput directory for a given model directory. When a simulation is run, the contents of the cache are copied into a timestamped directory in the model output directory, and the outputs of the simulation are stored only in the timestamped output. The cache is not updated during simulation or graphing. Graphing uses the most recent timestamp directory by default, though a different target directory can be passed as an argument. 

Output files
```
MODEL is the name of a given model
MODEL.h5  # tensorflow model
MODEL_parsed.h5  # parsed tensorflow model
MODEL_INI.h  # converted snn model, not loadable in the current version of snntoolbox
config  # config file used by snntoolbox internally, contains the non-default simulation parameters
graph  # pickled directed graph of the snn synapses
log.txt  # model and simulation details, human readable
sim_accuracies.json  # pickle of simulation results
spike_rate_sums.npz  # numpy array of the summed spike events, by neuron, for the entire simulation. used for graph creation
training_history.csv  # dump of the epoch history generated by tensorflow during training
training_metrics.json  # pickled file used for log generation internally
training_plot.png  # training loss and accuracy by epoch from the tensorflow training
.config  # the full config file used by snntoolbox for a simulation. This is where to look to determine exactly what parameters were used in a simulation
```

## Pipeline logic
basic overview 

`Pipeline` is the base class for all snn models created. To create new snn models, an intermediate "dataset" `Pipeline` child class should be created for the target dataset, if one is not already defined. This intermediate class should define the dataset setup and loading, and override `train()`, `parse()`, and `sim()` as required to adapt to the new dataset. The actual snn model pipelines which define a model architecture should inherit from the intermediate "dataset" pipeline. Ideally, the final pipeline class should only define the `model()` method and handle `__init__()` hyper-parameters.


# Working Models
The following networks are 'complete' with documented output and graphs.

#### ModelPipeline
- `Autoencoder_Mnist` [from kaggle tutorial](https://www.kaggle.com/shivamb/how-autoencoders-work-intro-and-usecases)
- `Mini_Network` a small example network

#### MnistPipeline
- `CNN_Mnist` (arch taken from Balaji et al. 2019)
- `LeNet_Mnist` (28x28 input)
- `Zambrano_Mnist` (taken from [Zambrano, Bohte 2016](https://arxiv.org/pdf/1609.02053.pdf))
- `MultiLayerPerceptron_Mnist` (MLP)

#### Cifar10Pipeline
- `LeNet_Cifar10`
- `Rueckauer_Cifar10` [Rueckauer et al 2017](https://www.frontiersin.org/articles/10.3389/fnins.2017.00682/full)
	Max pool layers replaced with Average pooling

#### CatsVsDogsPipeline
https://www.kaggle.com/c/dogs-vs-cats 
Setup for dataset is in `catsvsdogs.py`
- `CNN_CatsVsDogs`
- `AlexNet_CatsVsDogs`

### SqueezeNet architecture
[Squeezenet](https://arxiv.org/pdf/1602.07360.pdf)
1. Model uses version 1.1 of squeezenet
2. Maxpool layers are replaced with average pool
3. Dense layer added as output layer
Working:
- `SqueezeNet_CatsVsDogs`

### Fruits-360
[kaggle dataset](https://www.kaggle.com/moltean/fruits)
- `CNN_Fruits360`
- `SqueezeNet_Fruits360`


# Unstable / broken models

#### Mnist224Pipeline
- `Alexnet_Mnist224`

#### ImageNetPipeline
- `Alexnet_ImageNet`
- all other imagenet models

#### Mnist224Pipeline
- `SqueezeNet_Mnist224`


# Deprecated
#### MnistPipeline
- `LeNet_padded_Mnist` (32x32 input)



# References
- [SNN toolbox docs](https://snntoolbox.readthedocs.io/en/latest/index.html)
- [SNN toolbox github](https://github.com/NeuromorphicProcessorProject/snn_toolbox)
- [SNN toolbox article 1](https://www.frontiersin.org/articles/10.3389/fnins.2017.00682/full)
- [SNN toolbox article 2](https://ieeexplore.ieee.org/abstract/document/8351295)
- [guide to convolution arithmetic](https://arxiv.org/pdf/1603.07285.pdf)
- [computing receptive fields of CNNs](https://distill.pub/2019/computing-receptive-fields/)

## Notes
- [pooling windows](https://datascience.stackexchange.com/questions/67334/whats-the-purpose-of-padding-with-maxpooling)
- the snn toolbox calcuates number of synapses different from graph creation code. The toolbox does not account for padding in conv layers, inflating synapse count with "imaginary" pre-neurons. [Code here](https://github.com/NeuromorphicProcessorProject/snn_toolbox/blob/96b08e3708120dbf647e477916f1ccdb26d7bb96/snntoolbox/parsing/utils.py#L1167).
- snn toolbox does not include input neurons in total neuron count
- Batch normalization: batch normalization layers must be placed between the conv layer and the non-linear activation layer. Otherwise the parser cannot fuse the BN layer with the preceeding conv layer.
- for output graph: edge of 0 indicates either the synapse weight or the fire rate was 0 for the simulation. It does not mean 'no connection'.
- graph creation supports weight output only via `weights_only` arg to `graph.build_graph`
- reshaping layers (Flatten, Concatenate) can't be in sequence with current implementation
- if the final layer is `Flatten` (as in global average pooling replacement) the output shape will not take the flatten into account, will be the shape of the prior layer.
- `train()` and `parse()` use a single `0_cache` directory to store results. `sim()` creates a copy of the cache before performing the simulation. This saves the results for an entire experiment.
- batch size must be identical for tensorflow training, parsing, and simulation. Different batch sizes creates dependency conflicts inside snntoolbox. The simplest solution was to force a uniform batch size.


## Limitations of SNN Toolbox
The primary purpose of snntoolbox is to convert neual networks, defined and trained in tensorflow, into spiking neural networks. The conversion process has limitations which can prevent any snn from being created, or results in a snn with very poor accuracy.

### workflow 
- the source model must be from tensorflow
- the source model must be trained in tensorflow
- the output snn model *can not* be trained once converted into a snn
- once converted, the snn can only be simulated and have weights / biases extracted

### conversion 
- snntoolbox only supports the conversion of certain types of tensorflow layers. Only supported layers can be present in a model to be converted into snn.
- supported layers in snntoolbox 0.5 are as follows:
```
# Layers that have neurons with membrane potentials, from which we can measure spikes:
spiking_layers = {'Dense', 'Conv1D', 'Conv2D', 'DepthwiseConv2D', 'Conv2DTranspose', 'UpSampling2D', 'MaxPooling2D',
                  'AveragePooling2D', 'Sparse', 'SparseConv2D',
                  'SparseDepthwiseConv2D'}
# Layers that can be implemented by our spiking neuron simulators:
snn_layers = %(spiking_layers)s | {'Reshape', 'Flatten', 'Concatenate',
                                   'ZeroPadding2D'}
```
- of those layers supported by snntoolbox, this tool (snnbuilder) supports only a subset of layers. Each layer type supported requires graph creation code to generate the direct graph of synapses, and this has not been done for all layer types. Most layers used in convolutional networks are supported. see `./docs/conversion.md` for details.
- snntoolbox does not support recurrent layers
- the default spike rate encoding (`temporal_mean_rate`) used by the backend `INI` simulator does not support Max Pool layers. Other spike encoding do support max pool layers, but can cause crashes for some models. As most convolutional networks use max pooling, there are a few work arounds.
	1. replace all max pool layers with average pool layers in source models. This can cause issues for networks like SqueezeNet (either in tensorflow training, or accuracy crash during simulation), but works for the most part for simple models.
	2. enable `max2avg` layer conversion during parsing. All max pool layers are replaced with average pooling. Because this does not update any other weights, enabling this has thus far just resulted in poor accuracy models.
	3. use another spike encoding, untested
	4. Update snntoolbox to the most recent dev version. Max pool support for `temporal_mean_rate` has been added. This is untested for snnbuilder, good luck

### normalization 
- normalization of the parsed model is critical to accuracte snn, but normalization is limited by the avaliable system memory. This is because normalization is done with using an activation quantile (99% default) using numpy. If a layer has too many weights to be loaded into memory several times (numpy requires about 4n space, where n is number of weights), then the normalization will fail. Online quantile estimation techniques are slow, and generally not practical.

### simulation
- during simulation of the snn, snntoolbox writes the entire spike train for a batch to disk. This is done via a numpy compression call inside snntoolbox using pickle protocol 3. This protocol has an object size limit for serialization of 4GiB. This does not *exactly* equate to the bytes required by the numpy array, but there is a limit to the batch and network size which can be simulated. If the simulation crashes due to a numpy serialization crash, try to reduce the batch size for simulation. If that fails, then the network has too many neurons to be written to disk.
- rate encoding determines how the backend simulator handles spikes during simulation (default backend is `INI`). The default encoder is `temporal_mean_rate`. Other encoders exist for snntoolbox, but there are some bugs / strange interactions in the stable release of snntoolbox (0.5). The other spike encodings can cause errors during simulation, so `temporal_mean_rate` has been used for all simulations
- the above may not hold for all networks, and future versions (or dev release) of snntoolbox may solve these issues. More through experiments with simulators may be warranted
- very deep networks take time for spiking signals to propagate from the input layer to the output layer. The deeper the network, the longer a sample must be simulated to get high accuracy. Networks which are too deep (subjective, no hard data) seem to suffer from signal degredation. Complex models which should be able to memorize a dataset fail to achieve the accuracy of a simpler, shallower, network.



# dataset setup
To set up catdog, activate conda environment then run `python -m models.catsvsdogs`, optionally with source and target paths