# CNN to SNN Transformation
The theory behind CNN to SNN conversion is described by the SNN Toolbox authors in [Rueckauer et al., 2017](https://www.frontiersin.org/articles/10.3389/fnins.2017.00682/full). 

### Normalization
Normalization of CNN weights is critical for accurate SNN models. Unfortunately, the SNN Toolbox implementation requires the layer activations for the entire normalization dataset to be loaded into GPU memory at once. As a result, large models with large datasets can't be normalized on most GPU's. I created a normalization work around which generates identical results, but shifts the space constraint to system memory. The work around is only implemented for `npz` and `jpg` type datasets (as defined in SNN Toolbox documentation, not literal jpg files). The work around can be invoked via passing the `norm_workaround=True` kwarg to the `parse()` method in the tool. The work-around is the default behavior.

### Notes
#### Conversion
- the weight conversion uses a membrane equation for which the fire rate of spiking neurons is proportional to the weights in the CNN.
- weights and biases are normalized using a normalization factor set to a k-percentile of the total activity distribution of the layer.
- normalization prevents approximation errors due to very low and very high firing rates.
- SNN weights can be positive or negative

####Simulation
- input to the simulation can either be Poisson spike trains or constant current. The later produces better accuracy.
- after a spike the membrane potential resets by subtraction of the voltage threshold, not to zero. This prevents loss of detail and inaccuracies in deeper networks.
- normalized biases are used in the simulation in addition to weights

## Graph Creation
The final output is a directed graph of the SNN network synapses and a log file describing the network setup. The graph is written to file as raw bytes using `pickle`. A conversion utility exists to translate the byte dump into a csv file. Each synapse has the following attributes:
```
source_group  	# name of the pre-synaptic neuron layer
source_neruon  	# index of the pre-synaptic neuron. Index can be 1-3 dimensions depending on the layer
dest_group  	# name of the post-synaptic neuron layer
dest_neuron  	# index of the post-synaptic neuron. Index can be 1-3 dimensions depending on the layer
edge  			# dot product of the neuron's average fire rate and the synaptic weight. Can be positive or negative.
```


Graph creation requires two sets of input. First is the trained weights created by SNN Toolbox and second is the spike trains for each layer produced by the SNN simulation against test data. The spike train data is a record of every timestep in the simulation and which timesteps each neuron fired. The spike train data is summed into total spikes per neuron, and then into average spikes rates using the total simulation time. The spike rate data does not describe which example class the neuron spiked in response to, nor if the SNN correctly classified that example. The spike rate data only describes the average spike rate for each neuron across the entire simulation.


### Layer Shapes
Neuron groups created by the SNN Toolbox have between 1 and 3 dimensions. In the context of image data, the first two dimensions are rows and columns of image pixels. The 3rd dimension is channels, for example Red Green Blue colors which together form the image. Not all input images have multiple color channels, but they all have a third dimension. 


Convolutional 2D layers connect 3D input layers to 3D output layers. Each index in the 3rd dimension of a convolutional layer is a filter. Pooling layers maintain the channel count of their input neuron layers.


## Layer Types
SNN Toolbox supports a subset of possible Tensorflow layers for conversion into SNN. The layers can be divided into 4 categories: Parsable, Convertible, Spiking, and Reshaping. Of the layers supported by SNN Toolbox, a subset is supported by the graph creation tool. The layers listed below are supported both by SNN Toolbox and the graph creation tool.


**Parsable**: layers which can be present in the base Tensorflow model. 
`[Input, Conv2D, Dense, AveragePooling2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Activation, Concatenate, GlobalAveragePooling2D]`

**Convertible**: layers which can be converted to SNN after parsing by SNN Toolbox.
`[Input, Conv2D, Dense, AveragePooling2D, MaxPooling2D, Flatten, Concatenate]`

**Spiking**: layers which produce spikes during simulation.
`[Input, Conv2D, Dense, AveragePooling2D, MaxPooling2D]`

**Reshaping**: layers which alter the shape of the neuron layer, but do not generate spikes nor are represented in the final graph output.
`[Flatten, Concatenate]`


### Dense
Dense layers are fully connected. Each neuron in the pre-synaptic group is connected to each neuron in the post-synaptic group.


### Convolutional 2d
For each stride of the kernel across the input group, synapses are created from each input neuron within the kernel to the respective output neuron. This is done for each filter in the convolutional kernel, creating connections between input and output neurons for each position of the kernel. When the kernel size is larger than the stride, input neurons are connected to multiple output neurons. 


#### Notes
- kernel must have same number of channels as the input neuron group
- arbitrary strides are supported, but width and height dimensions must be identical
- all kernels must be square, ie (k, k)
- `same` padding is supported, but kernel size must be odd
- input layer dimensions must be square, ie height == width. This implies Conv layer must also be square


#### Mapping from pre-neurons to post-neurons:
```
pre-neuron group A with shape (height h', width w', channels c)
	where i(h,w,c) is a pre-neuron
post-neuron group B with shape (height h, width w, filters f)
	where o(h,w,f) is a post-neuron
kernel weights K with shape (k_h, k_w, c, f) where k_h == k_w == k

Requirement: height == width, referred to as 'input_dim' when calculating output dimensions

k = kernel size, which must be odd if padding is non-zero
s = stride length
p = padding  # depends on the padding type, either 'same' or 'valid'

Output layer dimensions depend on the input dimensions, kernel, stride, and padding type
	if padding type is 'valid'
		output dimensions = ceil((input_dim - k + 1) / s)  # height == width
		padding = 0
	if padding type is 'same'
		output dimensions = ceil(input_dim / strides)  # height == width

		if input_dim % strides == 0
			padding = max(k - s, 0) // 2
		else
			padding = max(k - (input_dim % s), 0) // 2


Synapses connect to a post-neuron o(h,w,f) with edges according to the following mapping:
S( o(h,w,f) ) = {i.fire_rate * K[k_h][k_w][c][f]} 
	For Every i(h,w,c) (element of) 
		{ i(h*s - p + k_h, w*s - p + k_c, c) | 
			k_h={0, ..., k-1}, 
			k_c={0, ..., k-1}, 
			c={0, ..., channels-1}
		}
	Such That i(h,w,c) (element of) (A) and therefore is a valid pre-neuron index 

Where K[k_h][k_w][c][f] is a connection weight in the kernel


All synapses are generated by iterating over all post-neurons and filters in the conv kernel:
For each o(h,w,f) in B
	Synapses += S(o(h,w,f))
```


####pseudocode
```
pre_neurons_shape = (rows_p, columns_p, channels)
post_neurons_shape = (rows, columns, filters)
kernel_shape = (kernel_rows, kernel_columns, channels, filters)

for each row in rows:
	for each col in columns:

		for each k_row in kernel_rows:
			for each k_col in kernel_columns:
				pre_neuron_row = ((row * stride) + k_row) - padding
				pre_neuron_col = ((col * stride) + k_col) - padding

				skip if pre neuron is invalid

				for each neuron in channels:
					for each filter in filters:
						create_synapse()
```


### Average Pooling 2d
Activations in an average pooling layer are not dependent on synaptic weights. The average fire rate of the pre-synaptic neurons connected to a post-synaptic neuron is applied as input to the post-synpatic neuron. Keras CNN does not train weights for pooling layers, and SNN Toolbox does not store weights. Pooling layers do have neurons and synapses in the output graph. The synapse weight are calculated with only the fire rate of the pre-synaptic neuron. The 'weight' of the synapse is set to a neutral value `1`. Pooling layers preserve channels like the original CNN. Synapses for average pooling connections use alternate calculations for destination neuron activation, see [this article](https://www.frontiersin.org/articles/10.3389/fnins.2017.00682/full).


#### Notes
- strides are supported
- padding is not supported
- pool shape must be square, ie (p, p)
- strides shape must be square, ie (s, s)
- strides != pool is supported


#### Pool Mapping
```
pre-neurons A with shape (height h, width w, channels c)
	where i(h,w,c) is a pre-neuron
post-neurons B with shape (height floor(h / pool), width floor(w / pool), channels c)
	where o(w,h,f) is a post-neuron
pool = pool size
pool_weight = 1  # placeholder with neutral effect

i(h,w,c) -> o(floor(h/pool), floor(w/pool), c)

Pool synapses from A -> B are creating according to the following mapping:
S( i(h,w,c) ) = {i.fire_rate * pool_weight} | o(floor(h/pool), floor(w/pool), c)

(pool * pool) pre-neurons connect to each post-neuron
```


### Flatten
Flatten layers in Keras are not represented in the SNN as neurons. Flatten layers instead serve to reshape the dimensions of the input layer to accomodate the shape of a following layer. Because Keras CNNs are dependent on the shape of layers in ways the converted SNN is not, SNN Toolbox does not create SNN layers. Flatten layers do not have weights or neurons associated with them. Flatten layers are in essence dimensional bookkeeping for Keras and not important in a SNN.

However, the weight tensors stored by SNN Toolbox have the shape of the flattened output group of neurons. Because neuron indexing for convolutional and pooling layers is 3D, flatten layers are un-flattened into the shape of the source neuron group. This is done to maintain a consistent neuron indexing scheme throughout the graph which matches the original CNN and preserves all spatial information.

The reshaping of the weights is done with built-in numpy methods.


### Dropout
Dropout layers do not appear in the parsed SNN model, they have neither weights or spike trains. SNN Toolbox removes the dropout layers completely before simulation.


### Batch Normalization
Normalization layers and their trainable parameters are merged into other SNN layers in the parsed model. Normalization layers do not have weights nor spike trains.

To apply batch normalization layers to a model, e.g. after a `Conv2D` layer, the following setup must be used. The layer preceding the `BatchNormalization` layer (in this case `Conv2D`) must have no activation function. The `BatchNormalization` layer must follow. The convolution -> normalization -> activation unit is concluded with an `Activation` layer. 
```
...
# conv2d -> batch normalization -> activation unit
x = layers.Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation(activation='relu')(x)

# model definition continues
x = layers.Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
...
```

SNN Toolbox will merge the batch normalized weights and activation function into the convolutional layer during the parsing stage.


### Concatenate 
Graph creation supports branching models through the use of Concatenate layers. Only channel axis concatenate is supported, and only Concatenate layers can have multiple input layers. 



### Activation
To support batch normalization, SNN Toolbox attempts to look ahead for `Activation` layers when it encounters `{'Dense', 'Conv1D', 'Conv2D', 'DepthwiseConv2D', 'Conv2DTranspose'}` layers (referenced as original layer in this section). If an `Activation` layer is found within the next 3 layers, the activation function from the `Activation` layer is merged into the original layer. `Activation` layers are never included in the parsed model, they are always skipped. When performing the look ahead, if a layer with an activation function is found and that layer is not an `Activation` layer (eg a `Conv2D` layer is found with `relu` activation), the search is terminated and the activation function (if any) of the original layer is used.


This can have unexpected outcomes. Consider the following layers:
```
x = layers.Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Activation(activation='softmax')(x)
model = keras.Model(input_layer, outputs, name=self.model_name)
```

Instead of creating a `softmax` activation function for the final layer of the model (which is how tensorflow will treat the example), SNN Toolbox will use the look ahead on the `Conv2D` layer. The `GlobalAveragePooling2D` layer does not have an activation function, so is skipped. The `Activation` layer with `softmax` is then encountered. SNN Toolbox will produce a parsed model with the `relu` activation function for the `Conv2D` layer replaced by `softmax`, and the final `Activaiton` layer will be removed entirely.


references:
[parsing logic](https://github.com/NeuromorphicProcessorProject/snn_toolbox/blob/master/snntoolbox/parsing/utils.py)
[layer absorption](https://github.com/NeuromorphicProcessorProject/snn_toolbox/blob/f7d89f339b1b55a95fa3b5b948f43b527f0fa7f0/snntoolbox/parsing/utils.py#L665)


### Global Average Pooling
If a `GlobalAveragePooling2D` pooling layer is present in the source model, SNN Toolbox will replace it with `AveragePooling2D` and `Flatten`. The pool size of the `AveragePooling2D` will be the input dimensions. The end result is identical. 


### Max Pooling 2D
Max pooling works similar to average pooling. There are no weights for a max pool layer. Keras CNN does not train weights for pooling layers, and SNN Toolbox does not store weights. There are several options for the max pool calcuation during simulation ('fir_max', 'exp_max', 'avg_max'), see snntoolbox documentation and literature for details. Pooling layers do have neurons and synapses in the output graph. The 'weight' of the synapse is set to a neutral value `1`. Pooling layers preserve channels like the original CNN. See `Limitations of SNN Toolbox` in the readme for more information on the limitations and issues with max pool layers.

