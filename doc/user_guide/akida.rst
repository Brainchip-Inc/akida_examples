
Akida user guide
================

Introduction
------------

Like many other machine learning frameworks, the core data structures of Akida
are layers and models, and users familiar with Keras, Tensorflow or Pytorch
should be on familiar grounds.

The main difference between Akida and other machine learning framework is that
instead of modeling traditional Artificial Neural Networks, Akida models aim at
representing `Spiking Neural Networks <https://en.wikipedia.org/wiki/Spiking_neural_network>`__,
i.e. interconnected graphs of neurons that *fire* when their potential reaches
a predefined *threshold*.

On another note, unlike other frameworks, Akida layers only use integer inputs,
outputs and weights.

Akida layers
^^^^^^^^^^^^^

Concretely, Akida layers can be represented by the combination of standard
machine learning layers into computation *blocks*:

- a Convolutional or Dense layer to evaluate the Spiking Neuron Potential,
- the addition of an inverted bias to represent the firing *threshold*,
- a ReLu activation to represent the neuron *spike*.

Three principal layer types are available:

* `FullyConnected <../api_reference/akida_apis.html#fullyconnected>`__
  – sometimes described as ‘dense’
* `Convolutional <../api_reference/akida_apis.html#convolutional>`__
  – or ‘weight-sharing’
* `SeparableConvolutional <../api_reference/akida_apis.html#separableconvolutional>`__,
  - a less computationally intensive convolutional layer.

The weights of Akida layers are N-bit integer: please refer to the `hardware
constraints <./hw_constraints.html>`__ for details of the supported bitwidth for
each layer.

Input Format
^^^^^^^^^^^^

Akida inputs and outputs are 4-dimensional tensors whose first dimension is the
index of a specific sample.

The inputs of Akida layers are N-bit integer: please refer to the `hardware
constraints <./hw_constraints.html>`__ for details of the supported bitwidth for
each layer.

A versatile machine learning framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Akida machine learning framework supports two main types of models:

- native SNN models and,
- deep-learning SNN models.

Native Spiking Neural Networks
""""""""""""""""""""""""""""""

Native SNN models are typically composed of a few Dense layers.
They require most of the time a specific feature extractor to feed the Dense
layers.
This feature extractor can either be completely external to the model, or be
a deep-learning SNN submodel as defined below.

The last `FullyConnected <../api_reference/akida_apis.html#fullyconnected>`__ layer
of a native SNN model can be trained online from individual samples using Akida
edge learning algorithm.
Please refer to `Using Akida Edge Learning <akida.html#id1>`_ for details.

Deep-learning Spiking Neural Networks
"""""""""""""""""""""""""""""""""""""

Deep-learning SNN models are genuine CNN models converted to Akida SNN models.

As a consequence, deep-learning professionals do not need to learn any new
framework to start using Akida: they can simply craft their models in
TensorFlow/Keras and convert them to Akida SNN models using the `CNN2SNN <./cnn2snn.html>`__
seamless conversion tool.

Unlike genuine CNN, deep-learning SNN cannot be trained online using
back-propagation: for deep models where online learning is required, it is
therefore recommended to import the weights of early layers from a pre-trained
CNN, and to apply Akida Edge learning only on the last layer.

The Sequential model
--------------------

Specifying the model
^^^^^^^^^^^^^^^^^^^^

Akida models are defined using the sequential API.

This comprises creating a ``Model`` object and adding layers to it using the
`.add() <../api_reference/akida_apis.html#akida.Model.add>`__ method.

The available layers are `InputData <../api_reference/akida_apis.html#inputdata>`__,
`InputConvolutional <../api_reference/akida_apis.html#inputconvolutional>`__,
`FullyConnected <../api_reference/akida_apis.html#fullyconnected>`__,
`Convolutional <../api_reference/akida_apis.html#convolutional>`__,
`SeparableConvolutional <../api_reference/akida_apis.html#separableconvolutional>`__
and `Concat <../api_reference/akida_apis.html#concat>`__.

Layers are built with a name and a list of named parameters that are described
in the sections below.

Example of sequential definition of a model:

.. code-block:: python

   from akida import Model, InputData, FullyConnected
   model = Model()
   model.add(InputData(name="input", input_shape=(32, 32, 1)))
   model.add(FullyConnected(name="fully", units=32, threshold=40))

The ``Model`` `.summary() <../api_reference/akida_apis.html#akida.Model.summary>`__
method prints a description of the model architecture.

Accessing layer parameters and weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The layers of a ``Model`` can be accessed either by their index or by their
name.

.. code-block:: python

   first_layer = model.layers[0]
   fc = model.get_layer("fully")

Each layer type has a different set of attributes, available through the ``Layer``
`.parameters <../api_reference/akida_apis.html#akida.Layer.parameters>`__ member:

.. code-block:: python

   fc = model.get_layer("fully")
   n = fc.parameters.units
   fc.parameters.weights_bits = 2

Some layer types also have variables containing weights and thresholds:

.. code-block:: python

   fc = model.get_layer("fully")
   weights = fc.variables["weights"]
   weights[0, 0, 0, 0] = 1
   fc.variables["weights"] = weights

Inference
^^^^^^^^^

The Akida ``Model`` `.forward <../api_reference/akida_apis.html#akida.Model.forward>`__
method allows to infer the outputs of a specific set of inputs.

Like inference methods in other machine learning frameworks, it simply returns
the integer potentials or activations of the last layer.

.. code-block:: python

    import numpy as np

    ...

    # Prepare one sample
    input_shape = (1,) + tuple(model.input_shape)
    inputs = np.ones(input_shape, dtype=np.uint8)
    # Inference
    outputs = model.forward(inputs)

The ``Model`` `.evaluate <../api_reference/akida_apis.html#akida.Model.evaluate>`__
method is very similar to the forward method, but is specifically designed to
replicate the float outputs of a converted CNN: instead of the integer potentials,
it returns float values representing the integer potentials shifted and rescaled using
per-axis constants evaluated during the CNN conversion.

After an inference, the ``Model`` `.statistics <../api_reference/akida_apis.html#akida.Model.statistics>`__ member provides relevant inference statistics.

.. code-block:: python

    import numpy as np

    ...

    # Prepare one sample
    input_shape = (1,) + tuple(model.input_shape)
    inputs = np.ones(input_shape, dtype=np.uint8)
    # Inference
    outputs = model.evaluate(inputs)
    assert outputs.dtype == np.float32

Saving and loading
^^^^^^^^^^^^^^^^^^

A ``Model`` object can be saved to disk for future use with the
`.save() <../api_reference/akida_apis.html#akida.Model.save>`__
method that needs a path for the model.

The model will be saved as a file with an .fbz extension that describes its
architecture and weights.

A saved model can be reloaded using the ``Model`` object constructor with the
full path of saved file as a string argument. This will automatically load the
weights associated to the model.

.. code-block:: python

   model.save("demo_CharacterDVS.fbz")
   loaded_model = Model("demo_CharacterDVS.fbz")

Input layer types
^^^^^^^^^^^^^^^^^

The first layer of a model must be one of two possible input layer
types:


* `InputData <../api_reference/akida_apis.html#inputdata>`__ – universal
  input layer type.
* `InputConvolutional <../api_reference/akida_apis.html#inputconvolutional>`__
  - image-specific input layer, taking either RGB or grayscale pixel input.

Data-Processing layer types
^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the input layer all subsequent layers will be data-processing layers.

Each layer contains several neurons that are connected to the layer inputs
according to different topologies defined by the layer type. A weight is
assigned to each connection, and that weight is combined with the input
to modify the neuron potential.

When the neuron potentials have been evaluated, the layer feeds them to an
activation function that may or may not emit a spike.

A data-processing layer can be one of three types:


* `FullyConnected <../api_reference/akida_apis.html#fullyconnected>`__ –
  each neuron is connected to members of the full set of possible inputs –
  hence ‘fully connected’, even though a much smaller number of connections
  are likely to be non-zero.
* `Convolutional <../api_reference/akida_apis.html#convolutional>`__ –
  each neuron’s connection weights express a localized filter – typically a
  region that is a small fraction of the input’s height and width. This filter
  is tested across all x and y positions.
* `SeparableConvolutional <../api_reference/akida_apis.html#separableconvolutional>`__
  - a variant of the `Convolutional <../api_reference/akida_apis.html#convolutional>`__
  layer that is less computationally intensive due to simplified filters.

The `FullyConnected <../api_reference/akida_apis.html#fullyconnected>`__
layers can be trained using the Akida Edge learning algorithm if they are the
last layer of a model.

Activation parameters
"""""""""""""""""""""

The Akida activation function uses a quantization scheme to evaluate the neuron
response when its potential goes beyond its firing threshold.
The intensity of the response is measured by dividing the difference between the
potential and the threshold in several quantization intervals that correspond to
a set of quantized spike values. The default quantization scheme is ``binary`` :
whenever the neuron potential is above the threshold, a spike with a value of
one is emitted.

More generally, if we denote:


* T the threshold,
* s the length of a quantization interval,
* p the neuron potential,
* Q the quantized activation values.

``T + n * s < p <= T + (n + 1)*s => response = Q[n]``

All data-processing layers share the following activation parameters:


* ``threshold``\ : integer value which defines the threshold for neurons to
  fire or generate an event. When using binary weights and activations, the
  activation level of neurons cannot exceed the ``num_weights`` value.
* ``act_bits``\ : < one of ``[1, 2, 4]``\ > Defines the number of
  bits used to quantize the neuron response (defaults to one bit for binary).
  Quantized activations are integers in the range ``[1, 2^(weights_bits) -1]``.
* ``act_step``\ : a float value, defining the length of the potential
  quantization intervals for act_bits = 4. For 2 bits, this is 1/4 of
  the length of the potentials intervals and it is not relevant for 1 bit.

Pooling parameters
""""""""""""""""""

The `InputConvolutional <../api_reference/akida_apis.html#inputconvolutional>`__,
`Convolutional <../api_reference/akida_apis.html#convolutional>`__ and
`SeparableConvolutional <../api_reference/akida_apis.html#separableconvolutional>`__
layer types share the following pooling parameters:


* [optional if ``pool_type = Average``] ``pool_size``: tuple of integer values,
  sets the width and height of the patch used to perform the pooling. If not
  specified it performs a global pooling.
* [optional] `pool_type`: `PoolType <../api_reference/akida_apis.html#pooltype>`__
  Sets the effective pooling type (defaults to `NoPooling`):

  * ``NoPooling`` – no pooling.
  * ``Max`` – computing the maximum of each region.
  * ``Average`` – computing the average values of each region.

* [optional] ``pool_stride``: tuple of integer values, sets the horizontal
  and vertical strides applied when sliding the pooling patches. If not
  specified, a stride of ``pool_size`` is applied.

Model Hardware Mapping
----------------------

By default, Akida models are implicitly mapped on a software backend: in other
words, their inference is computed on the host CPU.

Devices
^^^^^^^
In order to perform the inference of a model on hardware, the corresponding
``Model`` object must first be mapped on a specific ``Device``.

The Akida ``Device`` object represents an Akida device, which is entirely
characterized by:

- its `hardware version <../api_reference/akida_apis.html#hwversion>`__,
- the description of its `mesh <../api_reference/akida_apis.html#akida.NP.Mesh>`__ of
  processing nodes.

Discovering Hardware Devices
""""""""""""""""""""""""""""

The list of hardware devices detected on a specific host is available using the
`devices() <../api_reference/akida_apis.html#akida.devices>`__ method.

.. code-block:: python

    from akida import devices

    device = devices()[0]
    print(device.version)

Virtual Devices
"""""""""""""""

Most of the time, ``Device`` objects are real hardware devices, but virtual
devices can also be created to allow the mapping of a ``Model`` on a host that is
not connected to a hardware device.

Virtual devices are simply created by specifying their hardware revision and mesh
topology:

.. code-block:: python

    from akida import Device, NSoC_v2

    # Assuming mesh has been defined above
    device = Device(NSoC_v2, mesh)

It is possible to build a virtual device for known hardware devices, by calling
functions `AKD1000() <../api_reference/akida_apis.html#akida.AKD1000>`__ and
`TwoNodesIP() <../api_reference/akida_apis.html#akida.TwoNodesIP>`__.

Model mapping
^^^^^^^^^^^^^

Mapping a model on a specific device is as simple as calling the ``Model``
`.map() <../api_reference/akida_apis.html#akida.Model.map>`__ method.

.. code-block:: python

    model.map(device)

When mapping a model on a device, the information related to the layers and related
variables are processed in such way that the selected device can perform an inference.
If the Model contains layers that are not hardware compatible or is too big to fit on
the device, it will be split in multiple sequences.

The number of sequences, program size for each and how they are mapped are included in
the ``Model`` `.summary() <../api_reference/akida_apis.html#akida.Model.summary>`__ output
after it has been mapped on a device.

Advanced Mapping Details and Hardware Devices Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calling ``Model`` `.map() <../api_reference/akida_apis.html#akida.Model.map>`__ might create more
than one "sequence". In this case, when inference methods are used, each sequence will be chain
loaded on the device to process the given input. Sequences can be obtained using the ``Model``
`.sequences() <../api_reference/akida_apis.html#akida.Model.sequences>`__
property, that will return a list of sequence objects. The program used to load
one sequence can be obtained programmatically.

.. code-block:: python

    model.map(device)
    print(len(model.sequences))
    # Assume there is at least one sequence.
    sequence = model.sequences[0]
    # Check program size
    print(len(sequence.program))

The information found in the ``Model`` `.summary()
<../api_reference/akida_apis.html#akida.Model.summary>`__ can be used to
modify a model to make it fit into less sequences, and program size can be
used to estimate the flash and memory usage on an embedded system that would
use the device.

Once the model has been mapped, the sequences mapped in the Hardware run on the device,
and the sequences mapped in the Software run on the CPU.

One can also force the model to be mapped as one sequence in the hardware device
only by setting the parameter ``hw_only`` to True (by default the value is False).
See the `.map() <../api_reference/akida_apis.html#akida.Model.map>`__ method API for more details.

Note: an exception will be raised if the Model cannot be mapped entirely on the device.

.. code-block:: python

  model.map(device, hw_only=True)

Once the model has been mapped, the inference happens only on the device, and not on the host
CPU except for passing inputs and fetching outputs.

Using Akida Edge learning
-------------------------

The Akida Edge learning is a unique feature of the Akida IP.

In this mode, an Akida Layer will typically be compiled with specific learning
parameters and then undergo a period of feed-forward unsupervised or
semi-supervised training by letting it process inputs generated by previous
layers from a relevant dataset.

Once a layer has been compiled, new learning episodes can be resumed at any
time, even after the model has been saved and reloaded.

Learning constraints
^^^^^^^^^^^^^^^^^^^^

Only the last layer of a model can be trained with Akida Edge Learning and must
fulfill the following constraints:

* must be of type `FullyConnected <../api_reference/akida_apis.html#fullyconnected>`__,

* must have binary weight,

* must receive binary inputs.

Compiling a layer
^^^^^^^^^^^^^^^^^

For a layer to learn using Akida Edge Learning, it must first be compiled using
the ``Model`` `.compile <../api_reference/akida_apis.html#akida.Model.compile>`_ method.

The following learning parameters can be specified when compiling a layer:

* ``num_weights``: integer value which defines the number of connections for
  each neuron and is constant across neurons. When determining a value for
  ``num_weights`` note that the total number of available connections for a
  `Convolutional <../api_reference/akida_apis.html#convolutional>`__
  layer is not set by the dimensions of the input to the layer, but by the
  dimensions of the kernel. Total connections = ``kernel_size`` x
  ``num_features``, where ``num_features`` is typically the ``filters`` or
  ``units`` of the preceding layer. ``num_weights`` should be much smaller
  than this value – not more than half, and often much less.
* [optional] ``num_classes``: integer value, representing the number of
  classes in the dataset. Defining this value sets the learning to a ‘labeled’
  mode, when the layer is initialized. The neurons are divided into groups of
  equal size, one for each input data class. When an input packet is sent with a
  label included, only the neurons corresponding to that input class are allowed
  to learn.
* [optional] ``initial_plasticity``: floating point value, range 0–1 inclusive
  (defaults to 1). It defines the initial plasticity of each neuron’s
  connections or how easily the weights will change when learning occurs;
  similar in some ways to a learning rate. Typically, this can be set to 1,
  especially if the model is initialized with random weights. Plasticity can
  only decrease over time, never increase; if set to 0 learning will never occur
  in the model.
* [optional] ``min_plasticity``: floating point value, range 0–1 inclusive
  (defaults to 0.1). It defines the minimum level to which plasticity will decay.
* [optional] ``plasticity_decay``: floating point value, range 0–1 inclusive
  (defaults to 0.25). It defines the decay of plasticity with each learning
  step, relative to the ``initial_plasticity``.
* [optional] ``learning_competition``: floating point value, range 0–1 inclusive
  (defaults to 0). It controls competition between neurons. This is a rather
  subtle parameter since there is always substantial competition in learning
  between neurons. This parameter controls the competition from neurons that
  have already learned – when set to zero, a neuron that has already learned a
  given feature will not prevent other neurons from learning similar features.
  As ``learning_competition`` increases such neurons will exert more
  competition. This parameter can, however, have serious unintended consequences
  for learning stability; we recommend that it should be kept low, and probably
  never exceed 0.5.

The only mandatory parameter is the number of active (non-zero) connections that
each of the layer neurons has with the previous layer, expressed as the number
of active ``weights`` for each neuron.

Optimizing this value is key to achieving high accuracy in the Akida NSoC.
Broadly speaking, the number of weights should be related to the number of
events expected to compose the items’ or item’s sub-features of interest.

Tips to set Akida learning parameters are detailed in `the dedicated example
<../examples/edge/plot_2_edge_learning_parameters.html>`_.
