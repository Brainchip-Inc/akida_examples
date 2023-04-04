
Getting started
===============

The Akida MetaTF ML Tools can easily be installed using `pip
<https://pypi.org/project/pip/>`_ python installer (see `Installation <../installation.html>`_).

For beginners
-------------

MetaTF examples come with ready-to-use models for popular datasets.

Run the MNIST example below, then visit the `Akida examples <../examples/index.html>`_.

.. code-block::

   import numpy as np
   from tensorflow.keras.utils import get_file
   from tensorflow.keras.datasets import mnist

   # Akida specific imports
   from akida import Model

   # Retrieve MNIST dataset
   (train_set, train_label), (test_set, test_label) = mnist.load_data()

   # Load pre-trained MNIST model
   model_file = get_file("gxnor_mnist.fbz",
                         "http://data.brainchip.com/models/AkidaV1/gxnor/gxnor_mnist.fbz",
                         cache_subdir='models/gxnor')
   model = Model(model_file)

   # Test the first image of the test set
   sample_image = 0
   image = test_set[sample_image]
   labels = model.predict_classes(image.reshape(1,28,28,1))
   assert labels[0] == test_label[sample_image]

For users familiar with deep-learning
-------------------------------------

The best place to start is the `Model sequential API <../api_reference/akida_apis.html#model>`_.

As in `Keras <https://keras.io>`_, you can create models by plugging together
neural layers.

Run the XOR example below, then visit the `Akida examples <../examples/index.html>`_.

.. code-block::

   import numpy as np
   from akida import Model, InputData, FullyConnected

   # Instantiate xor model
   xor = Model()
   layer_input = InputData(name="input", input_shape=(1, 1, 2))
   xor.add(layer_input)
   layer_hidden = FullyConnected(name="hidden", units=2, weights_bits=1)
   xor.add(layer_hidden)
   layer_output = FullyConnected(name="output", units=1, weights_bits=2)
   xor.add(layer_output)

   # Display model structure and parameters
   xor.summary()

   # Set weights for hidden layer: both neurons accumulate the inputs
   h_weights = np.array([[[[1, 1], [1, 1]]]], dtype=np.int8)
   layer_hidden.set_variable("weights", h_weights)
   # Set thresholds for hidden layer:
   # - first neuron spikes if any of the two inputs is 1 (thresh = 0)
   # - second neuron spikes only if both inputs are 1 (thresh = 1)
   h_thresholds = np.array([0,1], dtype=np.int32)
   layer_hidden.set_variable("threshold", h_thresholds)

   # Set weights for output layer: first hidden neuron minus second
   o_weights = np.array([[[[1],[-1]]]], dtype=np.int8)
   layer_output.set_variable("weights", o_weights)
   # Set threshold for output layer: spike if neurons do not cancel each other
   o_thresholds = np.array([0], dtype=np.int32)
   layer_output.set_variable("threshold", o_thresholds)

   # XOR model table
   # +---+---+---------+
   # | A | B | A XOR B |
   # +---+---+---------+
   # | 0 | 0 |    0    |
   # +---+---+---------+
   # | 0 | 1 |    1    |
   # +---+---+---------+
   # | 1 | 0 |    1    |
   # +---+---+---------+
   # | 1 | 1 |    0    |
   # +---+---+---------+

   # 0, 0 -> no spikes generated
   # Not even evaluated since we don't have input spikes

   # 0, 1 -> spikes
   in_spikes = np.array([[[[0, 1]]]], dtype=np.uint8)
   out_spikes = xor.forward(in_spikes)
   assert (np.count_nonzero(out_spikes) == 1)

   # 1, 0 -> spikes
   in_spikes = np.array([[[[1, 0]]]], dtype=np.uint8)
   out_spikes = xor.forward(in_spikes)
   assert (np.count_nonzero(out_spikes) == 1)

   # 1, 1 -> no spikes
   in_spikes = np.array([[[[1, 1]]]], dtype=np.uint8)
   out_spikes = xor.forward(in_spikes)
   assert (np.count_nonzero(out_spikes) == 0)
