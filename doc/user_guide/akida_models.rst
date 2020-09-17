
Akida models zoo
================

Overview
--------

Brainchip akida_models package is a model zoo that offers a set of pre-built
akida compatible models (e.g Mobilenet or VGG), pretrained weights for those
models and training scripts.

See the `model zoo API reference
<../api_reference/akida_models_apis.html#model-zoo>`_ for a complete list of the
available models.

akida_models also contains a set of `quantization blocks
<../api_reference/akida_models_apis.html#quantization-blocks>`_ and
`layer blocks <../api_reference/akida_models_apis.html#layer-blocks>`_
that are used to define the above models.

Command-line interface for model creation
-----------------------------------------

In addition to `the programming API <../api_reference/akida_models_apis.html>`_,
the akida_models toolkit provides a command-line interface to instantiate and
save models from the zoo.

Instantiating models using the CLI makes use of the model definitions from the
programming interface with default values. To quantize a given model, the
`CNN2SNN quantize CLI <cnn2snn.html#command-line-interface>`_ should be used.

**Examples**

Instantiate a DS-CNN (MobileNet inspired) network for CIFAR10 (object
classification):

.. code-block:: bash

    akida_models create ds_cnn_cifar10

The model is automatically saved to ``ds_cnn_cifar10.h5``.

Instantiate a VGG model for CIFAR10 and save it to a specific location:

.. code-block:: bash

    akida_models create -s ./models/my_vgg_network.h5 vgg_cifar10

A model named ``my_vgg_network.h5`` is saved under the models directory
(providing the directory exists).

Current available models for creation are:

 * ds_cnn_cifar10
 * vgg_cifar10
 * vgg_utk_face
 * ds_cnn_kws

Command-line interface for model training
-----------------------------------------

The package also comes with a CLI to train models from the zoo.

Training models first requires that a model is created and saved using the CLI
described above. Once a model is ready, training will use dedicated scripts
to load and preprocess a dataset and perform training.

As shown in the examples below, the training CLI should be used along with
``akida_models create`` and ``cnn2snn quantize``.

If the quantized model offers acceptable performance, it can be converted into
an Akida model, ready to be loaded on the Akida NSoC using the
`CNN2SNN convert CLI <cnn2snn.html#command-line-interface>`_.

CIFAR10 training and tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two different network architectures are provided for CIFAR10 object
classification, namely ``ds_cnn_cifar10`` and ``vgg_cifar10`` and both can be
trained using the ``cifar10_train`` CLI.

``cifar10_train`` offers two actions:

 * ``train`` integrates data augmentation and a decreasing learning rate. It
   will generally be used for a large number of epochs on a model that has not
   been quantized yet.
 * ``tune`` has a lower learning rate and will early stop when loss reaches a
   plateau. It is intended for re-training after quantization.

See `typical training scenario <cnn2snn.html#typical-training-scenario>`_ for
more details about quantization aware training.

**Example**

Apply quantization-aware training to a VGG model for the CIFAR10 dataset by:

 * creating the model
 * training the full-precision model for 100 epochs
 * quantizing weights and activations to 4 bits
 * tuning the 4-4-4 quantized model for 15 epochs
 * quantizing weights and activations to 2 bits
 * tuning the 2-2-2 quantized model for 15 epochs

.. code-block:: bash

    akida_models create -s vgg_cifar10.h5 vgg_cifar10

    cifar10_train -m vgg_cifar10.h5 -s vgg_cifar10.h5 -e 100 train

    cnn2snn -m vgg_cifar10.h5 quantize -wq 4 -aq 4

    cifar10_train -m vgg_cifar10_iq4_wq4_aq4.h5 -s vgg_cifar10_iq4_wq4_aq4.h5 -e 15 tune

    cnn2snn -m vgg_cifar10_iq4_wq4_aq4.h5 quantize -wq 2 -aq 2

    cifar10_train -m vgg_cifar10_iq2_wq2_aq2.h5 -s vgg_cifar10_iq2_wq2_aq2.h5 -e 15 tune

Note that the model is saved and reloaded at each step.


Layer Blocks
------------

In order to ensure that the design of a Keras model is compatible for conversion
into an Akida model, a higher-level interface is proposed with the use of layer
blocks. These blocks are available in the package through:

.. code-block:: python

   import akida_models.layer_blocks

In Keras, when adding a core layer type (\ ``Dense`` or ``Conv2D``\ ) to a
model, an activation function is typically included:

.. code-block:: python

   x = Dense(64, activation='relu')(x)

or the equivalent, explicitly adding the activation function separately:

.. code-block:: python

   x = Dense(64)(x)
   x = Activation('relu'))(x)

It is very common for other functions to be included in this arrangement, e.g.,
a normalization of values before applying the activation function:

.. code-block:: python

   x = Dense(64)(x)
   x = BatchNormalization()(x)
   x = Activation('relu')(x)

This particular arrangement of layers is important for conversion and is
therefore reflected in the blocks API.

For instance, the following code snippet sets up the same trio of layers as
those above:

.. code-block:: python

   x = dense_block(x, 64, add_batchnorm=True)

The ``dense_block`` function will produce a group of layers that we call a
"block".

.. note::
    To avoid adding the activation layer, add the parameter
    ``add_activation = False`` to the block.


The option of including pooling, batchnorm layers or activation is directly
built into the provided block modules.
The layer block functions provided are:


* ``conv_block``\ ,
* ``separable_conv_block``\ ,
* ``dense_block``.

Most of the parameters for these blocks are identical to those passed to the
corresponding inner processing layers, such as strides and bias.

``conv_block``
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def conv_block(inputs,
                  filters,
                  kernel_size,
                  pooling=None,
                  pool_size=(2, 2),
                  add_batchnorm=False,
                  add_activation=True,
                  **kwargs):

``dense_block``
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def dense_block(inputs,
                   units,
                   add_batchnorm=False,
                   add_activation=True,
                   **kwargs)

``separable_conv_block``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def separable_conv_block(inputs,
                            filters,
                            kernel_size,
                            pooling=None,
                            pool_size=(2, 2),
                            add_batchnorm=False,
                            add_activation=True,
                            **kwargs)
