
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
`CNN2SNN quantize CLI <cnn2snn.html#command-line-interface>`__ should be used.

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

Some models come with additional parameters that allow a deeper configuration.
That is the case for the MobileNet, Mobilenet edge and YOLO models. Examples
are given below.

To build a MobileNet model with a 64x64 input size, alpha parameter (model
width) equal to 0.35 and 250 classes:

.. code-block:: bash

    akida_models create mobilenet_imagenet -i 64 -a 0.35 -c 250

To create a YOLO model with 20 classes, 5 anchors and a model width of 0.5:

.. code-block:: bash

    akida_models create yolo_base -c 20 -na 5 -a 0.5

The full parameter list with description can be obtained using the  ``-h`` or
``--help`` argument for each model:

.. code-block:: bash

    akida_models create mobilenet_imagenet -h

    usage: akida_models create mobilenet_imagenet [-h]
                                              [-i {32,64,96,128,160,192,224}]
                                              [-a ALPHA] [-c CLASSES]

    optional arguments:
      -h, --help            show this help message and exit
      -i {32,64,96,128,160,192,224}, --image_size {32,64,96,128,160,192,224}
                            The square input image size
      -a ALPHA, --alpha ALPHA
                            The width of the model
      -c CLASSES, --classes CLASSES
                            The number of classes

Current available models for creation are:

 * ds_cnn_cifar10
 * vgg_cifar10
 * vgg_utk_face
 * ds_cnn_kws
 * mobilenet_imagenet
 * mobilenet_imagenet_edge
 * yolo_base
 * vgg_imagenet

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
 * tuning the 8-4-4 quantized model for 15 epochs

.. code-block:: bash

    akida_models create -s vgg_cifar10.h5 vgg_cifar10

    cifar10_train train -m vgg_cifar10.h5 -s vgg_cifar10.h5 -e 100

    cnn2snn -m vgg_cifar10.h5 quantize -wq 4 -aq 4

    cifar10_train tune -m vgg_cifar10_iq4_wq4_aq4.h5 -s vgg_cifar10_iq4_wq4_aq4.h5 -e 15


Note that the model is saved and reloaded at each step.

UTK Face training
^^^^^^^^^^^^^^^^^

UTK Face training pipeline uses the ``vgg_utk_face`` model and the
CNN2SNN ``quantize`` CLI. Dataset loading and preprocessing is done within the
training script called by the ``utk_face_train`` CLI.

**Example**

Create a VGG model for UTK Face training and perfom step-wise quantization to
obtain a network with 2-bit weights and activations.

.. code-block:: bash

   akida_models create vgg_utk_face

   utk_face_train train -e 300 -m vgg_utk_face.h5 -s vgg_utk_face.h5

   cnn2snn -m vgg_utk_face.h5 quantize -iq 8 -wq 4 -aq 4

   utk_face_train train -e 30 -m vgg_utk_face_iq8_wq4_aq4.h5 -s vgg_utk_face_iq8_wq4_aq4.h5

   cnn2snn -m vgg_utk_face_iq8_wq4_aq4.h5 quantize -iq 8 -wq 2 -aq 2

   utk_face_train train -e 30 -m vgg_utk_face_iq8_wq2_aq2.h5 -s vgg_utk_face_iq8_wq2_aq2.h5

KWS training
^^^^^^^^^^^^

KWS training pipeline uses the ``ds_cnn_kws`` model and the CNN2SNN
``quantize`` CLI. Dataset loading and preprocessing is done within the
training script called by the ``kws_train`` CLI.

**Example**

Create a DS-CNN model for KWS training and perfom step-wise quantization to
obtain a network with 4-bit weights and activations. Note that the ``kws_train``
script takes the ``-laq`` which defines the bitwidth of the last activation
layer. It must be set to 1 for the last training step, since the model requires
binary activations for edge learning.

.. code-block:: bash

   akida_models create -s ds_cnn_kws.h5 ds_cnn_kws

   kws_train train -m ds_cnn_kws.h5 -s ds_cnn_kws.h5 -e 16

   cnn2snn -m ds_cnn_kws.h5 quantize -iq 0 -wq 0 -aq 4

   kws_train train -m ds_cnn_kws_iq0_wq0_aq4.h5 -s ds_cnn_kws_iq0_wq0_aq4_laq4.h5 -e 16

   cnn2snn -m ds_cnn_kws_iq0_wq0_aq4_laq4.h5 quantize -iq 8 -wq 4 -aq 4

   kws_train train -m ds_cnn_kws_iq8_wq4_aq4.h5 -s ds_cnn_kws_iq8_wq4_aq4_laq4.h5 -e 16

   kws_train train -m ds_cnn_kws_iq8_wq4_aq4_laq4.h5 -s ds_cnn_kws_iq8_wq4_aq4_laq3.h5 -e 16 -laq 3

   kws_train train -m ds_cnn_kws_iq8_wq4_aq4_laq3.h5 -s ds_cnn_kws_iq8_wq4_aq4_laq2.h5 -e 16 -laq 2

   kws_train train -m ds_cnn_kws_iq8_wq4_aq4_laq2.h5 -s ds_cnn_kws_iq8_wq4_aq4_laq1.h5 -e 16 -laq 1

YOLO training
^^^^^^^^^^^^^

YOLO training pipeline uses the ``yolo_base`` model and the CNN2SNN
``quantize`` CLI. Dataset preprocessing must be done beforehand using the
`processing toolbox <api_reference/akida_models_apis.html#processing>`__.

**Example**

Create a YOLO model for VOC car/person training, use transfer learning from
MobileNet weights trained on ImageNet and perform step-wise quantization to
obtain a network with 4-bit weights and activations. Note that the backbone
MobileNet layers are frozen (i.e not trainable) when performing float training
using the `--freeze_before` or `-fb` option.

.. code-block:: bash

   wget http://data.brainchip.com/models/mobilenet/mobilenet_imagenet_224_alpha_50.h5

   akida_models create -s yolo_voc.h5 yolo_base -c 2 -bw mobilenet_imagenet_alpha_50.h5

   yolo_train train -d voc_preprocessed.pkl -m yolo_voc.h5 -ap voc_anchors.pkl -e 25 -fb 1conv -s yolo_voc.h5

   cnn2snn -m yolo_voc.h5 quantize -iq 8 -wq 8 -aq 8

   yolo_train train -d voc_preprocessed.pkl -m yolo_voc_iq8_wq8_aq8.h5 -ap voc_anchors.pkl -e 20 -s yolo_voc_iq8_wq8_aq8.h5

   cnn2snn -m yolo_voc_iq8_wq8_aq8.h5 quantize -iq 8 -wq 4 -aq 4

   yolo_train train -d voc_preprocessed.pkl -m yolo_voc_iq8_wq4_aq4.h5 -ap voc_anchors.pkl -e 20 -s yolo_voc_iq8_wq4_aq4.h5


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
^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^

.. code-block:: python

   def dense_block(inputs,
                   units,
                   add_batchnorm=False,
                   add_activation=True,
                   **kwargs)

``separable_conv_block``
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def separable_conv_block(inputs,
                            filters,
                            kernel_size,
                            pooling=None,
                            pool_size=(2, 2),
                            add_batchnorm=False,
                            add_activation=True,
                            **kwargs)
