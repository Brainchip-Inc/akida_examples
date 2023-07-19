
Akida models zoo
================

Overview
--------

Brainchip akida_models package is a model zoo that offers a set of pre-built
akida compatible models (e.g Mobilenet, VGG or AkidaNet), pretrained weights for
those models and training scripts.

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

Instantiate a DS-CNN network for KWS (keyword spotting):

.. code-block:: bash

    akida_models create ds_cnn_kws

The model is automatically saved to ``ds_cnn_kws.h5``.

Some models come with additional parameters that allow a deeper configuration.
That is the case for the AkidaNet, AkidaNet edge, MobileNet, Mobilenet edge and
YOLO models. Examples are given below.

To build an AkidaNet model with a 64x64 input size, alpha parameter (model
width) equal to 0.35 and 250 classes:

.. code-block:: bash

    akida_models create akidanet_imagenet -i 64 -a 0.35 -c 250

To create a YOLO model with 20 classes, 5 anchors and a model width of 0.5:

.. code-block:: bash

    akida_models create yolo_base -c 20 -na 5 -a 0.5

The full parameter list with description can be obtained using the  ``-h`` or
``--help`` argument for each model:

.. code-block:: bash

    akida_models create akidanet_imagenet -h

    usage: akida_models create akidanet_imagenet [-h]
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

 * vgg_utk_face
 * ds_cnn_kws
 * pointnet_plus_modelnet40
 * akidanet_imagenet
 * akidanet_imagenet_edge
 * mobilenet_imagenet
 * yolo_base
 * gxnor_mnist
 * akidanet18_imagenet
 * centernet
 * akida_unet_portrait128
 * vit_ti16
 * bc_vit_ti16
 * deit_ti16
 * bc_deit_ti16

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

   cnn2snn quantize -m vgg_utk_face.h5 -iq 8 -wq 4 -aq 4

   utk_face_train train -e 30 -m vgg_utk_face_iq8_wq4_aq4.h5 -s vgg_utk_face_iq8_wq4_aq4.h5

   cnn2snn quantize -m vgg_utk_face_iq8_wq4_aq4.h5 -iq 8 -wq 2 -aq 2

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

   cnn2snn quantize -m ds_cnn_kws.h5 -iq 8 -wq 4 -aq 4

   kws_train train -m ds_cnn_kws_iq8_wq4_aq4.h5 -e 64 -laq 1 -s ds_cnn_kws_iq8_wq4_aq4_laq1.h5

YOLO training
^^^^^^^^^^^^^

YOLO training pipeline uses the ``yolo_base`` model and the CNN2SNN
``quantize`` CLI. Dataset preprocessing must be done beforehand using the
`processing toolbox <../api_reference/akida_models_apis.html#processing>`__.

**Example**

Create a YOLO model for VOC car/person training, use transfer learning from
AkidaNet weights trained on ImageNet and perform step-wise quantization to
obtain a network with 4-bit weights and activations. Note that the backbone
AkidaNet layers are frozen (i.e not trainable) when performing float training
using the `--freeze_before` or `-fb` option. Accuracy lost when quantizing is
partially recovered using Adaround calibration from CNN2SNN CLI, then tuning
is applied.

.. code-block:: bash

   wget https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_imagenet_224_alpha_50.h5

   akida_models create -s yolo_akidanet_voc.h5 yolo_base -c 2 -bw akidanet_imagenet_alpha_50.h5

   yolo_train train -d voc_preprocessed.pkl -m yolo_akidanet_voc.h5 -ap voc_anchors.pkl -e 25 -fb 1conv -s yolo_akidanet_voc.h5

   cnn2snn quantize -m yolo_akidanet_voc.h5 -iq 8 -wq 4 -aq 4

   yolo_train extract -d voc_preprocessed.pkl -ap voc_anchors.pkl -b 1024 -o voc_samples.npz -m yolo_akidanet_voc_iq8_wq4_aq4.h5

   cnn2snn calibrate adaround -sa voc_samples.npz -b 128 -e 500 -lr 1e-3 -m yolo_akidanet_voc_iq8_wq4_aq4.h5

   yolo_train tune -d voc_preprocessed.pkl -m yolo_akidanet_voc_iq8_wq4_aq4_adaround_calibrated.h5 -ap voc_anchors.pkl -e 10 -s yolo_akidanet_voc_iq8_wq4_aq4.h5


AkidaNet training
^^^^^^^^^^^^^^^^^

AkidaNet training pipeline uses the ``akidanet_imagenet`` model and the CNN2SNN
``quantize`` CLI. Dataset loading and preprocessing is done within the
training script called by the ``imagenet_train`` CLI. Note that ImageNet data must be downloaded
from `<https://www.image-net.org/>`__ first.

**Example**

Create an AkidaNet 0.5 with resolution 160, perform float training then quantize to 4-bit weights
and activations.

.. code-block:: bash

   akida_models create -s akidanet_imagenet_160_alpha_50.h5 akidanet_imagenet -a 0.5 -i 160

   imagenet_train train -d path/to/imagenet/ -e 90 -m akidanet_imagenet_160_alpha_50.h5 -s akidanet_imagenet_160_alpha_50.h5

   cnn2snn quantize -m akidanet_imagenet_160_alpha_50.h5 -iq 8 -wq 4 -aq 4

   imagenet_train tune -d path/to/imagenet/ -e 10 -m akidanet_imagenet_160_alpha_50_iq8_wq4_aq4.h5 -s akidanet_imagenet_160_alpha_50_iq8_wq4_aq4.h5


Command-line interface for model evaluation
-------------------------------------------

The CLI also comes with an ``eval`` action that allows to evaluate model
performances, the ``-ak`` or ``--akida`` option allows to evaluate the model
once converted to Akida.

.. code-block:: bash

   kws_train eval -m ds_cnn_kws_iq8_wq4_aq4_laq1.h5

   kws_train eval -m ds_cnn_kws_iq8_wq4_aq4_laq1.h5 -ak


Command-line interface to evaluate model MACS
---------------------------------------------

CLI comes with a ``macs`` action that allows to compute the number of multiply and accumulate (MACS)
in a model.

.. code-block:: bash

   akida_models macs -m akidanet_imagenet_224_alpha_50.h5 -v


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


* ``conv_block``
* ``separable_conv_block``
* ``dense_block``
* ``mlp_block``
* ``multi_head_attention``
* ``transformer_block``
* ``conv_transpose_block``
* ``sepconv_transpose_block``
* ``yolo_head_block``

Most of the parameters for these blocks are identical to those passed to the
corresponding inner processing layers, such as strides and bias. The detailed API is given in the
`dedicated section <../api_reference/akida_models_apis.html#layer-blocks>`__.
