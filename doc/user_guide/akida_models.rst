
Akida models zoo
================

Overview
--------

Brainchip Akida Models package is a model zoo that offers a set of pre-built akida compatible
models (e.g MobileNet, AkidaNet), pretrained weights for those models and training scripts.
See the `model zoo API reference <../api_reference/akida_models_apis.html#model-zoo>`_ for a
complete list of the available models. The performance of all models from the zoo are reported for
both Akida 1.0 and Akida 2.0 in the `model zoo performance page <../model_zoo_performance.html>`__.
Akida Models also contains a set of
`layer blocks <../api_reference/akida_models_apis.html#layer-blocks>`_ that are used to define the
above models.

Command-line interface for model creation
-----------------------------------------

In addition to `the programming API <../api_reference/akida_models_apis.html>`_,
the Akida Models toolkit provides a command-line interface to instantiate and
save models from the zoo.

Instantiating models using the CLI makes use of the model definitions from the
programming interface with default values. To quantize a given model, the
`QuantizeML quantize CLI <./quantizeml.html#command-line-interface>`__ should be used.

**Examples**

Instantiate a DS-CNN network for KWS (keyword spotting):

.. code-block:: bash

    akida_models create ds_cnn_kws

The model is automatically saved to ``ds_cnn_kws.h5``.

Some models come with additional parameters that allow a deeper configuration. Examples are given
below.

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
         -c CLASSES, --classes CLASSES
                              The number of classes, by default 1000.
         -i {32,64,96,128,160,192,224}, --image_size {32,64,96,128,160,192,224}
                              The square input image size, by default 224.
         -a ALPHA, --alpha ALPHA
                              The width of the model, by default 1.0.

Current available models for creation are:

 * vgg_utk_face
 * convtiny_dvs_handy
 * convtiny_dvs_gesture
 * ds_cnn_kws
 * pointnet_plus_modelnet40
 * mobilenet_imagenet
 * akidanet_imagenet
 * akidanet_imagenet_edge
 * akidanet18_imagenet
 * yolo_base
 * centernet
 * gxnor_mnist
 * akida_unet_portrait128
 * tenn_spatiotemporal_dvs128
 * tenn_spatiotemporal_eye
 * tenn_spatiotemporal_jester

Command-line interface for model training
-----------------------------------------

The package also comes with a CLI to train models from the zoo.

Training models first requires that a model is created and saved using the CLI described above. Once
a model is ready, training will use dedicated scripts to load and preprocess a dataset and perform
training.

As shown in the examples below, the training CLI should be used along with ``akida_models create``
and ``quantizeml quantize``.

If the quantized model offers acceptable performance, it can be converted into an Akida model,
ready to be loaded on the Akida NSoC using the
`CNN2SNN convert CLI <./cnn2snn.html#command-line-interface>`_.

KWS training
^^^^^^^^^^^^

KWS training pipeline uses the ``ds_cnn_kws`` model and the QuantizeML ``quantize`` CLI. Dataset
loading and preprocessing is done within the training script called by the ``kws_train`` CLI.

**Example**

Create a DS-CNN model for KWS, train it over 16 epochs, then quantize it to 4-bit weights and
activations (using a set of samples for calibration only), perform a 16 epochs QAT to recover
accuracy and evaluate.

.. code-block:: bash

   akida_models create -s ds_cnn_kws.h5 ds_cnn_kws
   kws_train train -m ds_cnn_kws.h5 -s ds_cnn_kws.h5 -e 16

   wget https://data.brainchip.com/dataset-mirror/samples/kws/kws_batch1024.npz
   quantizeml quantize -m ds_cnn_kws.h5 -w 4 -a 4 -e 2 -bs 100 -sa kws_batch1024.npz
   kws_train train -m ds_cnn_kws_i8_w4_a4.h5 -e 16 -s ds_cnn_kws_i8_w4_a4.h5
   kws_train eval -m ds_cnn_kws_i8_w4_a4.h5

AkidaNet training
^^^^^^^^^^^^^^^^^

AkidaNet training pipeline uses the ``akidanet_imagenet`` model and the QuantizeML ``quantize`` CLI.
Dataset loading and preprocessing is done within the training script called by the
``imagenet_train`` CLI. Note that ImageNet data must be downloaded from
`<https://www.image-net.org/>`__ first.

**Example**

Create an AkidaNet 0.5 with resolution 160, train it for 90 epochs then quantize to 4-bit weights
and activations, perform a 10 epochs QAT to recover accuracy, upscale to resolution 224 and
evaluate.


.. code-block:: bash

   akida_models create -s akidanet_imagenet_160_alpha_0.5.h5 akidanet_imagenet -a 0.5 -i 160
   imagenet_train train -d path/to/imagenet/ -e 90 -m akidanet_imagenet_160_alpha_0.5.h5 \
                        -s akidanet_imagenet_160_alpha_0.5.h5

   wget https://data.brainchip.com/dataset-mirror/samples/imagenet/imagenet_batch1024_160.npz
   quantizeml quantize -m akidanet_imagenet_160_alpha_0.5.h5 -w 4 -a 4 -e 2 -bs 100 \
                        -sa imagenet_batch1024_160.npz
   imagenet_train tune -d path/to/imagenet/ -e 10 -m akidanet_imagenet_160_alpha_0.5_i8_w4_a4.h5 \
                       -s akidanet_imagenet_160_alpha_50_i8_w4_a4.h5
   imagenet_train rescale -i 224 -m akidanet_imagenet_160_alpha_0.5_i8_w4_a4.h5 \
                          -s akidanet_imagenet_224_alpha_0.5_i8_w4_a4.h5
   imagenet_train eval -d path/to/imagenet/ -m akidanet_imagenet_224_alpha_0.5_i8_w4_a4.h5


Current training pipelines available are:

* utk_face_train
* kws_train
* modelnet40_train
* yolo_train
* dvs_train
* mnist_train
* imagenet_train
* portrait128_train
* centernet_train
* urbansound_train
* tenn_dvs128_train
* tenn_eye_train
* tenn_jester_train

Command-line interface for model evaluation
-------------------------------------------

The CLI also comes with an ``eval`` action that allows to evaluate model performance, the ``-ak``
or ``--akida`` option allows to convert to Akida then evaluate the model.

.. code-block:: bash

   kws_train eval -m ds_cnn_kws_i8_w8_a8.h5

   kws_train eval -m ds_cnn_kws_i8_w8_a8.h5 -ak

Command-line interface to display summary
-----------------------------------------

CLI comes with a ``summary`` action that allows to display a model summary (supports Keras, ONNX and
Akida model files).

.. code-block:: bash

   akida_models summary -m akidanet_imagenet_224_alpha_0.5.h5

Command-line interface to display sparsity
------------------------------------------

CLI comes with a ``sparsity`` action that allows to display a model sparsity (supports Keras, ONNX
and Akida model files).

.. code-block:: bash

   akida_models sparsity -m akidanet_imagenet_224_alpha_0.5.h5 -v

The ``-v`` option (or ``--verbose``) will display all layers sparsity and the average across all
layers. The ``--layer_names`` option allows to display sparsity for target layers.

.. code-block:: bash

   akida_models sparsity -m akidanet_imagenet_224_alpha_0.5.h5 -v \
                          --layer_names conv_0/relu,conv_1/relu

Layer Blocks
------------

In Keras, it is very common for activations or other functions to be defined along with the
processing layer, e.g.:

.. code-block:: python

   x = Dense(64)(x)
   x = BatchNormalization()(x)
   x = Activation('relu')(x)

In order to ease the design of a Keras model compatible for conversion into an Akida model, a
higher-level interface is proposed with the use of layer blocks. These blocks are available
in the package through:

.. code-block:: python

   import akida_models.layer_blocks

For instance, the following code snippet sets up the same trio of layers as
those above:

.. code-block:: python

   x = dense_block(x, 64, add_batchnorm=True, relu_activation='ReLU')

The ``dense_block`` function will produce a group of layers that we call a "block".

.. note::
   - To avoid adding the activation layer, add the parameter ``relu_activation = False`` to the
     block.
   - The ReLU activation max_value can be set in the parameter using a string expression, that is
     ``relu_activation='ReLU6'`` will create a ReLU activation with max_value set to 6.
   - The ReLu activation can also be defined as unbounded, that is ``relu_activation='ReLU'`` (only
     supported for models targeting Akida 2.0)

Separable layers can be defined as ``fused`` (Akida 1.0) or ``unfused`` (Akida 2.0):

.. code-block:: python

   x = separable_conv_block(x, 64, 3, add_batchnorm=True, relu_activation='ReLU6', fused=False)

Placement of the GlobalAveragePooling (GAP) operation is also configurable in layer blocks so that
it comes before the activation (``post_relu_gap=False`` for Akida 1.0) or after
(``post_relu_gap=True`` for Akida 2.0):

.. code-block:: python

   x = conv_block(x, 64, 3, relu_activation='ReLU', post_relu_gap=True)


The option of including pooling, BatchNormalization layers or activation is directly built into the
provided block modules.

The layer block functions provided are:

* `conv_block <../api_reference/akida_models_apis.html#akida_models.layer_blocks.conv_block>`__
* `separable_conv_block <../api_reference/akida_models_apis.html#akida_models.layer_blocks.separable_conv_block>`__
* `dense_block <../api_reference/akida_models_apis.html#akida_models.layer_blocks.dense_block>`__
* `conv_transpose_block <../api_reference/akida_models_apis.html#akida_models.layer_blocks.conv_transpose_block>`__
* `sepconv_transpose_block <../api_reference/akida_models_apis.html#akida_models.layer_blocks.sepconv_transpose_block>`__
* `yolo_head_block <../api_reference/akida_models_apis.html#akida_models.layer_blocks.yolo_head_block>`__
* `spatiotemporal_block <../api_reference/akida_models_apis.html#akida_models.layer_blocks.spatiotemporal_block>`__

Most of the parameters for these blocks are identical to those passed to the
corresponding inner processing layers, such as strides and bias. The detailed API is given in the
`dedicated section <../api_reference/akida_models_apis.html#layer-blocks>`__.


Handling Akida 1.0 and Akida 2.0 specificities
----------------------------------------------

Akida 1.0 and 2.0 specific model architecture requirements are embedded in the returned models
(pretrained or not). By default, the returned models and pretrained model target Akida 2.0. It is
however possible to build and instantiate Akida 1.0 models.

Using the programming interface:

.. code-block:: python

   from akida_models import ds_cnn_kws, ds_cnn_kws_pretrained
   from cnn2snn import set_akida_version, AkidaVersion

   with set_akida_version(AkidaVersion.v1):
      model = ds_cnn_kws()
      pretrained = ds_cnn_kws_pretrained()

Using the CLI interface:

.. code-block:: bash

   CNN2SNN_TARGET_AKIDA_VERSION=v1 akida_models create ds_cnn_kws
