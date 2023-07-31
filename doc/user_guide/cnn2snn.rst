
CNN2SNN toolkit
===============

Overview
--------

The Brainchip CNN2SNN toolkit provides a means to convert a quantized model obtained using
QuantizeML to a low-latency and low-power network for use with the Akida runtime.


Conversion flow
---------------

CNN2SNN offers a simple `convert <../api_reference/cnn2snn_apis.html#cnn2snn.convert>`__ function
that takes a quantized model as input and converts it into an Akida runtime compatible network.

Let's take the `DS-CNN <../api_reference/akida_models_apis.html#ds-cnn>`__ model from our zoo that
targets KWS task as an example:

.. code-block:: python

    from akida_models import ds_cnn_kws_pretrained
    from cnn2snn import convert

    # Load a pretrained 8/4/4 quantized model
    quantized_model = ds_cnn_kws_pretrained()
    model_akida = convert(quantized_model)


Conversion compatibility
^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to check if a quantized model is compatible with Akida conversion using the
`check_model_compatibility
<../api_reference/cnn2snn_apis.html#cnn2snn.quantizeml.compatibility_checks.check_model_compatibility>`__
helper. Note that this helper will not convert the model or check compatibility with an Akida
hardware, it only checks that the model quantization scheme is allowed and that building blocks are
compatible with Akida layers blocks.

.. note::
    `cnn2snn.quantizeml.compatibility_checks.check_model_compatibility
    <../api_reference/cnn2snn_apis.html#cnn2snn.quantizeml.compatibility_checks.check_model_compatibility>`__
    only applies to a model quantized via QuantizeML. For a CNN2SNN-quantized model, instead use the
    deprecated
    `cnn2snn.check_model_compatibility <../api_reference/cnn2snn_apis.html#cnn2snn.check_model_compatibility>`__.

Command-line interface
^^^^^^^^^^^^^^^^^^^^^^

In addition to the CNN2SNN programming API, the CNN2SNN toolkit provides a command-line interface to
perform conversion to an Akida runtime compatible model. Converting a quantized model into an Akida
model using the CLI makes use of the
`convert <../api_reference/cnn2snn_apis.html#cnn2snn.convert>`__ function.

**Examples**

Convert the DS-CNN/KWS 8/4/4 quantized model:

.. code-block:: bash

    wget https://data.brainchip.com/models/AkidaV2/ds_cnn/ds_cnn_kws_i8_w4_a4.h5

    cnn2snn convert -m ds_cnn_kws_i8_w4_a4.h5

An Akida ``.fbz`` model named ``ds_cnn_kws_i8_w4_a4.fbz`` is then saved. This model can be loaded
back into an `akida.Model <../api_reference/akida_apis.html#akida.Model>`__ and run on Akida runtime.

Deprecated CLI actions
~~~~~~~~~~~~~~~~~~~~~~

The ``scale`` and ``shift`` options of the ``convert`` CLI action that were used to set input
scaling parameters are now deprecated.

CNN2SNN CLI comes with additional actions that are also deprecated and should no longer be used:
``quantize``, ``reshape`` and  ``calibrate``. You should rather use
`QuantizeML <quantizeml.html#>`__ tool to perform the same operations.


Handling Akida 1.0 and Akida 2.0 specificities
----------------------------------------------

Conversion towards Akida 1.0 or Akida 2.0 is inherently different because the targeted SoC or IP
comes with different features. By default, a model is converted towards Akida 2.0. It is however
possible to convert towards Akida 1.0.

Using the programming interface:

.. code-block:: python

  from akida_models import ds_cnn_kws_pretrained
  from cnn2snn import convert, set_akida_version, AkidaVersion

  with set_akida_version(AkidaVersion.v1):
      quantized_model = ds_cnn_kws_pretrained()
      model_akida = convert(quantized_model)

Using the CLI interface:

.. code-block:: bash

  wget https://data.brainchip.com/models/AkidaV1/ds_cnn/ds_cnn_kws_iq8_wq4_aq4_laq1.h5

  CNN2SNN_TARGET_AKIDA_VERSION=v1 cnn2snn convert -m ds_cnn_kws_iq8_wq4_aq4_laq1.h5

.. note::
    - converting a model `quantized with QuantizeML <quantizeml.html>`__ will use the contextual
      `AkidaVersion <../api_reference/cnn2snn_apis.html#cnn2snn.AkidaVersion>`__ to target either
      1.0 or 2.0.
    - converting a model `quantized with CNN2SNN <cnn2snn.html#legacy-quantization-api>`__
      (deprecated path) will always target 1.0.


Legacy quantization API
-----------------------

.. warning::
    While it is possible to quantize Akida 1.0 models using cnn2snn legacy quantization blocks, such
    usage is deprecated. You should rather use `QuantizeML <../user_guide/quantizeml.html#>`__ tool
    to quantize a model whenever possible.


Typical quantization scenario
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CNN2SNN toolkit offers a turnkey solution to quantize a model:
the `quantize <../api_reference/cnn2snn_apis.html#cnn2snn.quantize>`_ function. It
replaces the neural Keras layers (Conv2D, SeparableConv2D and Dense) and
the ReLU layers with custom CNN2SNN layers, which are quantization-aware
derived versions of the base Keras layer types. The obtained quantized model is
still a Keras model with a mix of CNN2SNN quantized layers (QuantizedReLU,
QuantizedDense, etc.) and standard Keras layers (BatchNormalization, MaxPool2D,
etc.).

Direct quantization of a standard Keras model (also called post-training
quantization) generally introduces a drop in performance. This drop is usually
small for 8-bit or even 4-bit quantization of simple models, but it can be very
significant for low quantization bitwidth and complex models.

If the quantized model offers acceptable performance, it can be directly
converted into an Akida model, ready to be loaded on the Akida NSoC (see the
`convert <../api_reference/cnn2snn_apis.html#cnn2snn.convert>`_ function).

However, if the performance drop is too high, a quantization-aware training is
required to recover the performance prior to quantization. Since the quantized
model is a Keras model, it can then be trained using the standard Keras API.

Note that quantizing directly to the target bitwidth is not mandatory: it is
possible to proceed with quantization in a serie of smaller steps.
For example, it may be beneficial to keep float weights and only quantize
activations, retrain, and then, quantize weights.


Design compatibility constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When designing a tf.keras model, consider design compatibility at these
distinct levels before the quantization stage:


* Only serial and feedforward arrangements can be converted\ [#fn-1]_.
* Supported Keras layers are listed `below <#supported-layer-types>`_.
* Order of the layers is important, e.g. a BatchNormalization layer
  must be placed before the activation, and not after.
* Some constraints are needed about layer's parameters, e.g. a MaxPool2D layer
  must have the same padding as its corresponding convolutional layer.


All these design compatibility constraints are summarized in the CNN2SNN
`check_model_compatibility <../api_reference/cnn2snn_apis.html#cnn2snn.check_model_compatibility>`_
function. A good practice is to check model compatibility before going through
the training process [#fn-2]_.

Helpers (see `Layer Blocks <../api_reference/akida_models_apis.html#layer-blocks>`_) are available
in the ``akida_models`` PyPI package to easily create a compatible model from scratch.

Command-line interface
^^^^^^^^^^^^^^^^^^^^^^

In addition to the cnn2snn programming API, the CNN2SNN toolkit also provides a
command-line interface to perform quantization, conversion to an Akida NSoC
compatible model or model reshape.

Quantizing a standard Keras model or a CNN2SNN quantized model using the CLI
makes use of the ``cnn2snn.quantize`` Python function. The same arguments, i.e.
the quantization bitwidths for weights and activations, are required.

**Examples**

Quantize a standard Keras model with 4-bit weights and activations and 8-bit
input weights:

.. code-block:: bash

    cnn2snn quantize -m model_keras.h5 -wq 4 -aq 4 -iq 8

The quantized model is automatically saved to ``model_keras_iq8_wq4_aq4.h5``.

Quantize an already quantized model with different quantization bitwidths:

.. code-block:: bash

    cnn2snn quantize -m model_keras_iq8_wq4_aq4.h5 -wq 2 -aq 2

A new quantized model named ``model_keras_iq2_wq2_aq2.h5`` is saved.

A model can be reshaped (change of input shape) using CNN2SNN CLI that makes
use of the ``cnn2snn.transforms.reshape`` function. This will only apply to
Sequential models, a `sequentialize helper
<../api_reference/cnn2snn_apis.html#cnn2snn.transforms.sequentialize>`__ is
provided for convenience.

**Examples**

Reshape a model to 160x96:

.. code-block:: bash

    cnn2snn reshape -m model_keras.h5 -iw 160 -ih 96

A reshaped model will be saved as ``model_keras_160_96.h5``.


Layers Considerations
^^^^^^^^^^^^^^^^^^^^^

Supported layer types
~~~~~~~~~~~~~~~~~~~~~

The CNN2SNN toolkit provides quantization of Keras models with the following
Keras layer types:


* **Core Neural Layers**\ :

  * tf.keras `Dense <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`__
  * tf.keras `Conv2D <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D>`__

* **Specialized Layers**\ :

  * tf.keras `SeparableConv2D <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SeparableConv2D>`__

* **Other Layers (from tf.keras)**\ :

  * ReLU
  * BatchNormalization
  * MaxPooling2D
  * GlobalAveragePooling2D
  * Dropout
  * Flatten
  * Reshape
  * Input

CNN2SNN Quantization-aware layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several articles have reported\ [#fn-4]_ that the quantization of a pre-trained
float Keras model using 8-bit precision can be performed with a minimal loss
of accuracy for simple models, but that for lower bitwidth or complex models a
quantization-aware re-training of the quantized model may be required.

The CNN2SNN toolkit therefore includes quantization-aware versions of the base
Keras layers.

These layers are produced when quantizing a standard Keras model using the
``quantize`` function: it replaces the base Keras layers with their quantization-aware
counterparts (see the `quantize <../api_reference/cnn2snn_apis.html#cnn2snn.quantize>`_ function).

Quantization-aware training simulates the effect of quantization in the forward
pass, yet using a straight-through estimator for the quantization gradient in
the backward pass.
For the stochastic gradient descent to be efficient, the weights are stored as
float values and updated with high precision during back propagation.
This ensures sufficient precision in accumulating tiny weights adjustments.

The CNN2SNN toolkit includes two classes of quantization-aware layers:


* **quantized processing layers**\ :

  * `QuantizedDense <../api_reference/cnn2snn_apis.html#quantizeddense>`__\ ,
  * `QuantizedConv2D <../api_reference/cnn2snn_apis.html#quantizedconv2d>`__\ ,
  * `QuantizedSeparableConv2D <../api_reference/cnn2snn_apis.html#quantizedseparableconv2d>`__

* **quantized activation layers**\ :

  * `QuantizedReLU <../api_reference/cnn2snn_apis.html#quantizedrelu>`_

Most of the parameters for the quantized processing layers are identical to
those used when defining a model using standard Keras layers. However, each of
these layers also includes a ``quantizer`` parameter that specifies the
`WeightQuantizer <../api_reference/cnn2snn_apis.html#weightquantizer>`_
object to use during the quantization-aware training.

The quantized ReLU takes a single parameter corresponding to the
bitwidth of the quantized activations.

Training-Only Layers
~~~~~~~~~~~~~~~~~~~~~

Training is done within the Keras environment and training-only layers may be
added at will, such as BatchNormalization or Dropout layers. These are handled
fully by Keras during the training and do not need to be modified to be
Akida-compatible for inference.

As regards the implementation within the Akida neuromorphic IP: it may be
helpful to understand that the associated scaling operations (multiplication and
shift) are never performed during inference. The computational cost is reduced
by wrapping the (optional) batch normalization function and quantized activation
function into the spike generating thresholds and other parameters of the Akida model.
That process is completely transparent to the user. It does, however, have an
important consequence for the output of the final layer of the model; see
`Final Layers <#id6>`_ below.

First Layers
~~~~~~~~~~~~

Most layers of an Akida model only accept sparse inputs.
In order to support the most common classes of models in computer vision, a
special layer (`InputConvolutional <../api_reference/akida_apis.html#akida.InputConvolutional>`__)
is however able to receive image data (8-bit grayscale or RGB). See the
`Akida user guide <akida.html>`__ for further details.

The CNN2SNN toolkit supports any quantization-aware training layer as the first
layer in the model. The type of input accepted by that layer can be specified
during conversion, but only models starting with a QuantizedConv2D layer will
accept dense inputs, thanks to the special
`InputConvolutional <../api_reference/akida_apis.html#akida.InputConvolutional>`__ layer.

Input Scaling
+++++++++++++

The `InputConvolutional <../api_reference/akida_apis.html#akida.InputConvolutional>`_
layer only receives 8-bit input values:


* if the data is already in 8-bit format it can be sent to the Akida inputs
  without rescaling.
* if the data has been scaled to ease training, it is necessary to provide the
  scaling coefficients at model conversion.

This applies to the common case where input data are natively 8-bit. If input
data are not 8-bit, the process is more complex, and we recommend applying
rescaling in two steps:


#. Taking the data to an 8-bit unsigned integer format suitable for Akida
   architecture. Apply this step both for training and inference data.
#. Rescaling the 8-bit values to some unit or zero centered range suitable for
   CNN training, as above. This step should only be applied for the CNN training.
   Also, remember to provide those scaling coefficients when converting the
   trained model to an Akida-compatible format.

Final Layers
~~~~~~~~~~~~

As is typical for CNNs, the final layer of a model does not include the
standard activation nonlinearity. If that is the case, once converted to Akida
hardware, the model will give the potentials levels and in most cases, taking the
maximum among these values is sufficient to obtain the correct response from
the model.
However, if there is a difference in performance between the Keras and the
Akida-compatible implementations of the model, it is likely be at this step.


Tips and Tricks
^^^^^^^^^^^^^^^

In some cases, it may be useful to adapt existing CNN models in order to
simplify or enhance the converted model. Here's a short list of some possible
substitutions that might come in handy:


* `Substitute a fully connected layer with a convolutional layer
  <http://cs231n.github.io/convolutional-networks/#convert>`_.
* `Substitute a convolutional layer with stride 2 with a convolutional layer
  with stride 1 in combination with an additional pooling layer
  <https://arxiv.org/abs/1412.6806>`_.
* `Substitute a convolutional layer that has 1 large filter with multiple
  convolutional layers that contain smaller filters
  <http://cs231n.github.io/convolutional-networks/>`_.

____

.. [#fn-1] Parallel layers and "residual" connections are currently not
           supported.
.. [#fn-2] Check model compatibility must be applied on a quantized model. It
            then requires to quantize the model first.
.. [#fn-3] The spike value depends on the intensity of the potential, see the
           `Akida documentation <akida.html>`_ for details on the activation.
.. [#fn-4] See for instance `"Quantizing deep convolutional networks for
           efficient inference: A whitepaper"
           <https://arxiv.org/pdf/1806.08342.pdf>`_
           - Raghuraman Krishnamoorthi, 2018