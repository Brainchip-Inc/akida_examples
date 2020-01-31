
CNN2SNN toolkit
===============

Overview
--------

The Brainchip CNN2SNN toolkit provides a means to convert Convolutional Neural
Networks (CNN) that were trained using Deep Learning methods to a low-latency
and low-power Spiking Neural Network (SNN) for use with the Akida Execution
Engine (AEE). This document is a guide to that process.

The AEE provides Spiking Neural Networks (SNN) in which communications between
neurons take the form of “spikes” or impulses that are generated when a neuron
exceeds a threshold level of activation. Neurons that do not cross the threshold
generate no output and contribute no further computational cost downstream. This
feature is key to the efficiency of the AEE. The AEE further extends this
efficiency by operating with low bitwidth “synapses” or weights of connections
between neurons.

Despite the apparent fundamental differences between SNNs and CNNs, the
underlying mathematical operations performed by each may be rendered identical.
Consequently, the trained parameters of a CNN can be converted to be compatible
with those of the AEE, given only a small number of constraints [#fn-1]_. By
careful attention to specifics in the architecture and training of the CNN, an
overly complex conversion step from CNN to SNN can be avoided. The CNN2SNN
toolkit comprises a set of functions designed for the popular `Tensorflow Keras
<https://www.tensorflow.org/guide/keras>`_ framework, making it easy
to train a SNN-compatible network.

Conversion Workflow
^^^^^^^^^^^^^^^^^^^


.. image:: ../img/CNN2SNN_Flow.png
   :target: ../_images/CNN2SNN_Flow.png
   :alt: CNN2SNN Flow


Compatibility Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^

When designing a model from scratch, or converting an existing model to obtain a
final model compatible with the AEE, consider compatibility at these distinct
levels:


* Weights must be quantized (using `1, 2, 3, 4 or 8 bits <hw_constraints.html>`_).
* Activations should be quantized too (using 1, 2 or 4 bits, with maximum spike
  value set to 15\ [#fn-2]_\ ). This is necessary for preparing a CNN compatible
  with the Akida SNN model. Brainchip provides easy-to-use custom functions for
  this purpose. The use of tf.keras layers is described in the next chapter.
* Modification or substitution of a layer will be required when a method that is
  used in the CNN is not compatible with the Akida architecture. For instance,
  this can be the case for residual connections.

Only serial and feedforward arrangements can be converted\ [#fn-3]_.

Note that our experience reveals that although existing Keras models can be
quantized almost without loss to 8 bits, a quantization-aware training of the
model is required for lower quantization bitwidths.

For that reason, the CNN2SNN toolkit includes quantization-aware versions of
the base Keras layer types.

Typical training scenario
^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend preparing your model for the AEE in two (or more) training
episodes.
In the first episode, train with the standard Keras versions of the weights
and activations.

Once it is established that the overall model configuration prior to
quantization yields a satisfactory performance on the task, proceed with
quantization suitable for the AEE, changing only the quantization parameters.

We recommend saving the weights of your trained non-quantized model and using
those to initialize the quantized version; typically leading to both faster
convergence and better final performance values.
See the provided tutorials for examples.

Also, it is possible to proceed with quantization in a serie of smaller steps.
For example, it may be beneficial to train, first with all
standard/floating-point values, then retrain with quantized activations, and
then, finally, with quantized weights.

It is possible to subdivide even further – going step by step through the
individual layers and even modifying the targeted network sparsity\ [#fn-4]_ to
achieve an optimal trade-off between accuracy and speed.

Layers Considerations
---------------------

Supported layer types
^^^^^^^^^^^^^^^^^^^^^

The CNN2SNN toolkit currently supports the following layer types:


* **Core Neural Layers**\ :

  * `QuantizedDense <../api_reference/cnn2snn_apis.html#quantizeddense>`__\,
    a quantization-aware child of Keras `Dense <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`__
  * `QuantizedConv2D <../api_reference/cnn2snn_apis.html#quantizedconv2d>`__\,
    a quantization-aware child of Keras `Conv2D <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D>`__

* **Specialized Layers**\ :

  * `QuantizedSeparableConv2D <../api_reference/cnn2snn_apis.html#quantizedseparableconv2d>`__\,
    a quantization-aware child of Keras `SeparableConv2D <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SeparableConv2D>`_

* **Other Layers (from tf.keras)**\ :

  * BatchNormalization
  * MaxPooling2D
  * AveragePooling2D
  * GlobalAveragePooling2D
  * Dropout

.. note::
    AEE `check_model_compatibility <../api_reference/cnn2snn_apis.html#check-model-compatibility>`_
    function gives an indication regarding incompatible layers before conversion.
    A good practice is to check model compatibility before going through the
    training process.

Quantization-aware layers
^^^^^^^^^^^^^^^^^^^^^^^^^

Several articles have reported\ [#fn-5]_ that the quantization of a pre-trained
float Keras model using 8 bits precision can be performed with a minimal loss
of accuracy, but that for lower bitwidth a quantization-aware training of the
model is required. This is confirmed by our own experiments.

The CNN2SNN toolkit therefore includes quantization-aware versions of the base
Keras layers.

Quantization aware training simulates the effect of quantization in the forward
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

  * `ActivationDiscreteRelu <../api_reference/cnn2snn_apis.html#activationdiscreterelu>`_

Most of the parameters for the quantized processing layers are identical to
those used when defining a model using standard Keras layers. However, each of
these layers also includes a ``quantizer`` parameter that specifies the
`WeightQuantizer <../api_reference/cnn2snn_apis.html#weightquantizer>`_
object to use during the quantization-aware training.

.. note::
    `QuantizedConv2D <../api_reference/cnn2snn_apis.html#quantizedconv2d>`__\ ,
    supports convolutions with stride 1 only: to adapt an existing model with a
    higher convolution stride, we suggest substituting a convolution with stride
    1 followed by a pooling step of the appropriate size and stride. Only
    exception: if the first layer of the CNN model with image inputs is a
    QuantizedConv2D, a convolution stride is supported.

The quantized activation layer takes a single parameter corresponding to the
bitwidth of the quantized activations.

Training-Only Layers
^^^^^^^^^^^^^^^^^^^^

The AEE is used in CNN conversion for inference only. Training is done within
the Keras environment and training-only layers may be added at will, such as
BatchNormalization or Dropout layers. These are handled fully by Keras during
the training and do not need to be modified to be Akida-compatible for
inference.

As regards the implementation within the AEE: it may be helpful to understand
that the associated scaling operations (multiplication and shift) are never
performed within the AEE during inference. The computational cost is reduced by
wrapping the (optional) batch normalization function and quantized activation
function into the spike generating thresholds and other parameters of the Akida
SNN.
That process is completely transparent to the user. It does, however, have an
important consequence for the output of the final layer of the model; see
`Final Layers <#id6>`_ below.

First Layers
^^^^^^^^^^^^

Most layers of an Akida model only accept sparse inputs.
In order to support the most common classes of models in computer vision, a
special layer (`InputConvolutional <../api_reference/aee_apis.html#inputconvolutional>`__)
is however able to receive image data (8-bit grayscale or RGB). See the
`AEE user guide <aee.html>`__ for further details.

The CNN2SNN toolkit supports any quantization-aware training layer as the first
layer in the model. The type of input accepted by that layer can be specified
during conversion, but only models starting with a QuantizedConv2D layer will
accept dense inputs, thanks to the special (`InputConvolutional <../api_reference/aee_apis.html#inputconvolutional>`__)
layer.

Input Scaling
~~~~~~~~~~~~~~~

The `InputConvolutional <../api_reference/aee_apis.html#inputconvolutional>`_
layer only receives 8-bit input values:


* if the data is already in 8-bit format it can be sent to the AEE inputs
  without rescaling.
* if the data has been scaled to ease training, it is necessary to provide the
  scaling coefficients at model conversion.

This applies to the common case where input data are natively 8-bit. If input
data are not 8-bit, the process is more complex, and we recommend applying
rescaling in two steps:


#. Taking the data to an 8-bit unsigned integer format suitable for the AEE.
   Apply this step both for training and inference data.
#. Rescaling the 8-bit values to some unit or zero centered range suitable for
   CNN training, as above. This step should only be applied for the CNN training.
   Also, remember to provide those scaling coefficients when converting the
   trained model to an Akida-compatible format.

Final Layers
^^^^^^^^^^^^

As is typical for CNNs, the final layer of a model does not include the
standard activation nonlinearity. If that is the case, once converted to Akida,
the model will give the potentials levels and in most cases, taking the
maximum among these values is sufficient to obtain the correct response from
the model.
However, if there is a difference in performance between the Keras and the Akida
compatible implementations of the model, it is likely be at this step.

Layer Blocks
------------

Ensuring the conversion compatibility of a CNN model into an Akida model can
be tricky. Therefore, a higher-level interface is proposed with the use of
layer blocks. These blocks are available in the ``akida_models`` PyPi package:

.. code-block:: python

   import akida_models.quantization_blocks

Overview
^^^^^^^^

In Keras, when adding a core layer type (\ ``Dense`` or ``Conv2D``\ ) to a
model, an
activation function is typically included:

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

This particular arrangement of layers is important during the quantization-aware
training of Akida-compatible CNNs and is therefore reflected in the blocks
API.

For instance, the following code snippet sets up the same trio of layers as
those above:

.. code-block:: python

   x = dense_block(x, 64, add_batchnorm=True)

The ``dense_block`` function will produce a group of layers that we call a
"block".

.. note::
    **quantization_block = QuantizedConv2D/Dense/SeparableConv2D + (Pooling)
    + (BatchNorm) + (Activation)**

    To avoid adding the activation layer, add the parameter
    ``activ_quantization = None`` to the block.


The option of including pooling, batchnorm layers or activation is directly
built into the provided block modules.
The layer block functions provided are:


* ``conv_block``\ ,
* ``separable_conv_block``\ ,
* ``dense_block``.

Most of the parameters for these blocks are identical to those passed to the
corresponding inner quantized processing layers.

``conv_block``
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def conv_block(inputs,
                  filters,
                  kernel_size,
                  weight_quantization=0,
                  activ_quantization=0,
                  pooling=None,
                  pool_size=(2, 2),
                  add_batchnorm=False,
                  **kwargs):

``dense_block``
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def dense_block(inputs,
                   units,
                   weight_quantization=0,
                   activ_quantization=0,
                   add_batchnorm=False,
                   **kwargs)

``separable_conv_block``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def separable_conv_block(inputs,
                            filters,
                            kernel_size,
                            weight_quantization=0,
                            activ_quantization=0,
                            pooling=None,
                            pool_size=(2, 2),
                            add_batchnorm=False,
                            **kwargs)

Tips and Tricks
---------------

In some cases, it may be useful to adapt existing CNN models in order to
simplify or enhance the converted SNN. Here's a short list of some possible
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

.. [#fn-1] Typically, for the AEE – quantized weights and quantized activations.
.. [#fn-2] The spike value depends on the intensity of the potential, see the
           `AEE documentation <aee.html>`_ for details on the activation.
.. [#fn-3] Parallel layers and "residual" connections are currently not
           supported.
.. [#fn-4] Sparsity refers to the fraction of both weights and activations with
           value zero.
.. [#fn-5] See for instance "Quantizing deep convolutional networks for
           efficient inference: A whitepaper" - Raghuraman Krishnamoorthi, 2018
