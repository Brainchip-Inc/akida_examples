"""
Upgrading to Akida 2.0
======================

This tutorial targets Akida 1.0 users that are looking for advice on how to migrate their Akida 1.0
model towards Akida 2.0. It also lists the major differences in model architecture compatibilities
between 1.0 and 2.0.

"""

######################################################################
# 1. Workflow differences
# ---------------------------------
#
# .. figure:: ../../img/1.0vs2.0_flow.png
#    :target: ../../_images/1.0vs2.0_flow.png
#    :alt: 1.0 vs. 2.0 flow
#    :scale: 25 %
#    :align: center
#
#    Akida 1.0 and 2.0 workflows
#
# As shown in the figure above, the main difference between 1.0 and 2.0 workflows is the
# quantization step that was based on CNN2SNN and that is now based on QuantizeML.
#
# Providing your model architecture is 2.0 compatible (`next section
# <plot_1_upgrading_to_2.0.html#models-architecture-differences>`__ lists differences), upgrading to
# 2.0 is limited to moving from a `cnn2snn.quantize` call to a `quantizeml.models.quantize
# <../../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__ call. The code snippets
# below show the two different calls.
#

import keras

# Build a simple model that is cross-compatible
input = keras.layers.Input((32, 32, 3))
x = keras.layers.Conv2D(kernel_size=3, filters=32, strides=2, padding='same')(input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(units=10)(x)

model = keras.Model(input, x)
model.summary()

######################################################################

import cnn2snn

# Akida 1.0 flow
quantized_model_1_0 = cnn2snn.quantize(model, input_weight_quantization=8, weight_quantization=4,
                                       activ_quantization=4)
akida_model_1_0 = cnn2snn.convert(quantized_model_1_0)
akida_model_1_0.summary()

######################################################################

import quantizeml

# Akida 2.0 flow
qparams = quantizeml.models.QuantizationParams(input_weight_bits=8, weight_bits=4,
                                               activation_bits=4)
quantized_model_2_0 = quantizeml.models.quantize(model, qparams=qparams)
akida_model_2_0 = cnn2snn.convert(quantized_model_2_0)
akida_model_2_0.summary()


######################################################################
# .. note:: Here we use 8/4/4 quantization to match the CNN2SNN version above, but most users will
#           typically use the default 8-bit quantization that comes with QuantizeML.
#
# QuantizeML quantization API is close to the legacy CNN2SNN quantization API and further details on
# how to use it are given in the `global workflow tutorial
# <../general/plot_0_global_workflow.html>`__ and the `advanced QuantizeML tutorial
# <plot_0_advanced_quantizeml.html>`__.

######################################################################
# 2. Models architecture differences
# ----------------------------------
#
# 2.1. Separable convolution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In Akida 1.0, a Keras `SeparableConv2D
# <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SeparableConv2D>`__ used to be
# quantized as a `QuantizedSeparableConv2D` and converted to an Akida `SeparableConvolutional
# <../../api_reference/akida_apis.html#akida.SeparableConvolutional>`__ layer. These 3 layers each
# perform a "fused" operation where the depthwise and pointwise operations are grouped together in a
# single layer.
#
# In Akida 2.0, the fused separable layer support has been dropped in favor of a more commonly used
# unfused operation where the depthwise and the pointwise operations are computed in independent
# layers. The akida_models package offers a `separable_conv_block
# <../../api_reference/akida_models_apis.html#akida_models.layer_blocks.separable_conv_block>`__
# with a ``fused=False`` parameter that will create the `DepthwiseConv2D
# <https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D>`__ and the pointwise
# `Conv2D <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D>`__ layers under the
# hood. This block will then be quantized towards a `QuantizedDepthwiseConv2D
# <../../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedDepthwiseConv2D>`__ and a
# pointwise `QuantizedConv2D
# <../../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedConv2D>`__ before
# conversion into `DepthwiseConv2D <../../api_reference/akida_apis.html#akida.DepthwiseConv2D>`__
# and pointwise `Conv2D <../../api_reference/akida_apis.html#akida.Conv2D>`__ respectively.
#
# Note that while the resulting model topography is slightly different, the fused and unfused
# mathematical operations are strictly equivalent.
#
# In order to ease 1.0 to 2.0 migration of existing models, akida_models offers an
# `unfuse_sepconv2d <../../api_reference/akida_models_apis.html#akida_models.unfuse_sepconv2d>`__
# API that takes a model with fused layers and transforms it into an unfused equivalent version. For
# convenience, an ``unfuse`` CLI action is also provided.
#
# .. code-block:: bash
#
#    akida_models unfuse -m model.h5
#
# 2.2. Global average pooling operation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The supported position of the `GlobalAveragePooling2D
# <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D>`__ operation
# has changed in Akida 2.0 as it now must come after the ReLU activation (when there is one). In
# other words, in Akida 1.0 the layers were organized as follows:
#
# * ... > Neural layer > GlobalAveragePooling > (BatchNormalization) > ReLU >  Neural layer > ...
#
# In Akida 2.0 the supported sequence of layer is:
#
# * ... > Neural layer > (BatchNormalization) > (ReLU) > GlobalAveragePooling >  Neural layer > ...
#
# This can also be configured using the ``post_relu_gap`` parameter of akida_models `layer_blocks
# <../../api_reference/akida_models_apis.html#layer-blocks>`__.
#
# To migrate an existing model from 1.0 to 2.0, it is possible to load 1.0 weights into a 2.0
# oriented architecture using `Keras save and load APIs
# <https://www.tensorflow.org/tutorials/keras/save_and_load>`__ because the global average pooling
# position does not have an effect on model weights. However, the sequences between 1.0 and 2.0 are
# not mathematically equivalent so it might be required to tune or even retrain the model.
#

######################################################################
# 3. Using ``AkidaVersion``
# -------------------------
#
# It is still possible to build, quantize and convert models towards a 1.0 target using the
# `AkidaVersion API <../../api_reference/cnn2snn_apis.html#akida-version>`__.
#

# Reusing the previously defined 2.0 model but converting to a 1.0 target this time
with cnn2snn.set_akida_version(cnn2snn.AkidaVersion.v1):
    akida_model = cnn2snn.convert(quantized_model_2_0)
akida_model.summary()

######################################################################
# One will notice the different Akida layers types as detailed in `Akida user guide
# <../../user_guide/akida.html#akida-layers>`__.
