"""
Advanced QuantizeML tutorial
============================

This tutorial provides a comprehensive understanding of quantization in `QuantizeML python
package <../../user_guide/quantizeml.html#quantizeml-toolkit>`__. Refer to `QuantizeML user
guide <../../user_guide/quantizeml.html>`__  and `Global Akida workflow tutorial
<../general/plot_0_global_workflow.html>`__ for additional resources.

`QuantizeML python package <../../user_guide/quantizeml.html#quantizeml-toolkit>`__ provides
a user-friendly collection of functions for obtaining a quantized model. The `quantize
<../../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__ function replaces Keras
layers with quantized, integer only layers from `QuantizeML <../../user_guide/quantizeml.html>`__.

"""

######################################################################
# 1. Defining a quantization scheme
# ---------------------------------
#
# The quantization scheme refers to all the parameters used for quantization, that is the method of
# quantization such as per-axis or per-tensor, and the bitwidth used for inputs, outputs and
# weights.
#
# The first part in this section explains how to define a quantization scheme using
# `QuantizationParams
# <../../api_reference/quantizeml_apis.html#quantizeml.models.QuantizationParams>`__,
# which defines a homogeneous scheme that applies to all layers, and the second part explains how to
# fully customize the quantization scheme using a configuration file.
#

######################################################################
# 1.1. The quantization parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The easiest way to customize quantization is to use the ``qparams`` parameter of the `quantize
# <../../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`_ function. This is made
# possible by creating a `QuantizationParams
# <../../api_reference/quantizeml_apis.html#quantizeml.models.QuantizationParams>`__ object.

from quantizeml.models import QuantizationParams

qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8,
                             per_tensor_activations=False, output_bits=8, buffer_bits=32)

######################################################################
# By default, the quantization scheme adopted is 8-bit with per-axis activations, but it is possible
# to set every parameter with a different value. The following list is a detailed description of the
# parameters with tips on how to set them:
#
# - ``input_weight_bits`` is the bitwidth used to quantize weights of the first layer. It is usually
#   set to 8 which allows to better preserve the overall accuracy.
# - ``weight_bits`` is the bitwidth  used to quantize all other weights. It is usually set to 8
#   (Akida 2.0) or 4 (Akida 1.0).
# - ``activation_bits`` is the bitwidth used to quantize all ReLU activations. It is usually set to
#   8 (Akida 2.0) or 4 (Akida 1.0) but can be lower for edge learning (1-bit).
# - ``per_tensor_activations`` is a boolean that allows to define a per-axis (default) or per-tensor
#   quantization for ReLU activations. Per-axis quantization will usually provide more accurate
#   results (default ``False`` value) but it might be more challenging to `calibrate
#   <plot_0_advanced_quantizeml.html#calibration>`__ the model. Note that Akida 1.0 only supports
#   per-tensor activations.
# - ``output_bits`` is the bitwidth used to quantize intermediate results in
#   `OutputQuantizer <../../api_reference/quantizeml_apis.html#quantizeml.layers.OutputQuantizer>`__.
#   Go back to the `user guide quantization flow <../../user_guide/quantizeml.html#quantization-flow>`__
#   for details about this process.
# - ``buffer_bits`` is the maximum bitwidth allowed for low-level integer operations (e.g matrix
#   multiplications). It is set to 32 and should not be changed as this is what the Akida hardware
#   target will use.
#
# .. note:: It is recommended to quantize a model to 8-bit or 4-bit to ensure it is Akida hardware
#           compatible.
#
# .. warning:: ``QuantizationParams`` is only applied the first time a model is quantized.
#              If you want to re-quantize a model, you must to provide a complete ``q_config``.

######################################################################
# 1.2. Using a configuration file
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Quantization can be further customized via a JSON configuration passed to the ``q_config``
# parameter of the `quantize <../../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__
# function. This usage should be limited to targeted customization as writing a whole
# configuration from scratch is really error prone. An example of targeted customization is to set
# the quantization bitwidth of the output of a feature extractor to 1 which will allow edge learning.
#
# .. warning:: When provided, the configuration file has priority over arguments. As a result
#              however, the configuration file therefore must contain all parameters - you cannot
#              rely on argument defaults to set non-specified values.
#
# The following code snippets show what a configuration file looks like and how to edit it to
# customize quantization.
#

import keras
import json
from quantizeml.models import quantize, dump_config, QuantizationParams

# Define an example model with few layers to keep what follows readable
input = keras.layers.Input((28, 28, 3))
x = keras.layers.DepthwiseConv2D(kernel_size=3, name="dw_conv")(input)
x = keras.layers.Conv2D(filters=32, kernel_size=1, name="pw_conv")(x)
x = keras.layers.ReLU(name="relu")(x)
x = keras.layers.Dense(units=10, name="dense")(x)

model = keras.Model(input, x)

# Define QuantizationParams with specific values just for the sake of understanding the JSON
# configuration that follows.
qparams = QuantizationParams(input_weight_bits=16, weight_bits=4, activation_bits=6, output_bits=12,
                             per_tensor_activations=True, buffer_bits=24)

# Quantize the model
quantized_model = quantize(model, qparams=qparams)

######################################################################

quantized_model.summary()

######################################################################

# Dump the configuration
config = dump_config(quantized_model)

# Display in a JSON format for readability
print(json.dumps(config, indent=4))

######################################################################
# Explaining the above configuration:
#
# - the layer names are indexing the configuration dictionary.
# - the depthwise layer has an OutputQuantizer set to 12-bit (``output_bits=12``) to reduce
#   intermediate potentials bitwidth before the pointwise layer that follows (automatically added
#   when calling ``quantize``).
# - the depthwise layer weights are quantized to 16-bit because it is the first layer
#   (``input_weight_bits=16``) and are quantized per-axis (default for weights). The given axis is
#   -2 because of Keras depthwise kernel shape that is (Kx, Ky, F, 1), channel dimension is at index
#   -2.
# - the pointwise layer has weights quantized to 4-bit (``weight_bits=4``) but the quantization axis
#   is not specified as it defaults to -1 for a per-axis quantization. One would need to set it to
#   ``None`` for a per-tensor quantization.
# - the ReLU activation is quantized to 6-bit per-tensor (``activation_bits=6,
#   per_tensor_activations=True``)
# - all ``buffer_bitwidth`` are set to 24 (``buffer_bits=24``)
#
# The configuration will now be edited and used to quantize the float model with ``q_config``
# parameter.

# Edit the ReLU activation configuration
config["relu"]["output_quantizer"]['bitwidth'] = 1
config["relu"]["output_quantizer"]['axis'] = 'per-axis'
config["relu"]["output_quantizer"]['buffer_bitwidth'] = 32
config["relu"]['buffer_bitwidth'] = 32

# Drop other layers configurations
del config['dw_conv']
del config['pw_conv']
del config['dense']

# The configuration is now limited to the ReLU activation
print(json.dumps(config, indent=4))

######################################################################
# Now quantize with setting both ``qparams`` and ``q_config`` parameters: the activation will be
# quantized using the given configuration and the other layers will use what is provided in
# ``qparams``.

new_quantized_model = quantize(model, q_config=config, qparams=qparams)

######################################################################

# Dump the new configuration
new_config = dump_config(new_quantized_model)

# Display in a JSON format for readability
print(json.dumps(new_config, indent=4))

######################################################################
# The new configuration contains both the manually set configuration in the activation and the
# parameters defined configuration for other layers.

######################################################################
# 2. Calibration
# --------------

######################################################################
# 2.1. Why is calibration required?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# `OutputQuantizer <../../api_reference/quantizeml_apis.html#quantizeml.layers.OutputQuantizer>`__
# are added between layer blocks during quantization in order to decrease intermediate potential
# bitwidth and prevent saturation. Calibration is the process of defining the best quantization
# range possible for the OutputQuantizer.
#
# Calibration will statistically determine the quantization range by passing samples into the float
# model and observing the intermediate output values. The quantization range is stored in
# ``range_max`` variable. The calibration algorithm used in QuantizeML is based on a moving maximum:
# ``range_max`` is initialized with the maximum value of the first batch of samples (per-axis or
# per-tensor depending on the quantization scheme) and the following batches will update
# ``range_max`` with a moving momentum strategy (momentum is set to 0.9). Refer to the following
# pseudo code:
#
# .. code-block:: python
#
#   samples_max = reduce_max(samples)
#   delta = previous_range_max - new_range_max * (1 - momentum)
#   new_range_max = previous_range_max - delta
#
# In QuantizeML like in other frameworks, the calibration process happens simultaneously
# with quantization and the `quantize
# <../../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__ function thus comes with
# calibration parameters: ``samples``, ``num_samples``, ``batch_size`` and ``epochs``. Sections
# below describe how to set these parameters.
#
# .. note:: Calibration does not require any label or sample annotation and is therefore different
#           from training.

######################################################################
# 2.2. The samples
# ^^^^^^^^^^^^^^^^
#
# There are two types of calibration samples: randomly generated samples or real samples.
#
# When the ``samples`` parameter of ``quantize`` is left to the default ``None`` value, random
# samples will be generated using the ``num_samples`` value (default is 1024). When the model input
# shape has 1 or 3 channels, which corresponds to an image, the random samples value are unsigned
# 8-bit integers in the [0, 255] range. If the channel dimension is not 1 or 3, the generated
# samples are 8-bit signed integers in the [-128, 127] range.
# If that does not correspond to the range expected by your model, either add a `Rescaling
# <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling>`__ layer to your model
# using the `insert_rescaling helper
# <../../api_reference/quantizeml_apis.html#quantizeml.models.transforms.insert_rescaling>`__ or
# provide real samples.
#
# Real samples are often (but not necessarily) taken from the training dataset and should be the
# preferred option for calibration as it will always lead to better results.
#
# Samples are batched before being passed to the model for calibration. It is recommended to use at
# least 1024 samples for calibration. When providing samples, ``num_samples`` is only used to
# compute the number of steps during calibration.
#
# .. code-block:: python
#
#  if batch_size is None:
#      steps = num_samples
#  else:
#      steps = np.ceil(num_samples / batch_size)
#

######################################################################
# 2.3. Other calibration parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``batch_size``
# ~~~~~~~~~~~~~~
# Setting a large enough ``batch_size`` is important as it will impact ``range_max`` initialization
# that is made on the first batch of samples. The recommended value is 100.
#
# ``epochs``
# ~~~~~~~~~~
# It is the number of iterations over the calibration samples. Increasing the value will allow for
# more updates of the ``range_max`` variables thanks to the momentum policy without requiring a huge
# amount of samples. The recommended value is 2.
