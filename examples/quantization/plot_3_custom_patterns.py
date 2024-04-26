"""
Advanced ONNX models quantization
=================================

Akida, like any specialized hardware accelerator, sacrifices very generalized computational
ability in favor of highly optimized implementations of a subset of key operations. While
we strive to make sure that Akida directly supports the most important models, it isn't
feasible to support all possibilities. You may thus occasionally find yourself with a
model which is very nearly compatible with Akida, but which fails to convert due to just
a few incompatibilities. In this example, we will look at some simple workarounds and how
to implement them. The goal is to successfully convert the model to Akida without having
to retrain.

Preparing a model for Akida requires two steps: quantization, followed by conversion
for a specific target hardware device. We try to catch as many incompatibilities as
possible at the quantization step. However, some constraints depend on the specific
target device, and can only be caught at the conversion step. To illustrate, we will
simply walk through the process of preparing `ResNet50
<https://github.com/onnx/models/tree/main/archive/vision/classification/resnet#model>`__ for
acceleration on Akida - we'll run into several incompatibilities at different points
in that process, and see how to resolve them.

This example assumes a moderate level of experience with deep learning, and good familiarity
with the operations typically encountered in these types of models. For example, here we'll
use the following workarounds:

* to avoid some incompatible sequences of operations we'll insert layers with "identity"
  convolution kernels,
* in order to avoid an unusual kernel-size 1/stride 2 convolution, we'll substitute those
  kernels with equivalent size 3 kernels.
"""

######################################################################
# 1. Get model and data
# ---------------------
# Before diving into the model incompatibilities and how to resolve them, we'll need to acquire
# some sample data to test on, plus the pretrained model.

######################################################################
# 1.1 Data
# ^^^^^^^^
#
# Given that the reference model was trained on `ImageNet <https://www.image-net.org/>`__ dataset
# (which is not publicly available), this tutorial uses a set of 10 copyright free images.
# A helper function ``imagenet.preprocessing.get_preprocessed_samples`` loads
# and preprocesses (decodes, crops and extracts a square 224x224x3 patch from an input image)
# these images.
#

import numpy as np

from akida_models.imagenet import get_preprocessed_samples
from akida_models.imagenet.imagenet_utils import IMAGENET_MEAN, IMAGENET_STD

# Model specification and hyperparameters
NUM_CHANNELS = 3
IMAGE_SIZE = 224

# Load the preprocessed images and their corresponding labels for the test set
x_test_raw, labels_test = get_preprocessed_samples(IMAGE_SIZE, NUM_CHANNELS)
num_images = x_test_raw.shape[0]

# Normalize images as models expects
imagenet_mean_255 = np.array(IMAGENET_MEAN, dtype="float32") * 255.0
imagenet_std_255 = np.array(IMAGENET_STD, dtype="float32") * 255.0
x_test = ((x_test_raw - imagenet_mean_255) / imagenet_std_255)

# Transpose the channels to the first axis as per the default for ONNX models
x_test = np.transpose(x_test, (0, 3, 1, 2))

print(f'{num_images} images and their labels are loaded and preprocessed.')

######################################################################
# 1.2 Download the model
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We download ResNet50 from the `ONNX ZOO
# <https://github.com/onnx/models/tree/main/archive/vision/classification>`_,
#

import onnx
import onnx.hub
from onnxruntime import InferenceSession

onnx_model = onnx.hub.load("ResNet50")

######################################################################
# 1.3 Evaluate model performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The `ONNXRuntime <https://onnxruntime.ai>`__ package is a cross-platform
# accelerator capable of loading and running models described in ONNX format.
# We use this framework to evaluate the performance of the loaded ResNet50
# model.
#
# .. Note:: For example purposes, we only compute accuracy on 10 images.
#    Accuracy on the full ImageNet validation set is reported at the end.
#


def evaluate_onnx_model(model):
    sess = InferenceSession(model.SerializeToString())
    # Calculate outputs by running images through the session
    outputs = sess.run(None, {model.graph.input[0].name: x_test})
    # The class with the highest score is what we choose as prediction
    predicted = np.squeeze(np.argmax(outputs[0], 1))
    # Compute the number of valid predictions
    return int((predicted == labels_test).sum())


# Evaluate over test dataset
correctly_classified_floating = evaluate_onnx_model(onnx_model)
print(f'Floating point model accuracy: {correctly_classified_floating}/{num_images}.')

######################################################################
# 2. Quantize
# -----------
#
# Akida processes integer inputs, activations and weights. Therefore, the first step in
# preparing a floating point model to run on Akida is to quantize it using `QuantizeML quantize()
# <../../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__.
#
# .. Note::
#   Please refer to the `QuantizeML toolkit user guide <../../user_guide/quantizeml.html>`__
#   and the `Advanced QuantizeML tutorial <plot_0_advanced_quantizeml.html>`__ for further details.
#   In particular here, for simplicity, we pass only the small number of samples we already prepared
#   for calibration. Typically, you will want to use many more samples for calibration, say 1000 if
#   you have them available; and not drawn from your test data. The akida_models package provides a
#   helper function, `extract_samples() <../../api_reference/akida_models_apis.html#extract-samples>`__
#   which may be helpful in preparing those.

from quantizeml.models import quantize

model_quantized = quantize(onnx_model, samples=x_test)

######################################################################
# We can see that the model is not fully quantized, stopping at the first unrecognized
# pattern (node ``resnetv17_stage1_activation0 (Relu)``). We know that Akida can definitely
# handle ReLU activation functions, so we have to look more closely to understand the
# problem. Analyzing the model, the ReLU immediately follows an ``Add`` operator. It is
# this sequence of operations which is not supported by Akida.
#
# .. figure:: ../../img/unsupported_activation.png
#    :target: ../../_images/unsupported_activation.png
#    :alt: Unsupported activation
#    :scale: 70 %
#    :align: center
#
#    Unsupported pattern: [``Add``, ``Relu``].
#
#

######################################################################
# 2.1 About Patterns
# ^^^^^^^^^^^^^^^^^^
#
# For efficiency, Akida hardware actually groups certain commonly occuring
# operations together. For example, ReLU activation functions, where present,
# are almost always applied on the outputs of the hard-working computational
# layers (Convolutions, Depthwise Convolutions, Dense layers etc.). So the ReLU
# on Akida is tied to those operations. While efficient, this does mean that
# some sequences of operations will not by default be considered Akida-compatible,
# even though the individual operations are known to be handled. That's the
# cause of the problem encountered here.
#
#
# To properly see what's going on, and to resolve the problem, we'll need to
# understand the concept of "patterns". These are the objects that QuantizeML
# uses to map ONNX models to their Akida equivalents. A pattern is a sequence of
# continuous `ONNX operators <https://onnx.ai/onnx/operators/>`_ in a graph that
# **can be converted** to an
# `Akida V2 layer <../../api_reference/akida_apis.html#akida-v2-layers>`_.
# For example, the following model would be converted to an `akida.InputConv2D
# <../../api_reference/akida_apis.html#akida.InputConv2D>`_ layer:
#
# .. figure:: ../../img/onnx_input_conv2d.png
#    :target: ../../_images/onnx_input_conv2d.png
#    :alt: InputConv2D example model
#    :scale: 80 %
#    :align: center
#
#    One ONNX configuration that would map to an `InputConv2D
#    <../../api_reference/akida_apis.html#akida.InputConv2D>`_.
#
#
# The sequence of operators [``Conv``, ``Clip``, ``MaxPool``] **is one valid pattern**
# for conversion towards `InputConv2D <../../api_reference/akida_apis.html#akida.InputConv2D>`_.
#
#
# Crucially, we can check the list of the currently supported patterns:
#

from quantizeml.onnx_support.quantization.register_patterns import PATTERNS_MAP

print(*PATTERNS_MAP, sep='\n')

######################################################################
# Looking at that list, it should be apparent that a ``ReLU`` operation on its own or
# following an ``Add`` is not considered a compatible pattern.
#
# .. Note::
#   Before the conversion the following changes are automatically done to allow the
#   QuantizeML toolkit to see an ONNX graph suitable for quantization:
#
#       1. transforms the following operators for general purposes:
#
#          * ``Conv`` -> ``DepthwiseConv`` when kernel size is 1 x Kx x Ky and ``group`` is required
#          * ``Clip`` > ``Relu`` (if ``min = 0.0``)
#
#       2. uses `Graph Optimizations in ONNX Runtime
#          <https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html>`_
#          to optimize the graph (e.g. fuse BatchNorm into convolutions).
#


######################################################################
# 2.2. Custom quantization patterns
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The existing patterns won't allow us to map an isolated ReLU operation. But, for example,
# the ReLU operation can be mapped when following a Conv layer, and we can easily implement
# a Conv layer that performs an identity operation on its inputs, just by setting the kernel
# weights appropriately. We can implement this workaround by using custom quantization
# patterns to extend ``PATTERNS_MAP``.
#
# Every pattern includes an ONNX layer that stores the ONNX graph information for the matching
# sequence of nodes. QuantizeML also allows for a function to create a compatible layer from
# an initially incompatible pattern. This pattern function has two input parameters: the graph
# and the pattern-matched sequence of nodes extracted from it.
#
# Once a pattern function is defined for an unsupported pattern, both can be appended
# in the quantization context through the ``custom_pattern_scope`` function.

from quantizeml.onnx_support import layers
from quantizeml.onnx_support.quantization import custom_pattern_scope


class IdentityQuantizedConv2D(layers.QuantizedConv2D):
    def __build__(self, input_ts, downscale=True):
        # Produces a kernel such that the convolution does not modify the input.
        identity_kernel = np.identity(input_ts.shape[1], "float32")[..., None, None]
        self.set_weight("kernel", identity_kernel)
        return super().__build__(input_ts, downscale)


def relu_pattern_fn(block_nodes, graph):
    """Convert the incompatible patterns ['Relu'] and ['Relu', 'GlobalAveragePool'] into
    an IdentityQuantizedConv2D.
    """
    # Note that as 'quantization_pattern_map' is written, this function expects to receive
    # only the isolated ('Relu') that matches in the graph.
    block_ops = [x.op_type for x in block_nodes]
    if block_ops == ['Relu']:
        return IdentityQuantizedConv2D(activation=True)
    else:
        raise Exception(f"Unrecognized pattern: {block_ops}")


# Define a custom patterns map, as a new pattern and associated replacement function.
relu_pattern_map = {
    "Relu": relu_pattern_fn,
}

# Include relu_pattern_map in the quantization context
with custom_pattern_scope(relu_pattern_map):
    model_quantized = quantize(onnx_model, samples=x_test)


######################################################################
# With the isolated ReLU fixed, we managed to quantize much more of the model, but
# we hit a new problem, node ``resnetv17_pool1_fwd (GlobalAveragePool)``. Looking back
# at the list of compatible patterns, we can see that, like the ReLU, a GlobalAveragePooling
# (GAP) operation cannot be handled in isolation, but is compatible when it follows
# Conv or Conv + ReLU operations. The second of those will suit us better here,
# that way we can combine it with our solution for the ReLU operation (because
# the GAP here does indeed follow one of the isolated ReLU ops).
#

def activation_pattern_fn(block_nodes, graph):
    """Convert the incompatible patterns ['Relu'] and ['Relu', 'GlobalAveragePool'] into
    an IdentityQuantizedConv2D.
    """
    # Note that as 'quantization_pattern_map' is written, this function expects to receive
    # only the sequences ('Relu') or ('Relu', 'GlobalAveragePool').
    block_ops = [x.op_type for x in block_nodes]
    if block_ops == ['Relu']:
        return IdentityQuantizedConv2D(activation=True)
    elif block_ops == ['Relu', 'GlobalAveragePool']:
        return IdentityQuantizedConv2D(activation=True, pool_type="gap")
    else:
        raise Exception(f"Unrecognized pattern: {block_ops}")


# Define quantization custom patterns map, as a set of patterns and associated replacement function.
# activation_pattern_fn was designed to handle two similar incompatibilities present in ResNet50.
quantization_pattern_map = {
    ("Relu", "GlobalAveragePool"): activation_pattern_fn,
    "Relu": activation_pattern_fn,
}

# Include quantization_pattern_map in the quantization context
with custom_pattern_scope(quantization_pattern_map):
    model_quantized = quantize(onnx_model, samples=x_test)


######################################################################
# The full model is now quantized successfully.
# At this point we can re-check its accuracy:

correctly_classified = evaluate_onnx_model(model_quantized)
print(f'Quantized model accuracy: {correctly_classified}/{num_images}.')

######################################################################
# 3. Conversion
# -------------

######################################################################
# 3.1. Incompatibility at Conversion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As indicated above, while most imcompatibilities will be picked up at the
# quantization step, some constraints are specific to the target hardware
# device, and can only be applied at the conversion step. We can detect these
# either with the `check_model_compatibility <../../api_reference/cnn2snn_apis.html#cnn2snn.check_model_compatibility>`__ tool,
# or by trying to `convert the model into Akida <../../api_reference/cnn2snn_apis.html#cnn2snn.convert>`__.

from cnn2snn import convert

try:
    akida_model = convert(model_quantized)
except Exception as e:
    print(f"ResNet50 is not fully accelerated by Akida. Reason: {str(e)}")

######################################################################
# This error is raised because the ResNet50 padding scheme is very specific and differs
# from the Keras/Akida standard.
#
# Ideally, we should aim to swap incompatible operations with mathematically
# equivalent replacements. For issues of convolution kernel size or padding, we can
# often achieve that by putting the kernel weights within a larger kernel, placed
# eccentrically to compensate for any padding issues etc. More on that below - but
# we can't use that strategy here, because the kernel size for this layer (7x7) is
# already the maximum supported by the Akida input layer. In this case, we'll have to
# try simply modifying the padding to be Akida-compatible. Because this is the input
# layer, we could actually negate that change by padding the input image along two
# edges before passing to Akida. However, precisely because this is the very start of
# the network, and the consequence is only a single pixel of spatial offset, we might
# expect that the impact on model performance will be negligible, and that's precisely
# what we find on testing. So let's keep things simple in this case: simply replace the
# incompatible values with compatible ones.
#
# To achieve this, we'll again customize the pattern functions to modify the model before
# quantization. Rather than try to provide a general solution, we'll hard code this for
# the problem layer:

from quantizeml.onnx_support import graph_tools


def align_input_conv_with_akida(block_nodes, graph):
    """Pattern function that handles convolutions incompatible with Akida
    """
    # Recover initial ONNXLayer from block nodes and graph
    qconv = layers.get_qconv(block_nodes, graph)

    # Force the pads in first convolution to Akida compatible values
    if qconv.name == 'resnetv17_conv0_fwd':
        print("Setting Akida pads in first convolution...")
        # Note: pads in convolution include spatial dimension
        qconv.set_weight("pads", np.array([0, 0, 2, 2, 0, 0, 3, 3]))
        graph_tools.replace_field(qconv, "pool_pads", [0, 0, 1, 1])
    return qconv


# Infer intermediate shape: This is required for some custom pattern functions
onnx_model_temp = onnx.shape_inference.infer_shapes(onnx_model)

# Quantize model with custom patterns
quantization_pattern_map = {
    ("Conv", "Relu", "MaxPool"): align_input_conv_with_akida,
    ("Conv", "Relu"): align_input_conv_with_akida,
    ("Conv",): align_input_conv_with_akida,
    ("Relu", "GlobalAveragePool"): activation_pattern_fn,
    "Relu": activation_pattern_fn,
}
with custom_pattern_scope(quantization_pattern_map):
    model_quantized = quantize(onnx_model_temp, samples=x_test)

# Evaluate quantized model performance
correctly_classified = evaluate_onnx_model(model_quantized)
print(f'Quantized model accuracy: {correctly_classified}/{num_images}.')

######################################################################
# 3.2. Successful Conversion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Time to check conversion again

akida_model = convert(model_quantized)

######################################################################
# Great - the model is now both quantized successfully, and can be
# entirely converted for acceleration on Akida. To check its
# performance, we need to bear in mind that
#
# 1. images must be numpy-raw, with an 8-bit unsigned integer data type and
# 2. the channel dimension must be in the last dimension.

# Evaluate performance
akida_accuracy = akida_model.evaluate(x_test_raw, labels_test)
print(f'Akida model accuracy: {100 * akida_accuracy:.2f} %')

######################################################################
# 3.3. Performance on the full ImageNet validation set
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The table below summarizes the obtained accuracy at the various stages using the full
# ImageNet dataset. Note that forcing pads on the first layer decreases the performance
# of the model by 0.445% - as noted, that change could be rendered lossless by padding
# the input image prior to sending instead.
#
# +------------------------------------------+----------------+--------------------+----------------+
# | Float accuracy (before Akida adaptation) | Float accuracy | Quantized accuracy | Akida accuracy |
# +==========================================+================+====================+================+
# | 74.368                                   | 73.918         | 73.590             | 73.620         |
# +------------------------------------------+----------------+--------------------+----------------+
#
# .. Note::
#    The images shown in this tutorial are produced through `Netron <https://netron.app/>`_.
