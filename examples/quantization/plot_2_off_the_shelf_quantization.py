"""
Off-the-shelf models quantization
=================================

.. Warning::
   QuantizeML ONNX quantization is an **evolving feature**. Some models may not be compatible.

| The `Global Akida workflow <../general/plot_0_global_workflow.html>`__ and the
  `PyTorch to Akida workflow <../general/plot_8_global_pytorch_workflow.html>`__ guides
  describe all the steps required to create, train, quantize and convert a model for Akida,
  respectively using TensorFlow/Keras and PyTorch frameworks.
| Here we will illustrate off-the-shelf/pretrained CNN models quantization for Akida using
  `MobileNet V2 <https://huggingface.co/docs/transformers/model_doc/mobilenet_v2>`__ from
  the `Hugging Face Hub <https://huggingface.co/docs/hub/index>`__.

.. Note::
   | Off-the-shelf CNN models refer to already trained floating point models.
   | Their training recipe and framework have no importance as long as they can be exported
     to `ONNX <https://onnx.ai>`__.
   | Note however that this pathway offers slightly less flexibility than our default,
     TensorFlow-based pathway - specifically, fine tuning of the quantized model is
     not possible.
   | In most cases, that won't matter, there should be almost no performance drop when
     quantizing to 8-bit anyway.

.. Note::
   | This tutorial leverages the `Optimum toolkit
     <https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model>`__,
     an external tool, based on `PyTorch <https://pytorch.org/>`__, that allows models direct
     download and export to  ONNX.

     .. code-block::

        pip install optimum[exporters]

"""


######################################################################
# 1. Workflow overview
# ~~~~~~~~~~~~~~~~~~~~
#
# .. figure:: ../../img/off_the_shelf_flow.png
#    :target: ../../_images/off_the_shelf_flow.png
#    :alt: Off-the-shelf models quantization flow
#    :scale: 60 %
#    :align: center
#
#    Off-the-shelf CNN models Akida workflow
#
# As shown in the figure above, the `QuantizeML toolkit
# <../../api_reference/quantizeml_apis.html>`__ allows the Post Training Quantization of ONNX
# models.
#


######################################################################
# 2. Data preparation
# ~~~~~~~~~~~~~~~~~~~
#
# Given that the reference model was trained on `ImageNet <https://www.image-net.org/>`__ dataset
# (which is not publicly available), this tutorial used a subset of 10 copyright free images.
# A helper function ``imagenet.preprocessing.get_preprocessed_samples`` loads
# and preprocesses (decodes, crops and extracts a square 224x224x3 patch from an input image)
# these images.
#

import numpy as np
from akida_models.imagenet import get_preprocessed_samples

# Model specification and hyperparameters
NUM_CHANNELS = 3
IMAGE_SIZE = 224

# Load the preprocessed images and their corresponding labels for the test set
x_test_raw, labels_test = get_preprocessed_samples(IMAGE_SIZE, NUM_CHANNELS)
num_images = x_test_raw.shape[0]

# Get labels for the test set by index
# Note: Hugging Face models reserve the first index to null predictions
# (labeled as 'background' id). That is why we increase in '1' the original label id.
labels_test = labels_test + 1

print(f'{num_images} images and their labels are loaded and preprocessed.')

######################################################################
# As illustrated in `1. Workflow overview`_, the model's source is at the user's
# discretion. Here, we know a priori that MobileNet V2 was trained with
# images normalized within [-1, 1] interval. Also, ONNX models are usually
# saved with a `channels-first` format, input images are expected to be passed
# with the channels dimension on `axis = 1`.
#

# Project images in the range [-1, 1]
x_test = (x_test_raw / 127.5 - 1).astype('float32')

# Transpose the channels to the first axis
x_test = np.transpose(x_test, (0, 3, 1, 2))


######################################################################
# 3. Download and export
# ~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# 3.1. Download ONNX MobileNet V2
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# There are many repositories with models saved in ONNX format. In this example the
# `Optimum API <https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model>`__
# is used for downloading and exporting models to ONNX.
#

import onnx
from optimum.exporters.onnx import main_export

# Download and convert MobiletNet V2 to ONNX
main_export(model_name_or_path="google/mobilenet_v2_1.0_224",
            task="image-classification",
            output="./")

# Load the model in memory
model_onnx = onnx.load_model("./model.onnx")
print(onnx.helper.printable_graph(model_onnx.graph))


######################################################################
# 3.2. Evaluate model performances
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The `ONNXRuntime <https://onnxruntime.ai>`__ package is a cross-platform
# accelerator capable of loading and running models described in ONNX format.
# We take advantage of this framework to run the ONNX models and evaluate
# their performances.
#
# .. Note:: We only compute accuracy on 10 images.
#

from onnxruntime import InferenceSession


def evaluate_onnx_model(model):
    sess = InferenceSession(model.SerializeToString())
    # Calculate outputs by running images through the session
    outputs = sess.run(None, {model.graph.input[0].name: x_test})
    # The class with the highest score is what we choose as prediction
    predicted = np.squeeze(np.argmax(outputs[0], 1))
    # Compute the number of valid predictions
    return int((predicted == labels_test).sum())


correctly_classified_floating = evaluate_onnx_model(model_onnx)
print(f'Floating point model accuracy: {correctly_classified_floating}/{num_images}.')


######################################################################
# 4. Quantize
# ~~~~~~~~~~~
#
# | Akida processes integer activations and weights. Therefore, the floating point model
#   must be quantized in preparation to run on an Akida accelerator.
# | `QuantizeML quantize() <../../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__
#   function recognizes `ModelProto <https://onnx.ai/onnx/api/classes.html#modelproto>`__ objects
#   and quantizes them for Akida. The result is another ``ModelProto``, compatible with the
#   `CNN2SNN Toolkit <../../user_guide/cnn2snn.html>`__.
# | The table below summarizes the obtained accuracy at the various stages using the full
#   ImageNet dataset.
#
# +-------------------------------+----------------+--------------------+----------------+
# | Calibration parameters        | Float accuracy | Quantized accuracy | Akida accuracy |
# +===============================+================+====================+================+
# | Random samples / per-tensor   | 71.790         | 70.550             | 70.588         |
# +-------------------------------+----------------+--------------------+----------------+
# | Imagenet samples / per-tensor | 71.790         | 70.472             | 70.628         |
# +-------------------------------+----------------+--------------------+----------------+
#
# .. Note::
#    Please refer to the `QuantizeML toolkit user guide <../../user_guide/quantizeml.html>`__
#    and the `Advanced QuantizeML tutorial <./plot_0_advanced_quantizeml.html>`__ for details
#    about quantization parameters.
#

from quantizeml.models import quantize, QuantizationParams

# Quantize with activations quantized per tensor
qparams = QuantizationParams(per_tensor_activations=True)
model_quantized = quantize(model_onnx, qparams=qparams, num_samples=5)

# Evaluate the quantized model performance
correctly_classified = evaluate_onnx_model(model_quantized)
print(f'Quantized model accuracy: {correctly_classified_floating}/{num_images}.')

######################################################################
# 5. Convert to Akida
# ~~~~~~~~~~~~~~~~~~~

from cnn2snn import convert

# Convert the model
model_akida = convert(model_quantized)
model_akida.summary()

######################################################################

# Evaluate Akida performance using raw data
out = model_akida.forward(x_test_raw)
preds = np.squeeze(np.argmax(out, -1))
print(f'Akida model accuracy: {int((preds == labels_test).sum())}/{num_images}.')
