"""
Build Vision Transformers for Akida
===================================

The Vision Transformer, or ViT, is a model for image classification that employs a Transformer-like
architecture over patches of the image. An image is split into fixed-size patches, each of them are
then linearly embedded, position embeddings are added, and the resulting sequence of vectors are
fed to a standard Transformer encoder. Please refer to https://arxiv.org/abs/2010.11929 for further
details.

Akida 2.0 now supports patch and position embeddings, and the encoder block in hardware. This
tutorial explains how to build an optimized ViT using Akida models python API for Akida 2.0 hardware.

"""

######################################################################
# 1. Model selection
# ~~~~~~~~~~~~~~~~~~
# There are many variants of ViT. The choice of the model is typically influenced by the tradeoff
# among architecture size, accuracy, inference speed, and training capabilities.
#
# The following table shows few variants of commonly used ViT:
#
# +--------------+-------------------+---------+-------------------+
# | Architecture | Original accuracy | #Params | Architecture      |
# +==============+===================+=========+===================+
# | ViT Base     |  79.90%           |  86M    |  12 heads,        |
# |              |                   |         |  12 blocks,       |
# |              |                   |         |  hidden size 768  |
# +--------------+-------------------+---------+-------------------+
# | ViT Tiny     |  75.48%           |  5.8M   |  3 heads,         |
# |              |                   |         |  12 blocks,       |
# |              |                   |         |  hidden size 192  |
# +--------------+-------------------+---------+-------------------+
# | DeiT-dist    |  74.17%           |  5.8M   |  3 heads,         |
# | Tiny         |                   |         |  12 blocks,       |
# |              |                   |         |  hidden size 192  |
# +--------------+-------------------+---------+-------------------+
#
# .. note:: The Vision Transformers support has been introduced in Akida 2.0.
#
# The Akida model zoo provides tiny  ViT architectures that are optimized to run on Akida
# hardware:
#
#  - `ViT (tiny) <../../api_reference/akida_models_apis.html#akida_models.bc_vit_ti16>`__,
#  - `DeiT-dist (tiny) <../../api_reference/akida_models_apis.html#akida_models.bc_deit_ti16>`__.
#
# Both architectures have been modified so that their layers can be quantized to integer only
# operations.


######################################################################
# 2. Model optimization for Akida hardware
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ViT has many encoder blocks that perform self-attention to process visual data. Each encoder
# block consists of many different layers. To optimally run ViT at the edge using Akida requires
# transforming this encoder block in the following way:
#
#   - replace `LayerNormalization
#     <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization>`__ with
#     `LayerMadNormalization
#     <../../api_reference/quantizeml_apis.html#quantizeml.layers.LayerMadNormalization>`__,
#   - replace the last `LayerNormalization
#     <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization>`__ previous
#     to the classification head with a `BatchNormalization
#     <https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization>`__,
#   - replace `GeLU <https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GELU>`__
#     with `ReLU8 <https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU>`__ activations,
#   - replace `Softmax
#     <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax>`__ operation in
#     `Attention <../../api_reference/quantizeml_apis.html#quantizeml.layers.Attention>`__ with a
#     `shiftmax <../../api_reference/quantizeml_apis.html#quantizeml.layers.shiftmax>`__ operation.
#
# .. note:: Sections below show different ways to train a ViT for Akida which uses the above
#           transformations.


######################################################################
# 3. Model Training
# ~~~~~~~~~~~~~~~~~
# Akida accelerates ViT model that has the transformation mentioned in Section 2. Training a ViT
# that optimally runs on Akida can be made possible in the following two ways:


######################################################################
# 3.1 Option 1: Training a ViT (original) model first and then transforming each layer incrementally
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First, train a ViT (original) model on a custom dataset until satisfactory accuracy. It is then
# possible to transform this model into an Akida optimized one as per Section 2. The layers mentioned
# in Section 2 are functionally equivalent to each of the layers present in the original model.
#
# .. note:: To overcome the accuracy drop from the original when transforming the model as per Section 2,
#           it is recommended to replace the original layers one at a time and to fine-tune at every
#           step.
#
# The example below shows the transformation of ViT (tiny) into an optimized model that can run on
# the Akida hardware.
#
# The `akida_models <https://pypi.org/project/akida-models>`__ python package provides a Command Line
# Interface (CLI) to transform `vit_ti16 <../../_modules/akida_models/transformers/model_vit.html#vit_ti16>`__
# and `deit_ti16 <../../_modules/akida_models/transformers/model_deit.html#deit_ti16>`__ model architectures
# and fine-tune them respectively.
#
# .. code-block:: bash
#
#   $ akida_models create vit_ti16 -h
#   usage: akida_models create vit_ti16 [-h] [-c CLASSES] [-bw BASE_WEIGHTS] [--norm {LN,GN1,BN,LMN}]
#                                       [--last_norm {LN,BN}] [--softmax {softmax,softmax2}]
#                                       [--act {GeLU,ReLU8,swish}] [-i {224,384}]
#
#   optional arguments:
#     -h, --help            show this help message and exit
#     -c CLASSES, --classes CLASSES
#                           The number of classes, by default 1000.
#     -bw BASE_WEIGHTS, --base_weights BASE_WEIGHTS
#                           Optional keras weights to load in the model, by default None.
#     --norm {LN,GN1,BN,LMN}
#                           Replace normalization in model with a custom function, by default LN
#     --last_norm {LN,BN}   Replace last normalization in model with a custom function, by default LN
#     --softmax {softmax,softmax2}
#                           Replace softmax operation in model with custom function, by default softmax
#     --act {GeLU,ReLU8,swish}
#                           Replace activation function in model with custom function, by default GeLU
#     -i {224,384}, --image_size {224,384}
#                           The square input image size
#
# The following shows the transformation of a vit_ti16 model architecture which was trained on ImageNet. The
# same methods can be applied for other datasets.
#
# .. code-block:: bash
#
#   # download the pre-trained weights
#   wget https://data.brainchip.com/models/AkidaV2/vit/vit_ti16_224.h5
#
#   # transformation 1: replace layer normalization with mad norm layer and last layer normalization with batch normalization
#   akida_models create -s vit_ti16_lmnbn.h5 vit_ti16 -bw vit_ti16_224.h5 --norm LMN --last_norm BN
#   # fine-tuning
#   imagenet_train tune -m vit_ti16_lmnbn.h5 -e 15 --optim Adam --lr_policy cosine_decay \
#                       -lr 6e-5 -s vit_ti16_lmnbn_tuned.h5
#
#   # transformation 2: replace GeLU layer with ReLU
#   akida_models create -s vit_ti16_relu.h5 vit_ti16 -bw vit_ti16_lmnbn_tuned.h5 --norm LMN --last_norm BN --act ReLU8
#   # fine-tuning
#   imagenet_train tune -m vit_ti16_relu.h5 -e 15 --optim Adam --lr_policy cosine_decay \
#                       -lr 6e-5 -s vit_ti16_relu_tuned.h5
#
#   # transformation 3: replace softmax with shiftmax layer
#   akida_models create -s vit_ti16_shiftmax.h5 vit_ti16 -bw vit_ti16_relu_tuned.h5 --norm LMN --last_norm BN --act ReLU8 --softmax softmax2
#   # fine-tuning
#   imagenet_train tune -m vit_ti16_shiftmax.h5 -e 15 --optim Adam --lr_policy cosine_decay \
#                       -lr 6e-5 -s vit_ti16_transformed.h5
#
# The above transformation generates a ViT model that is optimized to run efficiently on Akida hardware.
# Similar steps can also be applied to deit_ti16. The table below highlights the accuracy of the original
# and transformed models.
#
# +--------------+-------------------+----------------------+
# | Architecture | Original accuracy | Transformed accuracy |
# +==============+===================+======================+
# | ViT          |  75.48%           | 74.25%               |
# +--------------+-------------------+----------------------+
# | DeiT-dist    |  74.17%           | 75.03%               |
# +--------------+-------------------+----------------------+
#
# .. note:: The models obtained above have floating point weights and are ready to be quantized.
#           See Section 4.


######################################################################
# 3.2 Option 2: Transfer Learning using Pre-trained transformed model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The `Akida models python package <../../api_reference/akida_models_apis.html>`__ has  `APIs for ViTs
# <../../api_reference/akida_models_apis.html#layer-blocks>`__ which provides pre-trained models for
# `vit_ti16 <../../_modules/akida_models/transformers/model_vit.html#vit_ti16>`__ and `deit_ti16
# <../../_modules/akida_models/transformers/model_deit.html#deit_ti16>`__. These models can be used
# for Transfer Learning on a custom dataset. Since the above models are already transformed, no
# further transformation is required.
#
# Visit our `Transfer Learning Example <plot_4_transfer_learning.html>`__ to learn more about Transfer
# Learning using the `Akida models python package <../../api_reference/akida_models_apis.html>`__. The
# following code snippet downloads a pre-trained model that can be used for Transfer Learning.

# The following is the API download the vit_t16 model trained on ImageNet dataset
from akida_models.model_io import load_model
from tensorflow.keras.utils import get_file

# Retrieve the float model with pretrained weights and load it
model_file = get_file(
    "bc_vit_ti16_224.h5",
    "https://data.brainchip.com/models/AkidaV2/vit/bc_vit_ti16_224.h5",
    cache_subdir='models/akidanet_imagenet')
model_keras = load_model(model_file)
model_keras.summary()

######################################################################
# .. note:: The models in Section 3 have floating point weights. Once the desired accuracy is obtained,
#           these models should go through quantization before converting to Akida.


######################################################################
# 4. Model quantization
# ~~~~~~~~~~~~~~~~~~~~~
# Akida 2.0 hardware adds efficient processing of 8-bit weights and activations for Vision Transformer
# models. This requires models in Section 3 to be quantized to 8-bit integer numbers. This means both
# weights and activation outputs become 8-bit integer numbers. This results in a smaller  model with
# minimal to no drop in accuracy and achieves improvements in latency and power when running on Akida
# hardware.
#
# Quantization of ViT models can be done using `QuantizeML python package <../../user_guide/quantizeml.html>`__
# using either Post Training Quantization (PTQ) or Quantization Aware Training (QAT) methods. The following
# section shows quantization an example, quantization of `vit_ti16
# <../../_modules/akida_models/transformers/model_vit.html#vit_ti16>`__ trained on ImageNet dataset.


######################################################################
# 4.1 Post-Training Quantization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Using `QuantizeML python package <../../user_guide/quantizeml.html>`__, ViT model can be quantized to
# 8-bit integer numbers (both weights and activation outputs). PTQ requires calibration (ideally using
# reference data) which helps to determine optimal quantization ranges. To learn more about PTQ, refer
# to `Advanced QuantizeML tutorial <../quantization/plot_0_advanced_quantizeml.html>`__.

# Using QuantizeML to perform quantization
from quantizeml.models import quantize
from quantizeml.layers import QuantizationParams

# Define the quantization parameters.
qparams = QuantizationParams(weight_bits=8, activation_bits=8)

# Quantize the model defined in Section 3.2
model_quantized = quantize(model_keras, qparams=qparams)
model_quantized.summary()

######################################################################
# The `bc_vit_ti16_imagenet_pretrained helper
# <../../api_reference/akida_models_apis.html#akida_models.bc_vit_ti16_imagenet_pretrained>`__
# was obtained with the same 8-bit quantization scheme but with an additional QAT step to further
# improve accuracy.


######################################################################
# 4.2 Quantization Aware Training (Optional)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In Section 4.1, we performed PTQ and converted the weights and activation outputs to 8-bit integer numbers.
# In most cases, there is no accuracy drop observed after quantization, however in cases where an accurary
# drop is observed, it is possible to further fine-tune this model using QAT.
#
# The model that is obtained through `QuantizeML python package <../../user_guide/quantizeml.html>`__ is an
# instance of Keras. This allows the model to be fine-tuned using the original dataset to regain accuracy.
#
# `Akida models python package <../../api_reference/akida_models_apis.html>`__  provides pre-trained models
# for vit_ti16 and deit_ti16 that have been trained using QAT method. It can be used in the following way:

from akida_models import bc_vit_ti16_imagenet_pretrained

# Load the pre-trained quantized model
model_quantized = bc_vit_ti16_imagenet_pretrained()
model_quantized.summary()


######################################################################
# 5. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~
#
# A model quantized through `QuantizeML python package <../../user_guide/quantizeml.html>`__ is ready to be
# converted to Akida. Once the quantized model has the desired accuracy `CNN2SNN toolkit <../../user_guide/cnn2snn.html>`__
# is used for conversion to Akida. There is no further optimization required and equivalent accuracy is
# observed upon converting the model to Akida.

from cnn2snn import convert

# Convert the model
model_akida = convert(model_quantized)
model_akida.summary()


######################################################################
# 6. Displaying results Attention Maps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Instead of showing predictions, here we propose to show attention maps on an image. This is
# derived from `Abnar et al. attention rollout <https://arxiv.org/abs/2005.00928>`__ as shown in the
# following `Keras tutorial
# <https://keras.io/examples/vision/probing_vits/#method-ii-attention-rollout>`__. This aims to
# highlight the model abilities to focus on relevant parts in the input image.
#
# Just like for the `AkidaNet example
# <plot_1_akidanet_imagenet.html#sphx-glr-examples-general-plot-1-akidanet-imagenet-py>`__, ImageNet
# images are not publicly available, this example uses a set of 10 copyright free images that were
# found on Google using ImageNet class names.
#
# Get sample images and preprocess them:

import os
import numpy as np

from tensorflow.io import read_file
from tensorflow.image import decode_jpeg

from akida_models.imagenet import preprocessing

# Model specification and hyperparameters
NUM_CHANNELS = 3
IMAGE_SIZE = 224

NUM_IMAGES = 10

# Retrieve dataset file from Brainchip data server
file_path = get_file(
    "imagenet_like.zip",
    "https://data.brainchip.com/dataset-mirror/imagenet_like/imagenet_like.zip",
    cache_subdir='datasets/imagenet_like',
    extract=True)
data_folder = os.path.dirname(file_path)

# Load images for test set
x_test_files = []
x_test = np.zeros((NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype('uint8')
for id in range(NUM_IMAGES):
    test_file = 'image_' + str(id + 1).zfill(2) + '.jpg'
    x_test_files.append(test_file)
    img_path = os.path.join(data_folder, test_file)
    base_image = read_file(img_path)
    image = decode_jpeg(base_image, channels=NUM_CHANNELS)
    image = preprocessing.preprocess_image(image, IMAGE_SIZE)
    x_test[id, :, :, :] = np.expand_dims(image, axis=0)

print(f'{NUM_IMAGES} images loaded and preprocessed.')


######################################################################
# Build and display the attention map for one selected sample:

import cv2
import matplotlib.pyplot as plt

from keras import Model
from quantizeml.layers import ClassToken, Attention
from quantizeml.tensors import FixedPoint
from quantizeml.models.transforms.transforms_utils import get_layers_by_type


def build_attention_map(model, image):
    # Get the Attention layers list
    attentions = get_layers_by_type(model, Attention)

    # Calculate the number of tokens and deduce the grid size
    num_tokens = sum(isinstance(ly, ClassToken) for ly in model.layers)
    grid_size = int(np.sqrt(attentions[0].output_shape[0][-2] - num_tokens))

    # Get the attention weights from each transformer
    outputs = [la.output[1] for la in attentions]
    weights = Model(inputs=model.inputs, outputs=outputs).predict(np.expand_dims(image, 0))

    # Converts to float if needed
    weights = [w.to_float() if isinstance(w, FixedPoint) else w for w in weights]
    weights = np.array(weights)

    # Heads number
    num_heads = weights.shape[2]
    num_layers = weights.shape[0]
    reshaped = weights.reshape((num_layers, num_heads, grid_size**2 + 1, grid_size**2 + 1))

    # Average the attention weights across all heads
    reshaped = reshaped.mean(axis=1)

    # To account for residual connections, we add an identity matrix to the attention matrix and
    # re-normalize the weights.
    reshaped = reshaped + np.eye(reshaped.shape[1])
    reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])

    # Attention from the output token to the input space
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = cv2.resize(mask / mask.max(), (image.shape[1], image.shape[0]))[..., np.newaxis]
    return (mask * image).astype("uint8")


# Using a specific image for which attention map is easier to observe
image = x_test[8]

# Compute the attention map
attention_float = build_attention_map(model_keras, image)
attention_quantized = build_attention_map(model_quantized, image)

# Display the attention map
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
ax1.axis('off')
ax1.set_title('Original')
ax1.imshow(image)

ax2.axis('off')
ax2.set_title('Float')
ax2.imshow(attention_float)

ax3.axis('off')
ax3.set_title('Quantized')
ax3.imshow(attention_quantized)
fig.suptitle('Attention masks', fontsize=10)
plt.show()
