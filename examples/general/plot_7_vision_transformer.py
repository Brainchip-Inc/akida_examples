"""
Vision transformers
===================

This tutorial demonstrates that Vision Transformers can be adapted and converted to Akida to perform
image classification.

Just like for the `AkidaNet example
<plot_1_akidanet_imagenet.html#sphx-glr-examples-general-plot-1-akidanet-imagenet-py>`__, ImageNet
images are not publicly available, performance is assessed using a set of 10 copyright free images
that were found on Google using ImageNet class names.

"""

######################################################################
# 1. Dataset preparation
# ~~~~~~~~~~~~~~~~~~~~~~
#
# See `AkidaNet example
# <plot_1_akidanet_imagenet.html#sphx-glr-examples-general-plot-1-akidanet-imagenet-py>`__ for
# details on dataset preparation.
#

import os
import csv
import numpy as np

from tensorflow.io import read_file
from tensorflow.image import decode_jpeg
from tensorflow.keras.utils import get_file

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

# Parse labels file
fname = os.path.join(data_folder, 'labels_validation.txt')
validation_labels = dict()
with open(fname, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        validation_labels[row[0]] = row[1]

# Get labels for the test set by index
labels_test = np.zeros(NUM_IMAGES)
for i in range(NUM_IMAGES):
    labels_test[i] = int(validation_labels[x_test_files[i]])

######################################################################
# 2. Create a transformer model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

######################################################################
# 2.1. Selecting an architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Vision Transformers is a hot-topic in AI and new architectures are being introduced regularly.
# When selecting an appropriate achitecture for Akida, some size, speed and training capabilities
# must be considered.
#
# The following table briefly shows what led to chose the ViT Tiny and DeiT-dist architectures:
#
# +--------------+-------------------+---------+-------------------+----------------------+
# | Architecture | Original accuracy | #Params | Architecture      | Commment             |
# +==============+===================+=========+===================+======================+
# | ViT Base     |  79.90%           |  86M    |  12 heads,        | base model but huge  |
# |              |                   |         |  12 blocks,       | amount of parameters |
# |              |                   |         |  hidden size 768  |                      |
# +--------------+-------------------+---------+-------------------+----------------------+
# | ViT Tiny     |  75.48%           |  5.8M   |  3 heads,         | edge compatible      |
# |              |                   |         |  12 blocks,       |                      |
# |              |                   |         |  hidden size 192  |                      |
# +--------------+-------------------+---------+-------------------+----------------------+
# | DeiT-dist    |  74.17%           |  5.8M   |  3 heads,         | easy to retrain      |
# | Tiny         |                   |         |  12 blocks,       | thanks to the        |
# |              |                   |         |  hidden size 192  | distilled token      |
# +--------------+-------------------+---------+-------------------+----------------------+
#
# The model zoo then comes with two vision transformers architectures:
#
#  - `BC ViT Ti16 <../../api_reference/akida_models_apis.html#akida_models.bc_vit_ti16>`__, which
#    is a modified version of `ViT TI16
#    <../../api_reference/akida_models_apis.html#akida_models.vit_ti16>`__ described in `the
#    original ViT paper <https://arxiv.org/abs/2010.11929>`__,
#  - `BC DeiT-dist Ti16 <../../api_reference/akida_models_apis.html#akida_models.bc_deit_ti16>`__,
#    which is a modified version of the original `DeiT TI16
#    <../../api_reference/akida_models_apis.html#akida_models.deit_ti16>`__ described in `the
#    original DeiT-dist paper <https://arxiv.org/abs/2012.12877>`__.
#
# .. note:: The Vision Transformers support has been introduced in Akida 2.0.

######################################################################
# 2.2. Model transformations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Both architectures have been modified so that their layers can be quantized to integer only
# operations. The detailed list of changes is:
#
#   - replace `LayerNormalization
#     <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization>`__ with
#     `LayerMadNormalization
#     <../../api_reference/quantizeml_apis.html#quantizeml.layers.LayerMadNormalization>`__ and
#     replace the last normalization previous to the classification head with a `BatchNormalization
#     <https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization>`__,
#   - replace `GeLU <https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GELU>`__
#     activations with `ReLU8 <https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU>`__,
#   - replace the `softmax
#     <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax>`__ operation in
#     `Attention <../../api_reference/quantizeml_apis.html#quantizeml.layers.Attention>`__ with a
#     `shiftmax <../../api_reference/quantizeml_apis.html#quantizeml.layers.shiftmax>`__ operation.
#
# .. note:: Details on the custom layers and operations are given in the API docstrings.
#
# Layer replacement is made possible through the ``akida_models create`` CLI that comes with
# dedicated options for the ``vit_ti16`` and ``deit_ti16`` architectures. See for example the helper
# for ViT:
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
# The replacement layers are functionaly equivalent to the base layers but an accuracy loss is
# introduced at each step. This is compensated by a tuning step after each change.
#
# For example, replacing activations layers can be done with:
#
# .. code-block:: bash
#
#   wget https://data.brainchip.com/models/AkidaV2/vit/vit_ti16_224.h5
#   akida_models create -s vit_ti16_relu.h5 vit_ti16 -bw vit_ti16_224.h5 --act ReLU8
#   imagenet_train tune -m vit_ti16_relu.h5 -e 15 --optim Adam --lr_policy cosine_decay \
#                       -lr 6e-5 -s vit_ti16_relu_tuned.h5
#
# After all changes, the model accuracy is close to (or better than) the original model and the
# "BC" transformer model is ready for quantization.
#
# +--------------+-------------------+---------------+
# | Architecture | Original accuracy | "BC" accuracy |
# +==============+===================+===============+
# | ViT          |  75.48%           | 74.25%        |
# +--------------+-------------------+---------------+
# | DeiT-dist    |  74.17%           | 75.03%        |
# +--------------+-------------------+---------------+
#
# .. note:: In the following sections, the ViT model will be used but the very same steps apply to
#           DeiT-dist.
#

######################################################################
# 2.3. Load a pre-trained native Keras model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from akida_models.model_io import load_model

# Retrieve the float model with pretrained weights and load it
model_file = get_file(
    "bc_vit_ti16_224.h5",
    "https://data.brainchip.com/models/AkidaV2/vit/bc_vit_ti16_224.h5",
    cache_subdir='models/akidanet_imagenet')
model_keras = load_model(model_file)
model_keras.summary()

######################################################################
# The perfomance given below uses the 10 ImageNet like images subset.


# Check model performance
def check_model_performance(model, x_test=x_test, labels_test=labels_test):
    outputs_keras = model.predict(x_test, batch_size=NUM_IMAGES)
    outputs_keras = np.squeeze(np.argmax(outputs_keras, 1))
    accuracy_keras = np.sum(np.equal(outputs_keras, labels_test)) / NUM_IMAGES
    print(f"Keras accuracy: {accuracy_keras*100:.2f} %")


######################################################################
check_model_performance(model_keras)

######################################################################
# 3. Quantization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

######################################################################
# 3.1. 8-bit PTQ
# ^^^^^^^^^^^^^^
# The above native Keras model is quantized to 8-bit (all weights and activations) and we compute
# the post-training quantization (PTQ) accuracy.
#

from akida_models import fetch_file

# Retrieve calibration samples
samples = fetch_file("https://data.brainchip.com/dataset-mirror/samples/imagenet/imagenet_batch1024_224.npz",
                     fname="imagenet_batch1024_224.npz")
samples = np.load(samples)
samples = np.concatenate([samples[item] for item in samples.files])

######################################################################

from quantizeml.models import quantize
from quantizeml.layers import QuantizationParams

# Quantize the model to 8-bit and calibrate using 1024 samples with a batch size of 100 over 2
# epochs.
model_quantized = quantize(model_keras,
                           qparams=QuantizationParams(weight_bits=8, activation_bits=8),
                           num_samples=1024, batch_size=100, epochs=2)

######################################################################
check_model_performance(model_quantized)

######################################################################
# 3.2. Load a pre-trained quantized Keras model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The `bc_vit_ti16_imagenet_pretrained helper
# <../../api_reference/akida_models_apis.html#akida_models.bc_vit_ti16_imagenet_pretrained>`__ was
# obtained with the same 8-bit quantization scheme but with an additional QAT step to further
# improve accuracy.
#

from akida_models import bc_vit_ti16_imagenet_pretrained

# Load the pre-trained quantized model
model_quantized = bc_vit_ti16_imagenet_pretrained()
model_quantized.summary()

######################################################################
check_model_performance(model_quantized)


######################################################################
# 4. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The quantized Keras model is now converted into an Akida model.

from cnn2snn import convert

# Convert the model
model_akida = convert(model_quantized)
model_akida.summary()

#######################################################################
accuracy_akida = model_akida.evaluate(x_test, labels_test)
print(f"Accuracy: {accuracy_akida*100:.2f} %")

# For non-regression purposes
assert accuracy_akida == 1

######################################################################
# 5.3 Attention maps
# ~~~~~~~~~~~~~~~~~~
#
# Instead of showing predictions, here we propose to show attention maps on an image. This is
# derived from `Abnar et al. attention rollout <https://arxiv.org/abs/2005.00928>`__ as shown in the
# following `Keras tutorial
# <https://keras.io/examples/vision/probing_vits/#method-ii-attention-rollout>`__. This aims to
# highlight the model abilities to focus on relevant parts in the input image.
#

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
