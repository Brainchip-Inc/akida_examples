"""
Transfer learning with AkidaNet for PlantVillage
================================================

This tutorial presents how to perform transfer learning for quantized models
targetting Akida NSoC.

The transfer learning example is derived from the `Tensorflow tutorial
<https://www.tensorflow.org/tutorials/images/transfer_learning>`__ where the
base  model is an AkidaNet 0.5 quantized model trained on ImageNet and the
target dataset is `PlantVillage
<https://www.tensorflow.org/datasets/catalog/plant_village>`__.
"""

######################################################################
# Transfer learning process
# -------------------------
#
# Transfer learning consists in customizing a pretrained model or feature
# extractor to fit another task.
#
# **Base model**
#
# The base model is a quantized version of AkidaNet 0.5 that was trained on the
# ImageNet dataset. Please refer to the `dedicated example
# <plot_2_akidanet_imagenet.html>`__ for more information on the model
# architecture and performances.
#
# Layers of this model have 4-bit weights (except for the first layer having
# 8-bit weights) and activations are quantized to 4 bits.
#
# **Classification head**
#
# Customization of the model happens by adding layers on top of the base model,
# which in AkidaNet case ends with a global average operation.
#
# The classification head is typically composed of two dense layers as follows:
#
#   - the first dense layer number of units is configurable and depends on the
#     task but is generally 512 or below,
#   - a BatchNormalization operation and ReLU activation follow the first layer,
#   - a dropout layer is placed between the two dense layers to prevent
#     overfitting,
#   - the second dense layer is the prediction layer and should have its units
#     value set to the number of classes to predict,
#   - a softmax activation ends the model.
#
# **Training process**
#
# The standard training process for transfer learning for AkidaNet is:
#
#   1. Get a trained AkidaNet base model
#   2. Add a float classification head to the model
#   3. Freeze the base model
#   4. Train for a few epochs
#   5. Quantize the classification head
#
# While this process will apply to most of the tasks, there might be cases where
# variants are needed:
#
#   - for some target datasets, freezing the base model will not produce the
#     best accuracy. In such a case, the base model with stay trainable and the
#     learning rate when tuning the model should be small enough to preserve
#     features learned by the features extractor.
#   - quantization in the 5th step might lead to drop in accuracy. In such a
#     case, an additional step of fine tuning is needed and consists in training
#     for a few additional epochs with a lower learning rate (e.g 10 to 100
#     times lower than the initial rate) and with the base model unfrozen.

######################################################################
# 1. Dataset preparation
# ----------------------
#

import tensorflow as tf
import tensorflow_datasets as tfds

# Define task specific variables
IMG_SIZE = 224
BATCH_SIZE = 32
CLASSES = 38

# Load the tensorflow dataset
(train_ds, validation_ds, test_ds), ds_info = tfds.load(
    'plant_village',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True)

# Visualize some data
_ = tfds.show_examples(test_ds, ds_info)

######################################################################


# Format test data
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


test_batches = test_ds.map(format_example).batch(BATCH_SIZE)

######################################################################
# 2. Get a trained AkidaNet base model
# ------------------------------------
#
# The AkidaNet architecture is available in the Akida model zoo as
# `akidanet_imagenet
# <../../api_reference/akida_models_apis.html#akida_models.akidanet_imagenet>`_.

from akida_models import akidanet_imagenet
from keras.utils.data_utils import get_file

# Create a quantized base model without top layers
base_model = akidanet_imagenet(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                               classes=CLASSES,
                               alpha=0.5,
                               include_top=False,
                               pooling='avg',
                               weight_quantization=4,
                               activ_quantization=4,
                               input_weight_quantization=8)

# Get pretrained quantized weights and load them into the base model
pretrained_weights = get_file(
    "akidanet_imagenet_224_alpha_50_iq8_wq4_aq4.h5",
    "http://data.brainchip.com/models/akidanet/akidanet_imagenet_224_alpha_50_iq8_wq4_aq4.h5",
    cache_subdir='models')

base_model.load_weights(pretrained_weights, by_name=True)
base_model.summary()

######################################################################
# 3. Add a float classification head to the model
# -----------------------------------------------
#
# As explained in `section 1 <#transfer-learning-process>`__, the classification
# head is defined as a dense layer with batch normalization and activation,
# which correspond to a `dense_block
# <../../api_reference/akida_models_apis.html#dense-block>`__, followed by a
# dropout layer and a second dense layer.

from keras import Model
from keras.layers import Activation, Dropout, Reshape, Flatten
from akida_models.layer_blocks import dense_block

x = base_model.output
x = Flatten(name='flatten')(x)
x = dense_block(x,
                units=512,
                name='fc1',
                add_batchnorm=True,
                add_activation=True)
x = Dropout(0.5, name='dropout_1')(x)
x = dense_block(x,
                units=CLASSES,
                name='predictions',
                add_batchnorm=False,
                add_activation=False)
x = Activation('softmax', name='act_softmax')(x)
x = Reshape((CLASSES,), name='reshape')(x)

# Build the model
model_keras = Model(base_model.input, x, name='akidanet_plantvillage')

model_keras.summary()

######################################################################
# 4. Freeze the base model
# ------------------------
#
# Freezing can be done by setting the `trainable` attribute of a layer to False.
# For convenience, a `freeze_model_before
# <../../api_reference/akida_models_apis.html#akida_models.training.freeze_model_before>`__
# API is provided in akida_models. It allows to freeze all layers before the
# classification head.

from akida_models.training import freeze_model_before

freeze_model_before(model_keras, 'flatten')

######################################################################
# 5. Train for a few epochs
# -------------------------
#
# Only giving textual information for training in this tutorial:
#
#   - the model is compiled with an Adam optimizer and the sparse categorical
#     crossentropy loss is used,
#   - the initial learning rate is set to 1e-2 and ends at 1e-4 with a linear
#     decay,
#   - the training lasts for 10 epochs.

######################################################################
# 6. Quantize the classification head
# -----------------------------------
#
# Quantization is done using `cnn2snn.quantize
# <../../api_reference/cnn2snn_apis.html#quantize>`__.

from cnn2snn import quantize

# Quantize weights and activation to 4 bits, first layer weights to 8 bits
model_quantized = quantize(model_keras, 4, 4, 8)

######################################################################
# After this step the model is trained and ready for use after for conversion to
# Akida (no extra fine tuning step needed for this task).

######################################################################
# 7. Compute accuracy
# -------------------
#
# Because training is not included in this tutorial, the pretrained Keras model
# is retrieved from the zoo.

from akida_models import akidanet_plantvillage_pretrained
from akida_models.training import evaluate_model

model = akidanet_plantvillage_pretrained()

# Evaluate Keras accuracy
model.compile(metrics=['accuracy'])
evaluate_model(model, test_batches)

######################################################################
# Convert the model and evaluate the Akida model.

import numpy as np
from cnn2snn import convert
from akida_models.training import evaluate_akida_model

model_akida = convert(model)

preds, labels = evaluate_akida_model(model_akida, test_batches, 'softmax')
accuracy = (np.squeeze(np.argmax(preds, 1)) == labels).mean()

print(f"Akida accuracy: {accuracy}")
