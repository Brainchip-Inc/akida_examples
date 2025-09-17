"""
Transfer learning with AkidaNet for PlantVillage
================================================

This tutorial presents how to perform transfer learning for quantized models targeting an Akida
accelerator.

The transfer learning example is derived from the `Tensorflow tutorial
<https://www.tensorflow.org/tutorials/images/transfer_learning>`__ where the
base model is an AkidaNet 0.5 quantized model trained on ImageNet and the
target dataset is `PlantVillage <https://www.tensorflow.org/datasets/catalog/plant_village>`__.
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
# The base model is an AkidaNet 0.5 that was trained on the
# ImageNet dataset. Please refer to the `dedicated example
# <./plot_1_akidanet_imagenet.html>`__ for more information on the model
# architecture and performance.
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
#   1. Get a trained float AkidaNet base model
#   2. Add a classification head to the model
#   3. Tune the whole model for a few epochs
#   4. Quantize the whole model
#   5. Optionally perform QAT for a few epochs to recover accuracy
#
# The transfer learning process operates in float precision,
# ensuring seamless integration with users' existing familiarity in setting
# hyperparameters, adding a new head, and deciding on layer freezing.
#
# While this process will apply to most of the tasks, there might be cases where
# variants are needed:
#
#   - Quantization in the 4th step might lead to drop in accuracy (especially for 4
#     bits quantization). In such a case, an additional step of fine tuning is needed
#     and consists in training for a few additional epochs with a lower learning rate
#     (e.g 10 to 100 times lower than the initial rate).

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
# `akidanet_imagenet <../../api_reference/akida_models_apis.html#akida_models.akidanet_imagenet>`_.

from akida_models import fetch_file, akidanet_imagenet

# Create a base model without top layers
base_model = akidanet_imagenet(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                               classes=CLASSES,
                               alpha=0.5,
                               include_top=False,
                               pooling='avg')

# Get pretrained quantized weights and load them into the base model
pretrained_weights = fetch_file(
    origin="https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_224_alpha_0.5.h5",
    fname="akidanet_imagenet_224_alpha_0.5.h5",
    cache_subdir='models')

base_model.load_weights(pretrained_weights, by_name=True)
base_model.summary()

######################################################################
# 3. Add a classification head to the model
# -----------------------------------------
#
# As explained in `section 1 <#transfer-learning-process>`__, the classification
# head is defined as a dense layer with batch normalization and activation,
# which correspond to a `dense_block
# <../../api_reference/akida_models_apis.html#akida_models.layer_blocks.dense_block>`__, followed by
# a dropout layer and a second dense layer.

from tf_keras import Model
from tf_keras.layers import Activation, Dropout, Reshape
from akida_models.layer_blocks import dense_block

x = base_model.output
x = dense_block(x,
                units=512,
                name='fc1',
                add_batchnorm=True,
                relu_activation='ReLU7.5')
x = Dropout(0.5, name='dropout_1')(x)
x = dense_block(x,
                units=CLASSES,
                name='predictions',
                add_batchnorm=False,
                relu_activation=False)
x = Activation('softmax', name='act_softmax')(x)
x = Reshape((CLASSES,), name='reshape')(x)

# Build the model
model_keras = Model(base_model.input, x, name='akidanet_plantvillage')

model_keras.summary()

######################################################################
# 4. Train for a few epochs
# -------------------------
#
# Only giving textual information for training in this tutorial:
#
#   - the model is compiled with an Adam optimizer and the sparse categorical
#     crossentropy loss is used,
#   - the initial learning rate is set to 1e-3 and ends at 1e-5 with an exponential decay,
#   - the training lasts for 10 epochs.

######################################################################
# 5. Quantize the model
# ---------------------
#
# Quantization is done using QuantizeML `quantize
# <../../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__.
#
# In order to get the best possible model, calibration samples should be provided to the model.
# Using here samples from the train set.

from quantizeml.models import quantize, QuantizationParams

train_batches = train_ds.map(format_example).batch(BATCH_SIZE)

# Prepare a quantization scheme: first layer weights to 8-bit, other weights and activation to 4-bit
qparams = QuantizationParams(input_weight_bits=8, weight_bits=4, activation_bits=4)

# Quantize the model, using the 1024 calibration samples from the train set and calibrate over 2
# epochs with a batch_size of 100.
model_quantized = quantize(model_keras, qparams=qparams,
                           samples=train_batches, epochs=2, batch_size=BATCH_SIZE, num_samples=1024)

######################################################################
# To recover the loss of accuracy introduced with 4-bit quantization, an extra QAT step with a lower
# learning rate (training rate divided by 10) is required. Note that you could also aim for 8-bit
# quantization and not require this extra QAT step.

######################################################################
# 6. Compute accuracy
# -------------------
#
# Because training is not included in this tutorial, the pretrained TF-Keras model
# is retrieved from the zoo.

from akida_models import akidanet_plantvillage_pretrained

model = akidanet_plantvillage_pretrained()

# Evaluate TF-Keras accuracy
model.compile(metrics=['accuracy'])
history = model.evaluate(test_batches, verbose=0)
print('TF-Keras accuracy: ', history[1])

######################################################################
# Convert the model and evaluate the Akida model.

import numpy as np
from cnn2snn import convert

model_akida = convert(model)

# Manual evaluation loop to retrieve activations and labels
labels, logits = None, None
for batch, label_batch in test_batches:
    logits_batch = model_akida.predict(batch.numpy().astype('uint8'))

    if labels is None:
        labels = label_batch
        logits = logits_batch.squeeze(axis=(1, 2))
    else:
        labels = np.concatenate((labels, label_batch))
        logits = np.concatenate((logits, logits_batch.squeeze(axis=(1, 2))))
preds = Activation("softmax")(logits)
accuracy = (np.argmax(preds, 1) == labels).mean()

print(f"Akida accuracy: {accuracy}")
