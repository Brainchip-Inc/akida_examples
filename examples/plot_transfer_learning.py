"""
Transfer learning with MobileNet for cats vs. dogs
==================================================

This tutorial presents a demonstration of transfer learning and the
conversion to an Akida model of a quantized Keras network.

The transfer learning example is derived from the `Tensorflow
tutorial <https://www.tensorflow.org/tutorials/images/transfer_learning>`__:

    * Our base model is an Akida-compatible version of **MobileNet v1**,
      trained on ImageNet.
    * The new dataset for transfer learning is **cats vs. dogs**
      (`link <https://www.tensorflow.org/datasets/catalog/cats_vs_dogs>`__).
    * We use transfer learning to customize the model to the new task of
      classifying cats and dogs.

.. Note:: This tutorial only shows the inference of the trained Keras
          model and its conversion to an Akida network. A textual explanation
          of the training is given below.

"""

######################################################################
# 1. Transfer learning process
# ----------------------------
# .. figure:: https://s2.qwant.com/thumbr/0x380/7/0/7b7386531ea24ab1294fdf9b8698b008a51e38a3c57e81427fbef626ff226c/1*6ACbDsBMeDZcLg9W8CFT_Q.png?u=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F1600%2F1%2A6ACbDsBMeDZcLg9W8CFT_Q.png&q=0&b=1&p=0&a=1
#    :alt: transfer_learning_image
#    :target: https://s2.qwant.com/thumbr/0x380/7/0/7b7386531ea24ab1294fdf9b8698b008a51e38a3c57e81427fbef626ff226c/1*6ACbDsBMeDZcLg9W8CFT_Q.png?u=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F1600%2F1%2A6ACbDsBMeDZcLg9W8CFT_Q.png&q=0&b=1&p=0&a=1
#    :align: center
#
# Transfer learning allows to classify on a specific task by using a
# pre-trained base model. For an introduction to transfer learning, please
# refer to the `Tensorflow
# tutorial <https://www.tensorflow.org/tutorials/images/transfer_learning>`__
# before exploring this tutorial. Here, we focus on how to quantize the
# Keras model in order to convert it to an Akida one.
#
# The model is composed of:
#
#   * a base quantized MobileNet model used to extract image features
#   * a top layer to classify cats and dogs
#   * a sigmoid activation function to interpret model outputs as a probability
#
# **Base model**
#
# The base model is an Akida-compatible version of MobileNet v1. This
# model was trained and quantized using the ImageNet dataset. Please refer
# to the corresponding `example <plot_mobilenet_imagenet.html>`__ for
# more information. The layers have 4-bit weights (except for the first
# layer having 8-bit weights) and the activations are quantized to 4 bits.
# This base model ends with a global average pooling whose output is (1,
# 1, 1024).
#
# In our transfer learning process, the base model is frozen, i.e., the
# weights are not updated during training. Pre-trained weights for the
# quantized model are provided on
# `<http://data.brainchip.com/models/mobilenet/>`__. These are
# loaded in our frozen base model.
#
# **Top layer**
#
# While the Tensorflow tutorial uses a fully-connected top layer with one
# output neuron, the only Akida layer supporting 4-bit weights is a separable
# convolutional layer (see `hardware compatibility
# <../user_guide/hw_constraints.html>`__).
#
# We thus decided to use a separable convolutional layer with one output
# neuron for the top layer of our model.
#
# **Final activation**
#
# ReLU6 is the only activation function that can be converted into an Akida SNN
# equivalent. The converted Akida model doesn't therefore include the 'sigmoid'
# activation, and we must instead apply it explicitly on the raw values returned
# by the model Top layer.
#
# **Training steps**
#
# The transfer learning process consists in two training phases:
#
#   1. **Float top layer training**: The base model is quantized using 4-bit
#      weights and activations. Pre-trained 4-bit weights of MobileNet/ImageNet
#      are loaded. Then a top layer is added with float weights. The base model
#      is frozen and the training is only applied on the top layer. After 10
#      epochs, the weights are saved. Note that the weights of the layers of
#      the frozen base model haven't changed; only those of the top layer are
#      updated.
#   2. **4-bit top layer training**: The base model is still
#      quantized using 4-bit weights and activations. The added top layer is
#      now quantized (4-bit weights). The weights saved at step 1 are used as
#      initialization. The base model is frozen and the training is only
#      applied on the top layer. After 10 epochs, the new quantized weights are
#      saved. This final weights are those used in the inference below.
#
# +----------+-------------------+------------+---------+------------+----+
# | Training | Frozen base model | Init.      | Top     | Init.      | E  |
# | step     |                   | weights    | layer   | weights    | p  |
# |          |                   | base model |         | top layer  | o  |
# |          |                   |            |         |            | c  |
# |          |                   |            |         |            | h  |
# |          |                   |            |         |            | s  |
# +==========+===================+============+=========+============+====+
# | step 1   | 4-bit weights /   | pre-trained| float   | random     | 10 |
# |          | activations       | 4-bit      | weights |            |    |
# |          |                   |            |         |            |    |
# +----------+-------------------+------------+---------+------------+----+
# | step 2   | 4-bit weights /   | pre-trained| 4-bit   | saved from | 10 |
# |          | activations       | 4-bit      | weights | step 1     |    |
# |          |                   |            |         |            |    |
# +----------+-------------------+------------+---------+------------+----+

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from akida_models import mobilenet_imagenet
from cnn2snn import convert
from akida_models.quantization_blocks import separable_conv_block

######################################################################
# 2. Load and preprocess data
# ---------------------------
#
# In this section, we will load the 'cats_vs_dogs' dataset preprocess
# the data to match the required model's inputs:
#
#   * **2.A - Load and split data**: we only keep the test set which represents
#     10% of the dataset.
#   * **2.B - Preprocess the test set** by resizing and rescaling the images.
#   * **2.C - Get labels**

######################################################################
# 2.A - Load and split data
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``cats_vs_dogs``
# `dataset <https://www.tensorflow.org/datasets/catalog/cats_vs_dogs>`__
# is loaded and split into train, validation and test sets. The train and
# validation sets were used for the transfer learning process. Here only
# the test set is used. We use here ``tf.Dataset`` objects to load and
# preprocess batches of data (one can look at the TensorFlow guide
# `here <https://www.tensorflow.org/guide/data>`__ for more information).
#
# .. Note:: The ``cats_vs_dogs`` dataset version used here is 2.0.1.
#

splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']

tfds.disable_progress_bar()
(raw_train, raw_validation,
 raw_test), metadata = tfds.load('cats_vs_dogs:2.0.1',
                                 split=splits,
                                 with_info=True,
                                 as_supervised=True)

######################################################################
# 2.B - Preprocess the test set
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We must apply the same preprocessing as for training: rescaling and
# resizing. Since Akida models directly accept integer-valued images, we
# also define a preprocessing function for Akida:
#
#   - for Keras: images are rescaled between 0 and 1, and resized to 160x160
#   - for Akida: images are only resized to 160x160 (uint8 values).
#
# Keras and Akida models require 4-dimensional (N,H,W,C) arrays as inputs.
# We must then create batches of images to feed the model. For inference,
# the batch size is not relevant; you can set it such that the batch of
# images can be loaded in memory depending on your CPU/GPU.

IMG_SIZE = 160
input_scaling = (127.5, 127.5)


def format_example_keras(image, label):
    image = tf.cast(image, tf.float32)
    image = (image - input_scaling[1]) / input_scaling[0]
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def format_example_akida(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.uint8)
    return image, label


######################################################################

BATCH_SIZE = 32
test_batches_keras = raw_test.map(format_example_keras).batch(BATCH_SIZE)
test_batches_akida = raw_test.map(format_example_akida).batch(BATCH_SIZE)

######################################################################
# 2.C - Get labels
# ~~~~~~~~~~~~~~~~
#
# Labels are contained in the test set as '0' for cats and '1' for dogs.
# We read through the batches to extract the labels.

labels = np.array([])
for _, label_batch in test_batches_keras:
    labels = np.concatenate((labels, label_batch))

get_label_name = metadata.features['label'].int2str
num_images = labels.shape[0]

print(f"Test set composed of {num_images} images: "
      f"{np.count_nonzero(labels==0)} cats and "
      f"{np.count_nonzero(labels==1)} dogs.")

######################################################################
# 3. Convert a quantized Keras model to Akida
# -------------------------------------------
#
# In this section, we will instantiate a quantized Keras model based on
# MobileNet and modify the last layers to specify the classification for
# ``cats_vs_dogs``. After loading the pre-trained weights, we will convert
# the Keras model to Akida.
#
# This section goes as follows:
#
#   * **3.A - Instantiate a Keras base model**
#   * **3.B - Modify the network and load pre-trained weights**
#   * **3.C - Convert to Akida**

######################################################################
# 3.A - Instantiate a Keras base model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here, we instantiate a quantized Keras model based on a MobileNet model.
# This base model was previously trained using the 1000 classes of the
# ImageNet dataset. For more information, please see the `ImageNet
# tutorial <plot_mobilenet_imagenet.html>`__.
#
# The quantized MobileNet model satisfies the Akida NSoC requirements:
#
#   * The model relies on a convolutional layer (first layer) and separable
#     convolutional layers, all being Akida-compatible.
#   * All the separable convolutional layers have 4-bit weights, the first
#     convolutional layer has 8-bit weights.
#   * The activations are quantized with 4 bits.
#
# Using the provided quantized MobileNet model, we create an instance
# without the top classification layer ('include_top=False').

base_model_keras = mobilenet_imagenet(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                      include_top=False,
                                      pooling='avg',
                                      weight_quantization=4,
                                      activ_quantization=4,
                                      input_weight_quantization=8)

######################################################################
# 3.B - Modify the network and load pre-trained weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As explained in `section 1 <plot_cats_vs_dogs_cnn2akida_demo.html#transfer-learning-process>`__,
# we add a separable convolutional layer as top layer with one output neuron.
# The new model is now appropriate for the ``cats_vs_dogs`` dataset and is
# Akida-compatible. Note that a sigmoid activation is added at the end of
# the model: the output neuron returns a probability between 0 and 1 that
# the input image is a dog.
#
# The transfer learning process has been run internally and the weights have
# been saved. In this tutorial, the pre-trained weights are loaded for inference
# and conversion.
#
# .. Note:: The pre-trained weights which are loaded corresponds to the
#           quantization parameters described as above. If you want to modify
#           these parameters, you must re-train the model and save weights.
#

# Add a top layer for classification
x = base_model_keras.output
x = tf.keras.layers.Reshape((1, 1, 1024), name='reshape_1')(x)
x = separable_conv_block(x,
                         filters=1,
                         kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         name='top_layer_separable',
                         weight_quantization=4,
                         activ_quantization=None)
x = tf.keras.layers.Activation('sigmoid')(x)
preds = tf.keras.layers.Reshape((1,), name='reshape_2')(x)
model_keras = tf.keras.Model(inputs=base_model_keras.input,
                             outputs=preds,
                             name="model_cats_vs_dogs")

model_keras.summary()

######################################################################

# Load pre-trained weights
pretrained_weights = tf.keras.utils.get_file(
    "mobilenet_cats_vs_dogs_wq4_aq4.h5",
    "http://data.brainchip.com/models/mobilenet/mobilenet_cats_vs_dogs_wq4_aq4.h5",
    cache_subdir='models/mobilenet')
model_keras.load_weights(pretrained_weights)

######################################################################
# 3.C - Convert to Akida
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The new Keras model with pre-trained weights is now converted to an
# Akida model. It only requires the quantized Keras model and the inputs
# scaling used during training.
# Note: the 'sigmoid' activation has no SNN equivalent and will be simply
# ignored during the conversion.

model_akida = convert(model_keras, input_scaling=input_scaling)

model_akida.summary()

######################################################################
# 4. Classify test images
# -----------------------
#
# This section gives a comparison of the results between the quantized
# Keras and the Akida models. It goes as follows:
#
#   * **4.A - Classify test images** with the quantized Keras and the Akida
#     models
#   * **4.B - Compare results**

######################################################################
# 4.A Classify test images
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here, we will predict the classes of the test images using the quantized
# Keras model and the converted Akida model. Remember that:
#
#   * Input images in Keras and Akida are not scaled in the same range, be
#     careful to use the correct inputs: uint8 images for Akida and float
#     rescaled images for Keras.
#   * The ``predict`` function of tf.keras can take a ``tf.data.Dataset``
#     object as argument. However, the Akida `evaluate <../api_reference/aee_apis.html#akida.Model.evaluate>`__
#     function takes a NumPy array containing the images. Though the Akida
#     `predict <../api_reference/aee_apis.html#akida.Model.predict>`__
#     function exists, it outputs a class label and not the raw predictions.
#   * The Keras ``predict`` function returns the probability to be a dog:
#     if the output is greater than 0.5, the model predicts a 'dog'. However,
#     the Akida `evaluate <../api_reference/aee_apis.html#akida.Model.evaluate>`__
#     function directly returns the potential before the 'sigmoid' activation, which has
#     no SNN equivalent. We must therefore apply it explicitly on the model outputs to obtain
#     the Akida probabilities.

# Classify test images with the quantized Keras model
from timeit import default_timer as timer

start = timer()
pots_keras = model_keras.predict(test_batches_keras)
end = timer()

preds_keras = pots_keras.squeeze() > 0.5
print(f"Keras inference on {num_images} images took {end-start:.2f} s.\n")

######################################################################

# Classify test images with the Akida model
from progressbar import ProgressBar
n_batches = num_images // BATCH_SIZE + 1
pbar = ProgressBar(maxval=n_batches)
i = 1
pbar.start()
start = timer()
pots_akida = np.array([], dtype=np.float32)
for batch, _ in test_batches_akida:
    pots_batch_akida = model_akida.evaluate(batch.numpy())
    pots_akida = np.concatenate((pots_akida, pots_batch_akida.squeeze()))
    pbar.update(i)
    i = i + 1
pbar.finish()
end = timer()

preds_akida = tf.keras.layers.Activation('sigmoid')(pots_akida) > 0.5
print(f"Akida inference on {num_images} images took {end-start:.2f} s.\n")

######################################################################

# Print model statistics
print("Model statistics")
stats = model_akida.get_statistics()
batch, _ = iter(test_batches_akida).get_next()
model_akida.evaluate(batch[:20].numpy())
for _, stat in stats.items():
    print(stat)

######################################################################
# 4.B Compare results
# ~~~~~~~~~~~~~~~~~~~
#
# The Keras and Akida accuracies are compared and the Akida confusion
# matrix is given (the quantized Keras confusion matrix is almost
# identical to the Akida one). Note that there is no exact equivalence
# between the quantized Keras and the Akida models. However, the
# accuracies are highly similar.

# Compute accuracies
n_good_preds_keras = np.sum(np.equal(preds_keras, labels))
n_good_preds_akida = np.sum(np.equal(preds_akida, labels))

keras_accuracy = n_good_preds_keras / num_images
akida_accuracy = n_good_preds_akida / num_images

print(f"Quantized Keras accuracy: {keras_accuracy*100:.2f} %  "
      f"({n_good_preds_keras} / {num_images} images)")
print(f"Akida accuracy:           {akida_accuracy*100:.2f} %  "
      f"({n_good_preds_akida} / {num_images} images)")

# For non-regression purpose
assert akida_accuracy > 0.97

######################################################################


def confusion_matrix_2classes(labels, predictions):
    tp = np.count_nonzero(labels + predictions == 2)
    tn = np.count_nonzero(labels + predictions == 0)
    fp = np.count_nonzero(predictions - labels == 1)
    fn = np.count_nonzero(labels - predictions == 1)

    return np.array([[tp, fn], [fp, tn]])


def plot_confusion_matrix_2classes(cm, classes):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)

    for i, j in zip([0, 0, 1, 1], [0, 1, 0, 1]):
        plt.text(j,
                 i,
                 f"{cm[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.autoscale()


######################################################################

# Plot confusion matrix for Akida
cm_akida = confusion_matrix_2classes(labels, preds_akida.numpy())
print("Confusion matrix quantized Akida:")
plot_confusion_matrix_2classes(cm_akida, ['dog', 'cat'])
plt.show()
