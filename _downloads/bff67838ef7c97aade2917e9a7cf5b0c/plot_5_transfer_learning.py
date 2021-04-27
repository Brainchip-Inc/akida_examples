"""
Transfer learning with MobileNet for cats vs. dogs
==================================================

This tutorial presents a demonstration of how transfer learning is applied
with our quantized models to get an Akida model.

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
# Transfer learning process
# ----------------------------
# .. figure:: https://s2.qwant.com/thumbr/0x380/7/0/7b7386531ea24ab1294fdf9b8698b008a51e38a3c57e81427fbef626ff226c/1*6ACbDsBMeDZcLg9W8CFT_Q.png?u=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F1600%2F1%2A6ACbDsBMeDZcLg9W8CFT_Q.png&q=0&b=1&p=0&a=1
#    :alt: transfer_learning_image
#    :target: https://s2.qwant.com/thumbr/0x380/7/0/7b7386531ea24ab1294fdf9b8698b008a51e38a3c57e81427fbef626ff226c/1*6ACbDsBMeDZcLg9W8CFT_Q.png?u=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F1600%2F1%2A6ACbDsBMeDZcLg9W8CFT_Q.png&q=0&b=1&p=0&a=1
#    :align: center
#
# Transfer learning allows to classify on a specific task by using a
# pre-trained base model. For an introduction to transfer learning, please
# refer to the `Tensorflow transfer learning
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
# The base model is a quantized version of MobileNet v1. This
# model was trained and quantized using the ImageNet dataset. Please refer
# to the corresponding `example <plot_2_mobilenet_imagenet.html>`__ for
# more information. The layers have 4-bit weights (except for the first
# layer having 8-bit weights) and the activations are quantized to 4 bits.
# This base model ends with a classification layer for 1000 classes. To
# classify cats and dogs, the feature extractor is preserved but the
# classification layer must be removed to be replaced by a new top layer
# focusing on the new task.
#
# In our transfer learning process, the base model is frozen, i.e., the
# weights are not updated during training. Pre-trained weights for the
# frozen quantized model are provided on our
# `data server <http://data.brainchip.com/models/mobilenet/>`__.
#
# **Top layer**
#
# While a fully-connected top layer is added in the Tensorflow tutorial, we
# decided to use a separable convolutional layer with one output neuron for the
# top layer of our model. The reason is that the separable convolutional layer
# is the only Akida layer supporting 4-bit weights (see `hardware compatibility
# <../../user_guide/hw_constraints.html>`__).
#
# **Training process**
#
# The transfer learning process for quantized models can be handled in different
# ways:
#
#   1. **From a quantized base model**, the new transferred model is composed
#      of a frozen base model and a float top layer. The top layer is trained.
#      Then, the top layer is quantized and fine-tuned. If necessary, the base
#      model can be unfrozen to be slightly trained to improve accuracy.
#   2. **From a float base model**, the new transferred model is also composed
#      of a frozen base model (with float weights/activations) and a float top
#      layer. The top layer is trained. Then the full model is quantized,
#      unfrozen and fine-tuned. This option requires longer training
#      operations since we don't take advantage of an already quantized base
#      model. Option 2 can be used alternatively if option 1 doesn't give
#      suitable performance.
#
# In this example, option 1 is chosen. The training steps are described below.

######################################################################
# 1. Load and preprocess data
# ---------------------------
#
# In this section, we will load and preprocess the 'cats_vs_dogs' dataset
# to match the required model's inputs.

######################################################################
# 1.A - Load and split data
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
# .. Note:: The ``cats_vs_dogs`` dataset version used here is 4.0.0.
#

import tensorflow_datasets as tfds

splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']

tfds.disable_progress_bar()
(raw_train, raw_validation,
 raw_test), metadata = tfds.load('cats_vs_dogs:4.0.0',
                                 split=splits,
                                 with_info=True,
                                 as_supervised=True)

######################################################################
# 1.B - Preprocess the test set
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

import tensorflow as tf

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
# 1.C - Get labels
# ~~~~~~~~~~~~~~~~
#
# Labels are contained in the test set as '0' for cats and '1' for dogs.
# We read through the batches to extract the labels.

import numpy as np

labels = np.array([])
for _, label_batch in test_batches_keras:
    labels = np.concatenate((labels, label_batch))

num_images = labels.shape[0]

print(f"Test set composed of {num_images} images: "
      f"{np.count_nonzero(labels==0)} cats and "
      f"{np.count_nonzero(labels==1)} dogs.")

######################################################################
# 2. Modify a pre-trained base Keras model
# -------------------------------------------
#
# In this section, we will describe how to modify a base model to specify
# the classification for ``cats_vs_dogs``.

######################################################################
# 2.A - Instantiate a Keras base model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here, we instantiate a quantized Keras model based on a MobileNet model.
# This base model was previously trained using the 1000 classes of the
# ImageNet dataset. For more information, please see the `ImageNet
# tutorial <plot_2_mobilenet_imagenet.html>`__.
#
# The quantized MobileNet model satisfies the Akida NSoC requirements:
#
#   * The model relies on a convolutional layer (first layer) and separable
#     convolutional layers, all being Akida-compatible.
#   * All the separable convolutional layers have 4-bit weights, the first
#     convolutional layer has 8-bit weights.
#   * The activations are quantized with 4 bits.

from akida_models import mobilenet_imagenet

# Instantiate a quantized MobileNet model
base_model_keras = mobilenet_imagenet(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                      weight_quantization=4,
                                      activ_quantization=4,
                                      input_weight_quantization=8)

# Load pre-trained weights for the base model
pretrained_weights = tf.keras.utils.get_file(
    "mobilenet_imagenet_iq8_wq4_aq4.h5",
    "http://data.brainchip.com/models/mobilenet/mobilenet_imagenet_224_iq8_wq4_aq4.h5",
    file_hash="d9eabb514a7db6d823ab108b0fbc64fe2872ad1113bd6c04c9a3329b6a41e135",
    cache_subdir='models/mobilenet')
base_model_keras.load_weights(pretrained_weights)

base_model_keras.summary()

######################################################################
# 2.B - Modify the network for the new task
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As explained in `section 1 <#transfer-learning-process>`__,
# we replace the 1000-class top layer with a separable convolutional layer with
# one output neuron.
# The new model is now appropriate for the ``cats_vs_dogs`` dataset and is
# Akida-compatible. Note that a sigmoid activation is added at the end of
# the model: the output neuron returns a probability between 0 and 1 that
# the input image is a dog.

from akida_models.layer_blocks import separable_conv_block

# Add a top layer for "cats_vs_dogs" classification
x = base_model_keras.get_layer('reshape_1').output
x = separable_conv_block(x,
                         filters=1,
                         kernel_size=(3, 3),
                         padding='same',
                         use_bias=False,
                         add_activation=False,
                         name='top_layer_separable')
x = tf.keras.layers.Activation('sigmoid')(x)
preds = tf.keras.layers.Reshape((1,), name='reshape_2')(x)
model_keras = tf.keras.Model(inputs=base_model_keras.input,
                             outputs=preds,
                             name="model_cats_vs_dogs")

model_keras.summary()

######################################################################
# 3. Train the transferred model for the new task
# -----------------------------------------------
#
# The transferred model must be trained to learn how to classify cats and dogs.
# The quantized base model is frozen: only the float top layer will effectively
# be trained. One can take a look at the
# `training section <https://www.tensorflow.org/tutorials/images/transfer_learning#compile_the_model>`__
# of the corresponding TensorFlow tutorial to reproduce the training stage.
#
# The float top layer is trained for 20 epochs. We don't illustrate the training
# phase in this tutorial; instead we directly load the pre-trained weights
# obtained after the 20 epochs.

# Freeze the base model part of the new model
base_model_keras.trainable = False

# Load pre-trained weights
pretrained_weights = tf.keras.utils.get_file(
    "mobilenet_cats_vs_dogs_iq8_wq4_aq4.h5",
    "http://data.brainchip.com/models/mobilenet/mobilenet_cats_vs_dogs_iq8_wq4_aq4.h5",
    file_hash="b021fccaba676de9549430336ff27875a85b2aea7ca767a5e70a76185362fa4b",
    cache_subdir='models')
model_keras.load_weights(pretrained_weights)

######################################################################

# Check performance on the test set
model_keras.compile(metrics=['accuracy'])
_, keras_accuracy = model_keras.evaluate(test_batches_keras)

print(f"Keras accuracy (float top layer): {keras_accuracy*100:.2f} %")

######################################################################
# 4 Quantize the top layer
# ------------------------
#
# To get an Akida-compatible model, the float top layer must be quantized.
# We decide to quantize its weights to 4 bits. The performance of the
# new quantized model is then assessed.
#
# Here, the quantized model gives suitable performance compared to the model
# with the float top layer. If that had not been the case, a fine-tuning step
# would have been necessary to recover the drop in accuracy.

from cnn2snn import quantize_layer

# Quantize the top layer to 4 bits
model_keras = quantize_layer(model_keras, 'top_layer_separable', bitwidth=4)

# Check performance for the quantized Keras model
model_keras.compile(metrics=['accuracy'])
_, keras_accuracy = model_keras.evaluate(test_batches_keras)

print(f"Quantized Keras accuracy: {keras_accuracy*100:.2f} %")

######################################################################
# 5. Convert to Akida
# -------------------
#
# The new quantized Keras model is now converted to an Akida model. The
# 'sigmoid' final activation has no SNN equivalent and will be simply ignored
# during the conversion.
#
# Performance of the Akida model is then computed. Compared to Keras inference,
# remember that:
#
#   * Input images in Akida are uint8 and not scaled like Keras inputs. But
#     remember that the conversion process needs to know what scaling was
#     applied during Keras training, in order to compensate (see
#     `CNN2SNN guide <../../user_guide/cnn2snn.html#input-scaling>`__)
#   * The Akida `evaluate <../../api_reference/aee_apis.html#akida.Model.evaluate>`__
#     function takes a NumPy array containing the images and returns potentials
#     before the sigmoid activation. We must therefore explicitly apply the
#     'sigmoid' activation on the model outputs to obtain the Akida
#     probabilities.
#
# Since activations sparsity has a great impact on Akida inference time, we
# also have a look at the average input and output sparsity of each layer on
# one batch of the test set.

from cnn2snn import convert

# Convert the model
model_akida = convert(model_keras, input_scaling=input_scaling)
model_akida.summary()

######################################################################

from timeit import default_timer as timer
from progressbar import ProgressBar

# Run inference with Akida model
n_batches = num_images // BATCH_SIZE + 1
pots_akida = np.array([], dtype=np.float32)

pbar = ProgressBar(maxval=n_batches)
pbar.start()
start = timer()
i = 1
for batch, _ in test_batches_akida:
    pots_batch_akida = model_akida.evaluate(batch.numpy())
    pots_akida = np.concatenate((pots_akida, pots_batch_akida.squeeze()))
    pbar.update(i)
    i = i + 1
pbar.finish()
end = timer()
print(f"Akida inference on {num_images} images took {end-start:.2f} s.\n")

# Compute predictions and accuracy
preds_akida = tf.keras.layers.Activation('sigmoid')(pots_akida) > 0.5
akida_accuracy = np.mean(np.equal(preds_akida, labels))
print(f"Akida accuracy: {akida_accuracy*100:.2f} %")

# For non-regression purpose
assert akida_accuracy > 0.97

######################################################################

# Print model statistics
stats = model_akida.get_statistics()

print("Model statistics")
for _, stat in stats.items():
    print(stat)

######################################################################
# Let's summarize the accuracy for the quantized Keras and the Akida model.
#
# +-----------------+----------+
# | Model           | Accuracy |
# +=================+==========+
# | quantized Keras | 98.28 %  |
# +-----------------+----------+
# | Akida           | 98.32 %  |
# +-----------------+----------+

######################################################################
# 6. Plot confusion matrix
# ------------------------

import matplotlib.pyplot as plt


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
plot_confusion_matrix_2classes(cm_akida, ['dog', 'cat'])
plt.show()
