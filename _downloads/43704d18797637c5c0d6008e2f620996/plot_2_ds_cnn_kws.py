"""
DS-CNN/KWS inference
=======================

This tutorial illustrates how to build a basic speech recognition
Akida network that recognizes thirty-two different words.

The model will be first defined as a CNN and trained in Keras, then
converted using the `CNN2SNN toolkit <../../user_guide/cnn2snn.html>`__.

This example uses a Keyword Spotting Dataset prepared using
**TensorFlow** `audio recognition
example <https://www.tensorflow.org/tutorials/audio/simple_audio>`__ utils.

The words to recognize are first converted to `spectrogram
images <https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md#how-does-this-model-work>`__
that allows us to use a model architecture that is typically used for
image recognition tasks.

"""

######################################################################
# 1. Load the preprocessed dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The TensorFlow `speech_commands <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>`__
# dataset is used for training and validation. All keywords except "backward",
# "follow" and "forward", are retrieved. These three words are kept to
# illustrate the edge learning in this
# `edge example <../edge/plot_1_edge_learning_kws.html>`__.
# The data are not directly used for training. They are preprocessed,
# transforming the audio files into MFCC features, well-suited for CNN networks.
# A pickle file containing the preprocessed data is available on our data
# server.
#
import pickle

from tensorflow.keras.utils import get_file

# Fetch pre-processed data for 32 keywords
fname = get_file(
    fname='kws_preprocessed_all_words_except_backward_follow_forward.pkl',
    origin="http://data.brainchip.com/dataset-mirror/kws/kws_preprocessed_all_words_except_backward_follow_forward.pkl",
    cache_subdir='datasets/kws')
with open(fname, 'rb') as f:
    [_, _, x_valid, y_valid, _, _, word_to_index, _] = pickle.load(f)

# Preprocessed dataset parameters
num_classes = len(word_to_index)

print("Wanted words and labels:\n", word_to_index)

######################################################################
# 2. Load a pre-trained native Keras model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The model consists of:
#
# * a first convolutional layer accepting dense inputs (images),
# * several separable convolutional layers preserving spatial dimensions,
# * a global pooling reducing the spatial dimensions to a single pixel,
# * a last separable convolutional to reduce the number of outputs
# * a final fully connected layer to classify words
#
# All layers are followed by a batch normalization and a ReLU activation.
#
# This model was obtained with unconstrained float weights and activations after
# 16 epochs of training.
#

from tensorflow.keras.models import load_model

# Retrieve the model file from the BrainChip data server
model_file = get_file("ds_cnn_kws.h5",
                      "http://data.brainchip.com/models/ds_cnn/ds_cnn_kws.h5",
                      cache_subdir='models')

# Load the native Keras pre-trained model
model_keras = load_model(model_file)
model_keras.summary()

######################################################################

import numpy as np

from sklearn.metrics import accuracy_score

# Check Keras Model performance
potentials_keras = model_keras.predict(x_valid)
preds_keras = np.squeeze(np.argmax(potentials_keras, 1))

accuracy = accuracy_score(y_valid, preds_keras)
print("Accuracy: " + "{0:.2f}".format(100 * accuracy) + "%")

######################################################################
# 3. Load a pre-trained quantized Keras model satisfying Akida NSoC requirements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The above native Keras model is quantized and fine-tuned to get a quantized
# Keras model satisfying the `Akida NSoC requirements
# <../../user_guide/hw_constraints.html>`__.
# The first convolutional layer uses 8 bits weights, but other layers use
# 4 bits weights.
#
# All activations are 4 bits except for the final Separable Convolutional that
# uses binary activations.
#
# Pre-trained weights were obtained after a few training episodes:
#
# * we train the model with quantized activations only, with weights initialized
#   from those trained in the previous episode (native Keras model),
# * then, we train the model with quantized weights, with both weights and
#   activations initialized from those trained in the previous episode,
# * finally, we train the model with quantized weights and activations and by
#   gradually increasing quantization in the last layer.
#

from akida_models import ds_cnn_kws_pretrained

# Load the pre-trained quantized model
model_keras_quantized = ds_cnn_kws_pretrained()
model_keras_quantized.summary()

# Check Model performance
potentials_keras_q = model_keras_quantized.predict(x_valid)
preds_keras_q = np.squeeze(np.argmax(potentials_keras_q, 1))

accuracy_q = accuracy_score(y_valid, preds_keras_q)
print("Accuracy: " + "{0:.2f}".format(100 * accuracy_q) + "%")

######################################################################
# 4. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~
#
# We convert the model to Akida and then evaluate the performances on the
# dataset.
#

from cnn2snn import convert

# Convert the model
model_akida = convert(model_keras_quantized)
model_akida.summary()

######################################################################

# Check Akida model performance
preds_akida = model_akida.predict_classes(x_valid, num_classes=num_classes)

accuracy = accuracy_score(y_valid, preds_akida)
print("Accuracy: " + "{0:.2f}".format(100 * accuracy) + "%")

# For non-regression purpose
assert accuracy > 0.9

######################################################################
# 5. Confusion matrix
# ~~~~~~~~~~~~~~~~~~~
#
# The confusion matrix provides a good summary of what mistakes the
# network is making.
#
# Per scikit-learn convention it displays the true class in each row (ie
# on each row you can see what the network predicted for the corresponding
# word).
#
# Please refer to the Tensorflow `audio
# recognition <https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md#confusion-matrix>`__
# example for a detailed explanation of the confusion matrix.
#
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_valid, preds_akida,
                      labels=list(word_to_index.values()))

# Normalize
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Display confusion matrix
plt.rcParams["figure.figsize"] = (16, 16)
plt.figure()

title = 'Confusion matrix'
cmap = plt.cm.Blues

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(word_to_index))
plt.xticks(tick_marks, word_to_index, rotation=45)
plt.yticks(tick_marks, word_to_index)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,
             i,
             format(cm[i, j], '.2f'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.autoscale()
plt.show()
