"""
DS-CNN/KWS inference
=======================

This tutorial illustrates the process of developing an Akida-compatible speech recognition
model that can identify thirty-two different keywords.

Initially, the model is defined as a CNN in TF-Keras and trained regularly. Next, it undergoes
quantization using `QuantizeML <../../user_guide/quantizeml.html>`__ and finally converted
to Akida using `CNN2SNN <../../user_guide/cnn2snn.html>`__.

This example uses a Keyword Spotting Dataset prepared using **TensorFlow** `audio recognition
example <https://www.tensorflow.org/tutorials/audio/simple_audio>`__ utils.

"""

######################################################################
# 1. Load the preprocessed dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The TensorFlow `speech_commands <https://www.tensorflow.org/datasets/catalog/speech_commands>`__
# dataset is used for training and validation. All keywords except "backward",
# "follow" and "forward", are retrieved. These three words are kept to
# illustrate the edge learning in this
# `edge example <../edge/plot_1_edge_learning_kws.html>`__.
#
# The words to recognize have been converted to `spectrogram images
# <https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md#how-does-this-model-work>`__
# that allows us to use a model architecture that is typically used for image recognition tasks.
# The raw audio data have been preprocessed, transforming the audio files into MFCC features,
# well-suited for CNN networks.
# A pickle file containing the preprocessed data is available on Brainchip data server.
#
import pickle

from akida_models import fetch_file

# Fetch pre-processed data for 32 keywords
fname = fetch_file(
    fname='kws_preprocessed_all_words_except_backward_follow_forward.pkl',
    origin="https://data.brainchip.com/dataset-mirror/kws/kws_preprocessed_all_words_except_backward_follow_forward.pkl",
    cache_subdir='datasets/kws')
with open(fname, 'rb') as f:
    [_, _, x_valid, y_valid, _, _, word_to_index, _] = pickle.load(f)

# Preprocessed dataset parameters
num_classes = len(word_to_index)

print("Wanted words and labels:\n", word_to_index)

######################################################################
# 2. Load a pre-trained native TF-Keras model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The model consists of:
#
# * a first convolutional layer accepting dense inputs (images),
# * several separable convolutional layers preserving spatial dimensions,
# * a global pooling reducing the spatial dimensions to a single pixel,
# * a final dense layer to classify words.
#
# All layers are followed by a batch normalization and a ReLU activation.
#

from tensorflow.keras.models import load_model

# Retrieve the model file from the BrainChip data server
model_file = fetch_file(fname="ds_cnn_kws.h5",
                        origin="https://data.brainchip.com/models/AkidaV2/ds_cnn/ds_cnn_kws.h5",
                        cache_subdir='models')

# Load the native TF-Keras pre-trained model
model_keras = load_model(model_file)
model_keras.summary()

######################################################################

import numpy as np

from sklearn.metrics import accuracy_score

# Check TF-Keras Model performance
potentials_keras = model_keras.predict(x_valid)
preds_keras = np.squeeze(np.argmax(potentials_keras, 1))

accuracy = accuracy_score(y_valid, preds_keras)
print("Accuracy: " + "{0:.2f}".format(100 * accuracy) + "%")

######################################################################
# 3. Load a pre-trained quantized TF-Keras model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The above native TF-Keras model has been quantized to 8-bit. Note that
# a 4-bit version is also available from the `model zoo <../../model_zoo_performance.html#id10>`_.
#

from quantizeml import load_model

# Load the pre-trained quantized model
model_file = fetch_file(
    fname="ds_cnn_kws_i8_w8_a8.h5",
    origin="https://data.brainchip.com/models/AkidaV2/ds_cnn/ds_cnn_kws_i8_w8_a8.h5",
    cache_subdir='models')
model_keras_quantized = load_model(model_file)
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
# The converted model is Akida 2.0 compatible and its performance
# evaluation is done using the Akida simulator.
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

# For non-regression purposes
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
