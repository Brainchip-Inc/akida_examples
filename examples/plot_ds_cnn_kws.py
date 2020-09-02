"""
DS-CNN/KWS inference
=======================

This tutorial illustrates how to build a basic speech recognition
Akida network that recognizes thirty different words.

The model will be first defined as a CNN and trained in Keras, then
converted using the `CNN2SNN toolkit <../user_guide/cnn2snn.html>`__.

This example uses a Keyword Spotting Dataset prepared using
**TensorFlow** `audio recognition
example <https://www.tensorflow.org/tutorials/sequences/audio_recognition>`__
utils.

The words to recognize are first converted to `spectrogram
images <https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md#how-does-this-model-work>`__
that allows us to use a model architecture that is typically used for
image recognition tasks.

"""

######################################################################
# 1. Load the preprocessed dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
import pickle

from tensorflow.keras.utils import get_file

# Fetch pre-processed data for 32 keywords
fname = get_file(
    fname='kws_preprocessed_all_words_except_backward_follow_forward.pkl',
    origin=
    "http://data.brainchip.com/dataset-mirror/kws/kws_preprocessed_all_words_except_backward_follow_forward.pkl",
    cache_subdir='datasets/kws')
with open(fname, 'rb') as f:
    [_, _, x_valid_akida, y_valid, _, _, word_to_index, _] = pickle.load(f)

# Preprocessed dataset parameters
num_classes = len(word_to_index)

print("Wanted words and labels:\n", word_to_index)

# For cnn2snn Keras training, data must be scaled (usually to [0,1])
a = 255
b = 0

x_valid_keras = (x_valid_akida.astype('float32') - b) / a

######################################################################
# 2. Create a Keras model satisfying Akida NSoC requirements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The model consists of:
#
# * a first Convolutional layer accepting dense inputs (images),
# * several Separable Convolutional layers preserving spatial dimensions,
# * a global pooling reducing the spatial dimensions to a single pixel,
# * a last Separable Convolutional to reduce the number of outputs
# * a final FullyConnected layer to classify words
#
# All layers are followed by a batch normalization and a ReLU activation.
#
# The first convolutional layer uses 8 bits weights, but other layers use
# 4 bits weights.
#
# All activations are 4 bits except for the final Separable Convolutional that
# uses binary activations.
#
# Pre-trained weights were obtained after a few training episodes:
#
# * first, we train the model with unconstrained float weights and activations
#   for 16 epochs,
# * then, we train the model with quantized activations only, with weights
#   initialized from those trained in the previous episode,
# * then, we train the model with quantized weights, with both weights and
#   activations initialized from those trained in the previous episode,
# * finally, we train the model with quantized weights and activations and by
#   gradually increasing quantization in the last layer.
#
# The table below summarizes the results obtained when preparing the
# weights stored under `<http://data.brainchip.com/models/ds_cnn/>`__ :
#
# +---------+----------------+----------------------------+----------+--------+
# | Episode | Weights Quant. | Activ. Quant. / last layer | Accuracy | Epochs |
# +=========+================+============================+==========+========+
# | 1       | N/A            | N/A                        | 93.35 %  | 16     |
# +---------+----------------+----------------------------+----------+--------+
# | 2       | N/A            | 4 bits / 4 bits            | 92.18 %  | 16     |
# +---------+----------------+----------------------------+----------+--------+
# | 3       | 8/4 bits       | 4 bits / 4 bits            | 91.92 %  | 16     |
# +---------+----------------+----------------------------+----------+--------+
# | 4       | 8/4 bits       | 4 bits / 3 bits            | 92.29 %  | 16     |
# +---------+----------------+----------------------------+----------+--------+
# | 5       | 8/4 bits       | 4 bits / 2 bits            | 92.31 %  | 16     |
# +---------+----------------+----------------------------+----------+--------+
# | 6       | 8/4 bits       | 4 bits / 1 bit             | 92.53 %  | 16     |
# +---------+----------------+----------------------------+----------+--------+
#

from akida_models import ds_cnn_kws_pretrained
model_keras = ds_cnn_kws_pretrained()
model_keras.summary()

######################################################################
# 3. Check performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

from sklearn.metrics import accuracy_score

# Check Model performance
potentials_keras = model_keras.predict(x_valid_keras)
preds_keras = np.squeeze(np.argmax(potentials_keras, 1))

accuracy = accuracy_score(y_valid, preds_keras)
print("Accuracy: " + "{0:.2f}".format(100 * accuracy) + "%")

######################################################################
# 4. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 4.1 Convert the trained Keras model to Akida
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We convert the model to Akida and verify that it is compatible with Akida
# NSoC.
#

# Convert the model
from cnn2snn import convert

model_akida = convert(model_keras, input_scaling=(a, b))
model_akida.summary()

# Check comptability
from akida.compatibility import model_hardware_incompatibilities
model_hardware_incompatibilities(model_akida)

######################################################################
# 4.2 Check prediction accuracy
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

preds_akida = model_akida.predict(x_valid_akida, num_classes=num_classes)

accuracy = accuracy_score(y_valid, preds_akida)
print("Accuracy: " + "{0:.2f}".format(100 * accuracy) + "%")

# For non-regression purpose
assert accuracy > 0.9

# Print model statistics
print("Model statistics")
stats = model_akida.get_statistics()
model_akida.predict(x_valid_akida[:20], num_classes=num_classes)
for _, stat in stats.items():
    print(stat)

######################################################################
# 4.3 Confusion matrix
# ^^^^^^^^^^^^^^^^^^^^
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
# example for a detailed explaination of the confusion matrix.
#
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_valid, preds_akida, list(word_to_index.values()))

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
