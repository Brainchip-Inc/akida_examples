"""
MobileNet/KWS inference
=======================

This tutorial illustrates how to build a basic speech recognition
Akida network that recognizes ten different words.

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
# 1. Load CNN2SNN tool dependencies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# System imports
import os
import sys
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# TensorFlow imports
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import get_file

# KWS model imports
from akida_models import mobilenet_kws


######################################################################
# 2. Load the preprocessed dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

wanted_words = ['down','go','left','no','off','on','right','stop','up','yes']
all_words = ['_silence_','_unknown_'] + wanted_words

# Preprocessed dataset parameters
CHANNELS = 1
CLASSES = len(all_words)
SPECTROGRAM_LENGTH = 49
FINGERPRINT_WIDTH = 10

input_shape = (SPECTROGRAM_LENGTH, FINGERPRINT_WIDTH, CHANNELS)

# Try to load pre-processed dataset
fname = get_file("preprocessed_data.pkl",
                 "http://data.brainchip.com/dataset-mirror/kws/preprocessed_data.pkl",
                 cache_subdir='datasets/kws')
if os.path.isfile(fname):
    print('Re-loading previously preprocessed dataset...')
    f = open(fname, 'rb')
    [x_train, y_train, x_valid, y_valid, train_files, val_files, word_to_index] = pickle.load(f)
    f.close()
else:
    raise ValueError("Unable to load the pre-processed KWS dataset.")

# Transform the data to uint8
x_train_min = x_train.min()
x_train_max = x_train.max()
max_int_value = 255.0

# For akida hardware training and validation range [0, 255] inclusive uint8
x_train_akida = ((x_train-x_train_min) * max_int_value / (x_train_max - x_train_min)).astype(np.uint8)
x_valid_akida = ((x_valid-x_train_min) * max_int_value / (x_train_max - x_train_min)).astype(np.uint8)

# For cnn2snn training and validation range [0,1] inclusive float32
x_train_rescaled_cnn = (x_train_akida.astype(np.float32))/max_int_value
x_valid_rescaled_cnn = (x_valid_akida.astype(np.float32))/max_int_value

input_scaling = (max_int_value, 0)


######################################################################
# 3. Create a Keras model satisfying Akida NSoC requirements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The model consists of:
#
# * a first Convolutional layer accepting dense inputs (images),
# * several Separable Convolutional layers preserving spatial dimensions,
# * a global pooling reducing the spatial dimensions to a single pixel,
# * a last Separable Convolutional layer to reduce the number of outputs
#   to the number of words to predict.
#
# All layers are followed by a batch normalization and a ReLU activation,
# except the last one that is followed by a SoftMax.
#
# The first convolutional layer uses 8 bits weights, but other layers use
# 4 bits weights.
#
# All activations are 4 bits.
#
# .. Note:: The reason why we do not use a simple FullyConnected layer as the
#           last layer is precisely because of the 4 bits activations, that are
#           only supported as inputs by the Separable Convolutional layers.
#
# Pre-trained weights were obtained after three training episodes:
#
# * first, we train the model with unconstrained float weights and
#   activations for 30 epochs,
# * then, we train the model with quantized activations only, with
#   weights initialized from those trained in the previous episode,
# * finally, we train the model with quantized weights and activations,
#   with weights initialized from those trained in the previous episode.
#
# The table below summarizes the results obtained when preparing the
# weights stored under `<http://data.brainchip.com/models/mobilenet/>`__ :
#
# +---------+----------------+---------------+----------+--------+
# | Episode | Weights Quant. | Activ. Quant. | Accuracy | Epochs |
# +=========+================+===============+==========+========+
# | 1       | N/A            | N/A           | 91.98 %  | 30     |
# +---------+----------------+---------------+----------+--------+
# | 2       | N/A            | 4 bits        | 92.13 %  | 30     |
# +---------+----------------+---------------+----------+--------+
# | 3       | 8/4 bits       | 4 bits        | 91.67 %  | 30     |
# +---------+----------------+---------------+----------+--------+
#

K.clear_session()
model_keras = mobilenet_kws(input_shape,
                            classes=CLASSES,
                            weights='kws',
                            weight_quantization=4,
                            activ_quantization=4,
                            input_weight_quantization=8)
model_keras.summary()


######################################################################
# 4. Check performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Check Model performance
potentials_keras = model_keras.predict(x_valid_rescaled_cnn)
preds_keras = np.squeeze(np.argmax(potentials_keras, 1))

accuracy = accuracy_score(y_valid, preds_keras)
print("Accuracy: "+"{0:.2f}".format(100*accuracy)+"%")


######################################################################
# 5. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 5.1 Convert the trained Keras model to Akida
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We convert the model to Akida and verify that it is compatible with the
# Akida NSoC (**HW** column in summary).
#

# Convert the model
from cnn2snn import convert

model_akida = convert(model_keras, input_scaling=input_scaling)
model_akida.summary()


######################################################################
# 5.2 Check prediction accuracy
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

preds_akida = model_akida.predict(x_valid_akida, num_classes = CLASSES)

accuracy = accuracy_score(y_valid, preds_akida)
print("Accuracy: "+"{0:.2f}".format(100*accuracy)+"%")

# For non-regression purpose
assert accuracy > 0.83

# Print model statistics
print("Model statistics")
stats = model_akida.get_statistics()
model_akida.predict(x_valid_akida[:20], num_classes = CLASSES)
for _, stat in stats.items():
    print(stat)


######################################################################
# 5.3 Confusion matrix
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

# Create confusion matrix
label_mapping = dict(zip(all_words, range(len(all_words))))

cm = confusion_matrix(y_valid, preds_akida, list(label_mapping.values()))

# Normalize
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Display confusion matrix
plt.rcParams["figure.figsize"] = (8,8)
plt.figure()

classes=label_mapping
title='Confusion matrix'
cmap = plt.cm.Blues

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], '.2f'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.autoscale()
plt.show()
