"""
Age estimation (regression) example
==================================================

This tutorial aims to demonstrate the comparable accuracy of the Akida-compatible
model to the traditional Keras model in performing an age estimation task.

It uses the `UTKFace dataset <https://susanqq.github.io/UTKFace/>`__, which
includes images of faces and age labels, to showcase how well akida compatible
model can predict the ages of individuals based on their facial features.

"""

######################################################################
# 1. Load the UTKFace Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The UTKFace dataset has 20,000+ diverse face images spanning 0 to 116 years.
# It includes age, gender, ethnicity annotations. This dataset is useful for
# various tasks like age estimation, face detection, and more.
#
# Load the dataset from Brainchip data server using the `load_data
# <../../api_reference/akida_models_apis.html#akida_models.utk_face.preprocessing.load_data>`__
# helper (decode JPEG images and load the associated labels).

from akida_models.utk_face.preprocessing import load_data

# Load the dataset
x_train, y_train, x_test, y_test = load_data()

######################################################################
# Akida models accept only `uint8 tensors <../../api_reference/akida_apis.html?highlight=uint8#akida.Model>`_
# as inputs. Use uint8 raw data for Akida performance evaluation.

# For Akida inference, use uint8 raw data
x_test_akida = x_test.astype('uint8')


######################################################################
# 2. Load a pre-trained native Keras model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The model is a simplified version inspired from `VGG <https://arxiv.org/abs/1409.1556>`__
# architecture. It consists of a succession of convolutional and pooling layers
# and ends with two dense layers that outputs a single value
# corresponding to the estimated age.
#
# The performance of the model is evaluated using the "Mean Absolute Error"
# (MAE). The MAE, used as a metric in regression problem, is calculated as an
# average of absolute differences between the target values and the predictions.
# The MAE is a linear score, i.e. all the individual differences are equally
# weighted in the average.

from akida_models import fetch_file
from tensorflow.keras.models import load_model

# Retrieve the model file from the BrainChip data server
model_file = fetch_file(fname="vgg_utk_face.h5",
                        origin="https://data.brainchip.com/models/AkidaV2/vgg/vgg_utk_face.h5",
                        cache_subdir='models')

# Load the native Keras pre-trained model
model_keras = load_model(model_file)
model_keras.summary()

######################################################################

# Compile the native Keras model (required to evaluate the MAE)
model_keras.compile(optimizer='Adam', loss='mae')

# Check Keras model performance
mae_keras = model_keras.evaluate(x_test, y_test, verbose=0)

print("Keras MAE: {0:.4f}".format(mae_keras))

######################################################################
# 3. Load a pre-trained quantized Keras model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The above native Keras model is quantized and fine-tuned (QAT). All weights and activations are
# quantized to 8-bit.
from akida_models import vgg_utk_face_pretrained

# Load the pre-trained quantized model
model_quantized_keras = vgg_utk_face_pretrained()
model_quantized_keras.summary()

######################################################################

# Compile the quantized Keras model (required to evaluate the MAE)
model_quantized_keras.compile(optimizer='Adam', loss='mae')

# Check Keras model performance
mae_quant = model_quantized_keras.evaluate(x_test, y_test, verbose=0)

print("Keras MAE: {0:.4f}".format(mae_quant))

######################################################################
# 4. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The quantized Keras model is now converted into an Akida model. After conversion, we evaluate the
# performance on the UTKFace dataset.
#

from cnn2snn import convert

# Convert the model
model_akida = convert(model_quantized_keras)
model_akida.summary()

#####################################################################

import numpy as np

# Check Akida model performance
y_akida = model_akida.predict(x_test_akida)

# Compute and display the MAE
mae_akida = np.sum(np.abs(y_test.squeeze() - y_akida.squeeze())) / len(y_test)
print("Akida MAE: {0:.4f}".format(mae_akida))

# For non-regression purposes
assert abs(mae_keras - mae_akida) < 0.5

######################################################################
# 5. Estimate age on a single image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Select a random image from the test set for age estimation.
# Print the Keras model's age prediction using the ``model_keras.predict()`` function.
# Print the Akida model's estimated age and the actual age associated with the image.

import matplotlib.pyplot as plt

# Estimate age on a random single image and display Keras and Akida outputs
id = np.random.randint(0, len(y_test) + 1)
age_keras = model_keras.predict(x_test[id:id + 1])

plt.imshow(x_test_akida[id], interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()

print("Keras estimated age: {0:.1f}".format(age_keras.squeeze()))
print("Akida estimated age: {0:.1f}".format(y_akida[id].squeeze()))
print(f"Actual age: {y_test[id].squeeze()}")
