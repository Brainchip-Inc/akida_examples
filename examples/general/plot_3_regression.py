"""
Regression tutorial
==================================================

This tutorial demonstrates that Akida models can perform regression tasks at the same accuracy level
as a native CNN network.

This is illustrated through an age estimation problem using the
`UTKFace dataset <https://susanqq.github.io/UTKFace/>`__.

"""

######################################################################
# 1. Load the dataset
# ~~~~~~~~~~~~~~~~~~~
#

from akida_models.utk_face.preprocessing import load_data

# Load the dataset using akida_models preprocessing tool
x_train, y_train, x_test, y_test = load_data()

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
# The pre-trained native Keras model loaded below was trained on 300 epochs.
#
# The performance of the model is evaluated using the "Mean Absolute Error"
# (MAE). The MAE, used as a metric in regression problem, is calculated as an
# average of absolute differences between the target values and the predictions.
# The MAE is a linear score, i.e. all the individual differences are equally
# weighted in the average.

from tensorflow.keras.utils import get_file
from tensorflow.keras.models import load_model

# Retrieve the model file from the BrainChip data server
model_file = get_file("vgg_utk_face.h5",
                      "https://data.brainchip.com/models/AkidaV2/vgg/vgg_utk_face.h5",
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
# The above native Keras model is quantized and fine-tuned over 30 epochs. The first convolutional
# layer of our model uses 8bit weights, other layers are quantized using 4bit weights, all
# activations are 4bit.
#

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

# For non-regression purpose
assert abs(mae_keras - mae_akida) < 0.5

######################################################################
# 5. Estimate age on a single image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
