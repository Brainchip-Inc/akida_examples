"""
Regression tutorial
==================================================

This tutorial demonstrates that hardware compatible Akida models can perform
regression tasks at the same accuracy level as a native CNN network.

This is illustrated through an age estimation problem using the
`UTKFace dataset <https://susanqq.github.io/UTKFace/>`__.

"""

######################################################################
# 1. Load dependencies
# ~~~~~~~~~~~~~~~~~~~~
#

# Various imports needed for the tutorial
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Akida imports
from cnn2snn import convert
from akida_models import vgg_utk_face
from akida_models.vgg.utk_face.utk_face_preprocessing import load_data

######################################################################
# 2. Load the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Load the dataset using akida_model tool function
x_train, y_train, x_test, y_test = load_data()

# Store the input shape that will later be used to create the model
input_shape = x_test.shape[1:]

# For CNN training and inference, normalize data by subtracting the mean value
# and dividing by the standard deviation
a = np.std(x_train)
b = np.mean(x_train)
input_scaling = (a, b)
x_test_keras = (x_test.astype('float32') - b) / a

# For akida training, use uint8 raw data
x_test_akida = x_test.astype('uint8')

######################################################################
# 3. Create a Keras model satisfying Akida NSoC requirements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The model is a simplified version inspired from `VGG <https://arxiv.org/abs/1409.1556>`__
# architecture. It consists of a succession of Convolutional and Pooling layers
# and ends with two Dense layers at the top that output a single value
# corresponding to the estimated age.
#
# The first convolutional layer uses 8 bit weights, but other layers are
# quantized using 2 bit weights.
# All activations are 2 bits.
#
# Pre-trained weights were obtained after four training episodes:
#
# * the model is first trained with unconstrained float weights and
#   activations for 30 epochs
# * the model is then progressively retrained with quantized activations and
#   weights during three steps: activations are set to 4 bits and weights to 8
#   bits, then both are set to 4 bits and finally both to 2 bits. At each step
#   weights are initialized from the previous step state.
#
model_keras = vgg_utk_face(input_shape,
                           weights='utkface',
                           weight_quantization=2,
                           activ_quantization=2,
                           input_weight_quantization=8)
model_keras.summary()

######################################################################
# 4. Check performance
# ~~~~~~~~~~~~~~~~~~~~

# Compile Keras model, use the mean absolute error (MAE) as a metric.
# MAE is calculated as an average of absolute differences between the target
# values and the predictions. The MAE is a linear score which means that all the
# individual differences are weighted equally in the average.

model_keras.compile(optimizer='Adam', loss='mae')

# Check Keras model performance
mae_keras = model_keras.evaluate(x_test_keras, y_test, verbose=0)

print("Keras MAE: {0:.4f}".format(mae_keras))

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
model_akida = convert(model_keras, input_scaling=input_scaling)
model_akida.summary()

#####################################################################
# 5.2 Check Akida model accuracy
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# Check Akida model performance
y_akida = model_akida.evaluate(x_test_akida)

# Compute and display the MAE
mae_akida = np.sum(np.abs(y_test.squeeze() - y_akida.squeeze())) / len(y_test)
print("Akida MAE: {0:.4f}".format(mae_akida))

# For non-regression purpose
assert abs(mae_keras - mae_akida) < 0.5

######################################################################

# Print model statistics
print("Model statistics")
stats = model_akida.get_statistics()
model_akida.evaluate(x_test_akida[:20])
for _, stat in stats.items():
    print(stat)

######################################################################
# 6. Estimate age on a single image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Estimate age on a random single image and display Keras and Akida outputs
id = random.randint(0, len(y_test))
age_keras = model_keras.predict([[x_test_keras[id]]])

plt.imshow(x_test_akida[id], interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()

print("Keras estimated age: {0:.1f}".format(age_keras.squeeze()))
print("Akida estimated age: {0:.1f}".format(y_akida[id].squeeze()))
print(f"Actual age: {y_test[id].squeeze()}")
