"""
CNN conversion flow tutorial
============================

This tutorial illustrates how to use the CNN2SNN toolkit to **convert CNN
networks to SNN networks** compatible with the **Akida NSoC** in a few steps.
You can refer to our `CNN2SNN toolkit user guide
<https://doc.brainchipinc.com/user_guide/cnn2snn.html>`__ for further
explanation.

The CNN2SNN tool is based on Keras, TensorFlow high-level API for building and
training deep learning models.

.. Note:: Please refer to TensorFlow  `tf.keras.models
          <https://www.tensorflow.org/api_docs/python/tf/keras/models>`__
          module for model creation/import details and `TensorFlow
          Guide <https://www.tensorflow.org/guide>`__ for details of how
          TensorFlow works.

          MNIST example below is light enough so you do not need a `GPU
          <https://www.tensorflow.org/install/gpu>`__ to run the CNN2SNN
          tool.

.. image:: ../img/cnn2snn_flow_small.jpg
   :scale: 35 %

"""

######################################################################
# 1. Load and reshape MNIST dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# After loading, we make 2 transformations on the dataset:
#
# 1. Reshape the sample content data (x values) into a num_samples x width x
#    height x channels matrix.
#
# .. Note:: At this point, we'll set aside the raw data for testing our
#           converted model in the Akida Execution Engine later.
#
# 2. Rescale the 8-bit loaded data to the range 0-to-1 for training.
#
# .. Note:: Input data normalization is a common step dealing with CNN
#           (rationale is to keep data in a range that works with selected
#           optimizers, some reading can be found
#           `here <https://www.jeremyjordan.me/batch-normalization/>`__.
#
#           This shift makes almost no difference in the current example, but
#           for some datasets rescaling the absolute values (and also shifting
#           to zero-mean) can make a really major difference.
#
#           Also note that we store the scaling values ``input_scaling`` for
#           use when preparing the model for the Akida Execution Engine. The
#           implementation of the Akida neural network allows us to completely
#           skip the rescaling step (i.e. the Akida model should be fed with
#           the raw 8-bit values) but that does require information about what
#           scaling was applied prior to training - see below for more details.
#

import tensorflow as tf
from tensorflow import keras

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape x-data
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Set aside raw test data for use with Akida Execution Engine later
raw_x_test = x_test.astype('uint8')
raw_y_test = y_test

# Rescale x-data
a = 255
b = 0
input_scaling = (a, b)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = (x_train - b) / a
x_test = (x_test - b) / a

######################################################################
# 2. Model definition
# ~~~~~~~~~~~~~~~~~~~
#
# Note that at this stage, there is nothing specific to the Akida NSoC.
# This start point is very much a completely standard CNN as defined
# within `Keras <https://www.tensorflow.org/api_docs/python/tf/keras>`__.
#
# An appropriate model for MNIST (inspired by `this
# example <https://www.tensorflow.org/model_optimization/guide/quantization/training_example#train_a_model_for_mnist_without_quantization_aware_training>`__)
# might look something like the following:
#

model_keras = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
    keras.layers.MaxPool2D(padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
], 'mnistnet')

model_keras.summary()

######################################################################
# The model defined above is compatible for conversion into an Akida model, i.e.
# the model doesn't include any layers or operations that aren't Akida-compatible
# (please refer to the `CNN2SNN toolkit <../user_guide/cnn2snn.html>`__ documentation for full
# details):
#
# * Standard Conv2D and Dense layers are supported
# * Hidden layers must be followed  by a ReLU layer.
# * BatchNormalization must always happen before activations.
# * Convolutional blocks can optionally be followed by a MaxPooling.
#
# The CNN2SNN toolkit provides the
# `check_model_compatibility <../api_reference/cnn2snn_apis.html#check-model-compatibility>`__
# function to ensure that the model can be converted into an Akida model. If
# the model is not fully compatible, substitutes will be needed for the
# relevant layers/operations (guidelines included in the documentation).

from cnn2snn import check_model_compatibility

print("Model compatible for Akida conversion:",
      check_model_compatibility(model_keras))

######################################################################
# 3. Model training
# ^^^^^^^^^^^^^^^^^^
#
# Before going any further, train the model and get its performance.
# The created model should have achieved a test accuracy a little over 99% after
# 10 epochs.
#

model_keras.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

model_keras.fit(x_train, y_train, epochs=10, validation_split=0.1)

score = model_keras.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

######################################################################
# 4. Model quantization
# ~~~~~~~~~~~~~~~~~~~~~
#
# We can now turn to quantization to get a discretized version of the model,
# where the weights and activations are quantized so as to be suitable for
# implementation in the Akida NSoC.
#
# For this, we just have to quantize the Keras model using the
# `quantize <../api_reference/cnn2snn_apis.html#quantize>`_
# function. Here, we decide to quantize to the maximum allowed bitwidths for
# the first layer weights (8-bit), the subsequent layer weights (4-bit) and the
# activations (4-bit).
#
# The quantized model is a Keras model where the neural layers (Conv2D, Dense)
# and the ReLU layers are replaced with custom CNN2SNN quantized layers
# (QuantizedConv2D, QuantizedDense, QuantizedReLU). All Keras API functions
# can be applied on this new model: ``summary()``, ``compile()``, ``fit()``. etc.
#
# .. Note:: The ``quantize`` function folds the batch normalization layers into
#           the corresponding neural layer. The new weights are computed
#           according to this folding operation.

from cnn2snn import quantize

model_quantized = quantize(model_keras,
                           input_weight_quantization=8,
                           weight_quantization=4,
                           activ_quantization=4)
model_quantized.summary()

######################################################################
# Check the quantized model accuracy.

model_quantized.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

score = model_quantized.evaluate(x_test, y_test, verbose=0)
print('Test accuracy after 8-4-4 quantization:', score[1])

######################################################################
# Since we used the maximum allowed bitwidths for weights and activations, the
# accuracy of the quantized model is equivalent to the one of the base model,
# but for lower bitwidth, the quantization  usually introduces a performance drop.
#
# Let's try this time with 2-bit for weights and 1-bit for activations.

model_quantized = quantize(model_keras,
                           weight_quantization=2,
                           activ_quantization=1)

model_quantized.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

score = model_quantized.evaluate(x_test, y_test, verbose=0)
print('Test accuracy after 2-2-1 quantization:', score[1])

# To recover the original model accuracy, a quantization-aware training phase
# is required.

######################################################################
# 5. Model fine tuning (quantization-aware training)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This quantization-aware training (fine tuning) allows to cover the
# performance drop due to the quantization step.
#
# Note that since this step is a fine tuning, the number of epochs can be
# lowered, compared to the training from scratch of the standard model.
#

model_quantized.fit(x_train, y_train, epochs=5, validation_split=0.1)

score = model_quantized.evaluate(x_test, y_test, verbose=0)
print('Test accuracy after fine tuning:', score[1])

######################################################################
# 6. Model conversion
# ~~~~~~~~~~~~~~~~~~~
#
# After having obtained a quantized model with satisfactory performance, it can
# be converted to a model suitable to be used in the Akida NSoC in inference
# mode. The `convert <../api_reference/cnn2snn_apis.html#convert>`__
# function returns a model in Akida format, ready for the Akida NSoC or the
# Akida Execution Engine.
#
# .. Note:: One needs to supply the coefficients used to rescale the input
#           dataset before the training - here ``input_scaling``.
#
# As with Keras, the summary() method provides a textual representation of the
# Akida model.
#

from cnn2snn import convert

model_akida = convert(model_quantized, input_scaling=input_scaling)
model_akida.summary()

results = model_akida.predict(raw_x_test)
accuracy = (raw_y_test == results).mean()

print('Test accuracy after conversion:', accuracy)

# For non-regression purpose
assert accuracy > 0.97

######################################################################
# Depending on the number of samples you run, you should find a
# performance of around 98% (better results can be achieved using more
# epochs for training).
#
