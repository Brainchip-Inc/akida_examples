"""
Global Akida workflow
=====================

Using the MNIST dataset, this example shows the definition and training of a TF-Keras
floating point model, its quantization to 8-bit with the help of calibration,
its quantization to 4-bit using QAT and its conversion to Akida.
Notice that the performance of the original TF-Keras floating point model is maintained
throughout the Akida flow.
Please refer to the `Akida user guide <../../user_guide/akida.html>`__ for further information.

.. Note:: Please refer to the TensorFlow  `tf_keras.models
          <https://www.tensorflow.org/api_docs/python/tf/keras/models>`__
          module for model creation/import details and the `TensorFlow Guide
          <https://www.tensorflow.org/guide>`__ for TensorFlow usage.

          The MNIST example below is light enough so that a `GPU
          <https://www.tensorflow.org/install/gpu>`__ is not needed for training.


.. figure:: ../../img/overall_flow.png
   :target: ../../_images/overall_flow.png
   :alt: Overall flow
   :scale: 60 %
   :align: center

   Global Akida workflow

"""

######################################################################
# 1. Create and train
# ~~~~~~~~~~~~~~~~~~~
#

######################################################################
# 1.1. Load and reshape MNIST dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from tf_keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Add a channels dimension to the image sets as Akida expects 4-D inputs (corresponding to
# (num_samples, width, height, channels). Note: MNIST is a grayscale dataset and is unusual
# in this respect - most image data already includes a channel dimension, and this step will
# not be necessary.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Display a few images from the test set
f, axarr = plt.subplots(1, 4)
for i in range(0, 4):
    axarr[i].imshow(x_test[i].reshape((28, 28)), cmap=cm.Greys_r)
    axarr[i].set_title('Class %d' % y_test[i])
plt.show()

######################################################################
# 1.2. Model definition
# ^^^^^^^^^^^^^^^^^^^^^
#
# Note that at this stage, there is nothing specific to the Akida IP.
# The model constructed below, as inspired by `this example
# <https://www.tensorflow.org/model_optimization/guide/quantization/training_example#train_a_model_for_mnist_without_quantization_aware_training>`__,
# is a completely standard `TF-Keras <https://www.tensorflow.org/api_docs/python/tf/keras>`__ CNN model.
#

import tf_keras as keras

model_keras = keras.models.Sequential([
    keras.layers.Rescaling(1. / 255, input_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=32, kernel_size=3, strides=2),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    # Separable layer
    keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', strides=2),
    keras.layers.Conv2D(filters=64, kernel_size=1, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
], 'mnistnet')

model_keras.summary()

######################################################################
# 1.3. Model training
# ^^^^^^^^^^^^^^^^^^^
#
# Given the model created above, train the model and check its accuracy. The model should achieve
# a test accuracy over 98% after 10 epochs.
#
from tf_keras.optimizers import Adam

model_keras.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=1e-3),
    metrics=['accuracy'])

_ = model_keras.fit(x_train, y_train, epochs=10, validation_split=0.1)

######################################################################
score = model_keras.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])

######################################################################
# 2. Quantize
# ~~~~~~~~~~~

######################################################################
# 2.1. 8-bit quantization
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# An Akida accelerator processes 8 or 4-bit integer activations and weights. Therefore,
# the floating point TF-Keras model must be quantized in preparation to run on an Akida accelerator.
#
# The QuantizeML `quantize <../../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__
# function can be used to quantize a TF-Keras model for Akida. For this step in this example, an
# “8/8/8” quantization scheme will be applied to the floating point TF-Keras model to produce 8-bit
# weights in the first layer, 8-bit weights in all other layers, and 8-bit activations.
#
# The quantization process results in a TF-Keras model with custom `QuantizeML quantized layers
# <../../api_reference/quantizeml_apis.html#layers>`__ substituted for the original TF-Keras layers.
# All TF-Keras API functions can be applied on this new model: ``summary()``, ``compile()``,
# ``fit()``. etc.
#
# .. Note:: The ``quantize`` function applies `several transformations
#           <../../api_reference/quantizeml_apis.html#transforms>`__ to
#           the original model. For example, it folds the batch normalization layers into the
#           corresponding neural layers. The new weights are computed according to this folding
#           operation.

from quantizeml.models import quantize, QuantizationParams

qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
model_quantized = quantize(model_keras, qparams=qparams)

######################################################################

model_quantized.summary()

######################################################################
# .. Note:: Note that the number of parameters for the floating and quantized models differs,
#           a consequence of the BatchNormalization folding and the additional parameters
#           added for quantization. For further details, please refer to their respective summary.
#

######################################################################
# Check the quantized model accuracy.


def compile_evaluate(model):
    """ Compiles and evaluates the model, then return accuracy score. """
    model.compile(metrics=['accuracy'])
    return model.evaluate(x_test, y_test, verbose=0)[1]


print('Test accuracy after 8-bit quantization:', compile_evaluate(model_quantized))


######################################################################
# 2.2. Effect of calibration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The previous call to ``quantize`` was made with random samples for calibration
# (default parameters). While the observed drop in accuracy is minimal, that is
# around 1%, it can be worse on more complex models. Therefore, it is advised to
# use a set of real samples from the training set for calibration during a call
# to ``quantize``.
# Note that this remains a calibration step rather than a training step in that
# no output labels are required. Furthermore, any relevant data could be used for
# calibration. The recommended settings for calibration that are widely used to
# obtain the `zoo performance <../../model_zoo_performance.html#akida-2-0-models>`__ are:
#
# - 1024 samples
# - a batch size of 100
# - 2 epochs

model_quantized = quantize(model_keras, qparams=qparams,
                           samples=x_train, num_samples=1024, batch_size=100, epochs=2)

######################################################################
# Check the accuracy for the quantized and calibrated model.

print('Test accuracy after calibration:', compile_evaluate(model_quantized))

######################################################################
# Calibrating with real samples on this model recovers the initial float accuracy.

######################################################################
# 2.3. 4-bit quantization
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The accuracy of the 8/8/8 quantized model is equal to that of the Keras floating point
# model. In some cases, a smaller memory size for the model is required. This can be
# accomplished through quantization of the model to smaller bitwidths.
#
# The model will now be quantized to 8/4/4, that is 8-bit weights in the first layer with
# 4-bit weights and activations in all other layers. Such a quantization scheme will usually
# introduce a performance drop.
#

qparams = QuantizationParams(input_weight_bits=8, weight_bits=4, activation_bits=4)
model_quantized = quantize(model_keras, qparams=qparams,
                           samples=x_train, num_samples=1024, batch_size=100, epochs=2)

######################################################################
# Check the 4-bit quantized accuracy.

print('Test accuracy after 4-bit quantization:', compile_evaluate(model_quantized))

######################################################################
# 2.4. Model fine-tuning (Quantization Aware Training)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When a model suffers from an accuracy drop after quantization, fine-tuning or Quantization
# Aware Training (QAT) may recover some or all of the original performance.
#
# Note that since this is a fine-tuning step, both the number of epochs and learning rate are
# expected to be lower than during the initial float training.
#
model_quantized.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy'])

model_quantized.fit(x_train, y_train, epochs=5, validation_split=0.1)

######################################################################
score = model_quantized.evaluate(x_test, y_test, verbose=0)[1]
print('Test accuracy after fine-tuning:', score)

######################################################################
# 3. Convert
# ~~~~~~~~~~
#

######################################################################
# 3.1 Convert to Akida model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When the quantized model produces satisfactory performance, it can be converted to the native
# Akida format. The `convert <../../api_reference/cnn2snn_apis.html#cnn2snn.convert>`__ function
# returns a model in Akida format ready for inference.
#
# As with TF-Keras, the summary() method provides a textual representation of the Akida model.
#

from cnn2snn import convert

model_akida = convert(model_quantized)
model_akida.summary()

######################################################################
# 3.2. Check performance
# ^^^^^^^^^^^^^^^^^^^^^^
accuracy = model_akida.evaluate(x_test, y_test.astype(np.int32))
print('Test accuracy after conversion:', accuracy)

# For non-regression purposes
assert accuracy > 0.96


######################################################################
# 3.3 Show predictions for a single image
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Display one of the test images, such as the first image in the dataset from above, to visualize
# the output of the model.
#

# Test a single example
sample_image = 0
image = x_test[sample_image]
outputs = model_akida.predict(image.reshape(1, 28, 28, 1))
print('Input Label: %i' % y_test[sample_image])

# sphinx_gallery_thumbnail_number = 2
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(x_test[sample_image].reshape((28, 28)), cmap=cm.Greys_r)
axarr[0].set_title('Class %d' % y_test[sample_image])
axarr[1].bar(range(10), outputs.squeeze())
axarr[1].set_xticks(range(10))
plt.show()

print(outputs.squeeze())

######################################################################
# Consider the output from the model above. As is typical in backprop-trained models, the final
# layer is a Dense layer with one neuron for each of the 10 classes in the dataset. The goal of
# training is to maximize the response of the neuron corresponding to the label of each training
# sample while minimizing the responses of the other neurons.
#
# In the bar chart above, you can see the outputs from all 10 neurons. It is easy to see that neuron
# 7 responds much more strongly than the others. The first sample is indeed a number 7.
#
