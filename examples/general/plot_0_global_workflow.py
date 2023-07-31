"""
Global Akida workflow tutorial
==============================

This tutorial illustrates how to use the QuantizeML and CNN2SNN toolkits to produce a model that can
be used with Akida accelerator. You can refer to our `Akida user guide
<../../user_guide/akida.html>`__ for further explanation.

.. Note:: Please refer to TensorFlow  `tf.keras.models
          <https://www.tensorflow.org/api_docs/python/tf/keras/models>`__
          module for model creation/import details and `TensorFlow Guide
          <https://www.tensorflow.org/guide>`__ for details of how TensorFlow works.

          MNIST example below is light enough so you do not need a `GPU
          <https://www.tensorflow.org/install/gpu>`__ to run the training steps.


.. figure:: ../../img/overall_flow.png
   :target: ../../_images/overall_flow.png
   :alt: Overall flow
   :scale: 25 %
   :align: center

   Akida workflow

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

from keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Add a channels dimension to the image sets as Akida expects 4-D inputs (corresponding to
# (num_samples, width, height, channels). Note: MNIST is unusual in this respect - most image data
# already includes a channel dimension, and this step will not be necessary.
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
# This start point is very much a completely standard CNN as defined
# within `Keras <https://www.tensorflow.org/api_docs/python/tf/keras>`__.
#
# An appropriate model for MNIST (inspired by `this example
# <https://www.tensorflow.org/model_optimization/guide/quantization/training_example#train_a_model_for_mnist_without_quantization_aware_training>`__)
# might look something like the following:
#

import keras

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
# Before going any further, train the model and get its performance. The created model should
# achieve a test accuracy over 98% after 10 epochs.
#
from keras.optimizers import Adam

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
# 2.1. 8bit quantization
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We can now turn to quantization to get a discretized version of the model, where the weights and
# activations are quantized so as to be suitable for conversion towards an Akida accelerator.
#
# For this, we just have to quantize the Keras model using the QuantizeML
# `quantize <../../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__
# function. The selected quantization scheme is 8/8/8 which stands for 8bit weights in the first
# layer, 8bit weights in other layers and 8bit activations respectively.
#
# The quantized model is a Keras model where the layers are replaced with custom `QuantizeML
# quantized layers <../../api_reference/quantizeml_apis.html#layers>`__. All Keras API functions
# can be applied on this new model: ``summary()``, ``compile()``, ``fit()``. etc.
#
# .. Note:: The ``quantize`` function applies `several transformations
#           <../../api_reference/quantizeml_apis.html#transforms>`__ to
#           the original model. For example, it folds the batch normalization layers into the
#           corresponding neural layers. The new weights are computed according to this folding
#           operation.

from quantizeml.models import quantize
from quantizeml.layers import QuantizationParams

qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
model_quantized = quantize(model_keras, qparams=qparams)
model_quantized.summary()

######################################################################
# Check the quantized model accuracy.


def compile_evaluate(model):
    """ Compiles and evaluates the model, then return accuracy score. """
    model.compile(metrics=['accuracy'])
    return model.evaluate(x_test, y_test, verbose=0)[1]


print('Test accuracy after 8bit quantization:', compile_evaluate(model_quantized))


######################################################################
# 2.2. Effect of calibration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The previous call to ``quantize`` was made with random samples for calibration (default
# parameters). While the observed accuracy drop is minimal, that is around 1%, it can be higher on
# more complex models and it is advised to use a set of real samples from the training set. Note
# that this remains a calibration rather than a training step: any relevant data could be used, and
# crucially, no labels are required.
# The recommended configuration for calibration that is widely used to obtain
# `zoo performances <../../zoo_performances.html#akida-2-0-models>`__ is:
#
# - 1024 samples
# - a batch size of 100
# - 2 epochs

model_quantized = quantize(model_keras, qparams=qparams,
                           samples=x_train, num_samples=1024, batch_size=100, epochs=2)

######################################################################
# Check the quantized and calibrated with real samples accuracy.

print('Test accuracy after calibration:', compile_evaluate(model_quantized))

######################################################################
# Calibrating with real samples on this model allows to recover the initial float accuracy.

######################################################################
# 2.3. 4bit quantization
# ^^^^^^^^^^^^^^^^^^^^^^
#
# The accuracy of the 8bit quantized model is equivalent to the one of the base model. In this
# section, a lower bitwidth quantization scheme that is still compatible with Akida accelerator is
# adopted.
# The accuracy of the 8bit quantized model is equal to that of the base model. That quantized model
# is already compatible with the Akida accelerator (following "conversion", see below), and for most
# users, no further quantization is required. In a few cases, it may be attractive to bring the
# model down to an even lower bitwidth quantization scheme, and here we show how to do that.
#
# The model will now be quantized to 8/4/4, that is 8bit weights in the first layer and 4bit weights
# and activations everywhere else. Such a quantization scheme will usually introduce a performance
# drop.

qparams = QuantizationParams(input_weight_bits=8, weight_bits=4, activation_bits=4)
model_quantized = quantize(model_keras, qparams=qparams,
                           samples=x_train, num_samples=1024, batch_size=100, epochs=2)

######################################################################
# Check the 4bit quantized accuracy.

print('Test accuracy after 4bit quantization:', compile_evaluate(model_quantized))

######################################################################
# 2.4. Model fine tuning (quantization-aware training)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When a model suffers from a large accuracy drop after quantization, fine tuning or "quantization
# aware training" (QAT) allows to recover some or all performance.
#
# Note that since this is a fine tuning step, both the number of epochs and learning rate are
# expected to be lower than during the initial float training.
#
model_quantized.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy'])

model_quantized.fit(x_train, y_train, epochs=5, validation_split=0.1)

######################################################################
score = model_quantized.evaluate(x_test, y_test, verbose=0)[1]
print('Test accuracy after fine tuning:', score)

######################################################################
# 3. Convert
# ~~~~~~~~~~
#

######################################################################
# 3.1 Convert to Akida model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When the quantized model achieves satisfactory performance, it can be converted to an Akida
# accelerator suitable format. The
# `convert <../../api_reference/cnn2snn_apis.html#cnn2snn.convert>`__ function returns a model in
# Akida format ready for inference.
#
# As with Keras, the summary() method provides a textual representation of the Akida model.
#

from cnn2snn import convert

model_akida = convert(model_quantized)
model_akida.summary()

######################################################################
# 3.2. Check performance
# ^^^^^^^^^^^^^^^^^^^^^^
accuracy = model_akida.evaluate(x_test, y_test)
print('Test accuracy after conversion:', accuracy)

# For non-regression purposes
assert accuracy > 0.96


######################################################################
# 3.3 Show predictions for a single image
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now try processing a single image, say, the first image in the dataset
# that we looked at above:
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
# Consider the output from the model, printed above. As is typical in backprop trained models, the
# final layer here comprises a Dense layer, with one neuron per class in the dataset (here, 10). The
# goal of training is to maximize the response of the neuron corresponding to the label of each
# training sample, while minimizing the responses of the other neurons.
#
# In the bar chart above, you can see the outputs from all 10 neurons. It is easy to see that neuron
# 7 responds much more strongly than the others. The first sample is indeed a number 7.
#

######################################################################
# 4. GXNOR/MNIST
# ~~~~~~~~~~~~~~
#
# A more robust model called GXNOR/MNIST is provided in `the model zoo
# <../../api_reference/akida_models_apis.html#akida_models.gxnor_mnist>`__ It is inspired from the
# `GXNOR-Net paper <https://arxiv.org/pdf/1705.09283.pdf>`__. It comes with its
# `pretrained 2/2/1 helper
# <../../api_reference/akida_models_apis.html#akida_models.gxnor_mnist_pretrained>`__ for which the
# float training was done for 20 epochs, then the model was then gradually quantized following:
# 4/4/4 --> 4/4/2 --> 2/2/2 --> 2/2/1 with a 15 epochs QAT step at each quantization stage.
