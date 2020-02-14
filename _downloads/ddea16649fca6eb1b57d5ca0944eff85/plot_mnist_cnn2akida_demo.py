"""
CNN conversion flow tutorial for MNIST
======================================

The CNN2SNN tool is based on Keras, TensorFlow high-level API for building and
training deep learning models.

.. Note:: Please refer to TensorFlow  `tf.keras.models
          <https://www.tensorflow.org/api_docs/python/tf/keras/models>`__
          module for model creation/import details and `TensorFlow
          Guide <https://www.tensorflow.org/guide>`__ for details of how
          TensorFlow works.

**CNN2SNN tool** allows you to **convert CNN networks to SNN networks**
compatible with the **Akida NSoC** in a few steps.

.. Note:: MNIST example below is light enough so you do not need a `GPU
          <https://www.tensorflow.org/install/gpu>`__ to run the CNN2SNN
          tool.

.. image:: ../../img/cnn2snn_flow_small.png

"""

######################################################################
# 1. System configuration
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# 1.1 Load CNN2SNN tool dependencies
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# System imports
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from tempfile import TemporaryDirectory

# TensorFlow imports
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, MaxPooling2D, Activation, ReLU, Flatten, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


######################################################################
# 1.2 Load and reshape MNIST dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# After loading, we make 3 transformations on the dataset:
#
# 1. Reshape the sample content data (x values) into a num_samples x width x
#    height x channels matrix.
#
# .. Note:: At this point, we'll set aside the raw data for testing our
#           converted model in the Akida Execution Engine later
#
# 2. Rescale the 8-bit loaded data to the range 0-to-1 for training.
#
# .. Note:: This shift makes almost no difference in the current example, but
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
# 3. Transform the loaded labels from a scalar representation (single integer
# value per sample) to a one-hot vector representation, appropriate for use
# with the squared hinge loss function used in the current model.
#
# .. Note:: Input data normalization is a common step dealing with CNN
#           (rationale is to keep data in a range that works with selected
#           optimizers, some interesting reading can be found
#           `here <https://www.jeremyjordan.me/batch-normalization/>`__.
#

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
x_train = (x_train - b)/a
x_test = (x_test - b)/a

# Transform scalar labels to one-hot representation, scaled to +/- 1 appropriate for squared hinge loss function
y_train = to_categorical(y_train, 10) * 2 - 1
y_test = to_categorical(y_test, 10) * 2 - 1


######################################################################
# 1.3 Set training parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Set some training parameters used across the different training sessions:
#

# Set dataset relative training parameters
epochs = 5
batch_size = 128

# Set the learning rate parameters
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start) ** (1. / epochs)


######################################################################
# 2. Model creation and performance check
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 2.1 Model creation
# ^^^^^^^^^^^^^^^^^^
#
# Note that at this stage, there is nothing specific to the Akida NSoC.
# This start point is very much a completely standard CNN as defined
# within `Keras <https://www.tensorflow.org/api_docs/python/tf/keras>`__.
#
# An appropriate model for MNIST (inspired by `this
# paper <https://arxiv.org/pdf/1705.09283.pdf>`__) might look something
# like the following:
#

img_input = Input(shape=(28, 28, 1))
x = Conv2D(filters=32,
           kernel_size=(5, 5),
           padding='same',
           use_bias=False,
           data_format='channels_last')(img_input)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU(6.)(x)

x = Conv2D(filters=64,
           kernel_size=(5, 5),
           padding='same',
           use_bias=False)(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU(6.)(x)

x = Flatten()(x)
x = Dense(512,
          use_bias=False)(x)
x = BatchNormalization()(x)
x = ReLU(6.)(x)
x = Dense(10,
          use_bias=False)(x)

model_keras = Model(img_input, x, name='mnistnet')

opt = Adam(lr=lr_start)
model_keras.compile(loss='squared_hinge', optimizer=opt, metrics=['accuracy'])
model_keras.summary()


######################################################################
# .. Note:: Adam optimizer is commonly used, more details can be found
#           `here <https://arxiv.org/abs/1609.04747>`__.
#


######################################################################
# 2.2 Performance check
# ^^^^^^^^^^^^^^^^^^^^^
#
# Before going any further, check the current model performance as a
# benchmark for CNN2SNN conversion.
# The created model should achieve a test accuracy a little over 99% after
# 5 epochs:
#

callbacks = []
lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
callbacks.append(lr_scheduler)
history = model_keras.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test),
                    callbacks=callbacks)
score = model_keras.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


######################################################################
# 3. Model Akida-compatibility check and changes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 3.1 Compatibility check
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The first step is to ensure that the model as defined doesn't include
# any layers or operations that aren't Akida-compatible (please refer to
# the `CNN2SNN toolkit <../../user_guide/cnn2snn.html>`__ documentation for full
# details):
#
# * Standard Conv2D and Dense layers are supported (note that
#   there is currently no support for skip, recursive and parallel layers).
# * Each of these trainable core layers except for the last one must be followed
#   by an Activation layer.
# * All blocks can optionally include a BatchNormalization layer.
# * Convolutional blocks can optionally include a MaxPooling type layer.
#
# .. Note:: This configuration of layers (Conv/Dense + BatchNormalization +
#           Activation) constitutes the basic building block of
#           Akida-compatible models and is widely used in deep learning.
#
# If the model defined is not fully compatible with the Akida NSoC,
# substitutes will be needed for the relevant layers/operations
# (guidelines included in the documentation).
#


######################################################################
# 3.2 Model adaptation
# ^^^^^^^^^^^^^^^^^^^^
#
# As noted above, the basic building blocks of Akida compatible models
# actually comprise a trio of layers: Conv/Dense + BatchNormalization +
# Activation (with, optionally, pooling). The CNN2SNN tool provides a set
# of functions that simplify using these building blocks, and subsequently
# enable easy application of Brainchip's custom quantization functions.
#

from akida_models.quantization_blocks import conv_block, dense_block


######################################################################
# The following code illustrates how to express the MNIST model defined
# above using the functions provided by Brainchip. A couple of points to
# avoid confusion when you look through it:
#
# * The ``weight_quantization`` in each block isn't used here, but will be used
#   later to apply a quantization method to the model weights.
# * The ``block_id`` is just used for naming the layers, and will be good
#   practice in enabling reloading of partially trained models in more advanced
#   training cases.
# * Note that in the final block, we set the nonlinearity
#   ``activ_quantization`` to ``None``. In that case, the block has no
#   Activation layer, and the output is simply the output from the
#   BatchNormalization layer.
#

# Removes all the nodes left over from the previous model and free memory
K.clear_session()

# Define the model.
# The commented code shows the sets of layers in the original definition
# that are being replaced by the provided conv_block and dense_blocks here

img_input = Input(shape=(28, 28, 1))

# x = Conv2D(filters=32,
#            kernel_size=(5, 5),
#            padding='same',
#            use_bias=False,
#            data_format='channels_last')(img_input)
# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
# x = BatchNormalization()(x)
# x = ReLU(6.)(x)
x = conv_block(img_input, filters=32,
               kernel_size=(5, 5),
               padding='same',
               use_bias=False,
               name='conv_0',
               pooling='max',
               add_batchnorm=True)

# x = Conv2D(filters=64,
#            kernel_size=(5, 5),
#            padding='same',
#            use_bias=False)(x)
# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
# x = BatchNormalization()(x)
# x = ReLU(6.)(x)
x = conv_block(x, filters=64,
               kernel_size=(5, 5),
               padding='same',
               use_bias=False,
               name='conv_1',
               pooling='max',
               add_batchnorm=True)

x = Flatten()(x)

# x = Dense(512,
#           use_bias=False)(x)
# x = BatchNormalization()(x)
# x = ReLU(6.)(x)
x = dense_block(x, units=512,
                use_bias=False,
                name='dense_2',
                add_batchnorm=True)

# x = Dense(10,
#           use_bias=False)(x)
# x = BatchNormalization()(x)
x = dense_block(x, units=10,
                use_bias=False,
                name='dense_3',
                activ_quantization=None)

model_keras = Model(img_input, x, name='mnistnet')

opt = Adam(lr=lr_start)
model_keras.compile(loss='squared_hinge', optimizer=opt, metrics=['accuracy'])
model_keras.summary()


######################################################################
# 3.3 Performance check
# ^^^^^^^^^^^^^^^^^^^^^
#
# Check modifed model performance:
#

callbacks = []
lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
callbacks.append(lr_scheduler)
history = model_keras.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test),
                    callbacks=callbacks)
score = model_keras.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


######################################################################
# Saving the model weights as ``mnistnet_act_fp_wgt_fp.hdf5`` to reload
# them as init weights for the quantization step:
#
# .. Note:: This is not mandatory but helps with training speed.
#

temp_dir = TemporaryDirectory()
model_keras.save_weights(os.path.join(temp_dir.name, 'mnistnet_act_fp_wgt_fp.hdf5'))


######################################################################
# 4. Model quantization and training
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 4.1 Quantize the model
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We can now turn to training a discretized version of the model, where
# the weights and activations are quantized so as to be suitable for
# implementation in the Akida NSoC.
#
# For this, we just have to change very slightly the definition of the
# model used above, changing just the values of ``weight_quantization``
# and ``activ_quantization`` used for the blocks (but still with no output
# nonlinearity for the final block). Additionally, we'll initialise the
# model using the set of pre-trained weights that we just saved (not so
# important here, but for more complex datasets can make a huge difference
# both to the accuracy level ultimately achieved and to the speed of
# convergence).
#
# Note that, for more challenging datasets, it may also be useful to make
# stepwise changes towards a fully quantized model - e.g. by first
# training with only the activations quantized, re-saving, and then adding
# the quantized weights. Additionally, the toolkit documentation describes
# how one can go further, optimizing the degree of sparsity in the model
# to reduce computational cost while maintaining accuracy. In this first
# example however, we'll stick to a one step conversion (mainly because
# the MNIST dataset simply isn't complex enough to see the benefit of the
# advanced techniques).
#

# Removes all the nodes left over from the previous model and free memory
K.clear_session()

img_input = Input(shape=(28, 28, 1))
x = conv_block(img_input, filters=32,
               kernel_size=(5, 5),
               padding='same',
               use_bias=False,
               name='conv_0',
               weight_quantization=2,
               activ_quantization=1,
               pooling='max',
               add_batchnorm=True)
x = conv_block(x, filters=64,
               kernel_size=(5, 5),
               padding='same',
               use_bias=False,
               name='conv_1',
               weight_quantization=2,
               activ_quantization=1,
               pooling='max',
               add_batchnorm=True)
x = Flatten()(x)
x = dense_block(x, units=512,
                use_bias=False,
                name='dense_2',
                weight_quantization=2,
                activ_quantization=1,
                add_batchnorm=True)
x = dense_block(x, units=10,
                use_bias=False,
                name='dense_3',
                weight_quantization=2,
                activ_quantization=None)

model_keras = Model(img_input, x, name='mnistnet_quantized')
lr_start = 1e-3
opt = Adam(lr=lr_start)
model_keras.compile(loss='squared_hinge', optimizer=opt, metrics=['accuracy'])
model_keras.summary()

# Reload previously computed weights as init weights for the quantization step
load_status = model_keras.load_weights(os.path.join(temp_dir.name, 'mnistnet_act_fp_wgt_fp.hdf5'))


######################################################################
# 4.2 Performance check
# ^^^^^^^^^^^^^^^^^^^^^
#
# Re-train and save the quantized model:
#

callbacks = []
lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
callbacks.append(lr_scheduler)
history = model_keras.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test),
                    callbacks=callbacks)
score = model_keras.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


######################################################################
# 5. Convert trained model for Akida and test
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 5.1 Final conversion
# ^^^^^^^^^^^^^^^^^^^^
#
# Convert the quantized model to a version suitable to be used in the Akida NSoC
# in inference mode:
#
# .. Note:: One needs to supply the coefficients used to rescale the input
#           dataset before the training - here ``input_scaling``.
#
# As with Keras, the summary() method provides a textual representation of the
# model.
#

from cnn2snn import convert

model_akida = convert(model_keras, input_scaling=input_scaling)
model_akida.summary()


######################################################################
# 5.2 Performances check with the Akida Execution Engine
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

num_samples = 1000

results = model_akida.predict(raw_x_test[:num_samples])
accuracy = accuracy_score(raw_y_test[:num_samples], results[:num_samples])

print("Accuracy: "+"{0:.2f}".format(100*accuracy)+"%")

# For non-regression purpose
assert accuracy > 0.95

# Print model statistics
print("Model statistics")
stats = model_akida.get_statistics()
model_akida.predict(raw_x_test[:20])
for _, stat in stats.items():
    print(stat)


######################################################################
# Depending on the number of samples you run, you should find a
# performance of around 99% (better results can be achieved using more
# epochs for training).
#
# .. Note:: Akida-compatible model first layer type is ``InputConvolutional``
#           and holds underlying data to spike conversion (please refer to
#           `Akida Execution Engine documentation
#           <../../user_guide/aee.html>`__ for more details).
#
