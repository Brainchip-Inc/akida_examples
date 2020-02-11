"""
Inference on CIFAR10 with VGG and MobileNet
===========================================

The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes,
with 6000 images per class. There are 50000 training images and 10000
test images.

This tutorial uses this dataset for a classic object classification task
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

We start from state-of-the-art CNN models, illustrating how these models
can be quantized, then converted to an Akida model with an equivalent
accuracy.

The neural networks used in this tutorial are inspired from the
`VGG <https://arxiv.org/abs/1409.1556>`__ and
`MobileNets <https://arxiv.org/abs/1704.04861>`__ architecture
respectively.

The VGG architecture uses Convolutional and Dense layers: these layers
must therefore be quantized with at most 2 bits of precision to be
compatible with the Akida NSoC. This causes a 2 % drop in accuracy.

The MobileNet architecture uses Separable Convolutional layers that can
be quantized using 4 bits of precision, allowing to preserve the base
Keras model accuracy.

+---------------------------+-------------+
| Model                     | Accuracy    |
+===========================+=============+
| VGG Keras                 | 93.15 %     |
+---------------------------+-------------+
| VGG Keras quantized       | 91.30 %     |
+---------------------------+-------------+
| VGG Akida                 | **91.59 %** |
+---------------------------+-------------+
| MobileNet Keras           | 93.49 %     |
+---------------------------+-------------+
| MobileNet Keras quantized | 93.07 %     |
+---------------------------+-------------+
| MobileNet Akida           | **93.22 %** |
+---------------------------+-------------+


"""

######################################################################
# 1. Load CNN2SNN tool dependencies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# System imports
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer

# TensorFlow imports
from tensorflow.keras.datasets import cifar10

# Akida models imports
from akida_models import mobilenet_cifar10, vgg_cifar10

# CNN2SNN
from cnn2snn import convert


######################################################################
# 2. Load and reshape CIFAR10 dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Reshape x-data
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)
input_shape = (32, 32, 3)

# Set aside raw test data for use with Akida Execution Engine later
raw_x_test = x_test.astype('uint8')

# Rescale x-data
a = 255
b = 0
input_scaling = (a, b)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = (x_train - b)/a
x_test = (x_test - b)/a


######################################################################
# 3. Create a quantized Keras VGG model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A Keras model based on the `VGG <https://arxiv.org/abs/1409.1556>`__
# architecture is instantiated with quantized weights and activations.
#
# This model relies only on FullyConnected and Convolutional layers:
#
#   * all the layers have 2-bit weights,
#   * all the layers have 2-bit activations.
#
# This model therefore satisfies the Akida NSoC requirements.
#
# This section goes as follows:
#
#   * **3.A - Instantiate a quantized Keras VGG model** according to above
#     specifications and load pre-trained weights** that performs 91 % accuracy
#     on the test dataset.
#   * **3.B - Check performance** on the test set.


######################################################################
# 3.A Instantiate Keras model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The ``vgg_cifar10`` function returns a VGG Keras model with custom
# quantized layers (see ``quantization_layers.py`` in the CNN2SNN module).
#
# .. Note:: The pre-trained weights which are loaded in the section 3.B
#           corresponds to the quantization parameters in the next cell. If you
#           want to modify some of these parameters, you must re-train the model
#           and save the weights.
#
# Pre-trained weights were obtained after a series of training episodes,
# starting from unconstrained float weights and activations and ending
# with quantized 2-bits weights and activations.
#
# For the first training episode, we train the model with unconstrained
# float weights and activations for 1000 epochs.
#
# For the subsequent training episodes, we start from the weights trained
# in the previous episode, progressively reducing the bitwidth of
# activations, then weights. We also stop the episode when the training
# loss has stopped decreasing for 20 epochs.
#
# The table below summarizes the results obtained when preparing the
# weights stored under ``http://data.brainchip.com/models/vgg/``:
#
# +---------+----------------+---------------+----------+--------+
# | Episode | Weights Quant. | Activ. Quant. | Accuracy | Epochs |
# +=========+================+===============+==========+========+
# | 1       | N/A            | N/A           | 93.15 %  | 1000   |
# +---------+----------------+---------------+----------+--------+
# | 2       | 4 bits         | 4 bits        | 93.24 %  | 30     |
# +---------+----------------+---------------+----------+--------+
# | 3       | 3 bits         | 4 bits        | 92.91 %  | 50     |
# +---------+----------------+---------------+----------+--------+
# | 4       | 3 bits         | 3 bits        | 92.38 %  | 64     |
# +---------+----------------+---------------+----------+--------+
# | 5       | 2 bits         | 3 bits        | 91.48 %  | 82     |
# +---------+----------------+---------------+----------+--------+
# | 6       | 2 bits         | 2 bits        | 91.31 %  | 74     |
# +---------+----------------+---------------+----------+--------+
#
# Please refer to `mnist_cnn2akida_demo example <mnist_cnn2akida_demo.html>`__
# and/or the `CNN2SNN toolkit <../../api_reference/cnn2snn_apis.html>`__
# documentation for flow and training steps details.

# Instantiate the quantized model
model_keras = vgg_cifar10(input_shape,
                          weights='cifar10',
                          weights_quantization=2,
                          activ_quantization=2,
                          input_weights_quantization=2)
model_keras.summary()

######################################################################
# 3.B Check performance
# ^^^^^^^^^^^^^^^^^^^^^
#
# We check the Keras model accuracy on the first *n* images of the test
# set.
#
# The table below summarizes the expected results:
#
# +---------+----------+
# | #Images | Accuracy |
# +=========+==========+
# | 100     | 94.00 %  |
# +---------+----------+
# | 1000    | 90.80 %  |
# +---------+----------+
# | 10000   | 91.30 %  |
# +---------+----------+
#
# .. Note:: Depending on your hardware setup, the processing time may vary
#           greatly.

num_images = 1000

# Check Model performance
start = timer()
potentials_keras = model_keras.predict(x_test[:num_images])
preds_keras = np.squeeze(np.argmax(potentials_keras, 1))

accuracy = accuracy_score(y_test[:num_images], preds_keras)
print("Accuracy: "+"{0:.2f}".format(100*accuracy)+"%")
end = timer()
print(f'Keras inference on {num_images} images took {end-start:.2f} s.\n')


######################################################################
# 4. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~

######################################################################
# 4.A Convert to Akida model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When converting to an Akida model, we just need to pass the Keras model
# and the input scaling that was used during training.

# Convert the model
model_akida = convert(model_keras, input_scaling=input_scaling)


######################################################################
# 4.B Check hardware compliancy
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The `Model.summary() <../../api_reference/aee_apis.html#akida.Model.summary>`__
# method provides a detailed description of the Model layers.
#
# It also indicates it they are hardware-compatible (see the ``HW`` third
# column).

model_akida.summary()


######################################################################
# 4.C Check performance
# ^^^^^^^^^^^^^^^^^^^^^
#
# We check the Akida model accuracy on the first *n* images of the test
# set.
#
# The table below summarizes the expected results:
#
# +---------+----------+
# | #Images | Accuracy |
# +=========+==========+
# | 100     | 95.00 %  |
# +---------+----------+
# | 1000    | 91.90 %  |
# +---------+----------+
# | 10000   | 91.59 %  |
# +---------+----------+
#
# Due to the conversion process, the predictions may be slightly different
# between the original Keras model and Akida on some specific images.
#
# This explains why when testing on a limited number of images the
# accuracy numbers between Keras and Akida may be quite different. On the
# full test set however, the two models accuracies are almost identical.
#
#  .. Note:: Depending on your hardware setup, the processing time may vary
#            greatly.

num_images = 1000

# Check Model performance
start = timer()
results = model_akida.predict(raw_x_test[:num_images])
accuracy = accuracy_score(y_test[:num_images], results)

print("Accuracy: "+"{0:.2f}".format(100*accuracy)+"%")
end = timer()
print(f'Akida inference on {num_images} images took {end-start:.2f} s.\n')

# For non-regression purpose
if num_images == 1000:
    assert accuracy == 0.919

######################################################################

# Print model statistics
print("Model statistics")
stats = model_akida.get_statistics()
model_akida.predict(raw_x_test[:20])
for _, stat in stats.items():
    print(stat)


######################################################################
# 5. Create a quantized Keras MobileNet model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A Keras model based on the
# `MobileNets <https://arxiv.org/abs/1704.04861>`__ architecture is
# instantiated with quantized weights and activations.
#
# This model relies on a first Convolutional layer followed by several
# Separable Convolutional layers:
#
#   * all the layers have 4-bit weights,
#   * all the layers have 4-bit activations.
#
# This model therefore satisfies the Akida NSoC requirements.
#
# This section goes as follows:
#
#   * **5.A - Instantiate a quantized Keras model** according to above
#     specifications
#   * **5.B - Check performance** on the test set.


######################################################################
# 5.A Instantiate Keras MobileNet model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The ``mobilenet_cifar10`` function returns a MobileNet Keras model with
# custom quantized layers (see ``quantization_layers.py`` in the CNN2SNN
# module).
#
#  .. Note:: The pre-trained weights which are loaded in the section 3.B
#            corresponds to the quantization parameters in the next cell. If you
#            want to modify some of these parameters, you must re-train the
#            model and save the weights.
#
# Pre-trained weights were obtained after two training episodes:
#
#   * first, we train the model with unconstrained float weights and
#     activations for 1000 epochs,
#   * then, we tune the model with quantized weights initialized from those
#     trained in the previous episode.
#
# We stop the second training episode when the training loss has stopped
# decreasing for 20 epochs.
#
# The table below summarizes the results obtained when preparing the
# weights stored under ``http://data.brainchip.com/models/mobilenet/``:
#
# +---------+----------------+---------------+----------+--------+
# | Episode | Weights Quant. | Activ. Quant. | Accuracy | Epochs |
# +=========+================+===============+==========+========+
# | 1       | N/A            | N/A           | 93.49 %  | 1000   |
# +---------+----------------+---------------+----------+--------+
# | 2       | 4 bits         | 4 bits        | 93.07 %  | 44     |
# +---------+----------------+---------------+----------+--------+
#
# Please refer to `mnist_cnn2akida_demo example <mnist_cnn2akida_demo.html>`__
# and/or the `CNN2SNN toolkit <../../api_reference/cnn2snn_apis.html>`__
# documentation for flow and training steps details.

# Use a quantized model with pretrained quantized weights (93.07% accuracy)
model_keras = mobilenet_cifar10(input_shape,
                                weights='cifar10',
                                weights_quantization=4,
                                activ_quantization=4,
                                input_weights_quantization=8)
model_keras.summary()


######################################################################
# 5.B Check performance
# ^^^^^^^^^^^^^^^^^^^^^
#
# We check the Keras model accuracy on the first *n* images of the test
# set.
#
# The table below summarizes the expected results:
#
# +---------+----------+
# | #Images | Accuracy |
# +=========+==========+
# | 100     | 95.00 %  |
# +---------+----------+
# | 1000    | 93.10 %  |
# +---------+----------+
# | 10000   | 93.07 %  |
# +---------+----------+
#
# .. Note:: Depending on your hardware setup, the processing time may vary
#           greatly.

num_images = 1000

# Check Model performance
start = timer()
potentials_keras = model_keras.predict(x_test[:num_images])
preds_keras = np.squeeze(np.argmax(potentials_keras, 1))

accuracy = accuracy_score(y_test[:num_images], preds_keras)
print("Accuracy: "+"{0:.2f}".format(100*accuracy)+"%")
end = timer()
print(f'Keras inference on {num_images} images took {end-start:.2f} s.\n')


######################################################################
# 6. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~


######################################################################
# 6.A Convert to Akida model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When converting to an Akida model, we just need to pass the Keras model
# and the input scaling that was used during training.

model_akida = convert(model_keras, input_scaling=input_scaling)


######################################################################
# 6.B Check hardware compliancy
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The `Model.summary() <../../api_reference/aee_apis.html#akida.Model.summary>`__
# method provides a detailed description of the Model layers.
#
# It also indicates it they are hardware-compatible (see the ``HW`` third
# column).

model_akida.summary()


######################################################################
# 6.C Check performance
# ^^^^^^^^^^^^^^^^^^^^^
#
# We check the Akida model accuracy on the first *n* images of the test
# set.
#
# The table below summarizes the expected results:
#
# +---------+----------+
# | #Images | Accuracy |
# +=========+==========+
# | 100     | 95.00 %  |
# +---------+----------+
# | 1000    | 93.10 %  |
# +---------+----------+
# | 10000   | 93.22 %  |
# +---------+----------+
#
# Due to the conversion process, the predictions may be slightly different
# between the original Keras model and Akida on some specific images.
#
# This explains why when testing on a limited number of images the
# accuracy numbers between Keras and Akida may be quite different. On the
# full test set however, the two models accuracies are almost identical.
#
#  .. Note:: Depending on your hardware setup, the processing time may vary
#            greatly.

num_images = 1000

# Check Model performance
start = timer()
results = model_akida.predict(raw_x_test[:num_images])
accuracy = accuracy_score(y_test[:num_images], results)

print("Accuracy: "+"{0:.2f}".format(100*accuracy)+"%")
end = timer()
print(f'Akida inference on {num_images} images took {end-start:.2f} s.\n')

# For non-regression purpose
if num_images == 1000:
    assert accuracy == 0.931

######################################################################

# Print model statistics
print("Model statistics")
stats = model_akida.get_statistics()
model_akida.predict(raw_x_test[:20])
for _, stat in stats.items():
    print(stat)


######################################################################
# 6D. Show predictions for a random image
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches

#%matplotlib notebook

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# prepare plot
barWidth = 0.75
pause_time = 1

fig = plt.figure(num='CIFAR10 Classification by Akida Execution Engine', figsize=(8, 4))
ax0 = plt.subplot(1, 3, 1)
imgobj = ax0.imshow(np.zeros((32, 32, 3), dtype=np.uint8))
ax0.set_axis_off()
# Results subplots
ax1 = plt.subplot(1, 2, 2)
ax1.xaxis.set_visible(False)
ax0.text(0, 34, 'Actual class:')
actual_class = ax0.text(16, 34, 'None')
ax0.text(0, 37, 'Predicted class:')
predicted_class = ax0.text(20, 37, 'None')

# Take a random test image
i = np.random.randint(y_test.shape[0])

true_idx = int(y_test[i])
pot =  model_akida.evaluate(np.expand_dims(raw_x_test[i], axis=0)).squeeze()

rpot = np.arange(len(pot))
ax1.barh(rpot, pot, height=barWidth)
ax1.set_yticks(rpot - 0.07*barWidth)
ax1.set_yticklabels(label_names)
predicted_idx = pot.argmax()
imgobj.set_data(raw_x_test[i])
if predicted_idx == true_idx:
    ax1.get_children()[predicted_idx].set_color('g')
else:
    ax1.get_children()[predicted_idx].set_color('r')
actual_class.set_text(label_names[true_idx])
predicted_class.set_text(label_names[predicted_idx])
ax1.set_title('Akida\'s predictions')
plt.show()
