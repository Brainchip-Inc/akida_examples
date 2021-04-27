"""
DS-CNN CIFAR10 inference
========================

This tutorial uses the CIFAR-10 dataset (60k training images distributed in 10
object classes) for a classic object classification task with a network built
around the Depthwise Separable Convolutional Neural Network (DS-CNN) which is
originated from `Zhang et al (2018) <https://arxiv.org/pdf/1711.07128.pdf>`_.

The goal of the tutorial is to provide users with an example of a complex model
that can be converted to an Akida model and that can be run on Akida NSoC
with an accuracy similar to a standard Keras floating point model.

"""

######################################################################
# 1. Dataset preparation
# ~~~~~~~~~~~~~~~~~~~~~~
from tensorflow.keras.datasets import cifar10

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

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = (x_train - b) / a
x_test = (x_test - b) / a

######################################################################
# 2. Create a Keras DS-CNN model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The DS-CNN architecture is available in the `Akida models zoo
# <../../api_reference/akida_models_apis.html#cifar-10>`_ along with pretrained
# weights.
#
#  .. Note:: The pre-trained weights were obtained after training the model with
#            unconstrained float weights and activations for 1000 epochs
#

from tensorflow.keras.utils import get_file
from tensorflow.keras.models import load_model

# Retrieve the float model with pretrained weights and load it
model_file = get_file(
    "ds_cnn_cifar10.h5",
    "http://data.brainchip.com/models/ds_cnn/ds_cnn_cifar10.h5",
    cache_subdir='models/ds_cnn_cifar10')
model_keras = load_model(model_file)
model_keras.summary()

######################################################################
# Keras model accuracy is checked against the first *n* images of the test set.
#
# The table below summarizes the expected results:
#
# +---------+----------+
# | #Images | Accuracy |
# +=========+==========+
# | 100     |  96.00 % |
# +---------+----------+
# | 1000    |  94.30 % |
# +---------+----------+
# | 10000   |  93.60 % |
# +---------+----------+
#
# .. Note:: Depending on your hardware setup, the processing time may vary.
#

import numpy as np

from sklearn.metrics import accuracy_score
from timeit import default_timer as timer


# Check Model performance
def check_model_performances(model, x_test, num_images=1000):
    start = timer()
    potentials_keras = model.predict(x_test[:num_images])
    preds_keras = np.squeeze(np.argmax(potentials_keras, 1))

    accuracy = accuracy_score(y_test[:num_images], preds_keras)
    print("Accuracy: " + "{0:.2f}".format(100 * accuracy) + "%")
    end = timer()
    print(f'Keras inference on {num_images} images took {end-start:.2f} s.\n')


check_model_performances(model_keras, x_test)

######################################################################
# 3. Quantized model
# ~~~~~~~~~~~~~~~~~~
#
# Quantizing a model is done using `cnn2snn.quantize
# <../../api_reference/cnn2snn_apis.html#quantize>`_. After the call, all the
# layers will have 4-bit weights and 4-bit activations.
#
# This model will therefore satisfy the Akida NSoC requirements but will suffer
# from a drop in accuracy due to quantization as shown in the table below:
#
# +---------+----------------+--------------------+
# | #Images | Float accuracy | Quantized accuracy |
# +=========+================+====================+
# | 100     |     96.00 %    |       96.00 %      |
# +---------+----------------+--------------------+
# | 1000    |     94.30 %    |       92.60 %      |
# +---------+----------------+--------------------+
# | 10000   |     93.66 %    |       92.58 %      |
# +---------+----------------+--------------------+
#

from cnn2snn import quantize

# Quantize the model to 4-bit weights and activations
model_keras_quantized = quantize(model_keras, 4, 4)

# Check Model performance
check_model_performances(model_keras_quantized, x_test)

######################################################################
# 4. Pretrained quantized model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The Akida models zoo also contains a `pretrained quantized helper
# <../../api_reference/akida_models_apis.html#akida_models.ds_cnn_cifar10_pretrained>`_
# that was obtained using the `tune <../../user_guide/akida_models.html#cifar10-training-and-tuning>`_
# action of ``akida_models`` CLI on the quantized model for 100 epochs.
#
# Tuning the model, that is training with a lowered learning rate, allows to
# recover performances up to the initial floating point accuracy.
#
# +---------+----------------+--------------------+--------------+
# | #Images | Float accuracy | Quantized accuracy | After tuning |
# +=========+================+====================+==============+
# | 100     |     96.00 %    |       96.00 %      |    97.00 %   |
# +---------+----------------+--------------------+--------------+
# | 1000    |     94.30 %    |       92.60 %      |    94.20 %   |
# +---------+----------------+--------------------+--------------+
# | 10000   |     93.66 %    |       92.58 %      |    93.08 %   |
# +---------+----------------+--------------------+--------------+
#

from akida_models import ds_cnn_cifar10_pretrained

# Use a quantized model with pretrained quantized weights
model_keras_quantized_pretrained = ds_cnn_cifar10_pretrained()

# Check Model performance
check_model_performances(model_keras_quantized_pretrained, x_test)

######################################################################
# 5. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~

######################################################################
# 5.1 Convert to Akida model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When converting to an Akida model, we just need to pass the Keras model
# and the input scaling that was used during training to `cnn2snn.convert
# <../../api_reference/cnn2snn_apis.html#convert>`_.

from cnn2snn import convert

model_akida = convert(model_keras_quantized_pretrained, input_scaling=(a, b))

######################################################################
# 5.2 Check hardware compliancy
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The `Model.summary <../../api_reference/aee_apis.html#akida.Model.summary>`__
# method provides a detailed description of the Model layers.
#
# It also indicates hardware-incompatibilities if there are any. Hardware
# compatibility can also be checked manually using
# `model_hardware_incompatibilities
# <../../api_reference/aee_apis.html#akida.compatibility.model_hardware_incompatibilities>`_.

model_akida.summary()

######################################################################
# 5.3 Check performance
# ^^^^^^^^^^^^^^^^^^^^^
#
# We check the Akida model accuracy on the first *n* images of the test
# set.
#
# The table below summarizes the expected results:
#
# +---------+----------------+----------------+
# | #Images | Keras accuracy | Akida accuracy |
# +=========+================+================+
# | 100     |     96.00 %    |     97.00 %    |
# +---------+----------------+----------------+
# | 1000    |     94.30 %    |     94.00 %    |
# +---------+----------------+----------------+
# | 10000   |     93.66 %    |     93.04 %    |
# +---------+----------------+----------------+
#
# Due to the conversion process, the predictions may be slightly different
# between the original Keras model and Akida on some specific images.
#
# This explains why when testing on a limited number of images the
# accuracy numbers between Keras and Akida may be quite different. On the
# full test set however, the two models accuracies are very close.
#

num_images = 1000

# Check Model performance
start = timer()
results = model_akida.predict(raw_x_test[:num_images])
accuracy = accuracy_score(y_test[:num_images], results)

print("Accuracy: " + "{0:.2f}".format(100 * accuracy) + "%")
end = timer()
print(f'Akida inference on {num_images} images took {end-start:.2f} s.\n')

# For non-regression purpose
if num_images == 1000:
    assert accuracy == 0.94

######################################################################
# Activations sparsity has a great impact on akida inference time. One can have
# a look at the average input and output sparsity of each layer using
# `Model.get_statistics() <../../api_reference/aee_apis.html#akida.Model.get_statistics>`_
# For convenience, it is called here on a subset of the dataset.
#

# Print model statistics
print("Model statistics")
stats = model_akida.get_statistics()
for _, stat in stats.items():
    print(stat)

######################################################################
# 5.4 Show predictions for a random image
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches

label_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]

# prepare plot
barWidth = 0.75
pause_time = 1

fig = plt.figure(num='CIFAR10 Classification by Akida Execution Engine',
                 figsize=(8, 4))
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
pot = model_akida.evaluate(np.expand_dims(raw_x_test[i], axis=0)).squeeze()

rpot = np.arange(len(pot))
ax1.barh(rpot, pot, height=barWidth)
ax1.set_yticks(rpot - 0.07 * barWidth)
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
