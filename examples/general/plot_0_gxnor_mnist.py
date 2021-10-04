"""
GXNOR/MNIST inference
=====================

The MNIST dataset is a handwritten digits database. It has a training
set of 60,000 samples, and a test set of 10,000 samples. Each sample
comprises a 28x28 pixel image and an associated label.

This tutorial illustrates how to use a pre-trained model to process the MNIST
dataset.

"""

######################################################################
# 1. Dataset preparation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

# Retrieve MNIST dataset
_, (test_set, test_label) = mnist.load_data()

# Add a dimension to images sets as akida expects 4 dimensions inputs
test_set = np.expand_dims(test_set, -1)

# Display a few images from the test set
f, axarr = plt.subplots(1, 4)
for i in range(0, 4):
    axarr[i].imshow(test_set[i].reshape((28, 28)), cmap=cm.Greys_r)
    axarr[i].set_title('Class %d' % test_label[i])
plt.show()

######################################################################
# 2. Create a Keras GXNOR model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The GXNOR architecture is available in the `Akida models zoo
# <../../api_reference/akida_models_apis.html#akida_models.gxnor_mnist>`_ along
# with pretrained weights.
#
#  .. Note:: The pre-trained weights were obtained with knowledge distillation
#            training, using the EfficientNet model from `this repository
#            <https://github.com/EscVM/Efficient-CapsNet>`_ and the `Distiller`
#            class from the `knowledge distillation toolkit
#            <../../api_reference/akida_models_apis.html#knowledge-distillation>`_.
#
#            The float training was done for 30 epochs, the model is then
#            gradually quantized following:
#            8-4-4 --> 4-4-4 --> 4-4-2 --> 2-2-2 --> 2-2-1
#            by tuning the model at each step with the same distillation
#            training method for 5 epochs.

from akida_models import gxnor_mnist_pretrained

model_keras = gxnor_mnist_pretrained()
model_keras.summary()

######################################################################

# Check Model performances
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model_keras.compile(optimizer='adam',
                    loss=SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
keras_accuracy = model_keras.evaluate(test_set, test_label, verbose=0)[1]
print(f"Keras accuracy : {keras_accuracy}")

######################################################################
# 3. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~

######################################################################
# 3.1 Convert to Akida model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When converting to an Akida model, we just need to pass the Keras model
# to `cnn2snn.convert <../../api_reference/cnn2snn_apis.html#convert>`_.

from cnn2snn import convert

model_akida = convert(model_keras)

######################################################################
# 3.2 Check hardware compliancy
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The `Model.summary <../../api_reference/aee_apis.html#akida.Model.summary>`__
# method provides a detailed description of the Model layers.

model_akida.summary()

######################################################################
# 3.3. Check performance
# ^^^^^^^^^^^^^^^^^^^^^^

from sklearn.metrics import accuracy_score

# Check performance against num_samples samples
num_samples = 10000

results = model_akida.predict(test_set[:num_samples])
accuracy = accuracy_score(test_label[:num_samples], results[:num_samples])

# For non-regression purpose
assert accuracy > 0.99

# Display results
print("Accuracy: " + "{0:.2f}".format(100 * accuracy) + "%")

######################################################################
# Depending on the number of samples you run, you should find a
# performance of around 99% (99.24% if you run all 10000 samples).
#

######################################################################

# Print model statistics
print("Model statistics")
print(model_akida.statistics)

######################################################################
# 3.4 Show predictions for a single image
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now try processing a single image, say, the first image in the dataset
# that we looked at above:
#

# Test a single example
sample_image = 0
image = test_set[sample_image]
outputs = model_akida.evaluate(image.reshape(1, 28, 28, 1))
print('Input Label: %i' % test_label[sample_image])

# sphinx_gallery_thumbnail_number = 2
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(test_set[sample_image].reshape((28, 28)), cmap=cm.Greys_r)
axarr[0].set_title('Class %d' % test_label[sample_image])
axarr[1].bar(range(10), outputs.squeeze())
axarr[1].set_xticks(range(10))
plt.show()

print(outputs.squeeze())

######################################################################
# Consider the output from the model, printed above. As is typical in
# backprop trained models, the final layer here comprises a
# 'fully-connected or 'dense' layer, with one neuron per class in the
# data (here, 10). The goal of training is to maximize the response of the
# neuron corresponding to the label of each training sample, while
# minimizing the responses of the other neurons.
#
# In the bar chart above, you can see the outputs from all 10 neurons. It
# is easy to see that neuron 7 responds much more strongly than the
# others. The first sample is indeed a number 7.
#
# Check this for some of the other samples by editing the value of
# sample_image in the script above (anything from 0 to 9999).
#
