"""
Akida vision edge learning
==========================

This tutorial demonstrates the Akida NSoC **edge learning** capabilities using
its built-in learning algorithm.
It focuses on an image classification example, where an existing Akida network
is re-trained to be able to classify images from 4 new classes.

Just a few samples (few-shot learning) of the new classes are sufficient
to augment the Akida model with extra classes, while preserving high accuracy.

Please refer to the `keyword spotting (KWS) tutorial <plot_1_edge_learning_kws.html>`__
for edge learning documentation, parameters fine tuning and steps details.

"""

##############################################################################
# 1. Dataset preparation
# ~~~~~~~~~~~~~~~~~~~~~~

from akida import FullyConnected

import tensorflow_datasets as tfds

# Retrieve TensorFlow `coil100 <https://www.tensorflow.org/datasets/catalog/coil100>`__
# dataset
ds, ds_info = tfds.load('coil100:2.*.*', split='train', with_info=True)
print(ds_info.description)

##############################################################################

# Select the 4 cup objects that will be used as new classes
object_ids = [15, 17, 24, 42]
object_dict = {k: [] for k in object_ids}
for data in ds:
    object_id = data['object_id'].numpy()
    if object_id in object_dict.keys():
        object_dict[object_id].append(data['image'].numpy())

##############################################################################

import matplotlib.pyplot as plt

# Display one image per selected object
f, axarr = plt.subplots(1, len(object_dict))
i = 0
for k in object_dict:
    axarr[i].axis('off')
    axarr[i].imshow(object_dict[k][0])
    axarr[i].set_title(k, fontsize=10)
    i += 1
plt.show()

##############################################################################
# 2. Prepare Akida model for learning
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from akida_models import akidanet_edge_imagenet_pretrained
from cnn2snn import convert

# Load a pre-trained model
model_keras = akidanet_edge_imagenet_pretrained()

# Convert it to akida
model_ak = convert(model_keras)

##############################################################################

from akida import AkidaUnsupervised

# Replace the last layer by a classification layer
num_classes = len(object_dict)
num_neurons_per_class = 1
num_weights = 350
model_ak.pop_layer()
layer_fc = FullyConnected(name='akida_edge_layer',
                          units=num_classes * num_neurons_per_class,
                          activation=False)
model_ak.add(layer_fc)
model_ak.compile(optimizer=AkidaUnsupervised(num_weights=num_weights,
                                             num_classes=num_classes,
                                             learning_competition=0.1))
model_ak.summary()

##############################################################################
# 3. Edge learning with Akida
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np

from tensorflow.image import resize_with_crop_or_pad
from time import time

# Learn objects in num_shots shot(s)
num_shots = 1
for i in range(len(object_ids)):
    start = time()
    train_images = object_dict[object_ids[i]][:num_shots]
    for image in train_images:
        padded_image = resize_with_crop_or_pad(image, 224, 224)
        model_ak.fit(np.expand_dims(padded_image, axis=0), i)
    end = time()
    print(f'Learned object {object_ids[i]} (class {i}) with \
            {len(train_images)} sample(s) in {end-start:.2f}s')

##############################################################################

import statistics as stat

# Check accuracy against remaining samples
accuracy = []
for i in range(len(object_ids)):
    test_images = object_dict[object_ids[i]][num_shots:]
    predictions = np.zeros(len(test_images))
    for j in range(len(test_images)):
        padded_image = resize_with_crop_or_pad(test_images[j], 224, 224)
        predictions[j] = model_ak.predict_classes(np.expand_dims(padded_image,
                                                                 axis=0),
                                                  num_classes=num_classes)
    accuracy.append(100 * np.sum(predictions == i) / len(test_images))
    print(f'Accuracy testing object {object_ids[i]} (class {i}) with \
            {len(test_images)} sample(s): {accuracy[i]:.2f}%')

mean_accuracy = stat.mean(accuracy)
print(f'Mean accuracy: {mean_accuracy:.2f}%')

# For non-regression purpose
assert mean_accuracy > 94
