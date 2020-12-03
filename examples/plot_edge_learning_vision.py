"""
Akida vision edge learning
==========================

Below is not a tutorial but just a code snippet for vision. Please refer to
the `keyword spotting (KWS) tutorial <plot_edge_learning_kws.html>`__ for edge
learning documentation, parameters fine tuning and steps details.

Note that unlike for the `KWS <plot_edge_learning_kws.html>`__ there is no
"offline" Akida training.
"""

##############################################################################
# 1. Load `coil100 <https://www.tensorflow.org/datasets/catalog/coil100>`__ dataset
# ---------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import statistics as stat
import tensorflow as tf
import tensorflow_datasets as tfds

from time import time
from tensorflow.keras.utils import get_file

from akida import FullyConnected

########################

# Retrieve TensorFlow dataset
tfds.disable_progress_bar()
ds, ds_info = tfds.load('coil100:1.*.*', split='train', with_info=True)
print(ds_info.description)

########################

# Select the 5 cup objects
object_ids = ['obj10', 'obj16', 'obj18', 'obj25', 'obj43']
object_dict = {k: [] for k in object_ids}
for data in ds:
    object_id = data['object_id'].numpy().decode('utf-8')
    if object_id in object_dict.keys():
        object_dict[object_id].append(data['image'].numpy())

########################

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
# -----------------------------------

from akida_models import mobilenet_edge_imagenet_pretrained
from cnn2snn import convert

# Load a pre-trained model
model_keras = mobilenet_edge_imagenet_pretrained()

# Convert it to akida
model_ak = convert(model_keras, input_scaling=(128, 128))

########################

# Replace the last layer by a classification layer
num_classes = len(object_dict)
num_neurons_per_class = 1
num_weights = 350
model_ak.pop_layer()
layer_fc = FullyConnected(name='akida_edge_layer',
                          num_neurons=num_classes * num_neurons_per_class,
                          activations_enabled=False)
model_ak.add(layer_fc)
model_ak.compile(num_weights=num_weights,
                 num_classes=num_classes,
                 learning_competition=0.1)
model_ak.summary()

##############################################################################
# 4. Learn with Akida
# -------------------

# Learn objects in num_shots shot(s)
num_shots = 1
for i in range(len(object_ids)):
    start = time()
    train_images = object_dict[object_ids[i]][:num_shots]
    for image in train_images:
        padded_image = tf.image.resize_with_crop_or_pad(image, 224, 224)
        model_ak.fit(np.expand_dims(padded_image, axis=0), i)
    end = time()
    print(
        f'Learned object {object_ids[i]} (class {i}) with {len(train_images)} sample(s) in {end-start:.2f}s'
    )

########################

# Check accuracy against remaining samples
accuracy = []
for i in range(len(object_ids)):
    test_images = object_dict[object_ids[i]][num_shots:]
    predictions = np.zeros(len(test_images))
    for j in range(len(test_images)):
        padded_image = tf.image.resize_with_crop_or_pad(test_images[j], 224,
                                                        224)
        predictions[j] = model_ak.predict(np.expand_dims(padded_image, axis=0),
                                          num_classes=num_classes)
    accuracy.append(100 * np.sum(predictions == i) / len(test_images))
    print(
        f'Accuracy testing object {object_ids[i]} (class {i}) with {len(test_images)} sample(s): {accuracy[i]:.2f}%'
    )

print(f'Total accuracy: {stat.mean(accuracy):.2f}%')
# For non-regression purpose
assert stat.mean(accuracy) > 98
