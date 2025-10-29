"""
Segmentation tutorial
==================================================

This example demonstrates image segmentation with an Akida-compatible model as
illustrated through person segmentation using the `Portrait128 dataset
<https://github.com/anilsathyan7/Portrait-Segmentation>`__.

Using pre-trained models for quick runtime, this example shows the evolution of
model performance for a trained TF-Keras floating point model, a TF-Keras quantized and
Quantization Aware Trained (QAT) model, and an Akida-converted model. Notice that
the performance of the original TF-Keras floating point model is maintained throughout
the model conversion flow.
"""

######################################################################
# 1. Load the dataset
# ~~~~~~~~~~~~~~~~~~~
#

import os
import numpy as np
from akida_models import fetch_file

# Download validation set from Brainchip data server, it contains 10% of the original dataset
data_path = fetch_file(fname="val.tar.gz",
                       origin="https://data.brainchip.com/dataset-mirror/portrait128/val.tar.gz",
                       cache_subdir=os.path.join("datasets", "portrait128"),
                       extract=True)

data_dir = os.path.join(os.path.dirname(data_path), "val")
x_val = np.load(os.path.join(data_dir, "val_img.npy"))
y_val = np.load(os.path.join(data_dir, "val_msk.npy")).astype('uint8')
batch_size = 32
steps = x_val.shape[0] // 32

# Visualize some data
import matplotlib.pyplot as plt

rng = np.random.default_rng()
id = rng.integers(0, x_val.shape[0] - 2)

fig, axs = plt.subplots(3, 3, constrained_layout=True)
for col in range(3):
    axs[0, col].imshow(x_val[id + col] / 255.)
    axs[0, col].axis('off')
    axs[1, col].imshow(1 - y_val[id + col], cmap='Greys')
    axs[1, col].axis('off')
    axs[2, col].imshow(x_val[id + col] / 255. * y_val[id + col])
    axs[2, col].axis('off')

fig.suptitle('Image, mask and masked image', fontsize=10)
plt.show()

######################################################################
# 2. Load a pre-trained native TF-Keras model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The model used in this example is AkidaUNet. It has an AkidaNet (0.5) backbone to extract
# features combined with a succession of `separable transposed convolutional
# <../../api_reference/akida_models_apis.html#akida_models.layer_blocks.sepconv_transpose_block>`__
# blocks to build an image segmentation map. A pre-trained floating point TF-Keras model is
# downloaded to save training time.
#
# .. note::
#   - The "transposed" convolutional feature is new in Akida 2.0.
#   - The "separable transposed" operation is realized through the combination of a QuantizeML custom
#     `DepthwiseConv2DTranspose
#     <../../api_reference/quantizeml_apis.html#quantizeml.layers.DepthwiseConv2DTranspose>`__ layer
#     with a standard pointwise convolution.
#
# The performance of the model is evaluated using both pixel accuracy and `Binary IoU
# <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryIoU>`__. The pixel
# accuracy describes how well the model can predict the segmentation mask pixel by pixel
# and the Binary IoU takes into account how close the predicted mask is to the ground truth.
#

from akida_models.model_io import load_model

# Retrieve the model file from Brainchip data server
model_file = fetch_file(fname="akida_unet_portrait128.h5",
                        origin="https://data.brainchip.com/models/AkidaV2/akida_unet/akida_unet_portrait128.h5",
                        cache_subdir='models')

# Load the native TF-Keras pre-trained model
model_keras = load_model(model_file)
model_keras.summary()

######################################################################

from tf_keras.metrics import BinaryIoU

# Compile the native TF-Keras model (required to evaluate the metrics)
model_keras.compile(loss='binary_crossentropy', metrics=[BinaryIoU(), 'accuracy'])

# Check Keras model performance
_, biou, acc = model_keras.evaluate(x_val, y_val, steps=steps, verbose=0)

print(f"TF-Keras binary IoU / pixel accuracy: {biou:.4f} / {100*acc:.2f}%")

######################################################################
# 3. Load a pre-trained quantized Keras model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The next step is to quantize and potentially perform Quantize Aware Training (QAT) on the
# TF-Keras model from the previous step. After the TF-Keras model is quantized to 8-bit for
# all weights and activations, QAT is used to maintain the performance of the quantized
# model. Again, a pre-trained model is downloaded to save runtime.
#

from akida_models import akida_unet_portrait128_pretrained

# Load the pre-trained quantized model
model_quantized_keras = akida_unet_portrait128_pretrained()
model_quantized_keras.summary()

######################################################################

# Compile the quantized TF-Keras model (required to evaluate the metrics)
model_quantized_keras.compile(loss='binary_crossentropy', metrics=[BinaryIoU(), 'accuracy'])

# Check Keras model performance
_, biou, acc = model_quantized_keras.evaluate(x_val, y_val, steps=steps, verbose=0)

print(f"TF-Keras quantized binary IoU / pixel accuracy: {biou:.4f} / {100*acc:.2f}%")

######################################################################
# 4. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Finally, the quantized TF-Keras model from the previous step is converted into an Akida
# model and its performance is evaluated. Note that the original performance of the TF-Keras
# floating point model is maintained throughout the conversion process in this example.
#

from cnn2snn import convert

# Convert the model
model_akida = convert(model_quantized_keras)
model_akida.summary()

#####################################################################

import tf_keras as keras

# Check Akida model performance
labels, pots = None, None

for s in range(steps):
    batch = x_val[s * batch_size: (s + 1) * batch_size, :]
    label_batch = y_val[s * batch_size: (s + 1) * batch_size, :]
    pots_batch = model_akida.predict(batch.astype('uint8'))

    if labels is None:
        labels = label_batch
        pots = pots_batch
    else:
        labels = np.concatenate((labels, label_batch))
        pots = np.concatenate((pots, pots_batch))
preds = keras.activations.sigmoid(pots)

m_binary_iou = keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
m_binary_iou.update_state(labels, preds)
binary_iou = m_binary_iou.result().numpy()

m_accuracy = keras.metrics.Accuracy()
m_accuracy.update_state(labels, preds > 0.5)
accuracy = m_accuracy.result().numpy()
print(f"Akida binary IoU / pixel accuracy: {binary_iou:.4f} / {100*accuracy:.2f}%")

# For non-regression purpose
assert binary_iou > 0.9

######################################################################
# 5. Segment a single image
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For visualization of the person segmentation performed by the Akida model, display a
# single image along with the segmentation produced by the original floating point model
# and the ground truth segmentation.
#

import matplotlib.pyplot as plt

# Estimate age on a random single image and display TF-Keras and Akida outputs
sample = np.expand_dims(x_val[id, :], 0)
keras_out = model_keras(sample)
akida_out = keras.activations.sigmoid(model_akida.forward(sample.astype('uint8')))

fig, axs = plt.subplots(1, 3, constrained_layout=True)
axs[0].imshow(keras_out[0] * sample[0] / 255.)
axs[0].set_title('Keras segmentation', fontsize=10)
axs[0].axis('off')

axs[1].imshow(akida_out[0] * sample[0] / 255.)
axs[1].set_title('Akida segmentation', fontsize=10)
axs[1].axis('off')

axs[2].imshow(y_val[id] * sample[0] / 255.)
axs[2].set_title('Expected segmentation', fontsize=10)
axs[2].axis('off')

plt.show()
