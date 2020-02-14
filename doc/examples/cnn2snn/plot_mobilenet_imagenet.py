"""
Inference on ImageNet with MobileNet
====================================

.. Note:: Please refer to `CNN2SNN Conversion Tutorial (MNIST)
          <../../examples/cnn2snn/mnist_cnn2akida_demo.html>`__ notebook
          and/or the `CNN2SNN documentation
          <../../user_guide/cnn2snn.html>`__ for flow and steps details of
          the CNN2SNN conversion.

This CNN2SNN tutorial presents how to convert a MobileNet pre-trained
model into Akida. As ImageNet images are not publicly available, performances
are assessed using a set of 10 copyright free images that were found on Google
using ImageNet class names.

"""

######################################################################
# 1. Load CNN2SNN tool dependencies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# System imports
import os
import numpy as np
import pickle
import csv
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import tensorflow as tf

from timeit import default_timer as timer

# ImageNet tutorial imports
from akida_models import mobilenet_imagenet
from akida_models.mobilenet.imagenet import imagenet_preprocessing

######################################################################
# 2. Load test images from ImageNet
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The inputs in the Keras MobileNet model must respect two requirements:
#
# * the input image size must be 224x224x3,
# * the input image values must be between -1 and 1.
#
# This section goes as follows:
#
# * **Load and preprocess images.** The test images all have at least 256 pixels
#   in the smallest dimension. They must be preprocessed to fit in the model.
#   The ``imagenet_preprocessing.preprocess_image`` function decodes, crops and
#   extracts a square 224x224x3 patch from an input image.
# * **Load corresponding labels.** The labels for test images are stored in the
#   akida_models package. The matching between names (*string*) and labels
#   (*integer*) is given through ``imagenet_preprocessing.index_to_label``
#   method.
#
# .. Note:: Akida Execution Engine is configured to take 8-bit inputs
#           without rescaling. For conversion, rescaling values used for
#           training the Keras model are needed.
#


######################################################################
# 2.1 Load test images and preprocess test images
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# Model specification and hyperparameters
NUM_CHANNELS = 3
IMAGE_SIZE = 224
NUM_CLASSES = 1000

num_images = 10

file_path = tf.keras.utils.get_file("imagenet_like.zip",
                                    "http://data.brainchip.com/dataset-mirror/imagenet_like/imagenet_like.zip",
                                    cache_subdir='datasets/imagenet_like',
                                    extract=True)
data_folder = os.path.dirname(file_path)

# Load images for test set
x_test_files = []
x_test = np.zeros((num_images, 224, 224, 3)).astype('uint8')
for id in range(num_images):
    test_file = 'image_' + str(id+1).zfill(2) + '.jpg'
    x_test_files.append(test_file)
    img_path = os.path.join(data_folder, test_file)
    base_image = tf.io.read_file(img_path)
    image = imagenet_preprocessing.preprocess_image(
        image_buffer=base_image,
        bbox=None,
        output_width=IMAGE_SIZE,
        output_height=IMAGE_SIZE,
        num_channels=NUM_CHANNELS,
        alpha=1.,
        beta=0.)
    x_test[id, :, :, :] = np.expand_dims(image.numpy(), axis=0)

# Rescale images for Keras model (normalization between -1 and 1)
# Assume rescaling format of (x - b)/a
a = 127.5
b = 127.5
input_scaling = (a, b)
x_test_preprocess = (x_test.astype('float32') - b) / a

print(f'{num_images} images loaded and preprocessed.')


######################################################################
# 2.2 Load labels
# ^^^^^^^^^^^^^^^
#

fname = os.path.join(data_folder, 'labels_validation.txt')
validation_labels = dict()
with open(fname, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        validation_labels[row[0]] = row[1]

# Get labels for the test set by index
labels_test = np.zeros(num_images)
for i in range(num_images):
    labels_test[i] = int(validation_labels[x_test_files[i]])

print('Labels loaded.')


######################################################################
# 3. Create a quantized Keras model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A Keras model based on a MobileNet model is instantiated with quantized
# weights and activations. This model satisfies the Akida NSoC
# requirements:
#
# * all the convolutional layers have 4-bit weights, except for the first
#   layer,
# * the first layer has 8-bit weights,
# * all the convolutional layers have 4-bit activations.
#
# This section goes as follows:
#
# * **Instantiate a quantized Keras model** according to above specifications.
# * **Load pre-trained weights** that performs a 65 % accuracy on the test
#   dataset.
# * **Check performance** on the test set. According to the number of test
#   images, the inference could last for several minutes.
#


######################################################################
# 3.1 Instantiate Keras model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The CNN2SNN module offers a way to easily instantiate a MobileNet model
# based on Keras with quantized weights and activations. Our ``MobileNet``
# function returns a Keras model with custom quantized layers (see
# ``quantization_layers.py`` in the CNN2SNN module).
#
# .. Note:: The pre-trained weights which are loaded correspond to the
#    parameters in the next cell. If you want to modify some of these
#    parameters, you must re-train the model and save the weights.
#

print("Instantiating MobileNet...")

input_shape = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
model_keras = mobilenet_imagenet(input_shape=input_shape,
                  classes=NUM_CLASSES,
                  weights='imagenet',
                  weights_quantization=4,
                  activ_quantization=4,
                  input_weights_quantization=8)

print("...done.")

model_keras.summary()

######################################################################
# 3.2 Check performance of the Keras model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

print(f'Predicting with Keras model on {num_images} images ...')

start = timer()
potentials_keras = model_keras.predict(x_test_preprocess, batch_size=100)
end = timer()
print(f'Keras inference on {num_images} images took {end-start:.2f} s.\n')

preds_keras = np.squeeze(np.argmax(potentials_keras, 1))
accuracy_keras = np.sum(np.equal(preds_keras, labels_test)) / num_images

print(f"Keras accuracy: {accuracy_keras*100:.2f} %")


######################################################################
# 4. Convert Keras model for Akida NSoC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here, the Keras quantized model is converted into a suitable version for
# the Akida NSoC. The `cnn2snn.convert <../../api_reference/cnn2snn_apis.html#convert>`__
# function needs as arguments the Keras model and the input scaling parameters.
# The Akida model is then saved in a YAML file with the corresponding weights
# binary files.
#
# This section goes as follows:
#
# * **Convert the Keras MobileNet model** to an Akida model compatible for
#   Akida NSoC. Print a summary of the model.
# * **Test performance** of the Akida model (this can take minutes).
# * **Show predictions** for some test images.
#


######################################################################
# 4.1 Convert Keras model to an Akida compatible model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# Convert to Akida and save model
from cnn2snn import convert

print("Converting Keras model for Akida NSoC...")
model_akida = convert(model_keras, input_scaling=input_scaling)
model_akida.summary()


######################################################################
# 4.2 Test performance of the Akida model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

print(f'Predicting with Akida model on {num_images} images ...')

start = timer()
preds_akida = model_akida.predict(x_test)
end = timer()
print(f'Inference on {num_images} images took {end-start:.2f} s.\n')

accuracy_akida = np.sum(np.equal(preds_akida, labels_test)) / num_images

print(f"Accuracy: {accuracy_akida*100:.2f} %")

# For non-regression purpose
assert accuracy_akida >= 0.9

# Print model statistics
print("Model statistics")
stats = model_akida.get_statistics()
model_akida.predict(x_test[:20])
for _, stat in stats.items():
    print(stat)


######################################################################
# 4.3 Show predictions for a random test image
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For a random test image, we predict the top 5 classes and display the
# results on a bar chart.
#


# Functions used to display the top5 results
def get_top5(potentials, true_label):
    """
    Returns the top 5 classes from the output potentials
    """
    tmp_pots = potentials.copy()
    top5 = []
    min_val = np.min(tmp_pots)
    for ii in range(5):
        best = np.argmax(tmp_pots)
        top5.append(best)
        tmp_pots[best] = min_val

    vals = np.zeros((6, ))
    vals[:5] = potentials[top5]
    if true_label not in top5:
        vals[5] = potentials[true_label]
    else:
        vals[5] = 0
    vals /= np.max(vals)

    class_name = []
    for ii in range(5):
        class_name.append(imagenet_preprocessing.index_to_label(top5[ii]).split(',')[0])
    if true_label in top5:
        class_name.append('')
    else:
        class_name.append(imagenet_preprocessing.index_to_label(true_label).split(',')[0])

    return top5, vals, class_name

def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def prepare_plots():
    fig = plt.figure(figsize=(8, 4))
    # Image subplot
    ax0 = plt.subplot(1, 3, 1)
    imgobj = ax0.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
    ax0.set_axis_off()
    # Top 5 results subplot
    ax1 = plt.subplot(1, 2, 2)
    bar_positions = (0, 1, 2, 3, 4, 6)
    rects = ax1.barh(bar_positions, np.zeros((6,)), align='center', height=0.5)
    plt.xlim(-0.2, 1.01)
    ax1.set(xlim=(-0.2, 1.15), ylim=(-1.5, 12))
    ax1.set_yticks(bar_positions)
    ax1.invert_yaxis()
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks([])
    adjust_spines(ax1, 'left')
    ax1.add_line(lines.Line2D((0, 0), (-0.5, 6.5), color=(0.0, 0.0, 0.0)))
    txt_axlbl = ax1.text(-1, -1, 'Top 5 Predictions:', size=12)
    # Adjust Plot Positions
    ax0.set_position([0.05, 0.055, 0.3, 0.9])
    l1, b1, w1, h1 = ax1.get_position().bounds
    ax1.set_position([l1*1.05, b1 + 0.09*h1, w1, 0.8*h1])
    # Add title box
    plt.figtext(0.5, 0.9, "Imagenet Classification by Akida", size=20, ha="center", va="center",
                bbox=dict(boxstyle="round", ec=(0.5, 0.5, 0.5), fc=(0.9, 0.9, 1.0)))

    return fig, imgobj, ax1, rects

def update_bars_chart(rects, vals, true_label):
    counter = 0
    for rect, h in zip(rects, yvals):
        rect.set_width(h)
        if counter<5:
            if top5[counter] == true_label:
                if counter==0:
                    rect.set_facecolor((0.0, 1.0, 0.0))
                else:
                    rect.set_facecolor((0.0, 0.5, 0.0))
            else:
                rect.set_facecolor('gray')
        elif counter == 5:
            rect.set_facecolor('red')
        counter+=1

# Prepare plots
fig, imgobj, ax1, rects = prepare_plots()

# Get a random image
img = np.random.randint(num_images)

# Predict image class
potentials_akida = model_akida.evaluate(np.expand_dims(x_test[img], axis=0)).squeeze()

# Get top 5 prediction labels and associated names
true_label = int(validation_labels[x_test_files[img]])
top5, yvals, class_name = get_top5(potentials_akida, true_label)

# Draw Plots
imgobj.set_data(x_test[img])
ax1.set_yticklabels(class_name, rotation='horizontal', size=9)
update_bars_chart(rects, yvals, true_label)
fig.canvas.draw()
plt.show()
