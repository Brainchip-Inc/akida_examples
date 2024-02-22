"""
YOLO/PASCAL-VOC detection tutorial
==================================

This tutorial demonstrates that Akida can perform object detection. This is illustrated using a
subset of the
`PASCAL-VOC 2007 dataset <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/htmldoc/index.html>`__
with "car" and "person" classes only. The YOLOv2 architecture from
`Redmon et al (2016) <https://arxiv.org/pdf/1506.02640.pdf>`_ has been chosen to
tackle this object detection problem.

"""

######################################################################
# 1. Introduction
# ~~~~~~~~~~~~~~~

######################################################################
# 1.1 Object detection
# ^^^^^^^^^^^^^^^^^^^^
#
# Object detection is a computer vision task that combines two elemental tasks:
#
#  - object classification that consists in assigning a class label to an image
#    like shown in the `AkidaNet/ImageNet inference <plot_1_akidanet_imagenet.html>`_
#    example
#  - object localization that consists of drawing a bounding box around one or
#    several objects in an image
#
# One can learn more about the subject by reading this `introduction to object
# detection blog article
# <https://machinelearningmastery.com/object-recognition-with-deep-learning/>`_.
#

######################################################################
# 1.2 YOLO key concepts
# ^^^^^^^^^^^^^^^^^^^^^
#
# You Only Look Once (YOLO) is a deep neural network architecture dedicated to
# object detection.
#
# As opposed to classic networks that handle object detection, YOLO predicts
# bounding boxes (localization task) and class probabilities (classification
# task) from a single neural network in a single evaluation. The object
# detection task is reduced to a regression problem to spatially separated boxes
# and associated class probabilities.
#
# YOLO base concept is to divide an input image into regions, forming a grid,
# and to predict bounding boxes and probabilities for each region. The bounding
# boxes are weighted by the prediction probabilities.
#
# YOLO also uses the concept of "anchors boxes" or "prior boxes". The network
# does not actually predict the actual bounding boxes but offsets from anchors
# boxes which are templates (width/height ratio) computed by clustering the
# dimensions of the ground truth boxes from the training dataset. The anchors
# then represent the average shape and size of the objects to detect. More
# details on the anchors boxes concept are given in `this blog article
# <https://medium.com/@andersasac/anchor-boxes-the-key-to-quality-object-detection-ddf9d612d4f9>`_.
#
# Additional information about YOLO can be found on the `Darknet website
# <https://pjreddie.com/darknet/yolov2/>`_ and source code for the preprocessing
# and postprocessing functions that are included in akida_models package (see
# the `processing section <../../api_reference/akida_models_apis.html#processing>`_
# in the model zoo) is largely inspired from
# `experiencor github <https://github.com/experiencor/keras-yolo2>`_.
#

######################################################################
# 2. Preprocessing tools
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As this example focuses on car and person detection only, a subset of VOC has
# been prepared with test images from VOC2007 that contains at least one
# of the two classes. The dataset is represented as a tfrecord file,
# containing images, labels, and bounding boxes.
#
# The `load_tf_dataset` function is a helper function that facilitates the loading
# and parsing of the tfrecord file.
#
# The `YOLO toolkit <../../api_reference/akida_models_apis.html#yolo-toolkit>`_
# offers several methods to prepare data for processing, see
# `load_image <../../api_reference/akida_models_apis.html#akida_models.detection.processing.load_image>`_,
# `preprocess_image <../../api_reference/akida_models_apis.html#akida_models.detection.processing.preprocess_image>`_.
#
#

import tensorflow as tf

from akida_models import fetch_file

# Download TFrecords test set from Brainchip data server
data_path = fetch_file(
    fname="voc_test_car_person.tfrecord",
    origin="https://data.brainchip.com/dataset-mirror/voc/voc_test_car_person.tfrecord",
    cache_subdir='datasets/voc',
    extract=True)


# Helper function to load and parse the Tfrecord file.
def load_tf_dataset(tf_record_file_path):
    tfrecord_files = [tf_record_file_path]

    # Feature description for parsing the TFRecord
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'objects/bbox': tf.io.VarLenFeature(tf.float32),
        'objects/label': tf.io.VarLenFeature(tf.int64),
    }

    def _count_tfrecord_examples(dataset):
        return len(list(dataset.as_numpy_iterator()))

    def _parse_tfrecord_fn(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)

        # Decode the image from bytes
        example['image'] = tf.io.decode_jpeg(example['image'], channels=3)

        # Convert the VarLenFeature to a dense tensor
        example['objects/label'] = tf.sparse.to_dense(example['objects/label'], default_value=0)

        example['objects/bbox'] = tf.sparse.to_dense(example['objects/bbox'])
        # Boxes were flattenned that's why we need to reshape them
        example['objects/bbox'] = tf.reshape(example['objects/bbox'],
                                             (tf.shape(example['objects/label'])[0], 4))
        # Create a new dictionary structure
        objects = {
            'label': example['objects/label'],
            'bbox': example['objects/bbox'],
        }

        # Remove unnecessary keys
        example.pop('objects/label')
        example.pop('objects/bbox')

        # Add 'objects' key to the main dictionary
        example['objects'] = objects

        return example

    # Create a TFRecordDataset
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    len_dataset = _count_tfrecord_examples(dataset)
    parsed_dataset = dataset.map(_parse_tfrecord_fn)

    return parsed_dataset, len_dataset


labels = ['car', 'person']

val_dataset, len_val_dataset = load_tf_dataset(data_path)
print("Loaded VOC2007 test data for car and person classes: "
      f"{len_val_dataset} images.")

######################################################################
# Anchors can also be computed easily using YOLO toolkit.
#
# .. Note:: The following code is given as an example. In a real use case
#           scenario, anchors are computed on the training dataset.

from akida_models.detection.generate_anchors import generate_anchors

num_anchors = 5
grid_size = (7, 7)
anchors_example = generate_anchors(val_dataset, num_anchors, grid_size)

######################################################################
# 3. Model architecture
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The `model zoo <../../api_reference/akida_models_apis.html#yolo>`_ contains a
# YOLO model that is built upon the `AkidaNet architecture
# <../../api_reference/akida_models_apis.html#akida_models.akidanet_imagenet>`_
# and 3 separable convolutional layers at the top for bounding box and class
# estimation followed by a final separable convolutional which is the detection
# layer. Note that for efficiency, the alpha parameter in AkidaNet (network
# width or number of filter in each layer) is set to 0.5.
#

from akida_models import yolo_base

# Create a yolo model for 2 classes with 5 anchors and grid size of 7
classes = 2

model = yolo_base(input_shape=(224, 224, 3),
                  classes=classes,
                  nb_box=num_anchors,
                  alpha=0.5)
model.summary()

######################################################################
# The model output can be reshaped to a more natural shape of:
#
#  (grid_height, grid_width, anchors_box, 4 + 1 + num_classes)
#
# where the "4 + 1" term represents the coordinates of the estimated bounding
# boxes (top left x, top left y, width and height) and a confidence score. In
# other words, the output channels are actually grouped by anchor boxes, and in
# each group one channel provides either a coordinate, a global confidence score
# or a class confidence score. This process is done automatically in the
# `decode_output <../../api_reference/akida_models_apis.html#akida_models.detection.processing.decode_output>`__
# function.

from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape

# Define a reshape output to be added to the YOLO model
output = Reshape((grid_size[1], grid_size[0], num_anchors, 4 + 1 + classes),
                 name="YOLO_output")(model.output)

# Build the complete model
full_model = Model(model.input, output)
full_model.output

######################################################################
# 4. Training
# ~~~~~~~~~~~
#
# As the YOLO model relies on Brainchip AkidaNet/ImageNet network, it is
# possible to perform transfer learning from ImageNet pretrained weights when
# training a YOLO model. See the `PlantVillage transfer learning example
# <plot_4_transfer_learning.html>`_ for a detail explanation on transfer
# learning principles.
#

######################################################################
# 5. Performance
# ~~~~~~~~~~~~~~
#
# The model zoo also contains an `helper method
# <../../api_reference/akida_models_apis.html#akida_models.yolo_voc_pretrained>`_
# that allows to create a YOLO model for VOC and load pretrained weights for the
# car and person detection task and the corresponding anchors. The anchors are
# used to interpret the model outputs.
#
# The metric used to evaluate YOLO is the mean average precision (mAP) which is
# the percentage of correct prediction and is given for an intersection over
# union (IoU) ratio. Scores in this example are given for the standard IoU of
# 0.5 meaning that a detection is considered valid if the intersection over
# union ratio with its ground truth equivalent is above 0.5.
#
#  .. Note:: A call to `evaluate_map <../../api_reference/akida_models_apis.html#akida_models.detection.map_evaluation.MapEvaluation.evaluate_map>`_
#            will preprocess the images, make the call to ``Model.predict`` and
#            use `decode_output <../../api_reference/akida_models_apis.html#akida_models.detection.processing.decode_output>`__
#            before computing precision for all classes.
#

from timeit import default_timer as timer
from akida_models import yolo_voc_pretrained
from akida_models.detection.map_evaluation import MapEvaluation

# Load the pretrained model along with anchors
model_keras, anchors = yolo_voc_pretrained()
model_keras.summary()

######################################################################

# Define the final reshape and build the model
output = Reshape((grid_size[1], grid_size[0], num_anchors, 4 + 1 + classes),
                 name="YOLO_output")(model_keras.output)
model_keras = Model(model_keras.input, output)

# Create the mAP evaluator object
num_images = 100

map_evaluator = MapEvaluation(model_keras, val_dataset.take(num_images),
                              num_images, labels, anchors)

# Compute the scores for all validation images
start = timer()
mAP, average_precisions = map_evaluator.evaluate_map()
end = timer()

for label, average_precision in average_precisions.items():
    print(labels[label], '{:.4f}'.format(average_precision))
print('mAP: {:.4f}'.format(mAP))
print(f'Keras inference on {num_images} images took {end-start:.2f} s.\n')

######################################################################
# 6. Conversion to Akida
# ~~~~~~~~~~~~~~~~~~~~~~

######################################################################
# 6.1 Convert to Akida model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# The last YOLO_output layer that was added for splitting channels into values
# for each box must be removed before Akida conversion.

# Rebuild a model without the last layer
compatible_model = Model(model_keras.input, model_keras.layers[-2].output)

######################################################################
# When converting to an Akida model, we just need to pass the Keras model
# to `cnn2snn.convert <../../api_reference/cnn2snn_apis.html#convert>`_.
#

from cnn2snn import convert

model_akida = convert(compatible_model)
model_akida.summary()

######################################################################
# 6.1 Check performance
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Akida model accuracy is tested on the first *n* images of the validation set.
#

# Create the mAP evaluator object
map_evaluator_ak = MapEvaluation(model_akida,
                                 val_dataset.take(num_images),
                                 num_images,
                                 labels,
                                 anchors,
                                 is_keras_model=False)

# Compute the scores for all validation images
start = timer()
mAP_ak, average_precisions_ak = map_evaluator_ak.evaluate_map()
end = timer()

for label, average_precision in average_precisions_ak.items():
    print(labels[label], '{:.4f}'.format(average_precision))
print('mAP: {:.4f}'.format(mAP_ak))
print(f'Akida inference on {num_images} images took {end-start:.2f} s.\n')

######################################################################
# 6.2 Show predictions for a random image
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from akida_models.detection.processing import preprocess_image, decode_output

# Shuffle the data to take a random test image
val_dataset = val_dataset.shuffle(buffer_size=num_images)

input_shape = model_akida.layers[0].input_dims

# Load the image
raw_image = next(iter(val_dataset))['image']

# Keep the original image size for later bounding boxes rescaling
raw_height, raw_width, _ = raw_image.shape

# Pre-process the image
image = preprocess_image(raw_image, input_shape)
input_image = image[np.newaxis, :].astype(np.uint8)

# Call evaluate on the image
pots = model_akida.predict(input_image)[0]

# Reshape the potentials to prepare for decoding
h, w, c = pots.shape
pots = pots.reshape((h, w, len(anchors), 4 + 1 + len(labels)))

# Decode potentials into bounding boxes
raw_boxes = decode_output(pots, anchors, len(labels))

# Rescale boxes to the original image size
pred_boxes = np.array([[
    box.x1 * raw_width, box.y1 * raw_height, box.x2 * raw_width,
    box.y2 * raw_height,
    box.get_label(),
    box.get_score()
] for box in raw_boxes])

fig = plt.figure(num='VOC2012 car and person detection by Akida')
ax = fig.subplots(1)
img_plot = ax.imshow(np.zeros(raw_image.shape, dtype=np.uint8))
img_plot.set_data(raw_image)

for box in pred_boxes:
    rect = patches.Rectangle((box[0], box[1]),
                             box[2] - box[0],
                             box[3] - box[1],
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)
    class_score = ax.text(box[0],
                          box[1] - 5,
                          f"{labels[int(box[4])]} - {box[5]:.2f}",
                          color='red')

plt.axis('off')
plt.show()
