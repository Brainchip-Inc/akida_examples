"""
Akida edge learning for keyword spotting
========================================

This tutorial demonstrates the Akida NSoC **edge learning** capabilities using
its built-in learning algorithm.

It focuses on a keyword spotting (KWS) example, where an existing Akida network
is re-trained to be able to classify new audio keywords.

Just a few samples (few-shot learning) of the new words are sufficient to
augment the Akida model with extra classes, while preserving high accuracy.
"""

##############################################################################
# 1. Edge learning process
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# By "edge learning", we mean the process of network learning in an edge device.
# Aside from technical requirements imposed by the device (low power, latency,
# etc.), the task itself will often present particular challenges:
#
# 1. The application cannot know which, or indeed, how many classes it will
#    be trained on ultimately, so it must be possible to **add new classes**
#    to the classifier online, i.e. requires **continual learning**.
# 2. Often, there will be no large labelled dataset for new classes, which
#    must instead be learned from just a few samples, i.e. requires **few-shot
#    learning**.
#
# The Akida NSoC has a built-in learning algorithm designed for training the
# last layer of a model and well suited for edge learning.
# The specific use case in this tutorial mimics the process of a mobile
# phone user who wants to add new speech commands, i.e. new keywords, to a
# pre-trained voice recognition system with a few preset keywords.
# To achieve this using the Akida NSoC, learning occurs in 3 stages:
#
# 1. The Akida model preparation: an Akida model must meet specific conditions
#    to be compatible for `Akida learning <../../user_guide/aee.html#id5>`__.
# 2. The "offline" Akida learning: the last layer of the Akida model is trained
#    from scratch with a large dataset. In this KWS case, the model is trained
#    with 32 keywords from the Google "Speech Commands dataset".
# 3. The "online" (edge learning) stage: new keywords are learned with few
#    samples, adding to the pre-trained words from stage 2.
#
#
# 1.1 Akida model preparation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The Akida NSoC embeds a native learning algorithm allowing training of the
# last layer of an Akida model. The overall model can then be seen as the
# combination of two parts:
#
# - a feature extractor (or spike generator) corresponding to all but the last
#   layer of a standard (back-propagation trained) neural network. This part of
#   the model cannot be trained on the Akida NSoC, and would typically be
#   prepared in advance, e.g. using the CNN2SNN conversion tool. Also, to be
#   compatible with Akida learning, the feature extractor must return binary
#   spikes (1-bit activations).
# - the classification layer (i.e. the last layer). It must have 1-bit weights
#   and usually has several output neurons per class. This layer will be the
#   only one trained using the built-in learning algorithm.
#
# Note that, unlike a standard CNN network where each class is represented by
# a single output neuron, an Akida native training requires several neurons
# for each class. The number of neurons per class can be seen as the number
# of centroids to represent a class; there is an analogy with k-means clustering
# applied to one-class samples, k being the number of neurons. The choice of the
# number of neurons is a trade-off: too many neurons per class may be
# computationally inefficient; in contrast too few neurons per class may have
# difficulty representing within-class heterogeneity. Like k-means
# clustering, the choice of k depends on the cluster representation of the data.
#
# Like any training process, hyper-parameters must be set appropriately.
# The only mandatory parameter is the number of weights (i.e. number of
# connections for each neuron) which must be correlated to the number of spikes
# at the end of the feature extractor. Other parameters, such as
# ``min_plasticity`` or ``learning_competition``, are optional and mainly used
# for model fine-tuning: one can set them to default for a first try.
#
#
# 1.2 "Offline" Akida learning
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The model is ready for training. Remember that the feature extractor has
# already been trained in stage 1. Here, only the last Akida layer is
# trainable. Training is still "offline" though, corresponding to the
# preparation of the network with the "preset" command keywords. The last layer
# is trained from scratch: its binary weights are randomly initialized.
#
# A large dataset is passed through the Akida network and the on-chip learning
# algorithm updates the weights of the classification layer accordingly.
# In this KWS example, we take a dataset containing 32 words + a "silence"
# class (33 classes) for a total of about 94,000 samples.
#
# Note that the dataset on which the feature extractor was trained does not need
# to be the same as the one used for "offline" training of the classification
# layer. What is important is that the features extracted are as good as
# possible for the expected inputs. Since the "edge" classes are, by
# definition, not known in advance, in practice that typically means making
# your feature extractor as general as possible.
#
#
# 1.3 "Online" edge learning
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# "Online" edge learning consists in adding and learning new classes to a
# former pre-trained model. This stage is meant to be performed on a chip with
# few examples for each new class.
#
# In practice, edge learning with Akida is similar to "offline" learning,
# except that:
#
# - the network has already been trained on a set of classes which need to be
#   kept, and so the novel classes are in addition to those.
# - only few samples are available for training.
#
# In this KWS example, 3 new keywords are learned using 4 samples per word from
# a single user. Applying data augmentation on these samples adds variability
# to improve generalization. After edge learning, the model is able to classify
# the 3 new classes with similar accuracy to the 33 existing classes (and
# performance on the existing classes is unaffected).

##############################################################################
# 2. Dataset preparation
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The data comes from the Google "Speech Commands" dataset containing audio
# files for 35 keywords. The number of utterances for each word varies from
# 1,500 to 4,000.
# 32 words are used to train the Akida model and 3 new words are added for
# edge learning.
#
# Two datasets are loaded:
#
# - The first dataset contains all samples of the 32 following keywords
#   extended with the "silence" samples (see the
#   `original paper <https://arxiv.org/pdf/1804.03209.pdf>`__ for details on
#   the dataset). In total, 94,252 samples are used. These are split into a
#   training set (90%) and a validation set (10%), used to train the model
#   "offline" (stage 2).
# - The second dataset contains samples of the 3 new keywords from a single
#   speaker: 'backward', 'follow' and 'forward'. Since the aim of edge learning
#   is to train with few samples, only 4 utterances will be used for
#   training and the rest for testing (ideally, one would test with many more
#   samples, but the number of repetitions per individual speaker in the
#   database makes this impossible). Data augmentation is applied with time
#   shift and additional background noise, generating 40 training samples per
#   utterances, therefore 4 x 40 = 160 training samples per new word.
#
# The audio files are pre-processed: the mel-frequency cepstral coefficients
# (MFCC) are computed as features to represent each audio sample. The obtained
# features for one sample are stored in an array of shape (49, 10, 1). This
# array of features is chosen as input in the Akida network.
#
# For the sake of simplicity, the pre-processing step is not detailed here;
# this tutorial directly fetches the pre-processed audio data for both datasets.
# The pre-processed utility methods to generate these MFCC data are available in
# the ``akida_models`` package.

import pickle

from tensorflow.keras.utils import get_file

# Fetch pre-processed data for 32 keywords
fname = get_file(
    fname='kws_preprocessed_all_words_except_backward_follow_forward.pkl',
    origin=
    "http://data.brainchip.com/dataset-mirror/kws/kws_preprocessed_all_words_except_backward_follow_forward.pkl",
    cache_subdir='datasets/kws')
with open(fname, 'rb') as f:
    [x_train, y_train, x_val, y_val, _, _, word_to_index,
     data_transform] = pickle.load(f)

# Fetch pre-processed data for the 3 new keywords
fname2 = get_file(
    fname='kws_preprocessed_edge_backward_follow_forward.pkl',
    origin=
    "http://data.brainchip.com/dataset-mirror/kws/kws_preprocessed_edge_backward_follow_forward.pkl",
    cache_subdir='datasets/kws')
with open(fname2, 'rb') as f:
    [
        x_train_new, y_train_new, x_val_new, y_val_new, files_train, files_val,
        word_to_index_new, dt2
    ] = pickle.load(f)

print("Wanted words and labels:\n", word_to_index)
print("New words:\n", word_to_index_new)

##############################################################################
# 3. Prepare Akida model for learning
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As explained above, to be compatible with Akida:
#
# - the feature extractor must return **binary spikes**.
# - the classification layer must have **binary weights**.
#
# For this example, we load a pre-trained model from which we keep the feature
# extractor, returning binary spikes. This model was previously trained and
# quantized with Keras and the CNN2SNN tool. The first dataset with 33 classes
# (32 keywords + "silence") was used for training.
#
# However, the last layer of this pre-trained model is not compatible for Akida
# learning since it doesn't have binary weights. We then remove this last layer
# and add a new classification layer with 33 classes and
# 15 neurons per class. One can try with different values of neurons per
# class, e.g. from 1 to 500 neurons per class, and see the effects on
# performance and time cost.
#
# Moreover, as for any training algorithm, the learning hyper-parameters have to
# be correctly set. For the Akida learning algorithm, one important parameter
# is the **number of weights**: because of the way the Akida learning algorithm
# works, the number of spikes at the end of the feature extractor provides a
# good starting point for this hyper-parameter. Here, we estimate this number
# of output spikes using 10% of the training set, which is enough to have a
# reasonable estimation.

from akida_models import ds_cnn_kws_pretrained

# Instantiate a quantized model with pretrained quantized weights
model = ds_cnn_kws_pretrained()
model.summary()

######################################################################

import numpy as np

from math import ceil

from cnn2snn import convert

#  Convert to an Akida model
model_ak = convert(model)
model_ak.summary()

######################################################################

# Measure Akida accuracy on validation set
batch_size = 1000
preds_ak = np.zeros(y_val.shape[0])
num_batches_val = ceil(x_val.shape[0] / batch_size)
for i in range(num_batches_val):
    s = slice(i * batch_size, (i + 1) * batch_size)
    preds_ak[s] = model_ak.predict(x_val[s])

acc_val_ak = np.sum(preds_ak == y_val) / y_val.shape[0]
print(f"Akida CNN2SNN validation set accuracy: {100 * acc_val_ak:.2f} %")

# For non-regression purpose
assert acc_val_ak > 0.88

######################################################################

from akida import FullyConnected

# Replace the last layer by a classification layer with binary weights
# Here, we choose to set 15 neurons per class.
num_classes = 33
num_neurons_per_class = 15

model_ak.pop_layer()
layer_fc = FullyConnected(name='akida_edge_layer',
                          units=num_classes * num_neurons_per_class,
                          activation=False)
model_ak.add(layer_fc)

######################################################################

from akida import evaluate_sparsity

# Compute sparsity information for the model using 10% of the training data
# which is enough for a good estimate
num_samples = ceil(0.1 * x_train.shape[0])
sparsities = evaluate_sparsity(model_ak, x_train[:num_samples])

# Retrieve the number of output spikes from the feature extractor output
output_density = 1 - sparsities[model_ak.get_layer('separable_4')]
avg_spikes = model.get_layer('separable_4').output_shape[-1] * output_density
print(f"Average number of spikes: {avg_spikes}")

# Fix the number of weights to 1.2 times the average number of output spikes
num_weights = int(1.2 * avg_spikes)
print("The number of weights is then set to:", num_weights)

##############################################################################
# 4. Learn with Akida using the training set
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This stage shows how to train the Akida model using the built-in learning
# algorithm in an "offline" stage, i.e. training the classification layer
# from scratch using a large training set.
# The dataset containing the 33 classes (32 keywords + "silence") is used.
#
# Now that the Akida model is ready for training, the hyper-parameters
# must be set using the `compile <../../api_reference/aee_apis.html#akida.Model.compile>`__
# method of the last layer. Compiling a layer means that this layer is
# configured for training and ready to be trained. For more information about
# the learning hyper-parameters, check the `user guide <../../user_guide/aee.html#id5>`__.
# Note that we set the `learning_competition` to 0.1, which gives a little
# competition between neurons to prevent learning similar features.
#
# Once the last layer is compiled, the
# `fit <../../api_reference/aee_apis.html#akida.Model.fit>`_ method is used to
# pass the dataset for training. This call is similar to the `fit` method in
# tf.keras.
#
# After training, the model is assessed on the validation set using the
# `predict <../../api_reference/aee_apis.html#akida.Model.predict>`_ method. It
# returns the estimated labels for the validation samples.
# The model is then saved to a ``.fbz`` file.
#
# Note that in this specific case, the same dataset was used to train the
# feature extractor using the CNN2SNN tool in an early stage, and to train this
# classification layer using the native learning algorithm. However, the edge
# learning in the next stage passes completely new data in the network.

# Compile Akida model with learning parameters
model_ak.compile(num_weights=num_weights,
                 num_classes=num_classes,
                 learning_competition=0.1)
model_ak.summary()

##############################################################################

from time import time

# Train the last layer using Akida `fit` method
print(f"Akida learning with {num_classes} classes... \
        (this step can take a few minutes)")
num_batches = ceil(x_train.shape[0] / batch_size)
start = time()
for i in range(num_batches):
    s = slice(i * batch_size, (i + 1) * batch_size)
    model_ak.fit(x_train[s], y_train[s].astype(np.int32))
end = time()

print(f"Elapsed time for Akida training: {end-start:.2f} s")

##############################################################################

# Measure Akida accuracy on validation set
preds_val_ak = np.zeros(y_val.shape[0])
for i in range(num_batches_val):
    s = slice(i * batch_size, (i + 1) * batch_size)
    preds_val_ak[s] = model_ak.predict(x_val[s], num_classes=num_classes)

acc_val_ak = np.sum(preds_val_ak == y_val) / y_val.shape[0]
print(f"Akida validation set accuracy: {100 * acc_val_ak:.2f} %")

##############################################################################

import os

from tempfile import TemporaryDirectory

# Save Akida model
temp_dir = TemporaryDirectory(prefix='edge_learning_kws')
model_file = os.path.join(temp_dir.name, 'ds_cnn_edge_kws.fbz')
model_ak.save(model_file)
del model_ak

##############################################################################
# 5. Edge learning
# ~~~~~~~~~~~~~~~~
#
# After the "offline" training stage, we emulate the use case where the
# pre-trained Akida model is loaded on an Akida chip, ready to learn new
# classes. Our previously saved Akida model has 33 output classes with learned
# weights.
# We now add 3 classes to the existing model using the
# `add_classes <../../api_reference/aee_apis.html#akida.Model.add_classes>`_ method
# and learn the 3 new keywords without changing the already learned weights.
#
# There is no need to compile the final layer again; the new neurons were
# initialized along with the old ones, based on the learning hyper-parameters
# given in the `compile <../../api_reference/aee_apis.html#akida.Model.compile>`_
# call. The edge learning then uses the same scheme as for the "offline" Akida
# learning - only the number of samples used is much more restricted.
#
# Here, each new class is trained using 160 samples, stored in the second
# dataset: 4 utterances per word from a single speaker, augmented 40 times each.
# The validation set for new words ['backward', 'follow', 'forward'] contains
# respectively 6, 7 and 6 utterances.

print(f"Validation set of new words ({y_val_new.shape[0]} samples):")
for word, label in word_to_index_new.items():
    print(f" - {word} (label {label}): {np.sum(y_val_new == label)} samples")

# Update new labels following the numbering of the old keywords, i.e, new word
# with label '0' becomes label '34', new word label '1' becomes '35', etc.
y_train_new += num_classes
y_val_new += num_classes

##############################################################################

from akida import Model

# Load the pre-trained model (no need to compile it again)
model_edge = Model(model_file)
model_edge.add_classes(3)

# Train the Akida model with new keywords; only few samples are used.
print("\nEdge learning with 3 new classes ...")
start = time()
model_edge.fit(x_train_new, y_train_new.astype(np.int32))
end = time()
print(f"Elapsed time for Akida edge learning: {end-start:.2f} s")

##############################################################################

# Predict on the new validation set
preds_ak_new = model_edge.predict(x_val_new, num_classes=num_classes + 3)
good_preds_val_new_ak = np.sum(preds_ak_new == y_val_new)
print(f"Akida validation set accuracy on 3 new keywords: \
        {good_preds_val_new_ak}/{y_val_new.shape[0]}")

# Predict on the old validation set. Edge learning of the 3 new keywords barely
# affects the accuracy of the old classes.
preds_ak_old = np.zeros(y_val.shape[0])
for i in range(num_batches_val):
    s = slice(i * batch_size, (i + 1) * batch_size)
    preds_ak_old[s] = model_edge.predict(x_val[s], num_classes=num_classes + 3)

acc_val_old_ak = np.sum(preds_ak_old == y_val) / y_val.shape[0]
print(f"Akida validation set accuracy on 33 old classes: \
        {100 * acc_val_old_ak:.2f} %")

# For non-regression purpose
assert acc_val_old_ak > 0.85
