"""
Native learning for pattern detection
=====================================


In this tutorial you will test the ability of a spiking neural network
to detect a small pattern (a smiley face) embedded in a noisy image
(random dots).

"""

######################################################################
# 1. Creating the dataset
# -----------------------

# Various imports needed for the tutorial
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

# Filter warnings
warnings.filterwarnings("ignore", module="matplotlib")

# Akida specific imports
from akida import Model, Sparse, InputData, FullyConnected, LearningType, dense_to_sparse


######################################################################
# 2. Creating random dot images
# -----------------------------

# Create a very simple data set of random dot array
def random_dot_pictures(pattern, size_image=32, n_dots=40, n_images=1000, n_targets=20):
    """This function generates a set of square random images, some of which
    include a specified pattern.

    Each image is actually a binary grid of pixels that are either active (1)
    or inactive (0).

    The pattern is also a binary grid of arbitrary dimensions that is randomly
    inserted in the middle of a subset of the generated images.

    Args:
        pattern: the two-dimensional pattern array
        size_image: the height and width of each square image
        n_dots: the number of active pixels in the image
        n_images: the number of images to generate
        n_targets: the number of images where the pattern should be inserted

    Returns:
        (random_dots, ind_targets)
        Where:
            random_dots is a numpy array containing the generated images
            ind_targets is another numpy array indicating which image contains
            the pattern
    """
    random_dots = np.zeros((size_image, size_image, n_images), dtype=np.uint8)

    # Randomly select num_targets images to insert the pattern
    ind_targets = np.zeros((n_images,), dtype=np.int32)
    ind_targets[np.random.choice(range(0, n_images), n_targets)] = 1

    # Find the middle of the array to insert the pattern
    x_ins = int(size_image/2 - pattern.shape[0]/2)
    y_ins = int(size_image/2 - pattern.shape[1]/2)

    # Generate images
    for im_ind in range(0, n_images):
        on_bits = np.zeros((size_image * size_image), dtype=np.uint8)
        if ind_targets[im_ind] == 1:
            # Include the pattern at the center of the image
            on_bits = np.zeros((size_image, size_image), dtype=np.uint8)
            on_bits[x_ins: x_ins + pattern.shape[0],
                    y_ins: y_ins + pattern.shape[1]] = pattern

            # Add the random noise
            on_bits = np.reshape(on_bits, size_image * size_image)
            on_bits[np.random.choice(np.setdiff1d(range(0, size_image * size_image), np.nonzero(on_bits)),
                                  n_dots - np.count_nonzero(on_bits))] = 1
        else:
            on_bits = np.zeros((size_image * size_image), dtype=np.uint8)
            on_bits[np.random.choice(range(0, size_image * size_image), n_dots)] = 1
        random_dots[:, :, im_ind] = np.reshape(on_bits, (size_image, size_image))
    return random_dots, ind_targets

######################################################################

# Create a target pattern
target_pattern = np.array([[0, 1, 0, 1, 0],
                           [0, 1, 0, 1, 0],
                           [0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1],
                           [0, 1, 1, 1, 0]])

# Create a set of random dots images for training
image_size = 32
nb_dots = 40
num_images_train = 1000
num_targets_train = 4
num_images_test = 10000
num_targets_test = 200

in_images_train, ind_targets_train = random_dot_pictures(target_pattern,image_size,nb_dots,num_images_train, num_targets_train)
in_images_test, ind_targets_test = random_dot_pictures(target_pattern,image_size, nb_dots, num_images_test, num_targets_test)


######################################################################
# 3. Take a look at some of the random dots images
# ------------------------------------------------

# Show the mean image across all images for the training set
fig1 = plt.figure(figsize=(12, 5))
spec3 = gridspec.GridSpec(ncols = 7, nrows = 3)
fig1.add_subplot(spec3[:3,:3])
plt.imshow(np.mean(in_images_train, axis=2), cmap=cm.Greys_r)
plt.title('Average input image (across {} training samples)'.format(num_images_train))
noise_images = in_images_train[:, :, ind_targets_train == 0]
pattern_images = in_images_train[:, :, ind_targets_train == 1]
im_ind = 0
# Show two rows of pure noise images
for i in range(0,2):
    for j in range(3,7):
        fig1.add_subplot(spec3[i,j])
        im_ind += 1
        plt.imshow(noise_images[:, :, im_ind], cmap=cm.Greys_r)
        plt.title('random')
# Show the four images with the inserted pattern
for i in range(0,4):
    fig1.add_subplot(spec3[2, 3+i])
    plt.imshow(pattern_images[:, :, i], cmap=cm.Greys_r)
    plt.title('target')
fig1.tight_layout()
plt.show()


######################################################################
# 4. Configuring the Akida model
# ------------------------------

# Initialize the model
model = Model()
model.add(InputData("input", input_width=32, input_height=32, input_features=1))
fully = FullyConnected("fully", num_neurons=1024, threshold_fire=8)
model.add(fully)
# Compile fully Layer for training
fully.compile(num_weights=30, min_plasticity=0.01, plasticity_decay=0.1)
model.summary()


######################################################################
# 5. Do the learning
# ------------------

# Define the function to test the ability of the model to find repeating pattern
def testing_performance(model, in_images, ind_targets, learning):
    """This function feeds a set of images to an Akida model instance and verifies if the
    model identified the embedded pattern.

    Args:
        model: the Akida model instance
        in_images: the input images
        ind_targets: an index of the images containing the pattern
        learning: a boolean indicating whether the model can learn or not

    Returns:
        (nrn_id, neurons_RF)
        Where:
            nrn_id is an array of neurons that spiked when the pattern was present
            neurons_RF is an array containing an image representing these neurons receptive fields
    """
    neurons_RF = np.zeros((1, image_size, image_size), dtype=int)
    spikes_to_targets = 0
    unnecessary_spikes = 0
    nrn_ids = np.empty(shape=[0, 0])
    for im_ind in range(0, in_images.shape[2]):
        # Transform the images into spikes
        spikes = dense_to_sparse(in_images[:, :, im_ind])
        # Propagate through the model
        if learning:
            out_spikes = model.fit(spikes)
        else:
            out_spikes = model.forward(spikes)
        # Each time there's a spike save the neuron ID to know who's learning
        if out_spikes.nnz != 0:
            if ind_targets[im_ind] == 1:
                spikes_to_targets += 1
                # Iterate over the coordinates of each neuron that spiked
                for coord in out_spikes.coords:
                    nrn_id = coord[3]

                    a = nrn_id == nrn_ids
                    if np.any(a):
                        # If this neuron has already spiked
                        b = np.argwhere(a)
                        # Add the image to it's RF
                        neurons_RF[b, :, :] = neurons_RF[b, :, :] + in_images[:, :,im_ind]
                    else:
                        # If this is a new spiking neuron
                        print("New spiking neuron ID = {} ({})".format(nrn_id, out_spikes.slice(coord).data[0]))
                        nrn_ids = np.append(nrn_ids, nrn_id)
                        # Create a new RF for the neuron
                        neurons_RF = np.append(neurons_RF,np.reshape(in_images[:, :, im_ind],[1, in_images[:, :, im_ind].shape[0], in_images[:, :, im_ind].shape[1]]), axis = 0)
            else:
                unnecessary_spikes += 1
    print("We recognized {} / {} presented targets ".format(spikes_to_targets, np.sum(ind_targets)))
    print("There was {} spikes to noise vectors".format(unnecessary_spikes))
    return nrn_ids, neurons_RF

######################################################################
nrn_ID, img_spiked = testing_performance(model, in_images_train,ind_targets_train, True)
print('')
print("let's have a look at the receptive field of the neuron")
for i in range(0,nrn_ID.shape[0]):
    plt.imshow(img_spiked[i,:,:], cmap=cm.Greys_r)
    plt.title('LEARNING neuron ID {}'.format(nrn_ID[i]))
    plt.show()


######################################################################
# 6. Test the performance
# -----------------------

# Do the testing
stats = model.get_statistics()
nrn_ID, img_spiked = testing_performance(model, in_images_test,ind_targets_test, False)

# Show the mean image across all images
# sphinx_gallery_thumbnail_number = 3
fig1 = plt.figure()
spec3 = gridspec.GridSpec(ncols = nrn_ID.shape[0], nrows = 1)
for i in range(0,nrn_ID.shape[0]):
    fig1.add_subplot(spec3[i])
    plt.imshow(img_spiked[i,:,:], cmap=cm.Greys_r)
    plt.title('ID {}'.format(nrn_ID[i]))
plt.show()

######################################################################

# Print model statistics
print("Model statistics")
for _, stat in stats.items():
    print(stat)
