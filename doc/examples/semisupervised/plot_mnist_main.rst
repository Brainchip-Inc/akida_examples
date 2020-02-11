.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_semisupervised_plot_mnist_main.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_semisupervised_plot_mnist_main.py:


Learning and inference on MNIST
===============================

The MNIST dataset is a handwritten digits database. It has a training
set of 60,000 images, and a test set of 10,000 images. An image has
28x28 pixels (784 features) and an associated label.

With this tutorial you will:

    * load the `MNIST dataset <http://yann.lecun.com/exdb/mnist/>`__
    * load an Akida Execution Engine configuration
    * train your neural model with the training set (unsupervised, using
      embedded spike generator) and check its performance

1. Loading the MNIST dataset
----------------------------


.. code-block:: default


    # Various imports needed for the tutorial
    import os
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from tensorflow.keras.datasets import mnist
    from sklearn.metrics import accuracy_score
    import warnings

    # Akida specific imports
    from akida import Model, InputBCSpike, FullyConnected, LearningType









.. code-block:: default


    #  Retrieve MNIST dataset
    (train_set, train_label), (test_set, test_label) = mnist.load_data()
    # add a dimension to images sets as akida expects 4 dimensions inputs
    train_set = np.expand_dims(train_set, -1)
    test_set = np.expand_dims(test_set, -1)









2. Look at some images from the dataset
---------------------------------------


.. code-block:: default


    # Display a few images from the train set
    f, axarr = plt.subplots(1, 4)
    for i in range (0, 4):
        axarr[i].imshow(train_set[i].reshape((28,28)), cmap=cm.Greys_r)
        axarr[i].set_title('Class %d' % train_label[i])
    plt.show()
    print ('Note that an MNIST image is a %s '  % (train_set[0].shape,) + "numpy array (filled with 8 bit values, i.e. grayscale)")





.. image:: /examples/semisupervised/images/sphx_glr_plot_mnist_main_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Note that an MNIST image is a (28, 28, 1) numpy array (filled with 8 bit values, i.e. grayscale)




3. Configuring Akida model
--------------------------

A neural network model can be sequentially defined. Check the `Akida
Execution Engine documentation <../../api_reference/aee_apis.html>`__ for a
full description of the parameters and layer types available.
Note that first layer matches MNIST image properties (InputBCSpike with
input_width: 28 and input_height: 28)


.. code-block:: default


    #Create a model
    model = Model()
    model.add(InputBCSpike("inputBC", input_width=28, input_height=28))
    fully = FullyConnected("fully", num_neurons=1000, activations_enabled=False)
    model.add(fully)
    # Configure the last layer for semi-supervised training
    fully.compile(num_weights=500, num_classes=10)
    model.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    -------------------------------------------------------------------------------------------------------------------------
    Layer (type)           HW  Input shape   Output shape  Kernel shape  Learning (#classes)       #InConn/#Weights/ThFire   
    =========================================================================================================================
    inputBC (InputConvolut yes [28, 28, 1]   [24, 24, 8]   (5 x 5 x 1)   N/A                       25 / 20 / 0               
    -------------------------------------------------------------------------------------------------------------------------
    fully (FullyConnected) yes [24, 24, 8]   [1, 1, 1000]  N/A           akidaUnsupervised (10)    4608 / 500 / 0            
    -------------------------------------------------------------------------------------------------------------------------




4. Testing performance
----------------------

The Akida Execution Engine provides a simple performance routine. We can
try a test of baseline performance without any training:


.. code-block:: default


    # Dumb try with an untrained model ...
    num_samples = 10000

    stats = model.get_statistics()
    pred_label = model.predict(test_set[:int(num_samples)], 10)
    accuracy = accuracy_score(test_label[:num_samples], pred_label[:num_samples])

    print("Accuracy: "+"{0:.2f}".format(100*accuracy)+"%")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Accuracy: 10.62%





.. code-block:: default


    # Print model statistics
    print("Model statistics")
    for _, stat in stats.items():
        print(stat)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Model statistics
    Layer (type)                  output sparsity     
    inputBC (InputConvolutional)  0.93                
    Layer (type)                  input sparsity      output sparsity     ops                 
    fully (FullyConnected)        0.93                0.00                335034              




5. Learning and inference
-------------------------

Let's train the model on the MNIST training dataset (60 000 images
available). Out of interest, we can run a performance test at regular
intervals, to see how it evolves as a function of the number of training
samples (and remember that the end of the training run here corresponds
to just one training 'epoch').


.. code-block:: default


    # Routine to update intermediate results plot
    def plot_update(performance, histogram, axes, figure, nb_samples):

        # Update performance subplot
        axes[0].plot(nb_samples, 100*performance, 'b.')
        axes[0].set(xlabel='\nTraining samples: '+str(nb_samples),
                    ylabel='Accuracy: '+'{0:.2f}'.format(100*performance))
        # Update learning rate subplot
        x_hist = [x[0] for x in histogram]
        y_hist = [x[1] for x in histogram]
        axes[1].plot(x_hist, y_hist, dash_joinstyle='round')

        figure.canvas.draw()
        return









.. code-block:: default


    # Check model performance every 'checkpoint' samples
    in_images = train_set
    checkpoints = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000]
    counter = 0

    # Change Matplotlib backend for dynamic display
    # %matplotlib notebook

    # Adjust Plot parameters
    plt.rcParams['axes.grid'] = True
    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(wspace = 0.5, right=0.9)

    # Set axis limit values and labels
    for n, subplot in np.ndenumerate(ax):
        subplot.set_xlim(0, 100)
        subplot.set_ylim(0, 100)
    ax[0].set_xlim(0, checkpoints[-1])
    ax[0].set(xlabel='\nTraining samples: ', ylabel='Accuracy', aspect=checkpoints[-1]/100)
    ax[0].tick_params(bottom=False, labelbottom=False)
    ax[1].set(xlabel='Neuron learning rate', ylabel='Number of neurons', aspect=100/100)

    fig.canvas.draw()

    # Get a reference to the layer for data extraction
    fully = model.get_layer('fully')

    # Start learning and plot performances all along
    for i in range(in_images.shape[0]):
        model.fit(in_images[i:i+1], input_labels=train_label[i])
        # Plot intermediate accuracy and learning rate for the defined checkpoints
        if counter < len(checkpoints) and i == (checkpoints[counter]):
            pred_label = model.predict(test_set[:int(num_samples)], 10)
            accuracy = accuracy_score(test_label[:num_samples], pred_label[:num_samples])
            hist       = fully.get_learning_histogram()
            plot_update(accuracy, hist, ax, fig, i)
            counter+=1

        # Then plot learning rate every 10000 samples
        elif (i+1)%10000 == 0:
            pred_label = model.predict(test_set[:int(num_samples)], 10)
            accuracy = accuracy_score(test_label[:num_samples], pred_label[:num_samples])
            hist       = fully.get_learning_histogram()
            plot_update(accuracy, hist, ax, fig, i+1)

    print("Accuracy: "+"{0:.2f}".format(100*accuracy)+"%")

    # For non-regression purpose
    assert accuracy > 0.93



.. image:: /examples/semisupervised/images/sphx_glr_plot_mnist_main_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Accuracy: 93.81%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  15.456 seconds)


.. _sphx_glr_download_examples_semisupervised_plot_mnist_main.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_mnist_main.py <plot_mnist_main.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_mnist_main.ipynb <plot_mnist_main.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
