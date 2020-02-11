.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_cnn2snn_plot_mnist_cnn2akida_main.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_cnn2snn_plot_mnist_cnn2akida_main.py:


Inference on MNIST
==================

The Akida Execution Engine includes a powerful native learning
algorithm. However, it is also possible to train an Akida-compatible
model externally, using specialized deep-learning techniques, and to
then implement that model within the Akida Execution Engine as an
efficient inference-only tool using the `CNN2SNN
toolkit <../../user_guide/cnn2snn.html>`__. In this tutorial, you will
simply load one such pre-trained model, use it to process the MNIST
dataset, and look at how to make sense of the outputs.

The MNIST dataset is a handwritten digits database. It has a training
set of 60,000 samples, and a test set of 10,000 samples. Each sample
comprises a 28x28 pixel image and an associated label.

**In this tutorial you will:**

    * load the MNIST test dataset only,
    * load a pre-trained neural network model,
    * run test samples through the model in inference-only mode (i.e. without
      any further learning),
    * check performance.

1. Loading the MNIST dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code-block:: default


    # Various imports needed for the tutorial
    import os
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import warnings
    from tensorflow.keras.utils import get_file
    from tensorflow.keras.datasets import mnist
    from sklearn.metrics import f1_score, accuracy_score

    # Filter warnings
    warnings.filterwarnings("ignore", module="matplotlib")

    # Akida specific imports
    from akida import Model









.. code-block:: default


    # Retrieve MNIST dataset
    (train_set, train_label), (test_set, test_label) = mnist.load_data()

    # Add a dimension to images sets as akida expects 4 dimensions inputs
    train_set = np.expand_dims(train_set, -1)
    test_set = np.expand_dims(test_set, -1)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
        8192/11490434 [..............................] - ETA: 0s     1359872/11490434 [==>...........................] - ETA: 0s     4358144/11490434 [==========>...................] - ETA: 0s     7839744/11490434 [===================>..........] - ETA: 0s    11116544/11490434 [============================>.] - ETA: 0s    11493376/11490434 [==============================] - 0s 0us/step




2. Look at some images from the test dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code-block:: default


    # Display a few images from the test set
    f, axarr = plt.subplots(1, 4)
    for i in range (0, 4):
        axarr[i].imshow(test_set[i].reshape((28,28)), cmap=cm.Greys_r)
        axarr[i].set_title('Class %d' % test_label[i])
    plt.show()





.. image:: /examples/cnn2snn/images/sphx_glr_plot_mnist_cnn2akida_main_001.png
    :class: sphx-glr-single-img





3. Load the pre-trained Akida model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pre-trained neural network model is included in the models/cnn2snn
directory. You only need to pass this .fbz file to the Akida Execution
Engine in order to instantiate the model.



.. code-block:: default


    # Load provided model configuration file
    model_file = get_file("gxnor_mnist.fbz",
                          "http://data.brainchip.com/models/gxnor/gxnor_mnist.fbz",
                          cache_subdir='models/gxnor')
    model_akida = Model(model_file)
    print (model_file + ' loaded...')
    model_akida.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Downloading data from http://data.brainchip.com/models/gxnor/gxnor_mnist.fbz
      8192/731972 [..............................] - ETA: 1s     81920/731972 [==>...........................] - ETA: 0s    622592/731972 [========================>.....] - ETA: 0s    737280/731972 [==============================] - 0s 0us/step
    /root/.keras/models/gxnor/gxnor_mnist.fbz loaded...
    -------------------------------------------------------------------------------------------------------------------------
    Layer (type)           HW  Input shape   Output shape  Kernel shape  Learning (#classes)       #InConn/#Weights/ThFire   
    =========================================================================================================================
    conv_0_conv (InputConv yes [28, 28, 1]   [14, 14, 32]  (5 x 5 x 1)   N/A                       25 / 16 / 0               
    -------------------------------------------------------------------------------------------------------------------------
    conv_1_conv (Convoluti yes [14, 14, 32]  [7, 7, 64]    (5 x 5 x 32)  N/A                       800 / 476 / 0             
    -------------------------------------------------------------------------------------------------------------------------
    block2_dense (FullyCon yes [7, 7, 64]    [1, 1, 512]   N/A           N/A                       3136 / 1937 / 0           
    -------------------------------------------------------------------------------------------------------------------------
    block3_dense (FullyCon yes [1, 1, 512]   [1, 1, 10]    N/A           N/A                       512 / 327 / 0             
    -------------------------------------------------------------------------------------------------------------------------




4. Classify a single image
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now try processing a single image, say, the first image in the dataset
that we looked at above:



.. code-block:: default


    # Test a single example
    sample_image = 0
    image = test_set[sample_image]
    outputs = model_akida.evaluate(image.reshape(1,28,28,1))
    print('Input Label: %i' % test_label[sample_image])

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(test_set[sample_image].reshape((28,28)), cmap=cm.Greys_r)
    axarr[0].set_title('Class %d' % test_label[sample_image])
    axarr[1].bar(range(10),outputs.squeeze())
    axarr[1].set_xticks(range(10))
    plt.show()

    print(outputs.squeeze())





.. image:: /examples/cnn2snn/images/sphx_glr_plot_mnist_cnn2akida_main_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Input Label: 7
    [-31. -44. -52. -36. -32. -44. -21.  44. -54. -48.]




Consider the output from the model, printed above. As is typical in
backprop trained models, the final layer here comprises a
'fully-connected or 'dense' layer, with one neuron per class in the
data (here, 10). The goal of training is to maximize the response of the
neuron corresponding to the label of each training sample, while
minimizing the responses of the other neurons.

In the bar chart above, you can see the outputs from all 10 neurons. It
is easy to see that neuron 7 responds much more strongly than the
others. The first sample is indeed a number 7.

Check this for some of the other samples by editing the value of
sample_image in the script above (anything from 0 to 9999).


5. Check performance across a number of samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We've included a utility to test performance across a large number of
samples. You can run this below.



.. code-block:: default


    # Check performance against num_samples samples
    num_samples = 10000

    results = model_akida.predict(test_set[:int(num_samples)], 10)
    accuracy = accuracy_score(test_label[:num_samples], results[:num_samples])
    f1 = f1_score(test_label[:num_samples],
                   results[:num_samples],
                   average='weighted')

    # For non-regression purpose
    assert accuracy > 0.99

    # Print model statistics
    print("Model statistics")
    stats = model_akida.get_statistics()
    model_akida.predict(test_set[:20], 10)
    for _, stat in stats.items():
        print(stat)

    # Display results
    print("Accuracy: "+"{0:.2f}".format(100*accuracy)+"% / "
           +"F1 score: "+"{0:.2f}".format(f1))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Model statistics
    Layer (type)                  output sparsity     
    conv_0_conv (InputConvolution 0.83                
    Layer (type)                  input sparsity      output sparsity     ops                 
    conv_1_conv (Convolutional)   0.83                0.84                1658320             
    Layer (type)                  input sparsity      output sparsity     ops                 
    block2_dense (FullyConnected) 0.84                0.83                251571              
    Layer (type)                  input sparsity      output sparsity     ops                 
    block3_dense (FullyConnected) 0.83                0.00                861                 
    Accuracy: 99.40% / F1 score: 0.99




Depending on the number of samples you run, you should find a
performance of around 99% (99.35% if you run all 10000 samples).

Note that classification here is done simply by identifying the neuron
with the highest activation level. Slightly higher performance is
actually possible for this model implementation (~99.1 %) if a very
slightly more complex final classification is applied (with a single
additional integer subtraction per neuron), but for simplicity we leave
those details aside here. See the cnn2snn training framework for a full
description.



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  3.817 seconds)


.. _sphx_glr_download_examples_cnn2snn_plot_mnist_cnn2akida_main.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_mnist_cnn2akida_main.py <plot_mnist_cnn2akida_main.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_mnist_cnn2akida_main.ipynb <plot_mnist_cnn2akida_main.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
