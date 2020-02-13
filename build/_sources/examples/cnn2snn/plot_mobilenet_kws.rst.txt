.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_cnn2snn_plot_mobilenet_kws.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_cnn2snn_plot_mobilenet_kws.py:


Inference on KWS with MobileNet
===============================


This tutorial illustrates how to build a basic speech recognition
Akida network that recognizes ten different words.

The model will be first defined as a CNN and trained in Keras, then
converted using the `CNN2SNN toolkit <../../user_guide/cnn2snn.html>`__.

This example uses a Keyword Spotting Dataset prepared using
**TensorFlow** `audio recognition
example <https://www.tensorflow.org/tutorials/sequences/audio_recognition>`__
utils.

The words to recognize are first converted to `spectrogram
images <https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md#how-does-this-model-work>`__
that allows us to use a model architecture that is typically used for
image recognition tasks.

Please refer to `Load and reshape Keyword Spotting dataset (KWS)
<../../examples/cnn2snn/kws_dataset.html>`__ example for details about the
dataset preparation.

1. Load CNN2SNN tool dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code-block:: default


    # System imports
    import os
    import sys
    import numpy as np
    import pickle
    from sklearn.metrics import accuracy_score
    import itertools
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    # TensorFlow imports
    import tensorflow.keras.backend as K
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.utils import get_file

    # KWS model imports
    from akida_models import mobilenet_kws









2. Load the preprocessed dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code-block:: default


    wanted_words = ['down','go','left','no','off','on','right','stop','up','yes']
    all_words = ['_silence_','_unknown_'] + wanted_words

    # Preprocessed dataset parameters
    CHANNELS = 1
    CLASSES = len(all_words)
    SPECTROGRAM_LENGTH = 49
    FINGERPRINT_WIDTH = 10

    input_shape = (SPECTROGRAM_LENGTH, FINGERPRINT_WIDTH, CHANNELS)

    # Try to load pre-processed dataset
    fname = get_file("preprocessed_data.pkl",
                     "http://data.brainchip.com/dataset-mirror/kws/preprocessed_data.pkl",
                     cache_subdir='datasets/kws')
    if os.path.isfile(fname):
        print('Re-loading previously preprocessed dataset...')
        f = open(fname, 'rb')
        [x_train, y_train, x_valid, y_valid, train_files, val_files, word_to_index] = pickle.load(f)
        f.close()
    else:
        raise ValueError("Unable to load the pre-processed KWS dataset.")

    # Transform the data to uint8
    x_train_min = x_train.min()
    x_train_max = x_train.max()
    max_int_value = 255.0

    # For akida hardware training and validation range [0, 255] inclusive uint8
    x_train_akida = ((x_train-x_train_min) * max_int_value / (x_train_max - x_train_min)).astype(np.uint8)
    x_valid_akida = ((x_valid-x_train_min) * max_int_value / (x_train_max - x_train_min)).astype(np.uint8)

    # For cnn2snn training and validation range [0,1] inclusive float32
    x_train_rescaled_cnn = (x_train_akida.astype(np.float32))/max_int_value
    x_valid_rescaled_cnn = (x_valid_akida.astype(np.float32))/max_int_value

    input_scaling = (max_int_value, 0)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Downloading data from http://data.brainchip.com/dataset-mirror/kws/preprocessed_data.pkl
         8192/175810841 [..............................] - ETA: 5:50        81920/175810841 [..............................] - ETA: 2:22       614400/175810841 [..............................] - ETA: 33s       1900544/175810841 [..............................] - ETA: 42s      4562944/175810841 [..............................] - ETA: 19s      4759552/175810841 [..............................] - ETA: 20s      5742592/175810841 [..............................] - ETA: 18s      7143424/175810841 [>.............................] - ETA: 15s      8650752/175810841 [>.............................] - ETA: 13s     10264576/175810841 [>.............................] - ETA: 12s     12001280/175810841 [=>............................] - ETA: 11s     13778944/175810841 [=>............................] - ETA: 10s     15474688/175810841 [=>............................] - ETA: 9s      17252352/175810841 [=>............................] - ETA: 8s     19120128/175810841 [==>...........................] - ETA: 8s     21045248/175810841 [==>...........................] - ETA: 7s     23019520/175810841 [==>...........................] - ETA: 7s     25018368/175810841 [===>..........................] - ETA: 7s     27099136/175810841 [===>..........................] - ETA: 6s     29310976/175810841 [====>.........................] - ETA: 6s     31662080/175810841 [====>.........................] - ETA: 6s     34095104/175810841 [====>.........................] - ETA: 5s     36511744/175810841 [=====>........................] - ETA: 5s     39059456/175810841 [=====>........................] - ETA: 5s     41771008/175810841 [======>.......................] - ETA: 4s     44564480/175810841 [======>.......................] - ETA: 4s     47316992/175810841 [=======>......................] - ETA: 4s     50053120/175810841 [=======>......................] - ETA: 4s     52789248/175810841 [========>.....................] - ETA: 4s     55590912/175810841 [========>.....................] - ETA: 3s     58310656/175810841 [========>.....................] - ETA: 3s     60940288/175810841 [=========>....................] - ETA: 3s     63545344/175810841 [=========>....................] - ETA: 3s     65757184/175810841 [==========>...................] - ETA: 3s     67788800/175810841 [==========>...................] - ETA: 3s     69869568/175810841 [==========>...................] - ETA: 3s     71974912/175810841 [===========>..................] - ETA: 3s     74145792/175810841 [===========>..................] - ETA: 2s     76292096/175810841 [============>.................] - ETA: 2s     78381056/175810841 [============>.................] - ETA: 2s     80453632/175810841 [============>.................] - ETA: 2s     82640896/175810841 [=============>................] - ETA: 2s     84918272/175810841 [=============>................] - ETA: 2s     87302144/175810841 [=============>................] - ETA: 2s     89710592/175810841 [==============>...............] - ETA: 2s     92192768/175810841 [==============>...............] - ETA: 2s     94748672/175810841 [===============>..............] - ETA: 2s     97443840/175810841 [===============>..............] - ETA: 2s    100188160/175810841 [================>.............] - ETA: 2s    102916096/175810841 [================>.............] - ETA: 1s    105529344/175810841 [=================>............] - ETA: 1s    108101632/175810841 [=================>............] - ETA: 1s    110804992/175810841 [=================>............] - ETA: 1s    113532928/175810841 [==================>...........] - ETA: 1s    116203520/175810841 [==================>...........] - ETA: 1s    118824960/175810841 [===================>..........] - ETA: 1s    121413632/175810841 [===================>..........] - ETA: 1s    124084224/175810841 [====================>.........] - ETA: 1s    126722048/175810841 [====================>.........] - ETA: 1s    129359872/175810841 [=====================>........] - ETA: 1s    130031616/175810841 [=====================>........] - ETA: 1s    131850240/175810841 [=====================>........] - ETA: 1s    133193728/175810841 [=====================>........] - ETA: 1s    134545408/175810841 [=====================>........] - ETA: 1s    135938048/175810841 [======================>.......] - ETA: 1s    137347072/175810841 [======================>.......] - ETA: 1s    138780672/175810841 [======================>.......] - ETA: 0s    140296192/175810841 [======================>.......] - ETA: 0s    141910016/175810841 [=======================>......] - ETA: 0s    143597568/175810841 [=======================>......] - ETA: 0s    145358848/175810841 [=======================>......] - ETA: 0s    147111936/175810841 [========================>.....] - ETA: 0s    148807680/175810841 [========================>.....] - ETA: 0s    150511616/175810841 [========================>.....] - ETA: 0s    152256512/175810841 [========================>.....] - ETA: 0s    154075136/175810841 [=========================>....] - ETA: 0s    156008448/175810841 [=========================>....] - ETA: 0s    158040064/175810841 [=========================>....] - ETA: 0s    160202752/175810841 [==========================>...] - ETA: 0s    162488320/175810841 [==========================>...] - ETA: 0s    164814848/175810841 [===========================>..] - ETA: 0s    167100416/175810841 [===========================>..] - ETA: 0s    169279488/175810841 [===========================>..] - ETA: 0s    171532288/175810841 [============================>.] - ETA: 0s    173867008/175810841 [============================>.] - ETA: 0s    175816704/175810841 [==============================] - 5s 0us/step
    Re-loading previously preprocessed dataset...




3. Create a Keras model satisfying Akida NSoC requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model consists of:

* a first Convolutional layer accepting dense inputs (images),
* several Separable Convolutional layers preserving spatial dimensions,
* a global pooling reducing the spatial dimensions to a single pixel,
* a last Separable Convolutional layer to reduce the number of outputs
  to the number of words to predict.

All layers are followed by a batch normalization and a ReLU activation,
except the last one that is followed by a SoftMax.

The first convolutional layer uses 8 bits weights, but other layers use
4 bits weights.

All activations are 4 bits.

.. Note:: The reason why we do not use a simple FullyConnected layer as the
          last layer is precisely because of the 4 bits activations, that are
          only supported as inputs by the Separable Convolutional layers.

Pre-trained weights were obtained after three training episodes:

* first, we train the model with unconstrained float weights and
  activations for 30 epochs,
* then, we train the model with quantized activations only, with
  weights initialized from those trained in the previous episode,
* finally, we train the model with quantized weights and activations,
  with weights initialized from those trained in the previous episode.

The table below summarizes the results obtained when preparing the
weights stored under `<http://data.brainchip.com/models/mobilenet/>`__ :

+---------+----------------+---------------+----------+--------+
| Episode | Weights Quant. | Activ. Quant. | Accuracy | Epochs |
+=========+================+===============+==========+========+
| 1       | N/A            | N/A           | 91.98 %  | 30     |
+---------+----------------+---------------+----------+--------+
| 2       | N/A            | 4 bits        | 92.13 %  | 30     |
+---------+----------------+---------------+----------+--------+
| 3       | 8/4 bits       | 4 bits        | 91.67 %  | 30     |
+---------+----------------+---------------+----------+--------+



.. code-block:: default


    K.clear_session()
    model_keras = mobilenet_kws(input_shape,
                                classes=CLASSES,
                                weights='kws',
                                weights_quantization=4,
                                activ_quantization=4,
                                input_weights_quantization=8)
    model_keras.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    WARNING: Keyword argument 'strides' is not supported in conv_block except for the first layer.
    Downloading data from http://data.brainchip.com/models/mobilenet/mobilenet_kws_wq4_aq4.hdf5
      8192/156592 [>.............................] - ETA: 0s     81920/156592 [==============>...............] - ETA: 0s    163840/156592 [===============================] - 0s 0us/step
    Model: "mobilenet_kws"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 49, 10, 1)]       0         
    _________________________________________________________________
    conv_0 (QuantizedConv2D)     (None, 25, 5, 32)         800       
    _________________________________________________________________
    conv_0_BN (BatchNormalizatio (None, 25, 5, 32)         128       
    _________________________________________________________________
    conv_0_relu (ActivationDiscr (None, 25, 5, 32)         0         
    _________________________________________________________________
    separable_1 (QuantizedSepara (None, 25, 5, 64)         2336      
    _________________________________________________________________
    separable_1_BN (BatchNormali (None, 25, 5, 64)         256       
    _________________________________________________________________
    separable_1_relu (Activation (None, 25, 5, 64)         0         
    _________________________________________________________________
    separable_2 (QuantizedSepara (None, 25, 5, 64)         4672      
    _________________________________________________________________
    separable_2_BN (BatchNormali (None, 25, 5, 64)         256       
    _________________________________________________________________
    separable_2_relu (Activation (None, 25, 5, 64)         0         
    _________________________________________________________________
    separable_3 (QuantizedSepara (None, 25, 5, 64)         4672      
    _________________________________________________________________
    separable_3_BN (BatchNormali (None, 25, 5, 64)         256       
    _________________________________________________________________
    separable_3_relu (Activation (None, 25, 5, 64)         0         
    _________________________________________________________________
    separable_4 (QuantizedSepara (None, 25, 5, 64)         4672      
    _________________________________________________________________
    separable_4_BN (BatchNormali (None, 25, 5, 64)         256       
    _________________________________________________________________
    separable_4_relu (Activation (None, 25, 5, 64)         0         
    _________________________________________________________________
    separable_5 (QuantizedSepara (None, 25, 5, 64)         4672      
    _________________________________________________________________
    separable_5_global_avg (Glob (None, 64)                0         
    _________________________________________________________________
    separable_5_BN (BatchNormali (None, 64)                256       
    _________________________________________________________________
    separable_5_relu (Activation (None, 64)                0         
    _________________________________________________________________
    reshape_1 (Reshape)          (None, 1, 1, 64)          0         
    _________________________________________________________________
    separable_6 (QuantizedSepara (None, 1, 1, 12)          1344      
    _________________________________________________________________
    act_softmax (Activation)     (None, 1, 1, 12)          0         
    _________________________________________________________________
    reshape_2 (Reshape)          (None, 12)                0         
    =================================================================
    Total params: 24,576
    Trainable params: 23,872
    Non-trainable params: 704
    _________________________________________________________________




4. Check performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: default


    # Check Model performance
    potentials_keras = model_keras.predict(x_valid_rescaled_cnn)
    preds_keras = np.squeeze(np.argmax(potentials_keras, 1))

    accuracy = accuracy_score(y_valid, preds_keras)
    print("Accuracy: "+"{0:.2f}".format(100*accuracy)+"%")






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Accuracy: 91.62%




5. Conversion to Akida
~~~~~~~~~~~~~~~~~~~~~~

5.1 Convert the trained Keras model to Akida
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We convert the model to Akida and verify that it is compatible with the
Akida NSoC (**HW** column in summary).



.. code-block:: default


    # Convert the model
    from cnn2snn import convert

    model_akida = convert(model_keras, input_scaling=input_scaling)
    model_akida.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    -------------------------------------------------------------------------------------------------------------------------
    Layer (type)           HW  Input shape   Output shape  Kernel shape  Learning (#classes)       #InConn/#Weights/ThFire   
    =========================================================================================================================
    conv_0 (InputConvoluti yes [10, 49, 1]   [5, 25, 32]   (5 x 5 x 1)   N/A                       25 / 24 / 0               
    -------------------------------------------------------------------------------------------------------------------------
    separable_1 (Separable yes [5, 25, 32]   [5, 25, 64]   (3 x 3 x 32)  N/A                       288 / 20 / 0              
    -------------------------------------------------------------------------------------------------------------------------
    separable_2 (Separable yes [5, 25, 64]   [5, 25, 64]   (3 x 3 x 64)  N/A                       576 / 30 / 0              
    -------------------------------------------------------------------------------------------------------------------------
    separable_3 (Separable yes [5, 25, 64]   [5, 25, 64]   (3 x 3 x 64)  N/A                       576 / 30 / 0              
    -------------------------------------------------------------------------------------------------------------------------
    separable_4 (Separable yes [5, 25, 64]   [5, 25, 64]   (3 x 3 x 64)  N/A                       576 / 30 / 0              
    -------------------------------------------------------------------------------------------------------------------------
    separable_5 (Separable yes [5, 25, 64]   [1, 1, 64]    (3 x 3 x 64)  N/A                       576 / 30 / 0              
    -------------------------------------------------------------------------------------------------------------------------
    separable_6 (Separable yes [1, 1, 64]    [1, 1, 12]    (3 x 3 x 64)  N/A                       576 / 15 / 0              
    -------------------------------------------------------------------------------------------------------------------------




5.2 Check prediction accuracy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code-block:: default


    preds_akida = model_akida.predict(x_valid_akida, num_classes = CLASSES)

    accuracy = accuracy_score(y_valid, preds_akida)
    print("Accuracy: "+"{0:.2f}".format(100*accuracy)+"%")

    # For non-regression purpose
    assert accuracy > 0.83

    # Print model statistics
    print("Model statistics")
    stats = model_akida.get_statistics()
    model_akida.predict(x_valid_akida[:20], num_classes = CLASSES)
    for _, stat in stats.items():
        print(stat)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Accuracy: 91.60%
    Model statistics
    Layer (type)                  output sparsity     
    conv_0 (InputConvolutional)   0.47                
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_1 (SeparableConvolu 0.47                0.55                1240873             
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_2 (SeparableConvolu 0.55                0.59                2127616             
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_3 (SeparableConvolu 0.59                0.64                1918741             
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_4 (SeparableConvolu 0.64                0.67                1705860             
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_5 (SeparableConvolu 0.67                0.46                1556217             
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_6 (SeparableConvolu 0.46                0.00                4066                




5.3 Confusion matrix
^^^^^^^^^^^^^^^^^^^^

The confusion matrix provides a good summary of what mistakes the
network is making.

Per scikit-learn convention it displays the true class in each row (ie
on each row you can see what the network predicted for the corresponding
word).

Please refer to the Tensorflow `audio
recognition <https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md#confusion-matrix>`__
example for a detailed explaination of the confusion matrix.



.. code-block:: default


    # Create confusion matrix
    label_mapping = dict(zip(all_words, range(len(all_words))))

    cm = confusion_matrix(y_valid, preds_akida, list(label_mapping.values()))

    # Normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Display confusion matrix
    plt.rcParams["figure.figsize"] = (8,8)
    plt.figure()

    classes=label_mapping
    title='Confusion matrix'
    cmap = plt.cm.Blues

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.autoscale()
    plt.show()



.. image:: /examples/cnn2snn/images/sphx_glr_plot_mobilenet_kws_001.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  8.079 seconds)


.. _sphx_glr_download_examples_cnn2snn_plot_mobilenet_kws.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_mobilenet_kws.py <plot_mobilenet_kws.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_mobilenet_kws.ipynb <plot_mobilenet_kws.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
