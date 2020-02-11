.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_cnn2snn_plot_cats_vs_dogs_cnn2akida_demo.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_cnn2snn_plot_cats_vs_dogs_cnn2akida_demo.py:


Transfer learning with MobileNet for cats vs. dogs
==================================================

This tutorial presents a demonstration of transfer learning and the
conversion to an Akida model of a quantized Keras network.

The transfer learning example is derived from the `Tensorflow
tutorial <https://www.tensorflow.org/tutorials/images/transfer_learning>`__:

    * Our base model is an Akida-compatible version of **MobileNet v1**,
      trained on ImageNet.
    * The new dataset for transfer learning is **cats vs. dogs**
      (`link <https://www.tensorflow.org/datasets/catalog/cats_vs_dogs>`__).
    * We use transfer learning to customize the model to the new task of
      classifying cats and dogs.

.. Note:: This tutorial only shows the inference of the trained Keras
          model and its conversion to an Akida network. A textual explanation
          of the training is given below.

This tutorial goes as follows:

    1. `Details of the transfer learning process <cats_vs_dogs_cnn2akida_demo.html#transfer-learning-process>`__
    2. `Load and preprocess data <cats_vs_dogs_cnn2akida_demo.html#load-and-preprocess-data>`__

        1. Load and split data
        2. Preprocess the test set
        3. Get labels
    3. `Convert a quantized Keras model to Akida <cats_vs_dogs_cnn2akida_demo.html#convert-a-quantized-keras-model-to-akida>`__

        1. Instantiate a Keras model
        2. Change top layer
        3. Convert to an Akida model
    4. `Compare performance <cats_vs_dogs_cnn2akida_demo.html#classify-test-images>`__

            1. Classify test images
            2. Compare results

1. Transfer learning process
----------------------------
.. figure:: https://s2.qwant.com/thumbr/0x380/7/0/7b7386531ea24ab1294fdf9b8698b008a51e38a3c57e81427fbef626ff226c/1*6ACbDsBMeDZcLg9W8CFT_Q.png?u=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F1600%2F1%2A6ACbDsBMeDZcLg9W8CFT_Q.png&q=0&b=1&p=0&a=1
   :alt: transfer_learning_image
   :target: https://s2.qwant.com/thumbr/0x380/7/0/7b7386531ea24ab1294fdf9b8698b008a51e38a3c57e81427fbef626ff226c/1*6ACbDsBMeDZcLg9W8CFT_Q.png?u=https%3A%2F%2Fcdn-images-1.medium.com%2Fmax%2F1600%2F1%2A6ACbDsBMeDZcLg9W8CFT_Q.png&q=0&b=1&p=0&a=1
   :align: center

Transfer learning allows to classify on a specific task by using a
pre-trained base model. For an introduction to transfer learning, please
refer to the `Tensorflow
tutorial <https://www.tensorflow.org/tutorials/images/transfer_learning>`__
before exploring this tutorial. Here, we focus on how to quantize the
Keras model in order to convert it to an Akida one.

The model is composed of:

  * a base quantized MobileNet model used to extract image features
  * a top layer to classify cats and dogs
  * a sigmoid activation function to interpret model outputs as a probability

**Base model**

The base model is an Akida-compatible version of MobileNet v1. This
model was trained and quantized using the ImageNet dataset. Please refer
to the corresponding `example <imagenet_cnn2akida_demo.html>`__ for
more information. The layers have 4-bit weights (except for the first
layer having 8-bit weights) and the activations are quantized to 4 bits.
This base model ends with a global average pooling whose output is (1,
1, 1024).

In our transfer learning process, the base model is frozen, i.e., the
weights are not updated during training. Pre-trained weights for the
quantized model are provided in the ``data/imagenet`` folder. These are
loaded in our frozen base model.

**Top layer**

While the Tensorflow tutorial uses a fully-connected top layer with one
output neuron, the only Akida layer supporting 4-bit weights is a separable
convolutional layer (see `hardware compatibility
<../../user_guide/hw_constraints.html>`__).

We thus decided to use a separable convolutional layer with one output
neuron for the top layer of our model.

**Final activation**

ReLU6 is the only activation function that can be converted into an Akida SNN
equivalent. The converted Akida model doesn't therefore include the 'sigmoid'
activation, and we must instead apply it explicitly on the raw values returned
by the model Top layer.

**Training steps**

The transfer learning process consists in two training phases:

  1. **Float top layer training**: The base model is quantized using 4-bit
     weights and activations. Pre-trained 4-bit weights of MobileNet/ImageNet
     are loaded. Then a top layer is added with float weights. The base model
     is frozen and the training is only applied on the top layer. After 10
     epochs, the weights are saved. Note that the weights of the layers of
     the frozen base model haven't changed; only those of the top layer are
     updated.
  2. **4-bit top layer training**: The base model is still
     quantized using 4-bit weights and activations. The added top layer is
     now quantized (4-bit weights). The weights saved at step 1 are used as
     initialization. The base model is frozen and the training is only
     applied on the top layer. After 10 epochs, the new quantized weights are
     saved. This final weights are those used in the inference below.

+----------+-------------------+------------+---------+------------+----+
| Training | Frozen base model | Init.      | Top     | Init.      | E  |
| step     |                   | weights    | layer   | weights    | p  |
|          |                   | base model |         | top layer  | o  |
|          |                   |            |         |            | c  |
|          |                   |            |         |            | h  |
|          |                   |            |         |            | s  |
+==========+===================+============+=========+============+====+
| step 1   | 4-bit weights /   | pre-trained| float   | random     | 10 |
|          | activations       | 4-bit      | weights |            |    |
|          |                   |            |         |            |    |
+----------+-------------------+------------+---------+------------+----+
| step 2   | 4-bit weights /   | pre-trained| 4-bit   | saved from | 10 |
|          | activations       | 4-bit      | weights | step 1     |    |
|          |                   |            |         |            |    |
+----------+-------------------+------------+---------+------------+----+


.. code-block:: default


    import os
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import matplotlib.pyplot as plt

    from akida_models import mobilenet_imagenet
    from cnn2snn import convert
    from akida_models.quantization_blocks import separable_conv_block









2. Load and preprocess data
---------------------------

In this section, we will load the 'cats_vs_dogs' dataset preprocess
the data to match the required model's inputs:

  * **2.A - Load and split data**: we only keep the test set which represents
    10% of the dataset.
  * **2.B - Preprocess the test set** by resizing and rescaling the images.
  * **2.C - Get labels**

2.A - Load and split data
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``cats_vs_dogs``
`dataset <https://www.tensorflow.org/datasets/catalog/cats_vs_dogs>`__
is loaded and split into train, validation and test sets. The train and
validation sets were used for the transfer learning process. Here only
the test set is used. We use here ``tf.Dataset`` objects to load and
preprocess batches of data (one can look at the TensorFlow guide
`here <https://www.tensorflow.org/guide/data>`__ for more information).

.. Note:: The ``cats_vs_dogs`` dataset version used here is 2.0.1.



.. code-block:: default


    SPLIT_WEIGHTS = (8, 1, 1)
    splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

    tfds.disable_progress_bar()
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        'cats_vs_dogs:2.0.1', split=list(splits),
        with_info=True, as_supervised=True)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [1mDownloading and preparing dataset cats_vs_dogs (786.68 MiB) to /root/tensorflow_datasets/cats_vs_dogs/2.0.1...[0m
    /usr/local/lib/python3.7/site-packages/urllib3/connectionpool.py:1004: InsecureRequestWarning: Unverified HTTPS request is being made to host 'download.microsoft.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
      InsecureRequestWarning,
    WARNING:absl:1738 images were corrupted and were skipped
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_datasets/core/file_format_adapter.py:210: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use eager execution and: 
    `tf.data.TFRecordDataset(path)`
    [1mDataset cats_vs_dogs downloaded and prepared to /root/tensorflow_datasets/cats_vs_dogs/2.0.1. Subsequent calls will reuse this data.[0m




2.B - Preprocess the test set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We must apply the same preprocessing as for training: rescaling and
resizing. Since Akida models directly accept integer-valued images, we
also define a preprocessing function for Akida: - for Keras: images are
rescaled between 0 and 1, and resized to 160x160 - for Akida: images are
only resized to 160x160 (uint8 values).

Keras and Akida models require 4-dimensional (N,H,W,C) arrays as inputs.
We must then create batches of images to feed the model. For inference,
the batch size is not relevant; you can set it such that the batch of
images can be loaded in memory depending on your CPU/GPU.


.. code-block:: default


    IMG_SIZE = 160
    input_scaling = (127.5, 127.5)

    def format_example_keras(image, label):
        image = tf.cast(image, tf.float32)
        image = (image - input_scaling[1]) / input_scaling[0]
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        return image, label

    def format_example_akida(image, label):
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, tf.uint8)
        return image, label









.. code-block:: default


    BATCH_SIZE = 32
    test_batches_keras = raw_test.map(format_example_keras).batch(BATCH_SIZE)
    test_batches_akida = raw_test.map(format_example_akida).batch(BATCH_SIZE)









2.C - Get labels
~~~~~~~~~~~~~~~~

Labels are contained in the test set as '0' for cats and '1' for dogs.
We read through the batches to extract the labels.


.. code-block:: default


    labels = np.array([])
    for _, label_batch in test_batches_keras:
        labels = np.concatenate((labels, label_batch))

    get_label_name = metadata.features['label'].int2str
    num_images = labels.shape[0]

    print(f"Test set composed of {num_images} images: "
          f"{np.count_nonzero(labels==0)} cats and "
          f"{np.count_nonzero(labels==1)} dogs.")






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Test set composed of 2320 images: 1127 cats and 1193 dogs.




3. Convert a quantized Keras model to Akida
-------------------------------------------

In this section, we will instantiate a quantized Keras model based on
MobileNet and modify the last layers to specify the classification for
'cats_vs_dogs'. After loading the pre-trained weights, we will convert
the Keras model to Akida.

This section goes as follows:

  * **3.A - Instantiate a Keras base model**
  * **3.B - Modify the network and load pre-trained weights**
  * **3.C - Convert to Akida**

3.A - Instantiate a Keras base model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we instantiate a quantized Keras model based on a MobileNet model.
This base model was previously trained using the 1000 classes of the
ImageNet dataset. For more information, please see the `ImageNet
tutorial <imagenet_cnn2akida_demo.html>`__.

The quantized MobileNet model satisfies the Akida NSoC requirements:

  * The model relies on a convolutional layer (first layer) and separable
    convolutional layers, all being Akida-compatible.
  * All the separable conv. layers have 4-bit weights, the first conv. layer
    has 8-bit weights.
  * The activations are quantized with 4 bits.

Using the provided quantized MobileNet model, we create an instance
without the top classification layer ('include_top=False').


.. code-block:: default



    base_model_keras = mobilenet_imagenet(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                include_top=False,
                                pooling='avg',
                                weights_quantization=4,
                                activ_quantization=4,
                                input_weights_quantization=8)









3.B - Modify the network and load pre-trained weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As explained in `section 1 <cats_vs_dogs_cnn2akida_demo.html#transfer-learning-process>`__,
we add a separable convolutional layer as top layer with one output neuron.
The new model is now appropriate for the ``cats_vs_dogs`` dataset and is
Akida-compatible. Note that a sigmoid activation is added at the end of
the model: the output neuron returns a probability between 0 and 1 that
the input image is a dog.

The transfer learning process has been run in the provided training
script and the weights have been saved. In this tutorial, the
pre-trained weights are loaded for inference and conversion.

.. Note:: The pre-trained weights which are loaded corresponds to the
          quantization parameters described as above. If you want to modify
          these parameters, you must re-train the model and save weights.



.. code-block:: default


    # Add a top layer for classification
    x = base_model_keras.output
    x = tf.keras.layers.Reshape((1, 1, 1024), name='reshape_1')(x)
    x = separable_conv_block(x, filters=1,
                             kernel_size=(3, 3),
                             padding='same',
                             use_bias=False,
                             name='top_layer_separable',
                             weight_quantization=4,
                             activ_quantization=None)
    x = tf.keras.layers.Activation('sigmoid')(x)
    preds = tf.keras.layers.Reshape((1,), name='reshape_2')(x)
    model_keras = tf.keras.Model(inputs=base_model_keras.input, outputs=preds, name="model_cats_vs_dogs")

    model_keras.summary()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Model: "model_cats_vs_dogs"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_4 (InputLayer)         [(None, 160, 160, 3)]     0         
    _________________________________________________________________
    conv_0 (QuantizedConv2D)     (None, 80, 80, 32)        864       
    _________________________________________________________________
    conv_0_BN (BatchNormalizatio (None, 80, 80, 32)        128       
    _________________________________________________________________
    conv_0_relu (ActivationDiscr (None, 80, 80, 32)        0         
    _________________________________________________________________
    separable_1 (QuantizedSepara (None, 80, 80, 64)        2336      
    _________________________________________________________________
    separable_1_BN (BatchNormali (None, 80, 80, 64)        256       
    _________________________________________________________________
    separable_1_relu (Activation (None, 80, 80, 64)        0         
    _________________________________________________________________
    separable_2 (QuantizedSepara (None, 80, 80, 128)       8768      
    _________________________________________________________________
    separable_2_maxpool (MaxPool (None, 40, 40, 128)       0         
    _________________________________________________________________
    separable_2_BN (BatchNormali (None, 40, 40, 128)       512       
    _________________________________________________________________
    separable_2_relu (Activation (None, 40, 40, 128)       0         
    _________________________________________________________________
    separable_3 (QuantizedSepara (None, 40, 40, 128)       17536     
    _________________________________________________________________
    separable_3_BN (BatchNormali (None, 40, 40, 128)       512       
    _________________________________________________________________
    separable_3_relu (Activation (None, 40, 40, 128)       0         
    _________________________________________________________________
    separable_4 (QuantizedSepara (None, 40, 40, 256)       33920     
    _________________________________________________________________
    separable_4_maxpool (MaxPool (None, 20, 20, 256)       0         
    _________________________________________________________________
    separable_4_BN (BatchNormali (None, 20, 20, 256)       1024      
    _________________________________________________________________
    separable_4_relu (Activation (None, 20, 20, 256)       0         
    _________________________________________________________________
    separable_5 (QuantizedSepara (None, 20, 20, 256)       67840     
    _________________________________________________________________
    separable_5_BN (BatchNormali (None, 20, 20, 256)       1024      
    _________________________________________________________________
    separable_5_relu (Activation (None, 20, 20, 256)       0         
    _________________________________________________________________
    separable_6 (QuantizedSepara (None, 20, 20, 512)       133376    
    _________________________________________________________________
    separable_6_maxpool (MaxPool (None, 10, 10, 512)       0         
    _________________________________________________________________
    separable_6_BN (BatchNormali (None, 10, 10, 512)       2048      
    _________________________________________________________________
    separable_6_relu (Activation (None, 10, 10, 512)       0         
    _________________________________________________________________
    separable_7 (QuantizedSepara (None, 10, 10, 512)       266752    
    _________________________________________________________________
    separable_7_BN (BatchNormali (None, 10, 10, 512)       2048      
    _________________________________________________________________
    separable_7_relu (Activation (None, 10, 10, 512)       0         
    _________________________________________________________________
    separable_8 (QuantizedSepara (None, 10, 10, 512)       266752    
    _________________________________________________________________
    separable_8_BN (BatchNormali (None, 10, 10, 512)       2048      
    _________________________________________________________________
    separable_8_relu (Activation (None, 10, 10, 512)       0         
    _________________________________________________________________
    separable_9 (QuantizedSepara (None, 10, 10, 512)       266752    
    _________________________________________________________________
    separable_9_BN (BatchNormali (None, 10, 10, 512)       2048      
    _________________________________________________________________
    separable_9_relu (Activation (None, 10, 10, 512)       0         
    _________________________________________________________________
    separable_10 (QuantizedSepar (None, 10, 10, 512)       266752    
    _________________________________________________________________
    separable_10_BN (BatchNormal (None, 10, 10, 512)       2048      
    _________________________________________________________________
    separable_10_relu (Activatio (None, 10, 10, 512)       0         
    _________________________________________________________________
    separable_11 (QuantizedSepar (None, 10, 10, 512)       266752    
    _________________________________________________________________
    separable_11_BN (BatchNormal (None, 10, 10, 512)       2048      
    _________________________________________________________________
    separable_11_relu (Activatio (None, 10, 10, 512)       0         
    _________________________________________________________________
    separable_12 (QuantizedSepar (None, 10, 10, 1024)      528896    
    _________________________________________________________________
    separable_12_maxpool (MaxPoo (None, 5, 5, 1024)        0         
    _________________________________________________________________
    separable_12_BN (BatchNormal (None, 5, 5, 1024)        4096      
    _________________________________________________________________
    separable_12_relu (Activatio (None, 5, 5, 1024)        0         
    _________________________________________________________________
    separable_13 (QuantizedSepar (None, 5, 5, 1024)        1057792   
    _________________________________________________________________
    separable_13_global_avg (Glo (None, 1024)              0         
    _________________________________________________________________
    separable_13_BN (BatchNormal (None, 1024)              4096      
    _________________________________________________________________
    separable_13_relu (Activatio (None, 1024)              0         
    _________________________________________________________________
    reshape_1 (Reshape)          (None, 1, 1, 1024)        0         
    _________________________________________________________________
    top_layer_separable (Quantiz (None, 1, 1, 1)           10240     
    _________________________________________________________________
    activation (Activation)      (None, 1, 1, 1)           0         
    _________________________________________________________________
    reshape_2 (Reshape)          (None, 1)                 0         
    =================================================================
    Total params: 3,219,264
    Trainable params: 3,207,296
    Non-trainable params: 11,968
    _________________________________________________________________





.. code-block:: default


    # Load pre-trained weights
    pretrained_weights = tf.keras.utils.get_file("mobilenet_cats_vs_dogs_wq4_aq4.h5",
                                                 "http://data.brainchip.com/models/mobilenet/mobilenet_cats_vs_dogs_wq4_aq4.h5",
                                                 cache_subdir='models/mobilenet')
    model_keras.load_weights(pretrained_weights)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Downloading data from http://data.brainchip.com/models/mobilenet/mobilenet_cats_vs_dogs_wq4_aq4.h5
        8192/12997744 [..............................] - ETA: 25s       81920/12997744 [..............................] - ETA: 10s      622592/12997744 [>.............................] - ETA: 2s      2187264/12997744 [====>.........................] - ETA: 0s     4300800/12997744 [========>.....................] - ETA: 1s     4571136/12997744 [=========>....................] - ETA: 1s     4603904/12997744 [=========>....................] - ETA: 1s     4644864/12997744 [=========>....................] - ETA: 1s     4694016/12997744 [=========>....................] - ETA: 1s     4751360/12997744 [=========>....................] - ETA: 1s     4816896/12997744 [==========>...................] - ETA: 1s     4890624/12997744 [==========>...................] - ETA: 1s     4980736/12997744 [==========>...................] - ETA: 1s     5079040/12997744 [==========>...................] - ETA: 1s     5177344/12997744 [==========>...................] - ETA: 1s     5292032/12997744 [===========>..................] - ETA: 1s     5414912/12997744 [===========>..................] - ETA: 1s     5545984/12997744 [===========>..................] - ETA: 1s     5685248/12997744 [============>.................] - ETA: 1s     5824512/12997744 [============>.................] - ETA: 1s     5980160/12997744 [============>.................] - ETA: 1s     6144000/12997744 [=============>................] - ETA: 1s     6316032/12997744 [=============>................] - ETA: 1s     6512640/12997744 [==============>...............] - ETA: 1s     6717440/12997744 [==============>...............] - ETA: 1s     6938624/12997744 [===============>..............] - ETA: 1s     7176192/12997744 [===============>..............] - ETA: 1s     7438336/12997744 [================>.............] - ETA: 1s     7716864/12997744 [================>.............] - ETA: 1s     8019968/12997744 [=================>............] - ETA: 1s     8339456/12997744 [==================>...........] - ETA: 1s     8699904/12997744 [===================>..........] - ETA: 0s     9093120/12997744 [===================>..........] - ETA: 0s     9535488/12997744 [=====================>........] - ETA: 0s    10018816/12997744 [======================>.......] - ETA: 0s    10543104/12997744 [=======================>......] - ETA: 0s    11124736/12997744 [========================>.....] - ETA: 0s    11755520/12997744 [==========================>...] - ETA: 0s    12435456/12997744 [===========================>..] - ETA: 0s    13000704/12997744 [==============================] - 2s 0us/step




3.C - Convert to Akida
~~~~~~~~~~~~~~~~~~~~~~

The new Keras model with pre-trained weights is now converted to an
Akida model. It only requires the quantized Keras model and the inputs
scaling used during training.
Note: the 'sigmoid' activation has no SNN equivalent and will be simply
ignored during the conversion.


.. code-block:: default


    model_akida = convert(model_keras, input_scaling=input_scaling)

    model_akida.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    -------------------------------------------------------------------------------------------------------------------------
    Layer (type)           HW  Input shape   Output shape  Kernel shape  Learning (#classes)       #InConn/#Weights/ThFire   
    =========================================================================================================================
    conv_0 (InputConvoluti yes [160, 160, 3] [80, 80, 32]  (3 x 3 x 3)   N/A                       27 / 26 / 0               
    -------------------------------------------------------------------------------------------------------------------------
    separable_1 (Separable yes [80, 80, 32]  [80, 80, 64]  (3 x 3 x 32)  N/A                       288 / 19 / 0              
    -------------------------------------------------------------------------------------------------------------------------
    separable_2 (Separable yes [80, 80, 64]  [40, 40, 128] (3 x 3 x 64)  N/A                       576 / 39 / 0              
    -------------------------------------------------------------------------------------------------------------------------
    separable_3 (Separable yes [40, 40, 128] [40, 40, 128] (3 x 3 x 128) N/A                       1152 / 61 / 0             
    -------------------------------------------------------------------------------------------------------------------------
    separable_4 (Separable yes [40, 40, 128] [20, 20, 256] (3 x 3 x 128) N/A                       1152 / 79 / 0             
    -------------------------------------------------------------------------------------------------------------------------
    separable_5 (Separable yes [20, 20, 256] [20, 20, 256] (3 x 3 x 256) N/A                       2304 / 121 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_6 (Separable yes [20, 20, 256] [10, 10, 512] (3 x 3 x 256) N/A                       2304 / 158 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_7 (Separable yes [10, 10, 512] [10, 10, 512] (3 x 3 x 512) N/A                       4608 / 240 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_8 (Separable yes [10, 10, 512] [10, 10, 512] (3 x 3 x 512) N/A                       4608 / 242 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_9 (Separable yes [10, 10, 512] [10, 10, 512] (3 x 3 x 512) N/A                       4608 / 243 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_10 (Separabl yes [10, 10, 512] [10, 10, 512] (3 x 3 x 512) N/A                       4608 / 243 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_11 (Separabl yes [10, 10, 512] [10, 10, 512] (3 x 3 x 512) N/A                       4608 / 244 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_12 (Separabl yes [10, 10, 512] [5, 5, 1024]  (3 x 3 x 512) N/A                       4608 / 323 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_13 (Separabl yes [5, 5, 1024]  [1, 1, 1024]  (3 x 3 x 1024 N/A                       9216 / 485 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    top_layer_separable (S yes [1, 1, 1024]  [1, 1, 1]     (3 x 3 x 1024 N/A                       9216 / 9 / 0              
    -------------------------------------------------------------------------------------------------------------------------




4. Classify test images
-----------------------

This section gives a comparison of the results between the quantized
Keras and the Akida models. It goes as follows:

  * **4.A - Classify test images** with the quantized Keras and the Akida
    models
  * **4.B - Compare results**

4.A Classify test images
~~~~~~~~~~~~~~~~~~~~~~~~

Here, we will predict the classes of the test images using the quantized
Keras model and the converted Akida model. Remember that:

  * Input images in Keras and Akida are not scaled in the same range, be
    careful to use the correct inputs: uint8 images for Akida and float
    rescaled images for Keras.
  * The ``predict`` function of tf.keras can take a ``tf.data.Dataset``
    object as argument. However, the Akida `evaluate <../../api_reference/aee_apis.html#akida.Model.evaluate>`__
    function takes a NumPy array containing the images. Though the Akida
    `predict <../../api_reference/aee_apis.html#akida.Model.predict>`__
    function exists, it outputs a class label and not the raw predictions.
  * The Keras ``predict`` function returns the probability to be a dog:
    if the output is greater than 0.5, the model predicts a 'dog'. However,
    the Akida `evaluate <../../api_reference/aee_apis.html#akida.Model.evaluate>`__
    function directly returns the potential before the 'sigmoid' activation, which has
    no SNN equivalent. We must therefore apply it explicitly on the model outputs to obtain
    the Akida probabilities.


.. code-block:: default


    # Classify test images with the quantized Keras model
    from timeit import default_timer as timer

    start = timer()
    pots_keras = model_keras.predict(test_batches_keras)
    end = timer()

    preds_keras = pots_keras.squeeze() > 0.5
    print(f"Keras inference on {num_images} images took {end-start:.2f} s.\n")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Keras inference on 2320 images took 11.70 s.






.. code-block:: default


    # Classify test images with the Akida model
    from progressbar import ProgressBar
    n_batches = num_images // BATCH_SIZE + 1
    pbar = ProgressBar(maxval=n_batches)
    i = 1
    pbar.start()
    start = timer()
    pots_akida = np.array([], dtype=np.float32)
    for batch, _ in test_batches_akida:
        pots_batch_akida = model_akida.evaluate(batch.numpy())
        pots_akida = np.concatenate((pots_akida, pots_batch_akida.squeeze()))
        pbar.update(i)
        i = i + 1
    pbar.finish()
    end = timer()

    preds_akida = tf.keras.layers.Activation('sigmoid')(pots_akida) > 0.5
    print(f"Akida inference on {num_images} images took {end-start:.2f} s.\n")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

      0% |                                                                        |      1% |                                                                        |      2% |#                                                                       |      4% |##                                                                      |      5% |###                                                                     |      6% |####                                                                    |      8% |#####                                                                   |      9% |######                                                                  |     10% |#######                                                                 |     12% |########                                                                |     13% |#########                                                               |     15% |##########                                                              |     16% |###########                                                             |     17% |############                                                            |     19% |#############                                                           |     20% |##############                                                          |     21% |###############                                                         |     23% |################                                                        |     24% |#################                                                       |     26% |##################                                                      |     27% |###################                                                     |     28% |####################                                                    |     30% |#####################                                                   |     31% |######################                                                  |     32% |#######################                                                 |     34% |########################                                                |     35% |#########################                                               |     36% |##########################                                              |     38% |###########################                                             |     39% |############################                                            |     41% |#############################                                           |     42% |##############################                                          |     43% |###############################                                         |     45% |################################                                        |     46% |#################################                                       |     47% |##################################                                      |     49% |###################################                                     |     50% |####################################                                    |     52% |#####################################                                   |     53% |######################################                                  |     54% |#######################################                                 |     56% |########################################                                |     57% |#########################################                               |     58% |##########################################                              |     60% |###########################################                             |     61% |############################################                            |     63% |#############################################                           |     64% |##############################################                          |     65% |###############################################                         |     67% |################################################                        |     68% |#################################################                       |     69% |##################################################                      |     71% |###################################################                     |     72% |####################################################                    |     73% |#####################################################                   |     75% |######################################################                  |     76% |#######################################################                 |     78% |########################################################                |     79% |#########################################################               |     80% |##########################################################              |     82% |###########################################################             |     83% |############################################################            |     84% |#############################################################           |     86% |##############################################################          |     87% |###############################################################         |     89% |################################################################        |     90% |#################################################################       |     91% |##################################################################      |     93% |###################################################################     |     94% |####################################################################    |     95% |#####################################################################   |     97% |######################################################################  |     98% |####################################################################### |    100% |########################################################################|    100% |########################################################################|
    Akida inference on 2320 images took 40.09 s.






.. code-block:: default


    # Print model statistics
    print("Model statistics")
    stats = model_akida.get_statistics()
    batch, _  = iter(test_batches_akida).get_next()
    model_akida.evaluate(batch[:20].numpy())
    for _, stat in stats.items():
        print(stat)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Model statistics
    Layer (type)                  output sparsity     
    conv_0 (InputConvolutional)   0.32                
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_1 (SeparableConvolu 0.32                0.33                81918194            
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_2 (SeparableConvolu 0.33                0.32                318317407           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_3 (SeparableConvolu 0.32                0.34                162690059           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_4 (SeparableConvolu 0.34                0.47                314136393           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_5 (SeparableConvolu 0.47                0.36                125307470           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_6 (SeparableConvolu 0.36                0.54                302731842           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_7 (SeparableConvolu 0.54                0.58                109525399           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_8 (SeparableConvolu 0.58                0.63                99751209            
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_9 (SeparableConvolu 0.63                0.70                88339831            
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_10 (SeparableConvol 0.70                0.69                70196407            
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_11 (SeparableConvol 0.69                0.67                72911203            
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_12 (SeparableConvol 0.67                0.85                156677864           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_13 (SeparableConvol 0.85                0.57                34937381            
    Layer (type)                  input sparsity      output sparsity     ops                 
    top_layer_separable (Separabl 0.57                0.00                7960                




4.B Compare results
~~~~~~~~~~~~~~~~~~~

The Keras and Akida accuracies are compared and the Akida confusion
matrix is given (the quantized Keras confusion matrix is almost
identical to the Akida one). Note that there is no exact equivalence
between the quantized Keras and the Akida models. However, the
accuracies are highly similar.


.. code-block:: default


    # Compute accuracies
    n_good_preds_keras = np.sum(np.equal(preds_keras, labels))
    n_good_preds_akida = np.sum(np.equal(preds_akida, labels))

    keras_accuracy = n_good_preds_keras / num_images
    akida_accuracy = n_good_preds_akida / num_images

    print(f"Quantized Keras accuracy: {keras_accuracy*100:.2f} %  "
          f"({n_good_preds_keras} / {num_images} images)")
    print(f"Akida accuracy:           {akida_accuracy*100:.2f} %  "
          f"({n_good_preds_akida} / {num_images} images)")

    # For non-regression purpose
    assert akida_accuracy > 0.97





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Quantized Keras accuracy: 97.07 %  (2252 / 2320 images)
    Akida accuracy:           97.11 %  (2253 / 2320 images)





.. code-block:: default


    def confusion_matrix_2classes(labels, predictions):
        tp = np.count_nonzero(labels + predictions == 2)
        tn = np.count_nonzero(labels + predictions == 0)
        fp = np.count_nonzero(predictions - labels == 1)
        fn = np.count_nonzero(labels - predictions == 1)

        return np.array([[tp, fn], [fp, tn]])

    def plot_confusion_matrix_2classes(cm, classes):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.xticks([0, 1], classes)
        plt.yticks([0, 1], classes)

        for i, j in zip([0,0,1,1],[0,1,0,1]):
            plt.text(j, i, f"{cm[i, j]:.2f}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2. else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.autoscale()









.. code-block:: default


    # Plot confusion matrix for Akida
    cm_akida = confusion_matrix_2classes(labels, preds_akida.numpy())
    print("Confusion matrix quantized Akida:")
    plot_confusion_matrix_2classes(cm_akida, ['dog', 'cat'])
    plt.show()



.. image:: /examples/cnn2snn/images/sphx_glr_plot_cats_vs_dogs_cnn2akida_demo_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Confusion matrix quantized Akida:





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  52.591 seconds)


.. _sphx_glr_download_examples_cnn2snn_plot_cats_vs_dogs_cnn2akida_demo.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_cats_vs_dogs_cnn2akida_demo.py <plot_cats_vs_dogs_cnn2akida_demo.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_cats_vs_dogs_cnn2akida_demo.ipynb <plot_cats_vs_dogs_cnn2akida_demo.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
