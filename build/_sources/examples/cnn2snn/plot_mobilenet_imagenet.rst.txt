.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_cnn2snn_plot_mobilenet_imagenet.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_cnn2snn_plot_mobilenet_imagenet.py:


Inference on ImageNet with MobileNet
====================================

.. Note:: Please refer to `CNN2SNN Conversion Tutorial (MNIST)
          <../../examples/cnn2snn/mnist_cnn2akida_demo.html>`__ notebook
          and/or the `CNN2SNN documentation
          <../../user_guide/cnn2snn.html>`__ for flow and steps details of
          the CNN2SNN conversion.

This CNN2SNN tutorial presents how to convert a MobileNet pre-trained
model into Akida. The performances are assessed using the ImageNet
dataset. This example goes as follow:

1. Load CNN2SNN tool dependencies.
2. Load test images and labels.
3. Create a quantized Keras model satifying Akida NSoC requirements

   * Instantiate a Keras model with pre-trained weights
   * Check performance

4. Convert to Akida model

   * Convert Keras model to Akida model compatible for Akida NSoC
   * Test performance
   * Show predictions for some images

1. Load CNN2SNN tool dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code-block:: default


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








2. Load test images from ImageNet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The inputs in the Keras MobileNet model must respect two requirements:

* the input image size must be 224x224x3,
* the input image values must be between -1 and 1.

This section goes as follows:

* **Load and preprocess images.** The test images all have at least 256 pixels
  in the smallest dimension. They must be preprocessed to fit in the model.
  The ``imagenet_preprocessing.preprocess_image`` function decodes, crops and
  extracts a square 224x224x3 patch from an input image.
* **Load corresponding labels.** The labels for test images are stored in the
  akida_models package. The matching between names (*string*) and labels
  (*integer*) is given through ``imagenet_preprocessing.index_to_label``
  method.

.. Note:: A set of 10 copyright free images extracted from Google using
          ImageNet labels are used here as ImageNet images are not free of
          rights.

          Akida Execution Engine is configured to take 8-bit inputs
          without rescaling. For conversion, rescaling values used for
          training the Keras model are needed.


2.1 Load test images and preprocess test images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code-block:: default


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






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Downloading data from http://data.brainchip.com/dataset-mirror/imagenet_like/imagenet_like.zip
        8192/20418307 [..............................] - ETA: 40s       81920/20418307 [..............................] - ETA: 16s      622592/20418307 [..............................] - ETA: 3s      2203648/20418307 [==>...........................] - ETA: 3s     5160960/20418307 [======>.......................] - ETA: 1s     5332992/20418307 [======>.......................] - ETA: 1s     6332416/20418307 [========>.....................] - ETA: 1s     7938048/20418307 [==========>...................] - ETA: 1s     9650176/20418307 [=============>................] - ETA: 0s    11485184/20418307 [===============>..............] - ETA: 0s    13385728/20418307 [==================>...........] - ETA: 0s    15360000/20418307 [=====================>........] - ETA: 0s    17391616/20418307 [========================>.....] - ETA: 0s    19472384/20418307 [===========================>..] - ETA: 0s    20422656/20418307 [==============================] - 1s 0us/step
    10 images loaded and preprocessed.




2.2 Load labels
^^^^^^^^^^^^^^^



.. code-block:: default


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






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Labels loaded.




3. Create a quantized Keras model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Keras model based on a MobileNet model is instantiated with quantized
weights and activations. This model satisfies the Akida NSoC
requirements:

* all the convolutional layers have 4-bit weights, except for the first
  layer,
* the first layer has 8-bit weights,
* all the convolutional layers have 4-bit activations.

This section goes as follows:

* **Instantiate a quantized Keras model** according to above specifications.
* **Load pre-trained weights** that performs a 65 % accuracy on the test
  dataset.
* **Check performance** on the test set. According to the number of test
  images, the inference could last for several minutes.


3.1 Instantiate Keras model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CNN2SNN module offers a way to easily instantiate a MobileNet model
based on Keras with quantized weights and activations. Our ``MobileNet``
function returns a Keras model with custom quantized layers (see
``quantization_layers.py`` in the CNN2SNN module).

.. Note:: The pre-trained weights which are loaded correspond to the
   parameters in the next cell. If you want to modify some of these
   parameters, you must re-train the model and save the weights.



.. code-block:: default


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





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Instantiating MobileNet...
    Downloading data from http://data.brainchip.com/models/mobilenet/mobilenet_imagenet_wq4_aq4.h5
        8192/17090328 [..............................] - ETA: 33s       81920/17090328 [..............................] - ETA: 13s      606208/17090328 [>.............................] - ETA: 3s      2097152/17090328 [==>...........................] - ETA: 3s     4964352/17090328 [=======>......................] - ETA: 1s     5144576/17090328 [========>.....................] - ETA: 1s     6127616/17090328 [=========>....................] - ETA: 1s     7684096/17090328 [============>.................] - ETA: 0s     9338880/17090328 [===============>..............] - ETA: 0s    11108352/17090328 [==================>...........] - ETA: 0s    13008896/17090328 [=====================>........] - ETA: 0s    15032320/17090328 [=========================>....] - ETA: 0s    17096704/17090328 [==============================] - 1s 0us/step
    ...done.
    Model: "mobilenet_1.00_224"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_5 (InputLayer)         [(None, 224, 224, 3)]     0         
    _________________________________________________________________
    conv_0 (QuantizedConv2D)     (None, 112, 112, 32)      864       
    _________________________________________________________________
    conv_0_BN (BatchNormalizatio (None, 112, 112, 32)      128       
    _________________________________________________________________
    conv_0_relu (ActivationDiscr (None, 112, 112, 32)      0         
    _________________________________________________________________
    separable_1 (QuantizedSepara (None, 112, 112, 64)      2336      
    _________________________________________________________________
    separable_1_BN (BatchNormali (None, 112, 112, 64)      256       
    _________________________________________________________________
    separable_1_relu (Activation (None, 112, 112, 64)      0         
    _________________________________________________________________
    separable_2 (QuantizedSepara (None, 112, 112, 128)     8768      
    _________________________________________________________________
    separable_2_maxpool (MaxPool (None, 56, 56, 128)       0         
    _________________________________________________________________
    separable_2_BN (BatchNormali (None, 56, 56, 128)       512       
    _________________________________________________________________
    separable_2_relu (Activation (None, 56, 56, 128)       0         
    _________________________________________________________________
    separable_3 (QuantizedSepara (None, 56, 56, 128)       17536     
    _________________________________________________________________
    separable_3_BN (BatchNormali (None, 56, 56, 128)       512       
    _________________________________________________________________
    separable_3_relu (Activation (None, 56, 56, 128)       0         
    _________________________________________________________________
    separable_4 (QuantizedSepara (None, 56, 56, 256)       33920     
    _________________________________________________________________
    separable_4_maxpool (MaxPool (None, 28, 28, 256)       0         
    _________________________________________________________________
    separable_4_BN (BatchNormali (None, 28, 28, 256)       1024      
    _________________________________________________________________
    separable_4_relu (Activation (None, 28, 28, 256)       0         
    _________________________________________________________________
    separable_5 (QuantizedSepara (None, 28, 28, 256)       67840     
    _________________________________________________________________
    separable_5_BN (BatchNormali (None, 28, 28, 256)       1024      
    _________________________________________________________________
    separable_5_relu (Activation (None, 28, 28, 256)       0         
    _________________________________________________________________
    separable_6 (QuantizedSepara (None, 28, 28, 512)       133376    
    _________________________________________________________________
    separable_6_maxpool (MaxPool (None, 14, 14, 512)       0         
    _________________________________________________________________
    separable_6_BN (BatchNormali (None, 14, 14, 512)       2048      
    _________________________________________________________________
    separable_6_relu (Activation (None, 14, 14, 512)       0         
    _________________________________________________________________
    separable_7 (QuantizedSepara (None, 14, 14, 512)       266752    
    _________________________________________________________________
    separable_7_BN (BatchNormali (None, 14, 14, 512)       2048      
    _________________________________________________________________
    separable_7_relu (Activation (None, 14, 14, 512)       0         
    _________________________________________________________________
    separable_8 (QuantizedSepara (None, 14, 14, 512)       266752    
    _________________________________________________________________
    separable_8_BN (BatchNormali (None, 14, 14, 512)       2048      
    _________________________________________________________________
    separable_8_relu (Activation (None, 14, 14, 512)       0         
    _________________________________________________________________
    separable_9 (QuantizedSepara (None, 14, 14, 512)       266752    
    _________________________________________________________________
    separable_9_BN (BatchNormali (None, 14, 14, 512)       2048      
    _________________________________________________________________
    separable_9_relu (Activation (None, 14, 14, 512)       0         
    _________________________________________________________________
    separable_10 (QuantizedSepar (None, 14, 14, 512)       266752    
    _________________________________________________________________
    separable_10_BN (BatchNormal (None, 14, 14, 512)       2048      
    _________________________________________________________________
    separable_10_relu (Activatio (None, 14, 14, 512)       0         
    _________________________________________________________________
    separable_11 (QuantizedSepar (None, 14, 14, 512)       266752    
    _________________________________________________________________
    separable_11_BN (BatchNormal (None, 14, 14, 512)       2048      
    _________________________________________________________________
    separable_11_relu (Activatio (None, 14, 14, 512)       0         
    _________________________________________________________________
    separable_12 (QuantizedSepar (None, 14, 14, 1024)      528896    
    _________________________________________________________________
    separable_12_maxpool (MaxPoo (None, 7, 7, 1024)        0         
    _________________________________________________________________
    separable_12_BN (BatchNormal (None, 7, 7, 1024)        4096      
    _________________________________________________________________
    separable_12_relu (Activatio (None, 7, 7, 1024)        0         
    _________________________________________________________________
    separable_13 (QuantizedSepar (None, 7, 7, 1024)        1057792   
    _________________________________________________________________
    separable_13_global_avg (Glo (None, 1024)              0         
    _________________________________________________________________
    separable_13_BN (BatchNormal (None, 1024)              4096      
    _________________________________________________________________
    separable_13_relu (Activatio (None, 1024)              0         
    _________________________________________________________________
    reshape_1 (Reshape)          (None, 1, 1, 1024)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 1, 1, 1024)        0         
    _________________________________________________________________
    separable_14 (QuantizedSepar (None, 1, 1, 1000)        1033216   
    _________________________________________________________________
    act_softmax (Activation)     (None, 1, 1, 1000)        0         
    _________________________________________________________________
    reshape_2 (Reshape)          (None, 1000)              0         
    =================================================================
    Total params: 4,242,240
    Trainable params: 4,230,272
    Non-trainable params: 11,968
    _________________________________________________________________




3.2 Check performance of the Keras model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code-block:: default


    print(f'Predicting with Keras model on {num_images} images ...')

    start = timer()
    potentials_keras = model_keras.predict(x_test_preprocess, batch_size=100)
    end = timer()
    print(f'Keras inference on {num_images} images took {end-start:.2f} s.\n')

    preds_keras = np.squeeze(np.argmax(potentials_keras, 1))
    accuracy_keras = np.sum(np.equal(preds_keras, labels_test)) / num_images

    print(f"Keras accuracy: {accuracy_keras*100:.2f} %")






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Predicting with Keras model on 10 images ...
    Keras inference on 10 images took 0.82 s.

    Keras accuracy: 100.00 %




4. Convert Keras model for Akida NSoC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, the Keras quantized model is converted into a suitable version for
the Akida NSoC. The `cnn2snn.convert <../../api_reference/cnn2snn_apis.html#convert>`__
function needs as arguments the Keras model and the input scaling parameters.
The Akida model is then saved in a YAML file with the corresponding weights
binary files.

This section goes as follows:

* **Convert the Keras MobileNet model** to an Akida model compatible for
  Akida NSoC. Print a summary of the model.
* **Test performance** of the Akida model (this can take minutes).
* **Show predictions** for some test images.


4.1 Convert Keras model to an Akida compatible model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code-block:: default


    # Convert to Akida and save model
    from cnn2snn import convert

    print("Converting Keras model for Akida NSoC...")
    model_akida = convert(model_keras, input_scaling=input_scaling)
    model_akida.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Converting Keras model for Akida NSoC...
    -------------------------------------------------------------------------------------------------------------------------
    Layer (type)           HW  Input shape   Output shape  Kernel shape  Learning (#classes)       #InConn/#Weights/ThFire   
    =========================================================================================================================
    conv_0 (InputConvoluti yes [224, 224, 3] [112, 112, 32 (3 x 3 x 3)   N/A                       27 / 26 / 0               
    -------------------------------------------------------------------------------------------------------------------------
    separable_1 (Separable yes [112, 112, 32 [112, 112, 64 (3 x 3 x 32)  N/A                       288 / 19 / 0              
    -------------------------------------------------------------------------------------------------------------------------
    separable_2 (Separable yes [112, 112, 64 [56, 56, 128] (3 x 3 x 64)  N/A                       576 / 39 / 0              
    -------------------------------------------------------------------------------------------------------------------------
    separable_3 (Separable yes [56, 56, 128] [56, 56, 128] (3 x 3 x 128) N/A                       1152 / 61 / 0             
    -------------------------------------------------------------------------------------------------------------------------
    separable_4 (Separable yes [56, 56, 128] [28, 28, 256] (3 x 3 x 128) N/A                       1152 / 79 / 0             
    -------------------------------------------------------------------------------------------------------------------------
    separable_5 (Separable yes [28, 28, 256] [28, 28, 256] (3 x 3 x 256) N/A                       2304 / 121 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_6 (Separable yes [28, 28, 256] [14, 14, 512] (3 x 3 x 256) N/A                       2304 / 158 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_7 (Separable yes [14, 14, 512] [14, 14, 512] (3 x 3 x 512) N/A                       4608 / 240 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_8 (Separable yes [14, 14, 512] [14, 14, 512] (3 x 3 x 512) N/A                       4608 / 242 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_9 (Separable yes [14, 14, 512] [14, 14, 512] (3 x 3 x 512) N/A                       4608 / 243 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_10 (Separabl yes [14, 14, 512] [14, 14, 512] (3 x 3 x 512) N/A                       4608 / 243 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_11 (Separabl yes [14, 14, 512] [14, 14, 512] (3 x 3 x 512) N/A                       4608 / 244 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_12 (Separabl yes [14, 14, 512] [7, 7, 1024]  (3 x 3 x 512) N/A                       4608 / 323 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_13 (Separabl yes [7, 7, 1024]  [1, 1, 1024]  (3 x 3 x 1024 N/A                       9216 / 485 / 0            
    -------------------------------------------------------------------------------------------------------------------------
    separable_14 (Separabl yes [1, 1, 1024]  [1, 1, 1000]  (3 x 3 x 1024 N/A                       9216 / 485 / 0            
    -------------------------------------------------------------------------------------------------------------------------




4.2 Test performance of the Akida model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code-block:: default


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






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Predicting with Akida model on 10 images ...
    Inference on 10 images took 0.39 s.

    Accuracy: 90.00 %
    Model statistics
    Layer (type)                  output sparsity     
    conv_0 (InputConvolutional)   0.35                
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_1 (SeparableConvolu 0.35                0.35                153628897           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_2 (SeparableConvolu 0.35                0.33                609465079           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_3 (SeparableConvolu 0.33                0.34                314257389           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_4 (SeparableConvolu 0.34                0.51                616725372           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_5 (SeparableConvolu 0.51                0.35                228292400           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_6 (SeparableConvolu 0.35                0.59                606560686           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_7 (SeparableConvolu 0.59                0.58                189793331           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_8 (SeparableConvolu 0.58                0.65                196439959           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_9 (SeparableConvolu 0.65                0.71                163780690           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_10 (SeparableConvol 0.71                0.69                135076801           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_11 (SeparableConvol 0.69                0.69                143066057           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_12 (SeparableConvol 0.69                0.88                288952832           
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_13 (SeparableConvol 0.88                0.55                53289132            
    Layer (type)                  input sparsity      output sparsity     ops                 
    separable_14 (SeparableConvol 0.55                0.00                4172969             




4.3 Show predictions for a random test image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a random test image, we predict the top 5 classes and display the
results on a bar chart.



.. code-block:: default



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

    # %matplotlib notebook

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



.. image:: /examples/cnn2snn/images/sphx_glr_plot_mobilenet_imagenet_001.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  8.798 seconds)


.. _sphx_glr_download_examples_cnn2snn_plot_mobilenet_imagenet.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_mobilenet_imagenet.py <plot_mobilenet_imagenet.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_mobilenet_imagenet.ipynb <plot_mobilenet_imagenet.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
