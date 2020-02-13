.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_semisupervised_plot_dvs_main.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_semisupervised_plot_dvs_main.py:


Learning and inference on Characters DVS
========================================

The Characters_DVS dataset comprises a set of recordings made using a
Dynamic Vision Sensor (DVS). This is a type of event-based camera, where
each event indicates the lightening or darkening of a given pixel at a
specific time. The stimuli are the 36 (latin) alphanumeric characters,
printed on paper and affixed to a rotating drum so that they drift
across the camera's field of view. The dataset includes 2 samples for
each character. For a full description of the dataset, see `Orchard et
al, (2015), doi:10.1109/TPAMI.2015.2392947
<https://www.researchgate.net/publication/273308877_HFirst_A_Temporal_Approach_to_Object_Recognition>`__.

For this simple demonstration, we use the ExtractedStabilized version of
the dataset, where, rather than use the full 128 x 128 pixel scene,
activity related to individual characters has been extracted and
centered in a 32 x 32 pixel scene.

Each DVS event is characterized by 3 values: x- and y-coordinates, and
the polarity of the luminance change (increment or decrement). Note that
in the current demo, we've discarded the polarity information for
simplicity (and because it's not useful in a task where we're only
interested in object shape and not direction of movement).

1. Loading the Characters DVS dataset
-------------------------------------


.. code-block:: default


    # Various imports needed for the tutorial
    import os

    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import warnings
    from tensorflow.keras.utils import get_file
    import csv

    # Filter warnings
    warnings.filterwarnings("ignore", module="matplotlib")

    # Akida specific imports
    from akida import Model, Sparse, InputData, FullyConnected, LearningType, coords_to_sparse









.. code-block:: default


    # Retrieve Characters DVS data set
    file_path = get_file("CharDVS.tar.gz", "http://data.brainchip.com/dataset-mirror/charDVS/CharDVS.tar.gz", cache_subdir='datasets/charDVS', extract=True)
    working_dir = os.path.dirname(file_path)

    datafilenames = []
    dvs_labels = []
    lbl_filepath = os.path.join(working_dir, "CharDVS_data", "CharDVS_labels.csv")
    if os.path.exists(lbl_filepath):
        with open(lbl_filepath, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                datafilenames.append(row[0])
                dvs_labels.append(row[1])
    else:
        print("Failed to find labels file " + lbl_filepath)

    dvs_events = []
    for fn in datafilenames[:]:
        fname = os.path.join(working_dir, "CharDVS_data", fn)
        if os.path.exists(fname):
            dvs_events.append(np.genfromtxt(fname, dtype=np.int32, delimiter=','))
        else:
            print("Failed to find data file " + fname)

    print ('Using charDVS dataset')
    # Using 32 x 32 images
    dvs_sz = (32, 32)
    dvs_shape = (*dvs_sz, 1)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Downloading data from http://data.brainchip.com/dataset-mirror/charDVS/CharDVS.tar.gz
       8192/1982663 [..............................] - ETA: 3s      81920/1982663 [>.............................] - ETA: 1s     655360/1982663 [========>.....................] - ETA: 0s    1990656/1982663 [==============================] - 0s 0us/step
    Using charDVS dataset




2. Look at some events from the dataset
---------------------------------------

As described above, each DVS event is characterized by 3 values.
Although we've discarded the polarity information, we've kept a third
channel for each event (always set to zero), because that's the input
event format expected by the Akida Execution Engine.


.. code-block:: default


    # Cherry-pick an abitrary event
    test_events = dvs_events[27]
    print(test_events[1,:])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [22 25  0]




The Akida Execution Engine processes groups of events, which we'll refer
to as 'packets'. But how many events should be in a packet?

In the current case, it's helpful to visualize the input data: we're
going to want to group together enough events to generate recognizable
features, but without allowing too many duplicate events (multiple
events occurring at the same input location). Try varying the
packet_size in the following:

[Note that this reconstruction of the events into an image is purely for
visualization here - this is not at all what happens inside the Akida
Execution Engine]


.. code-block:: default


    # Displaying a 'packet' of events as an image
    packet_size = 150

    test_img = np.zeros(dvs_sz, dtype=np.int32)
    test_events = dvs_events[27]
    xx = test_events[:packet_size,0]
    yy = test_events[:packet_size,1]
    for i in range(packet_size):
        test_img[yy[i],xx[i]] += 1

    plt.imshow(test_img, cmap=cm.Greys_r)
    plt.title('Displaying a packet of %i events' % packet_size)
    plt.show()





.. image:: /examples/semisupervised/images/sphx_glr_plot_dvs_main_001.png
    :class: sphx-glr-single-img





Ultimately, this is a variable that can be optimized according to the
task.

Here, we'll go forward with a packet_size of 150. You can try other
values.


.. code-block:: default


    # Set packet size to 150 from now on
    packet_size = 150









3. Configuring Akida model
--------------------------

A neural network model can be sequentially defined. Check the `Akida
Execution Engine documentation <../../api_reference/aee_apis.html>`__ for a
full description of the parameters and layer types available.

Note that we've defined the expected packet size to be 150 events. A
related value is num_weights, here also set to 150. Typically, those two
values will be similar - there are specific cases where more or fewer
weights will yield improved performance but setting them equal to the
packet size is a reasonable starting point.

With the neural network model in place, it's a simple matter to launch
the Akida Execution Engine:


.. code-block:: default


    #Create a model
    model = Model()
    model.add(InputData("input", input_width=32, input_height=32, input_features=1))
    fully = FullyConnected("fully", num_neurons=32, threshold_fire=40)
    model.add(fully)
    # Configure fully connected layer for training
    fully.compile(num_weights=150)
    model.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    -------------------------------------------------------------------------------------------------------------------------
    Layer (type)           HW  Input shape   Output shape  Kernel shape  Learning (#classes)       #InConn/#Weights/ThFire   
    =========================================================================================================================
    input (InputData)      yes [32, 32, 1]   [32, 32, 1]   N/A           N/A                       N/A / N/A / N/A           
    -------------------------------------------------------------------------------------------------------------------------
    fully (FullyConnected) yes [32, 32, 1]   [1, 1, 32]    N/A           akidaUnsupervised         1024 / 150 / 40           
    -------------------------------------------------------------------------------------------------------------------------




4. Learning and inference
-------------------------

A key feature of the Akida Execution Engine and the Akida NSoC is its
unsupervised learning algorithm, emulating the plasticity found between
biological neurons. As a result, we can send unlabeled data to the model
and it will learn to recognize patterns in the data.


.. code-block:: default


    # Define a simple function that iterates over a set of events
    def evaluate_events(events, num_packets, packet_size, learn):
        for pk in range(num_packets):
            pk_start = pk*packet_size
            event_packet = coords_to_sparse(coords=events[pk_start:(pk_start+packet_size), :], shape=dvs_shape)
            # This is where we call akida
            if learn:
                out_spikes = model.fit(event_packet)
            else:
                out_spikes = model.forward(event_packet)
            print("Packet " + str(pk))
            if out_spikes.nnz > 0:
                print("Output events:")
                print(out_spikes.coords)
            else:
                print('Zero output spikes generated')









First, send the neural network a few packets of data from the letter 'A'
sample, and let it learn.


.. code-block:: default


    # learning 'A' samples
    events_A = dvs_events[27]
    num_packets = 5

    stats = model.get_statistics()
    evaluate_events(events_A, num_packets, packet_size, learn=True)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Packet 0
    Zero output spikes generated
    Packet 1
    Output events:
    [[ 0  0  0 15]]
    Packet 2
    Output events:
    [[ 0  0  0 15]]
    Packet 3
    Output events:
    [[ 0  0  0 15]
     [ 0  0  0 22]]
    Packet 4
    Output events:
    [[ 0  0  0 15]
     [ 0  0  0 22]]





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
    input (InputData)             0.85                
    Layer (type)                  input sparsity      output sparsity     ops                 
    fully (FullyConnected)        0.85                0.96                4800                




The output events generated by the Akida Execution Engine are similar to
the input events we looked at above, in that each event comprises a
n-coordinate, then an x-coordinate, then a y-coordinate, then a feature
index. For output events, the x- and y-coordinates are only meaningful
for Convolutional layer types, so here, with a FullyConnected layer,
they'll always be zero. The fourth value, the feature index, is the
important one: in this case, it tells us which neuron in the model
generated the event. You can see that over the course of the packets
sent to the model, the same neurons kept responding: those are the
neurons that learned to recognize the presented input (here, the letter
'A').

Now, try sending the model some events from a stimulus that it hasn't
learned yet, say, the letter 'B' (and note that here, we've kept
learning turned off for now):


.. code-block:: default


    # Inference with 'B' samples
    events_B = dvs_events[29]
    num_packets = 5

    evaluate_events(events_B, num_packets, packet_size, learn=False)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Packet 0
    Zero output spikes generated
    Packet 1
    Zero output spikes generated
    Packet 2
    Zero output spikes generated
    Packet 3
    Zero output spikes generated
    Packet 4
    Zero output spikes generated




In most cases, no neurons will have responded. If any have (e.g. if
you've increased the packet size without adjusting the firing threshold
in the configuration file), it should be apparent that they are much
less activated (the 4th value in each output event) than they were for
the 'A' inputs.

Now send those same events again, but this time with learning enabled:


.. code-block:: default


    # learning 'B' samples
    evaluate_events(events_B, num_packets, packet_size, learn=True)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Packet 0
    Zero output spikes generated
    Packet 1
    Output events:
    [[ 0  0  0 17]]
    Packet 2
    Output events:
    [[ 0  0  0 17]]
    Packet 3
    Output events:
    [[ 0  0  0 17]]
    Packet 4
    Output events:
    [[ 0  0  0 17]
     [ 0  0  0 28]]




Some neurons should have started to respond to the 'B'. Importantly,
note that these are different neurons from those that learned the 'A'.
That means that, if we send some unknown events, depending on which
neurons respond, we should be able to infer whether the stimulus was an
'A' or a 'B'. Try it with some new packets of events, first whith letter
'A':


.. code-block:: default



    # Inference with 'A' samples
    jump_events = 5000

    evaluate_events(events_A[jump_events:,:], num_packets, packet_size, learn=False)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Packet 0
    Output events:
    [[ 0  0  0 15]
     [ 0  0  0 22]]
    Packet 1
    Output events:
    [[ 0  0  0 15]
     [ 0  0  0 22]]
    Packet 2
    Output events:
    [[ 0  0  0 15]
     [ 0  0  0 22]]
    Packet 3
    Output events:
    [[ 0  0  0 15]
     [ 0  0  0 22]]
    Packet 4
    Output events:
    [[ 0  0  0 15]
     [ 0  0  0 22]]




and now letter 'B':


.. code-block:: default


    # Inference with 'B' samples - forward(xxx, False, xxx)
    jump_events = 5000

    evaluate_events(events_B[jump_events:,:], num_packets, packet_size, learn=False)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Packet 0
    Output events:
    [[ 0  0  0 17]
     [ 0  0  0 28]]
    Packet 1
    Output events:
    [[ 0  0  0 17]
     [ 0  0  0 28]]
    Packet 2
    Output events:
    [[ 0  0  0 17]
     [ 0  0  0 28]]
    Packet 3
    Output events:
    [[ 0  0  0 17]
     [ 0  0  0 28]]
    Packet 4
    Output events:
    [[ 0  0  0 17]
     [ 0  0  0 28]]




5. Unsupervised learning with supervised classification
-------------------------------------------------------

Up to now, we've been learning in a purely unsupervised manner. That's
fine, but recognizing these inputs is a fundamentally supervised task:
we can look at the outputs and see that different neurons respond to
different inputs, but, by definition since it's unsupervised, we can't
attach any meaning to its activity. It would be relatively simple for us
to go back, look at which inputs drove which outputs and add labels
ourselves.

However, with a small change in the way the model is trained, we can
automate that process: we simply have to tell Akida how many different
classes to expect (in the neural network model file), and then send a
label with each training sample.

Replace the Akida Execution Engine instance we've been using up to now
by a new one with a slightly different neural network model:


.. code-block:: default


    # Create a different model
    model = Model()
    model.add(InputData("input", input_width=32, input_height=32, input_features=1))
    # Add a fully connected layer to the model, without activations so that we can
    # evaluate potentials directly
    fully = FullyConnected("fully", num_neurons=288, activations_enabled=False)
    model.add(fully)
    # Configure the fully connected layer for semi-supervised training by specifying
    # a number of classes
    fully.compile(num_weights=150, num_classes=36)
    model.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    -------------------------------------------------------------------------------------------------------------------------
    Layer (type)           HW  Input shape   Output shape  Kernel shape  Learning (#classes)       #InConn/#Weights/ThFire   
    =========================================================================================================================
    input (InputData)      yes [32, 32, 1]   [32, 32, 1]   N/A           N/A                       N/A / N/A / N/A           
    -------------------------------------------------------------------------------------------------------------------------
    fully (FullyConnected) yes [32, 32, 1]   [1, 1, 288]   N/A           akidaUnsupervised (36)    1024 / 150 / 0            
    -------------------------------------------------------------------------------------------------------------------------




Now let's train over a few hundred events from each input sample
(actually, only the first repeat of each character, so that we can come
back and use the second repeats for testing).


.. code-block:: default


    # Learn with the input label as an argument
    for inchar in range(36):
        events = dvs_events[inchar]
        label = inchar
        num_packets = 5

        for pk in range(num_packets):
            pk_start = pk*packet_size
            event_packet = coords_to_sparse(events[pk_start:(pk_start+packet_size), :], dvs_shape)
            model.fit(event_packet, input_labels=label)









We now use a different Akida API to retrieve the most active label among
the spiking neurons:


.. code-block:: default


    # Check updated output with a few samples
    events_B = dvs_events[29]
    num_packets = 5

    stats = model.get_statistics()
    for pk in range(num_packets):
        pk_start = pk*packet_size
        event_packet = coords_to_sparse(events_B[pk_start:(pk_start+packet_size), :], dvs_shape)
        out_label = model.predict(event_packet, num_classes=36)
        if out_label is not None:
            print("Output label:")
            print(out_label)
        else:
            print('Zero output spikes generated')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Output label:
    [29]
    Output label:
    [29]
    Output label:
    [29]
    Output label:
    [29]
    Output label:
    [29]





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
    input (InputData)             0.85                
    Layer (type)                  input sparsity      output sparsity     ops                 
    fully (FullyConnected)        0.85                0.00                43200               




It should be apparent that the predicted label corresponds to the
'label' of the input events that we sent (29).

Now let's run a full test on events that we didn't train on:


.. code-block:: default


    # Final check
    inLabels = []
    outLabels = []

    for inchar in range(36):
        events = dvs_events[inchar+36]
        num_packets = 5

        for pk in range(num_packets):
            print("Sample " + dvs_labels[inchar+36] + " In (number " + str(pk+1)  + "), Out", end="")
            inLabels.append(dvs_labels[inchar+36])
            pk_start = pk*packet_size
            event_packet = coords_to_sparse(events[pk_start:(pk_start+packet_size), :], dvs_shape)
            out_label = model.predict(event_packet, 36)[0]
            if out_label != -1:
                print(" " + dvs_labels[out_label], end="\n")
            else:
                print(" ?", end="\n")

        print()



.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Sample X In (number 1), Out X
    Sample X In (number 2), Out X
    Sample X In (number 3), Out X
    Sample X In (number 4), Out X
    Sample X In (number 5), Out X

    Sample F In (number 1), Out F
    Sample F In (number 2), Out F
    Sample F In (number 3), Out F
    Sample F In (number 4), Out F
    Sample F In (number 5), Out F

    Sample Y In (number 1), Out Y
    Sample Y In (number 2), Out Y
    Sample Y In (number 3), Out Y
    Sample Y In (number 4), Out Y
    Sample Y In (number 5), Out Y

    Sample G In (number 1), Out G
    Sample G In (number 2), Out G
    Sample G In (number 3), Out G
    Sample G In (number 4), Out G
    Sample G In (number 5), Out G

    Sample Z In (number 1), Out Z
    Sample Z In (number 2), Out Z
    Sample Z In (number 3), Out Z
    Sample Z In (number 4), Out Z
    Sample Z In (number 5), Out Z

    Sample H In (number 1), Out H
    Sample H In (number 2), Out H
    Sample H In (number 3), Out H
    Sample H In (number 4), Out H
    Sample H In (number 5), Out H

    Sample 0 In (number 1), Out 0
    Sample 0 In (number 2), Out 0
    Sample 0 In (number 3), Out 0
    Sample 0 In (number 4), Out 0
    Sample 0 In (number 5), Out 0

    Sample I In (number 1), Out 1
    Sample I In (number 2), Out 1
    Sample I In (number 3), Out 1
    Sample I In (number 4), Out 1
    Sample I In (number 5), Out 1

    Sample 1 In (number 1), Out 1
    Sample 1 In (number 2), Out 1
    Sample 1 In (number 3), Out 1
    Sample 1 In (number 4), Out 1
    Sample 1 In (number 5), Out 1

    Sample J In (number 1), Out J
    Sample J In (number 2), Out J
    Sample J In (number 3), Out J
    Sample J In (number 4), Out J
    Sample J In (number 5), Out J

    Sample 2 In (number 1), Out 2
    Sample 2 In (number 2), Out 2
    Sample 2 In (number 3), Out 2
    Sample 2 In (number 4), Out 2
    Sample 2 In (number 5), Out 2

    Sample K In (number 1), Out K
    Sample K In (number 2), Out K
    Sample K In (number 3), Out K
    Sample K In (number 4), Out K
    Sample K In (number 5), Out K

    Sample 3 In (number 1), Out 3
    Sample 3 In (number 2), Out 3
    Sample 3 In (number 3), Out 3
    Sample 3 In (number 4), Out 3
    Sample 3 In (number 5), Out 3

    Sample L In (number 1), Out L
    Sample L In (number 2), Out L
    Sample L In (number 3), Out L
    Sample L In (number 4), Out L
    Sample L In (number 5), Out L

    Sample 4 In (number 1), Out 4
    Sample 4 In (number 2), Out 4
    Sample 4 In (number 3), Out 4
    Sample 4 In (number 4), Out 4
    Sample 4 In (number 5), Out 4

    Sample M In (number 1), Out M
    Sample M In (number 2), Out M
    Sample M In (number 3), Out M
    Sample M In (number 4), Out M
    Sample M In (number 5), Out M

    Sample 5 In (number 1), Out 5
    Sample 5 In (number 2), Out 5
    Sample 5 In (number 3), Out 5
    Sample 5 In (number 4), Out 5
    Sample 5 In (number 5), Out 5

    Sample N In (number 1), Out N
    Sample N In (number 2), Out N
    Sample N In (number 3), Out N
    Sample N In (number 4), Out N
    Sample N In (number 5), Out N

    Sample 6 In (number 1), Out 6
    Sample 6 In (number 2), Out 6
    Sample 6 In (number 3), Out 6
    Sample 6 In (number 4), Out 6
    Sample 6 In (number 5), Out 6

    Sample O In (number 1), Out O
    Sample O In (number 2), Out O
    Sample O In (number 3), Out O
    Sample O In (number 4), Out O
    Sample O In (number 5), Out O

    Sample 7 In (number 1), Out 7
    Sample 7 In (number 2), Out 7
    Sample 7 In (number 3), Out 7
    Sample 7 In (number 4), Out 7
    Sample 7 In (number 5), Out 7

    Sample P In (number 1), Out P
    Sample P In (number 2), Out P
    Sample P In (number 3), Out P
    Sample P In (number 4), Out P
    Sample P In (number 5), Out P

    Sample 8 In (number 1), Out 8
    Sample 8 In (number 2), Out 8
    Sample 8 In (number 3), Out 8
    Sample 8 In (number 4), Out 8
    Sample 8 In (number 5), Out 8

    Sample Q In (number 1), Out Q
    Sample Q In (number 2), Out Q
    Sample Q In (number 3), Out Q
    Sample Q In (number 4), Out Q
    Sample Q In (number 5), Out Q

    Sample 9 In (number 1), Out 9
    Sample 9 In (number 2), Out 9
    Sample 9 In (number 3), Out 9
    Sample 9 In (number 4), Out 9
    Sample 9 In (number 5), Out 9

    Sample R In (number 1), Out R
    Sample R In (number 2), Out R
    Sample R In (number 3), Out R
    Sample R In (number 4), Out R
    Sample R In (number 5), Out R

    Sample S In (number 1), Out S
    Sample S In (number 2), Out S
    Sample S In (number 3), Out S
    Sample S In (number 4), Out S
    Sample S In (number 5), Out S

    Sample A In (number 1), Out A
    Sample A In (number 2), Out A
    Sample A In (number 3), Out A
    Sample A In (number 4), Out A
    Sample A In (number 5), Out A

    Sample T In (number 1), Out T
    Sample T In (number 2), Out T
    Sample T In (number 3), Out T
    Sample T In (number 4), Out T
    Sample T In (number 5), Out T

    Sample B In (number 1), Out B
    Sample B In (number 2), Out B
    Sample B In (number 3), Out B
    Sample B In (number 4), Out B
    Sample B In (number 5), Out B

    Sample U In (number 1), Out U
    Sample U In (number 2), Out U
    Sample U In (number 3), Out U
    Sample U In (number 4), Out U
    Sample U In (number 5), Out U

    Sample C In (number 1), Out C
    Sample C In (number 2), Out C
    Sample C In (number 3), Out C
    Sample C In (number 4), Out C
    Sample C In (number 5), Out C

    Sample V In (number 1), Out V
    Sample V In (number 2), Out V
    Sample V In (number 3), Out V
    Sample V In (number 4), Out V
    Sample V In (number 5), Out V

    Sample D In (number 1), Out D
    Sample D In (number 2), Out D
    Sample D In (number 3), Out D
    Sample D In (number 4), Out D
    Sample D In (number 5), Out D

    Sample W In (number 1), Out W
    Sample W In (number 2), Out W
    Sample W In (number 3), Out W
    Sample W In (number 4), Out W
    Sample W In (number 5), Out W

    Sample E In (number 1), Out E
    Sample E In (number 2), Out E
    Sample E In (number 3), Out E
    Sample E In (number 4), Out E
    Sample E In (number 5), Out E






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  3.603 seconds)


.. _sphx_glr_download_examples_semisupervised_plot_dvs_main.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_dvs_main.py <plot_dvs_main.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_dvs_main.ipynb <plot_dvs_main.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
