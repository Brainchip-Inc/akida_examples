See the power number
====================

Akida's differentiator is efficiency: neural network inference measured in
milliwatts. This page gets you to that number. Three commands take a
pretrained model from a Keras file to inference on an Akida device — and
report the power it draws, read from a sensor on the silicon.

**Requirements:** a `MetaTF installation <./installation.html>`__ and an Akida
device, such as the `AKD1000 reference SoC
<https://brainchip.com/dev-tools/>`__ on a PCIe board. No
device at hand? `Without a device`_ below shows what you can still see.

.. warning::
    Power measurement is currently supported on the AKD1000 only. On other
    devices (AKD1500 included) the same commands run inference but print
    ``Power measurement disabled...`` instead of the power figures — support
    for these devices will be added later.

Three commands
--------------

.. code-block:: bash

    wget https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_imagenet_224_alpha_50_iq8_wq4_aq4.h5
    CNN2SNN_TARGET_AKIDA_VERSION=v1 cnn2snn convert -m akidanet_imagenet_224_alpha_50_iq8_wq4_aq4.h5
    akida run -m akidanet_imagenet_224_alpha_50_iq8_wq4_aq4.fbz

Line by line:

1. download a pretrained, quantized `AkidaNet image classification
   <./examples/general/plot_2_akidanet_imagenet.html>`__ model,
2. convert it to an Akida binary (``.fbz``) — the environment variable targets
   Akida 1.0, the hardware generation of the AKD1000,
3. map the binary onto the device and run one inference on random data.

You should see the mapped model summary — how the layers spread over the
Neural Processors (NPs) — followed by the measurement:

.. code-block:: text

                                          Model Summary
    _________________________________________________________________________________________
    Input shape    Output shape  Sequences  Layers  NPs  Skip DMAs  External Memory (Bytes)
    =========================================================================================
    [224, 224, 3]  [1, 1, 1000]  1          15      68   0          400000
    _________________________________________________________________________________________
    ...

    No input provided, using random data.

    Floor power (mW): 907.01
    Average framerate = 10.99 fps
    Last inference clock: 13733572
    Last program clock: 999868

    Model metrics:
      inference_frames: 1
      inference_clk: 13733572
      program_clk: 999868


``Floor power`` is the idle draw of the board — a measured value, not an
estimate, like every power figure on this page.

If you get an error instead:

* ``cnn2snn: command not found`` — the environment where MetaTF was installed
  is not the one running the command. Activate your virtual environment, or
  set up the framework on the `Installation <./installation.html>`__ page.
* ``IndexError: No devices detected...`` — no Akida device is attached; see
  `Without a device`_.

Energy per inference
--------------------

A single random sample is processed in milliseconds — too brief for the
sensor to collect a meaningful inference reading, so only the floor power is
reported above. Feed the model you just converted a batch of ten images and
the statistics extend to the full measurement:

.. code-block:: bash

    wget https://data.brainchip.com/dataset-mirror/imagenet_like/imagenet_like.npy
    akida run -m akidanet_imagenet_224_alpha_50_iq8_wq4_aq4.fbz -i imagenet_like.npy

The tail of the output now reads:

.. code-block:: text

    ...

    Floor power (mW): 905.21
    Average framerate = 51.28 fps
    Last inference power range (mW):  Avg 981.00 / Min 981.00 / Max 981.00
    Last inference energy consumed (mJ/frame): 19.13
    Last inference clock: 43001443
    Last program clock: 1000756

    Model metrics:
      inference_frames: 10
      inference_clk: 43001443
      program_clk: 1000756


There it is: ImageNet-scale image classification at about one watt, costing 19
millijoules per frame.

.. note::
    The power figures above were captured on an AKD1000 reference board; your
    exact values will vary with the board's clock settings, the model, the
    inputs and the mapping mode used. Reported power and energy figures
    include the floor power.

Without a device
----------------

Power is read from a sensor on the silicon: the software simulator cannot
measure it, and ``akida run`` requires a device — without one it stops with
``No devices detected...``.

What works on any machine is checking how a model occupies the hardware, by
mapping it onto a `virtual device <./user_guide/akida.html#virtual-devices>`__:

.. code-block:: python

    import akida

    model = akida.Model("akidanet_imagenet_224_alpha_50_iq8_wq4_aq4.fbz")
    model.map(akida.AKD1000())
    model.summary()

This prints the same model summary and NP allocation as the device run —
everything except the power lines. Until a device is plugged in, the outputs
above tell you the numbers to expect from one.

Where to go next
----------------

* Measure power from Python — enable it with one line
  (`performance measurement <./user_guide/akida.html#performance-measurement>`__)
  and see it in action in the `AkidaNet/ImageNet example
  <./examples/general/plot_2_akidanet_imagenet.html#hardware-mapping-and-performance>`__.
* The complete CLI outputs are in the user guide's `command-line interface
  <./user_guide/akida.html#command-line-interface-for-model-evaluation>`__
  section.
* To bring your own model to Akida, start from the `Global Akida workflow
  <./examples/general/plot_0_global_workflow.html>`__ (TF-Keras) or the
  `PyTorch to Akida workflow
  <./examples/general/plot_1_global_pytorch_workflow.html>`__.
