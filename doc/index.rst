
Overview
========

.. toctree::
   :hidden:
   :maxdepth: 2

   self
   Installation <installation.rst>
   User guide <user_guide/user_guide.rst>
   API reference <api_reference/api_reference.rst>
   Examples <examples/index.rst>
   Support <https://support.brainchip.com/portal/home>
   license.rst


The Akida processor
-------------------

The Akida chip is a unique design, optimized for low-power edge AI, that
emulates, in hardware, the functions of neurons and synapses. The device is
event-driven, and thus inherently sparse, producing fast inference at a low
power consumption.

Built around a mesh-connected array of 80 neural processor units (NPUs), as
*Figure 1* shows, the chip includes a conversion complex and allows to run
popular convolutional neural networks (CNNs) such as MobileNet [#fn-1]_.
Designers can use the Akida chip as a native SNN processor, or they can use the
BrainChip `CNN2SNN <user_guide/cnn2snn.html>`__ tool to retrain their CNNs,
reducing power by changing convolutions to event based computations.

.. figure:: img/Akida_Block_Diagram.png
   :target: _images/Akida_Block_Diagram.png
   :alt: Brainchip
   :align: center

   Figure 1. BrainChip Akida processor

The Akida chip includes a number of key features that differentiate it from
other neural networks and SNN implementations. These are:

* Event-based computing leveraging SNN inherent sparsity
* Fully configurable neural processing cores, supporting convolutional,
  separable-convolutional, pooling and fully connected layers
* Incremental learning after off-line training
* On-chip few-shot training
* On-chip unsupervised learning


The Akida Development Environment
---------------------------------

The Akida Development Environment (ADE) relies on a high-level neural networks
API, written in Python, and largely inspired by the `Keras API
<https://keras.io>`_.

The core data structure used by the Akida Execution Engine is a neural network
**model**\ , which itself is a linear stack of **layers**.

The ADE leverages `TensorFlow <https://www.tensorflow.org/>`_ framework and
`PyPI <https://pypi.org/>`_ for BrainChip tools installation.
The major difference with other machine learning frameworks is that the data
exchanged between layers is not the usual **dense** multidimensional arrays,
but sets of spatially organized events that can be modelled as **sparse**
multidimensional arrays.

Throughout this documentation, those events will often be referred as "spikes",
due to their close similarity with the signals exchanged by biological neurons.

.. note::
    Although the preferred input of an Akida model is a set of spikes, dense
    inputs are also supported through dedicated adaptation layers that convert
    the dense input data frames into spikes.

.. figure:: img/ade.png
   :target: _images/ade.png
   :alt: Brainchip
   :align: center

   Figure 2. Akida Development Environment

The Akida Development Environment comprises three main python packages:

* the `Akida Execution Engine <https://pypi.org/project/akida>`_ is an interface
  to the Brainchip Akida Neuromorphic System-on-Chip (NSoC). To allow the
  development of Akida models without an actual Akida hardware, it includes a
  runtime, an Hardware Abstraction Layer (HAL) and a software backend that
  simulates the Akida NSoC (see *Figure 3*\ ).

.. figure:: img/AEE.png
   :target: _images/AEE.png
   :alt: Brainchip
   :align: center

   Figure 3. Akida Execution Engine

* the `CNN2SNN tool <https://pypi.org/project/cnn2snn>`_ provides means to
  convert Convolutional Neural Networks (CNN) that were trained using Deep
  Learning methods to a low-latency and low-power Spiking Neural Network (SNN)
  or use with the Akida Execution Engine.

* the `Akida model zoo <https://pypi.org/project/akida-models>`_ contains
  pre-created spiking neural network (SNN) models built with the Akida
  sequential API and the CNN2SNN tool using quantized Keras models.


The Akida examples
------------------

The `examples section <examples/index.html>`_ comprises a zoo of event-based CNN
and SNN tutorials. One can check models performances against MNIST, CIFAR10,
ImageNet and Google Speech Commands (KWS) datasets.

.. note::
    | While the Akida examples are provided under an
      `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0.txt>`_,
      the underlying Akida library is proprietary.
    | Please refer to the `End User License Agreement <license.html>`__ for
      terms and conditions.

____

.. [#fn-1] In most cases the entire network can be accommodated using the
   on-chip SRAM.
   Even the large MobileNet network used to classify 1000 classes of ImageNet fits
   comfortably.
