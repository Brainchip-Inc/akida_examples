
Overview
========

.. toctree::
   :hidden:
   :maxdepth: 2

   self
   Getting started <./getting_started.rst>
   See the power number <./power_number.rst>
   Installation <./installation.rst>
   User guide <./user_guide/user_guide.rst>
   API reference <./api_reference/api_reference.rst>
   Examples <./examples/index.rst>
   Model zoo performance <./model_zoo_performance.rst>
   Changelog <./changelog.rst>
   Support <https://developer.brainchip.com/support/>
   ./license.rst


The MetaTF ML Framework
-----------------------

| MetaTF is a complete machine learning framework enabling the seamless
  creation, training, and testing of neural networks on the Akida
  Neuromorphic Processor Platform. MetaTF includes an
  `Akida Neuromorphic Processor IP <https://brainchip.com/akida2-0//>`_
  simulator for execution of models in addition to Akida hardware implementations
  such as the `AKD1000 reference SoC <https://brainchip.com/dev-tools/>`_.
| Inspired by the `Keras API <https://keras.io>`_, MetaTF provides a high-level
  Python API for neural networks. This API facilitates early evaluation,
  design, final tuning, and productization of neural network models.

.. tip::
    New to MetaTF? Run your first model on the Akida simulator in under a
    minute with the `Getting started <./getting_started.html>`__ page.

.. figure:: ./img/Akida_Neural_Processor.png
  :target: ./_images/Akida_Neural_Processor.png
  :alt: Brainchip
  :scale: 40%
  :align: center

  AKD1000 reference SoC (left), Akida 2\ :sup:`nd` Generation IP (right)

|
|
| MetaTF is composed of four Python packages which leverage both the
  `TensorFlow <https://www.tensorflow.org/>`_ (through
  `TF-Keras <https://github.com/keras-team/tf-keras>`__) and `ONNX <https://onnx.ai/>`__
  frameworks, and are installed together from the `PyPI <https://pypi.org/>`_ repository through
  the single `metatf <https://pypi.org/project/metatf>`_ meta-package.
| The four MetaTF packages contain:

  * a Model zoo (`akida-models <https://pypi.org/project/akida-models>`_) to
    directly load quantized models or to easily instantiate and train
    Akida-compatible models,

  * a quantization tool (`quantizeml <https://pypi.org/project/quantizeml>`_)
    for quantization of models using low-bitwidth weights and outputs,

  * a conversion tool (`cnn2snn <https://pypi.org/project/cnn2snn>`_) to convert
    models to a binary format for model execution on an Akida platform,

  * and an interface to the Akida Neuromorphic Processor (`akida <https://pypi.org/project/akida>`_)
    including a runtime, a Hardware Abstraction Layer (HAL) and a software
    backend. It allows the simulation of the Akida Neuromorphic Processor and
    use of the AKD1000 reference SoC.

 .. figure:: ./img/metatf.png
   :target: ./_images/metatf.png
   :alt: Brainchip
   :scale: 40%
   :align: center

   MetaTF ML Framework

|
|
| The Akida package introduced above allows one to simulate the Akida Neuromorphic
  Processor IP without the need for any hardware. Furthermore, the interface to the
  Akida runtime enables seamless integration with Python-based, machine learning
  frameworks for easy prototyping with the Akida Neuromorphic Processor IP.
| It includes:

  * the Akida model API - a library supporting the native development of Akida models,
    the inference of instantiated models, their serialization (program sequences)
    and their mapping for a targeted hardware device,

  * a simulator (software backend) - a CPU implementation of the Akida Neuromorphic
    Processor IP,

  * and the `Akida Engine Library <./user_guide/engine.html>`_ - a C++ library supporting the instantiation of model
    programs produced by the model library on actual hardware devices and inference on
    programmed devices.

.. figure:: ./img/akida_runtime.png
   :target: ./_images/akida_runtime.png
   :alt: Brainchip
   :scale: 40%
   :align: center

   Akida runtime configurations


The Akida examples
------------------

The `examples section <./examples/index.html>`_ includes tutorials and examples to easily
get started with Akida technology. This section illustrates the use of Akida technology
in a variety of inference and incremental, on-device learning applications.

Two workflows are available as equal entry points, depending on the framework you train in:

* the `Global Akida workflow <./examples/general/plot_0_global_workflow.html>`_ for
  TF-Keras models,
* the `PyTorch to Akida workflow <./examples/general/plot_1_global_pytorch_workflow.html>`_
  for PyTorch models, going through the ONNX format.

.. warning::
    | While the Akida examples are provided under an
      `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0.txt>`_,
      the underlying Akida library is proprietary.
    | Please refer to the `End User License Agreement <./license.html>`__ for
      terms and conditions.
