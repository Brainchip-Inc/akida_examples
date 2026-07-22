Getting started
===============

Run your first model on the Akida Neuromorphic Processor — in simulation, on
your own machine — in under a minute. No hardware, no training and no
TensorFlow required: you will build a tiny network from Akida 2.0 layers, wire
its weights by hand and watch it compute XOR.

**Requirements:** Python 3.10 to 3.12 (see the
`supported configurations <./installation.html#supported-configurations>`__).

Install the akida package
-------------------------

The `akida <https://pypi.org/project/akida>`_ package contains everything this
page needs: the Akida model API and the software simulator. Its only
dependency is NumPy.

.. code-block:: bash

    pip install akida=={AKIDA_VERSION}

.. note::
    The full MetaTF framework (training, quantization and conversion tools) is
    not needed here — the `Installation <./installation.html>`__ page covers
    its complete setup.

Run the XOR network
-------------------

Save the following script as ``xor_akida.py`` and run it with
``python xor_akida.py``:

.. code-block:: python

    import numpy as np
    import akida

    # 1. Build the network: 2 inputs -> 2 hidden neurons (ReLU) -> 1 output
    model = akida.Model([
        akida.InputData(input_shape=(1, 1, 2), input_bits=8),
        akida.Dense1D(units=2, activation=akida.ActivationType.ReLU, name="hidden"),
        akida.Dense1D(units=1, name="output"),
        akida.Dequantizer(),
    ])

    # 2. Hand-wire the weights: XOR(a, b) = ReLU(a + b) - 2 * ReLU(a + b - 1)
    model.get_layer("hidden").variables["weights"] = np.array([[1, 1], [1, 1]], dtype=np.int8)
    model.get_layer("hidden").variables["bias"] = np.array([0, -1], dtype=np.int8)
    model.get_layer("output").variables["weights"] = np.array([[1], [-2]], dtype=np.int8)

    # 3. Run the four XOR input pairs through the model
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8).reshape(4, 1, 1, 2)
    outputs = model.predict(inputs)

    for pair, result in zip(inputs.reshape(4, 2), outputs.flatten()):
        print(f"XOR({pair[0]}, {pair[1]}) = {result:.0f}")

You should see:

.. code-block:: text

    XOR(0, 0) = 0
    XOR(0, 1) = 1
    XOR(1, 0) = 1
    XOR(1, 1) = 0

If you get an error instead:

* ``ModuleNotFoundError: No module named 'akida'`` — the environment where
  akida was installed is not the one running the script. Activate your virtual
  environment and run again.
* ``Unsupported input type`` — the inputs must be ``int8`` NumPy arrays: with
  ``input_bits=8``, Akida 2.0 layers take 8-bit *signed* integers.

How it works
------------

The model stacks three `Akida 2.0 layer types <./user_guide/akida.html#akida-2-0-layers>`__:

* `InputData <./api_reference/akida_apis.html#akida.InputData>`__ declares the
  input tensor: here two values, presented as 8-bit signed integers.
* `Dense1D <./api_reference/akida_apis.html#akida.Dense1D>`__ is a
  fully-connected layer. Akida executes integer-only arithmetic: weights,
  biases and activations are low-bitwidth integers — the reason for the
  ``int8`` types above, and a key ingredient of the processor's efficiency.
* `Dequantizer <./api_reference/akida_apis.html#akida.Dequantizer>`__ converts
  the final integer outputs back to floating point values for ``predict``.

The hand-wired weights implement the classic two-neuron solution:

.. math::

    \text{XOR}(a, b) = \text{ReLU}(a + b) - 2 \cdot \text{ReLU}(a + b - 1)

===  ===  ================  ====================  =======
`a`  `b`  `ReLU(a + b)`     `ReLU(a + b - 1)`     output
===  ===  ================  ====================  =======
0    0    0                 0                     **0**
0    1    1                 0                     **1**
1    0    1                 0                     **1**
1    1    2                 1                     **0**
===  ===  ================  ====================  =======

Besides ``weights`` and ``bias``, each layer carries quantization variables
(``input_shift``, ``bias_shift``, ``output_scales``, ``output_shift``) that
align integer computations with their floating point equivalents. Their
defaults are neutral (shifts of 0, scales of 1), so a hand-wired model can
ignore them — in a real workflow they are set automatically when a trained
model is quantized and converted.

The script ran on the Akida software simulator, on your CPU. The same model —
unchanged — maps onto Akida silicon when a device is present.

Where to go next
----------------

In practice you will not wire weights by hand: you train a model in your usual
framework, quantize it and convert it to Akida. Set up the full framework on
the `Installation <./installation.html>`__ page, then pick your entry point:

* the `Global Akida workflow <./examples/general/plot_0_global_workflow.html>`__
  for TF-Keras models,
* the `PyTorch to Akida workflow <./examples/general/plot_1_global_pytorch_workflow.html>`__
  for PyTorch models, going through the ONNX format.
