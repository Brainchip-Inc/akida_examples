
QuantizeML toolkit
==================

Overview
--------

QuantizeML package provides base layers and quantization tools for deep-learning models. It  allows
the quantization of CNN, Transformer and TENN models using low-bitwidth weights and outputs. Once
quantized with the provided tools, CNN2SNN toolkit will be able to convert the model and execute it
with Akida runtime.

The FixedPoint representation
-----------------------------

QuantizeML uses a FixedPoint representation in place of float values for layers inputs, outputs and
weights.

FixedPoint numbers are actually integers with a static number of fractional bits so that:

.. math::
    x_{float} \approx x_{int}.2^{-x_{frac\_bits}}

The precision of the representation is directly related to the number of fractional bits. For
example, representing PI using an 8bit FixedPoint with varying fractional bits:

+-----------+-------+-------------+
| frac_bits | x_int | float value |
+===========+=======+=============+
|     1     |   6   |     3.0     |
+-----------+-------+-------------+
|     3     |  25   |    3.125    |
+-----------+-------+-------------+
|     6     |  201  |  3.140625   |
+-----------+-------+-------------+

Further details are available in the
`FixedPoint API <../api_reference/quantizeml_apis.html#fixedpoint>`__ documentation.

Thanks to the FixedPoint representation, all operations within layers are implemented as integer
only operations [#fn-1]_.


Quantization flow
-----------------

The first step in the workflow is to train a standard Keras model. This trained model is the
starting point for the quantization stage. Once it is established that the overall model
configuration prior to quantization yields a satisfactory performance on the task, one can proceed
with quantization.

Let's take the `DS-CNN <../api_reference/akida_models_apis.html#ds-cnn>`__ model from our zoo that
targets KWS task as an example:

.. code-block:: python

    from akida_models import fetch_file
    from quantizeml.models import load_model
    model_file = fetch_file("https://data.brainchip.com/models/AkidaV2/ds_cnn/ds_cnn_kws.h5",
                            fname="ds_cnn_kws.h5")
    model = load_model(model_file)

The QuantizeML toolkit offers a turnkey solution to quantize a model: the
`quantize <../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__ function. It
replaces the Keras layers (or custom QuantizeML layers) with quantized, integer only layers. The
obtained quantized model is still a Keras model that can be evaluated with a standard Keras
pipeline.

The quantization scheme used by
`quantize <../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__ can be configured
using
`QuantizationParams <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizationParams>`__.
If none is given, an 8bit configuration scheme will be selected.

Here's an example for 8bit quantization:

.. code-block:: python

    from quantizeml.layers import QuantizationParams
    qparams8 = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)

Here's an example for 4bit quantization (with first layer weights set to 8bit):

.. code-block:: python

    from quantizeml.layers import QuantizationParams
    qparams4 = QuantizationParams(input_weight_bits=8, weight_bits=4, activation_bits=4)

Note that quantizating the first weights to 8bit helps preserving accuracy.

QuantizeML uses a uniform quantization scheme centered on zero. During quantization, the floating
point values are mapped to a given bitwidth quantization space of the form:

.. math::
    data_{float32} = data_{fixed\_point} * scales

`scales` is a real number used to map the FixedPoint numbers to a quantization space. It is
calculated as follows:

.. math::
    scales = \frac {max(abs(data))}{2^{bitwidth} - 1}

Inputs, weights and outputs scales are folded into a single output scale vector.

To avoid saturation in downstream operations throughout a model graph, the bitwidth of intermediary
results is decreased using
`OutputQuantizer <../api_reference/quantizeml_apis.html#quantizeml.layers.OutputQuantizer>`__. The
`quantize <../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__ function has
built-in rules to automatically isolate building blocks of layers after which such quantization is
required and will insert the
`OutputQuantizer <../api_reference/quantizeml_apis.html#quantizeml.layers.OutputQuantizer>`__
objects during the quantization process.

To properly operate, an
`OutputQuantizer <../api_reference/quantizeml_apis.html#quantizeml.layers.OutputQuantizer>`__ must
be calibrated so that it determines an adequate quantization range. Calibration will dertermine the
quantization range statistically. It is possible to pass down samples to the
`quantize <../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__ function so that
calibration and quantization are performed simultaneously.

Calibration samples are available on
`Brainchip data server <https://data.brainchip.com/dataset-mirror/samples/>`__ for datasets used in
our zoo. They must be downloaded and deserialized before being used for calibration.

.. code-block:: python

    import numpy as np
    from akida_models import fetch_file
    samples = fetch_file("https://data.brainchip.com/dataset-mirror/samples/kws/kws_batch1024.npz",
                         fname="kws_batch1024.npz")
    samples = np.load(samples)
    samples = np.concatenate([samples[item] for item in samples.files])

Quantizing the DS-CNN model to 8bit is then done with:

.. code-block:: python

    from quantizeml.models import quantize
    quantized_model = quantize(model, qparams=qparams8, samples=samples)

Please refer to `calibrate <../api_reference/quantizeml_apis.html#quantizeml.models.calibrate>`__
for more details on calibration.

Direct quantization of a standard Keras model (also called post-training quantization, PTQ)
generally introduces a drop in performance. This drop is usually small for 8bit or even 4bit
quantization of simple models, but it can be very significant for low quantization bitwidth and
complex models (`AkidaNet <../api_reference/akida_models_apis.html#akida_models.akidanet_imagenet>`_
or `transformers <../api_reference/akida_models_apis.html#transformers>`_ architectures).

If the quantized model offers acceptable performance, it can be directly converted into an Akida
model (see the `convert <../api_reference/cnn2snn_apis.html#cnn2snn.convert>`_ function).

However, if the performance drop is too high, a quantization-aware training (QAT) step is required
to recover the performance prior to quantization. Since the quantized model is a Keras model, it can
then be trained using the standard Keras API.

Check out the `examples section <../examples/index.html>`__ for tutorials on quantization, PTQ and
QAT.

Compatibility constraints
~~~~~~~~~~~~~~~~~~~~~~~~~

The tookit supports a wide range of layers (see the
`supported type section <quantizeml.html#supported-layer-types>`__). When hitting a non-compatible
layer, QuantizeML will simply stop the quantization before this layer and add a
`Dequantizer <../api_reference/quantizeml_apis.html#quantizeml.layers.Dequantizer>`__ before it so
that inference is still possible. When such an event occurs, a warning is raised to the user with the
faulty layer name.

While quantization comes with some restrictions on layer order (e.g. MaxPool2D operation should be
placed before ReLU activation), the
`sanitize <../api_reference/quantizeml_apis.html#quantizeml.models.transforms.sanitize>`__ helper is
called before quantization to deal with such restrictions and edit the model accordingly.
`sanitize <../api_reference/quantizeml_apis.html#quantizeml.models.transforms.sanitize>`__ will also
handle some layers that are not in the
`supported layer types <quantizeml.html#supported-layer-types>`__ such as:

- ZeroPadding2D which is replaced with 'same' padding convolution when possible
- Lambda layers:
    - Lambda(relu) or Activation('relu') → ReLU,
    - Lambda(transpose) → Permute,
    - Lambda(reshape) → Reshape,
    - Lambda(add) → Add.


Model loading
~~~~~~~~~~~~~

The toolkit offers a
`keras.models.load_model <https://www.tensorflow.org/api_docs/python/tf/keras/saving/load_model>`__
wrapper that allows to load models with quantized layers:
`quantizeml.models.load_model <../api_reference/quantizeml_apis.html#quantizeml.models.load_model>`__

Command line interface
----------------------

In addition to the programming interface, QuantizeML toolkit also provides a command-line interface
to perform quantization, dump a quantized model configuration, check a quantized model and insert a
rescaling layer.

quantize CLI
~~~~~~~~~~~~

Quantizing a model through the CLI uses almost the same arguments as the programing interface but
the quantization parameters are split into the parameters: input weight quantization with "-i",
weight bitwidth with "-w" and activation bitwidth with the "-a" options.

.. code-block:: bash

    quantizeml quantize -m model_keras.h5 -i 8 -w 8 -a 8

Note that without calibration options explicitly given, calibration will happen with 1024 randomly
generated samples. It is generally advised to use real samples serialized in a numpy `.npz` file.

.. code-block:: bash

    quantizeml quantize -m model_keras.h5 -i 8 -w 8 -a 8 -sa some_samples.npz -bs 128 -e 2

For akida 1.0 compatibility, it is mandatory to have activations quantized per-tensor instead of
the default per-axis quantization:

.. code-block:: bash

    quantizeml quantize -m model_keras.h5 -i 8 -w 4 -a 4 --per_tensor_activations


config CLI
~~~~~~~~~~

Advanced users might want to customize the default quantization pattern and this is made possible by
dumping a quantized model configuration to a `.json` file and quantizing again using the "-c"
option.

.. code-block:: bash

    quantizeml config -m model_keras_i8_w8_a8.h5 -o config.json

    ... manual configuration changes ...

    quantizeml quantize -m model_keras.h5 -c config.json

.. warning::
    Editing a model configuration can be complicated and might have negative effects on quantized
    accuracy or even model graph. This should be reserved to users deeply familiar with QuantizeML
    concepts.


check CLI
~~~~~~~~~

It is possible to check for quantization errors using the `check` CLI that will report inaccurate
weight scales quantization or saturation in integer operations.

.. code-block:: bash

    quantizeml check -m model_keras_i8_w8_a8.h5

insert_rescaling CLI
~~~~~~~~~~~~~~~~~~~~

Some models might not include a Rescaling layer in their architecture and have a separated
preprocessing pipeline (ie. moving from [0, 255] images to a [-1, 1] normalized representation). As
having a rescaling layer might be useful, QuantizeML offers the `insert_rescaling` CLI that will add
a Rescaling layer at the beginning of a given model.

.. code-block:: bash

    quantizeml insert_rescaling -m model_keras.h5 -s 0.007843 -o -1 -d model_updated.h5

where :math:`0.007843 = 1/127.5`.

Supported layer types
---------------------

The QuantizeML toolkit provides quantization of the following layer types which are standard Keras
layers for most part and custom QuantizeML layers for some of them:

- Neural layers
    - `Conv2D <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedConv2D>`__
    - `Conv2DTranspose <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedConv2DTranspose>`__
    - `DepthwiseConv2D <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedDepthwiseConv2D>`__
    - `DepthwiseConv2DTranspose <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedDepthwiseConv2DTranspose>`__
      (custom QuantizeML layer)
    - `SeparableConv2D <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedSeparableConv2D>`__
    - `Dense <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedDense>`__

- Transformers
    - `Attention <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedAttention>`__
      (custom QuantizeML layer)
    - `ClassToken <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedClassToken>`__
      (custom QuantizeML layer)
    - `AddPositionEmbs <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedAddPositionEmbs>`__
      (custom QuantizeML layer)
    - `ExtractToken <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedExtractToken>`__
      (custom QuantizeML layer)

- Skip connections
    - `Add <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedAdd>`__
    - `Concatenate <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedConcatenate>`__

- Normalization
    - `BatchNormalization <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedBatchNormalization>`__
    - `LayerMadNormalization <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedLayerNormalization>`__
      (custom QuantizeML layer)

- Activations
    - `ReLU <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedReLU>`__
      (both unbounded and with a max value)
    - `Shiftmax <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedShiftmax>`__
      (custom QuantizeML layer)

- Pooling
    - `MaxPool2D <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedMaxPool2D>`__
    - `GlobalAveragePooling2D <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedGlobalAveragePooling2D>`__

- Reshaping
    - `Flatten <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedFlatten>`__
    - `Permute <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedPermute>`__
    - `Reshape <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedReshape>`__

- Others
    - `Rescaling <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedRescaling>`__
    - `Dropout <../api_reference/quantizeml_apis.html#quantizeml.layers.QuantizedDropout>`__

____

.. [#fn-1] See https://en.wikipedia.org/wiki/Fixed-point_arithmetic for more details on the
    arithmetics.
