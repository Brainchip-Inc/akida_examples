
CNN2SNN toolkit
===============

Overview
--------

The Brainchip CNN2SNN toolkit provides a means to convert a quantized model obtained using
QuantizeML to a low-latency and low-power network for use with the Akida runtime.


Conversion flow
---------------

CNN2SNN offers a simple `convert <../api_reference/cnn2snn_apis.html#cnn2snn.convert>`__ function
that takes a quantized model as input and converts it into an Akida runtime compatible network.

Let's take the `DS-CNN <../api_reference/akida_models_apis.html#ds-cnn>`__ model from our zoo that
targets KWS task as an example:

.. code-block:: python

    from akida_models import ds_cnn_kws_pretrained
    from cnn2snn import convert

    # Load a pretrained 8/8/8 quantized model
    quantized_model = ds_cnn_kws_pretrained()
    model_akida = convert(quantized_model)


Conversion compatibility
^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to check if a float model is compatible with Akida conversion using the
`check_model_compatibility <../api_reference/cnn2snn_apis.html#cnn2snn.check_model_compatibility>`__
helper. This helper will check that the model quantization scheme is allowed and that building
blocks are compatible with Akida layers blocks, convert the model and optionally map on an Akida
hardware.

Command-line interface
^^^^^^^^^^^^^^^^^^^^^^

In addition to the CNN2SNN programming API, the CNN2SNN toolkit provides a command-line interface to
perform conversion to an Akida runtime compatible model. Converting a quantized model into an Akida
model using the CLI makes use of the
`convert <../api_reference/cnn2snn_apis.html#cnn2snn.convert>`__ function.

**Examples**

Convert the DS-CNN/KWS 8/8/8 quantized model:

.. code-block:: bash

    wget https://data.brainchip.com/models/AkidaV2/ds_cnn/ds_cnn_kws_i8_w8_a8.h5

    cnn2snn convert -m ds_cnn_kws_i8_w8_a8.h5

An Akida ``.fbz`` model named ``ds_cnn_kws_i8_w8_a8.fbz`` is then saved. This model can be loaded
back into an `akida.Model <../api_reference/akida_apis.html#akida.Model>`__ and run on Akida runtime.

Deprecated CLI actions
~~~~~~~~~~~~~~~~~~~~~~

The ``scale`` and ``shift`` options of the ``convert`` CLI action that were used to set input
scaling parameters are now deprecated.

CNN2SNN CLI comes with additional actions that are also deprecated and should no longer be used:
``quantize``, ``reshape`` and  ``calibrate``. You should rather use
`QuantizeML <./quantizeml.html#>`__ tool to perform the same operations.


Handling Akida 1.0 and Akida 2.0 specificities
----------------------------------------------

Conversion towards Akida 1.0 or Akida 2.0 is inherently different because the targeted SoC or IP
comes with different features. By default, a model is converted towards Akida 2.0. It is however
possible to convert towards Akida 1.0.

Using the programming interface:

.. code-block:: python

  from akida_models import ds_cnn_kws_pretrained
  from cnn2snn import convert, set_akida_version, AkidaVersion

  with set_akida_version(AkidaVersion.v1):
      quantized_model = ds_cnn_kws_pretrained()
      model_akida = convert(quantized_model)

Using the CLI interface:

.. code-block:: bash

  wget https://data.brainchip.com/models/AkidaV1/ds_cnn/ds_cnn_kws_iq8_wq4_aq4_laq1.h5

  CNN2SNN_TARGET_AKIDA_VERSION=v1 cnn2snn convert -m ds_cnn_kws_iq8_wq4_aq4_laq1.h5

.. note::
    - converting a model `quantized with QuantizeML <./quantizeml.html>`__ will use the contextual
      `AkidaVersion <../api_reference/cnn2snn_apis.html#cnn2snn.AkidaVersion>`__ to target either
      1.0 or 2.0.
    - converting a model `quantized with CNN2SNN <./cnn2snn.html#legacy-quantization-api>`__
      (deprecated path) will always target 1.0.
