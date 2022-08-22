
CNN2SNN Toolkit API
===================

.. automodule:: cnn2snn

    Tool functions
    ==============

    quantize
    --------
    .. autofunction:: quantize

    quantize_layer
    --------------
    .. autofunction:: quantize_layer

    convert
    -------
    .. autofunction:: convert

    A detailed description of the input_scaling parameter is given in the
    `user guide <../user_guide/cnn2snn.html#input-scaling>`__.

    check_model_compatibility
    -------------------------
    .. autofunction:: check_model_compatibility

    load_quantized_model
    --------------------
    .. autofunction:: load_quantized_model

    Transforms
    ----------
    .. autofunction:: cnn2snn.transforms.sequentialize
    .. autofunction:: cnn2snn.transforms.syncretize
    .. autofunction:: cnn2snn.transforms.invert_batchnorm_pooling
    .. autofunction:: cnn2snn.transforms.fold_batchnorm
    .. autofunction:: cnn2snn.transforms.weights_homogeneity
    .. autofunction:: cnn2snn.transforms.normalize_separable_layer
    .. autofunction:: cnn2snn.transforms.normalize_separable_model
    .. autofunction:: cnn2snn.transforms.reshape

    Calibration
    -----------
    .. autofunction:: cnn2snn.calibration.QuantizationSampler
    .. autofunction:: cnn2snn.calibration.bias_correction
    .. autofunction:: cnn2snn.calibration.adaround

    Quantizers
    ==========

    WeightQuantizer
    ---------------
    .. autoclass:: cnn2snn.quantization_ops.WeightQuantizer
        :members:
        :show-inheritance:

    LinearWeightQuantizer
    ---------------------
    .. autoclass:: cnn2snn.quantization_ops.LinearWeightQuantizer
        :members:
        :show-inheritance:

    StdWeightQuantizer
    ------------------
    .. autoclass:: StdWeightQuantizer
        :members:
        :show-inheritance:

    StdPerAxisQuantizer
    -------------------
    .. autoclass:: StdPerAxisQuantizer
        :members:
        :show-inheritance:

    MaxQuantizer
    ------------
    .. autoclass:: MaxQuantizer
        :members:
        :show-inheritance:

    MaxPerAxisQuantizer
    -------------------
    .. autoclass:: MaxPerAxisQuantizer
        :members:
        :show-inheritance:

    Quantized layers
    ================

    QuantizedConv2D
    ---------------
    .. autoclass:: QuantizedConv2D
        :members:
        :show-inheritance:

    QuantizedDense
    --------------
    .. autoclass:: QuantizedDense
        :members:
        :show-inheritance:

    QuantizedSeparableConv2D
    ------------------------
    .. autoclass:: QuantizedSeparableConv2D
        :members:
        :show-inheritance:

    QuantizedActivation
    -------------------
    .. autoclass:: QuantizedActivation
        :members:
        :show-inheritance:

    ActivationDiscreteRelu
    ----------------------
    .. autoclass:: ActivationDiscreteRelu
        :members:
        :show-inheritance:

    QuantizedReLU
    -------------
    .. autoclass:: QuantizedReLU
        :members:
        :show-inheritance:
