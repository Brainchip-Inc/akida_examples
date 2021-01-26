
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

    load_partial_weights
    --------------------
    .. autofunction:: load_partial_weights

    Quantizers
    ==========

    WeightQuantizer
    ---------------
    .. autoclass:: cnn2snn.quantization_ops.WeightQuantizer
        :members:

    LinearWeightQuantizer
    ---------------------
    .. autoclass:: cnn2snn.quantization_ops.LinearWeightQuantizer
        :members:

    StdWeightQuantizer
    ------------------
    .. autoclass:: StdWeightQuantizer
        :members:

    TrainableStdWeightQuantizer
    ---------------------------
    .. autoclass:: TrainableStdWeightQuantizer
        :members:

    MaxQuantizer
    ------------
    .. autoclass:: MaxQuantizer
        :members:

    MaxPerAxisQuantizer
    -------------------
    .. autoclass:: MaxPerAxisQuantizer
        :members:

    Quantized layers
    ================

    QuantizedConv2D
    ---------------
    .. autoclass:: QuantizedConv2D
        :members:

    QuantizedDepthwiseConv2D
    ------------------------
    .. autoclass:: QuantizedDepthwiseConv2D
        :members:

    QuantizedDense
    --------------
    .. autoclass:: QuantizedDense
        :members:

    QuantizedSeparableConv2D
    ------------------------
    .. autoclass:: QuantizedSeparableConv2D
        :members:

    QuantizedActivation
    -------------------
    .. autoclass:: QuantizedActivation
        :members:

    ActivationDiscreteRelu
    ----------------------
    .. autoclass:: ActivationDiscreteRelu
        :members:

    QuantizedReLU
    -------------
    .. autoclass:: QuantizedReLU
        :members:
