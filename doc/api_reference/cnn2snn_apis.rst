
CNN2SNN Toolkit API
===================

.. automodule:: cnn2snn

    quantize
    ========
    .. autofunction:: quantize

    quantize_layer
    ==============
    .. autofunction:: quantize_layer

    convert
    =======
    .. autofunction:: convert

    A detailed description of the input_scaling parameter is given in the
    `user guide <../user_guide/cnn2snn.html#input-scaling>`__.

    check_model_compatibility
    =========================
    .. autofunction:: check_model_compatibility

    load_quantized_model
    ====================
    .. autofunction:: load_quantized_model

    WeightQuantizer
    ===============
    .. autoclass:: WeightQuantizer
        :members:

        .. automethod:: __init__

    TrainableWeightQuantizer
    ========================
    .. autoclass:: TrainableWeightQuantizer
        :members:

        .. automethod:: __init__

    MaxQuantizer
    ============
    .. autoclass:: MaxQuantizer
        :members:

        .. automethod:: __init__

    MaxPerAxisQuantizer
    ===================
    .. autoclass:: MaxPerAxisQuantizer
        :members:

        .. automethod:: __init__

    WeightFloat
    ===========
    .. autoclass:: WeightFloat
        :members:

        .. automethod:: __init__

    QuantizedConv2D
    ===============
    .. autoclass:: QuantizedConv2D
        :members:

        .. automethod:: __init__

    QuantizedDepthwiseConv2D
    ========================
    .. autoclass:: QuantizedDepthwiseConv2D
        :members:

        .. automethod:: __init__

    QuantizedDense
    ==============
    .. autoclass:: QuantizedDense
        :members:

        .. automethod:: __init__

    QuantizedSeparableConv2D
    ========================
    .. autoclass:: QuantizedSeparableConv2D
        :members:

        .. automethod:: __init__

    ActivationDiscreteRelu
    ======================
    .. autoclass:: ActivationDiscreteRelu
        :members:

        .. automethod:: __init__


    QuantizedReLU
    =============
    .. autoclass:: QuantizedReLU
        :members:

        .. automethod:: __init__