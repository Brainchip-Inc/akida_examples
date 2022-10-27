
QuantizeML API
==============

.. automodule:: quantizeml

    Layers
    ======

    Reshaping
    ---------

    QuantizedFlatten
    ~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedFlatten

    QuantizedPermute
    ~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedPermute

    QuantizedReshape
    ~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedReshape

    Activations
    -----------

    QuantizedGELU
    ~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedGELU

    QuantizedReLU
    ~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedReLU

    Attention
    ---------
    .. autoclass:: quantizeml.layers.Attention

    QuantizedAttention
    ~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedAttention

    string_to_softmax
    ~~~~~~~~~~~~~~~~~
    .. autofunction:: quantizeml.layers.string_to_softmax

    CNN
    ---

    PaddedConv2D
    ~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.PaddedConv2D

    QuantizedConv2D
    ~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedConv2D

    QuantizedConv2DTranspose
    ~~~~~~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedConv2DTranspose

    QuantizedDepthwiseConv2D
    ~~~~~~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedDepthwiseConv2D

    QuantizedSeparableConv2D
    ~~~~~~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedSeparableConv2D

    SeparableConv2DTranspose
    ~~~~~~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.SeparableConv2DTranspose

    QuantizedSeparableConv2DTranspose
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedSeparableConv2DTranspose

    QuantizedDense
    ~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedDense

    Dropout
    -------
    .. autoclass:: quantizeml.layers.QDropout

    Add
    ---
    .. autoclass:: quantizeml.layers.Add

    QuantizedAdd
    ~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedAdd

    .. autofunction:: quantizeml.layers.deserialize_quant_object

    Normalization
    -------------

    LayerMadNormalization
    ~~~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.LayerMadNormalization

    QuantizedLayerNormalization
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedLayerNormalization

    Pooling
    -------

    QuantizedMaxPool2D
    ~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedMaxPool2D

    QuantizedGlobalAveragePooling2D
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedGlobalAveragePooling2D

    Quantizers
    ----------

    Quantizer
    ~~~~~~~~~
    .. autoclass:: quantizeml.layers.Quantizer
        :members:
        :show-inheritance:

    WeightQuantizer
    ~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.WeightQuantizer
        :members:
        :show-inheritance:

    FixedPointQuantizer
    ~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.FixedPointQuantizer
        :members:
        :show-inheritance:

    Dequantizer
    ~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.Dequantizer
        :members:
        :show-inheritance:

    Reciprocal
    ----------
    .. autoclass:: quantizeml.layers.Reciprocal

    QuantizedReciprocal
    ~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedReciprocal

    Shiftmax
    --------
    .. autoclass:: quantizeml.layers.Shiftmax

    QuantizedShiftmax
    ~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedShiftmax

    .. autofunction:: quantizeml.layers.shiftmax

    Transformers
    ------------

    ClassToken
    ~~~~~~~~~~
    .. autoclass:: quantizeml.layers.ClassToken

    QuantizedClassToken
    ~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedClassToken

    AddPositionEmbs
    ~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.AddPositionEmbs

    QuantizedAddPositionEmbs
    ~~~~~~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedAddPositionEmbs

    ExtractToken
    ~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.ExtractToken

    QuantizedExtractToken
    ~~~~~~~~~~~~~~~~~~~~~
    .. autoclass:: quantizeml.layers.QuantizedExtractToken

    Models
    ======

    Transforms
    ----------
    .. autofunction:: quantizeml.models.transforms.invert_batchnorm_pooling
    .. autofunction:: quantizeml.models.transforms.fold_batchnorms
    .. autofunction:: quantizeml.models.transforms.fold_rescaling

    Quantization
    ------------
    .. autofunction:: quantizeml.models.quantize
    .. autofunction:: quantizeml.models.dump_config

    Utils
    -----
    .. autofunction:: quantizeml.models.load_model
    .. autofunction:: quantizeml.models.deep_clone_model
    .. autofunction:: quantizeml.models.set_calibrate
    .. autofunction:: quantizeml.models.insert_layer
    .. autofunction:: quantizeml.models.load_weights
    .. autofunction:: quantizeml.models.save_weights

    Tensors
    =======

    QTensor
    -------
    .. autoclass:: quantizeml.tensors.QTensor
        :members:
        :show-inheritance:

    FixedPoint
    ----------
    .. autoclass:: quantizeml.tensors.FixedPoint
        :members:
        :show-inheritance:

    QFloat
    ------
    .. autoclass:: quantizeml.tensors.QFloat
        :members:
        :show-inheritance:

    Reciprocal
    ----------
    .. autofunction:: quantizeml.tensors.reciprocal_lut
