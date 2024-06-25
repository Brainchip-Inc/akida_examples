
QuantizeML API
==============

.. automodule:: quantizeml

    Layers
    ======

    Reshaping
    ---------
    .. autoclass:: quantizeml.layers.QuantizedFlatten
    .. autoclass:: quantizeml.layers.QuantizedPermute
    .. autoclass:: quantizeml.layers.QuantizedReshape

    Activations
    -----------
    .. autoclass:: quantizeml.layers.QuantizedReLU

    Attention
    ---------
    .. autoclass:: quantizeml.layers.Attention
    .. autoclass:: quantizeml.layers.QuantizedAttention
    .. autofunction:: quantizeml.layers.string_to_softmax

    Normalization
    -------------
    .. autoclass:: quantizeml.layers.QuantizedBatchNormalization
    .. autoclass:: quantizeml.layers.LayerMadNormalization
    .. autoclass:: quantizeml.layers.QuantizedLayerNormalization

    Convolution
    -----------
    .. autoclass:: quantizeml.layers.PaddedConv2D
    .. autoclass:: quantizeml.layers.QuantizedConv2D
    .. autoclass:: quantizeml.layers.QuantizedConv2DTranspose

    Depthwise convolution
    ---------------------
    .. autoclass:: quantizeml.layers.QuantizedDepthwiseConv2D
    .. autoclass:: quantizeml.layers.DepthwiseConv2DTranspose
    .. autoclass:: quantizeml.layers.QuantizedDepthwiseConv2DTranspose

    Separable convolution
    ---------------------
    .. autoclass:: quantizeml.layers.QuantizedSeparableConv2D

    Dense
    -----
    .. autoclass:: quantizeml.layers.QuantizedDense

    Skip connection
    ---------------
    .. autoclass:: quantizeml.layers.Add
    .. autoclass:: quantizeml.layers.QuantizedAdd
    .. autoclass:: quantizeml.layers.QuantizedConcatenate

    Pooling
    -------
    .. autoclass:: quantizeml.layers.QuantizedMaxPool2D
    .. autoclass:: quantizeml.layers.QuantizedGlobalAveragePooling2D

    Shiftmax
    --------
    .. autoclass:: quantizeml.layers.Shiftmax
    .. autoclass:: quantizeml.layers.QuantizedShiftmax
    .. autofunction:: quantizeml.layers.shiftmax

    Transformers
    ------------
    .. autoclass:: quantizeml.layers.ClassToken
    .. autoclass:: quantizeml.layers.QuantizedClassToken
    .. autoclass:: quantizeml.layers.AddPositionEmbs
    .. autoclass:: quantizeml.layers.QuantizedAddPositionEmbs
    .. autoclass:: quantizeml.layers.ExtractToken
    .. autoclass:: quantizeml.layers.QuantizedExtractToken

    Rescaling
    ---------
    .. autoclass:: quantizeml.layers.QuantizedRescaling

    Dropout
    -------
    .. autoclass:: quantizeml.layers.QuantizedDropout

    Quantizers
    ----------
    .. autoclass:: quantizeml.layers.Quantizer
    .. autoclass:: quantizeml.layers.WeightQuantizer
        :members:
        :show-inheritance:
    .. autoclass:: quantizeml.layers.AlignedWeightQuantizer
        :members:
        :show-inheritance:
    .. autoclass:: quantizeml.layers.OutputQuantizer
        :members:
        :show-inheritance:
    .. autoclass:: quantizeml.layers.Dequantizer
        :members:
        :show-inheritance:

    Calibration
    -----------
    .. autoclass::  quantizeml.layers.OutputObserver

    Recording
    ---------
    .. autofunction:: quantizeml.layers.recording
    .. autoclass:: quantizeml.layers.TensorRecorder
    .. autoclass:: quantizeml.layers.FixedPointRecorder
    .. autoclass:: quantizeml.layers.QFloatRecorder
    .. autoclass:: quantizeml.layers.NonTrackVariable
    .. autoclass:: quantizeml.layers.NonTrackFixedPointVariable

    Models
    ======

    Transforms
    ----------
    .. autofunction:: quantizeml.models.transforms.align_rescaling
    .. autofunction:: quantizeml.models.transforms.invert_batchnorm_pooling
    .. autofunction:: quantizeml.models.transforms.fold_batchnorms
    .. autofunction:: quantizeml.models.transforms.insert_layer
    .. autofunction:: quantizeml.models.transforms.insert_rescaling
    .. autofunction:: quantizeml.models.transforms.invert_relu_maxpool
    .. autofunction:: quantizeml.models.transforms.remove_zeropadding2d
    .. autofunction:: quantizeml.models.transforms.replace_lambda
    .. autofunction:: quantizeml.models.transforms.sanitize

    Quantization
    ------------
    .. autofunction:: quantizeml.models.quantize
    .. autofunction:: quantizeml.models.dump_config
    .. autofunction:: quantizeml.models.record_quantization_variables

    Quantization parameters
    -----------------------
    .. autoclass:: quantizeml.models.QuantizationParams
        :members:
    .. autofunction:: quantizeml.models.get_quantization_params
    .. autofunction:: quantizeml.models.quantization

    Calibration
    -----------
    .. autofunction:: quantizeml.models.calibrate
    .. autofunction:: quantizeml.models.calibration_required

    Utils
    -----
    .. autofunction:: quantizeml.models.apply_weights_to_model

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

    ONNX support
    ============

    Layers
    ------
    .. autofunction:: quantizeml.onnx_support.layers.OnnxLayer
    .. autofunction:: quantizeml.onnx_support.layers.QuantizedConv2D
    .. autofunction:: quantizeml.onnx_support.layers.QuantizedDepthwise2D
    .. autofunction:: quantizeml.onnx_support.layers.QuantizedDense1D
    .. autofunction:: quantizeml.onnx_support.layers.QuantizedAdd
    .. autofunction:: quantizeml.onnx_support.layers.InputQuantizer
    .. autofunction:: quantizeml.onnx_support.layers.Dequantizer

    Custom patterns
    ---------------
    .. autofunction:: quantizeml.onnx_support.quantization.custom_pattern_scope

    Model I/O
    =========
    .. autofunction:: quantizeml.load_model
    .. autofunction:: quantizeml.save_model

    Analysis
    ========

    Kernel distribution
    -------------------
    .. autofunction:: quantizeml.analysis.plot_kernel_distribution

    Quantization error
    ------------------
    .. autofunction:: quantizeml.analysis.measure_layer_quantization_error
    .. autofunction:: quantizeml.analysis.measure_cumulative_quantization_error

    Metrics
    -------
    .. autofunction:: quantizeml.analysis.tools.WeightedMAPE
    .. autofunction:: quantizeml.analysis.tools.Saturation
    .. autofunction:: quantizeml.analysis.tools.print_metric_table
