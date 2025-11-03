
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
    .. autoclass:: quantizeml.layers.QuantizedActivation

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

    Temporal convolution
    --------------------
    .. autoclass:: quantizeml.layers.BufferTempConv
    .. autoclass:: quantizeml.layers.QuantizedBufferTempConv
    .. autoclass:: quantizeml.layers.DepthwiseBufferTempConv
    .. autoclass:: quantizeml.layers.QuantizedDepthwiseBufferTempConv

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

    Rescaling
    ---------
    .. autoclass:: quantizeml.layers.QuantizedRescaling

    Dropout
    -------
    .. autoclass:: quantizeml.layers.QuantizedDropout

    Quantizer/Dequantizer
    ---------------------
    .. autoclass:: quantizeml.layers.InputQuantizer
        :members:
    .. autoclass:: quantizeml.layers.Dequantizer
        :members:

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

    Quantization
    ------------
    .. autofunction:: quantizeml.models.transforms.sanitize
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

    Utils
    -----
    .. autofunction:: quantizeml.models.apply_weights_to_model

    Reset buffers
    -------------
    .. autofunction:: quantizeml.models.reset_buffers


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
    .. autoclass:: quantizeml.onnx_support.layers.OnnxLayer
    .. autoclass:: quantizeml.onnx_support.layers.QuantizedConv2D
    .. autoclass:: quantizeml.onnx_support.layers.QuantizedDepthwise2D
    .. autoclass:: quantizeml.onnx_support.layers.QuantizedConv2DTranspose
    .. autoclass:: quantizeml.onnx_support.layers.QuantizedDepthwise2DTranspose
    .. autoclass:: quantizeml.onnx_support.layers.QuantizedBufferTempConv
    .. autoclass:: quantizeml.onnx_support.layers.QuantizedDepthwiseBufferTempConv
    .. autoclass:: quantizeml.onnx_support.layers.QuantizedDense1D
    .. autoclass:: quantizeml.onnx_support.layers.QuantizedAdd
    .. autoclass:: quantizeml.onnx_support.layers.QuantizedConcat
    .. autoclass:: quantizeml.onnx_support.layers.InputQuantizer
    .. autoclass:: quantizeml.onnx_support.layers.Dequantizer

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
    .. autofunction:: quantizeml.analysis.measure_weight_quantization_error

    Metrics
    -------
    .. autofunction:: quantizeml.analysis.tools.SMAPE
    .. autofunction:: quantizeml.analysis.tools.Saturation
    .. autofunction:: quantizeml.analysis.tools.print_metric_table
