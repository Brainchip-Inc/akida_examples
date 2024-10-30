
Akida runtime API
=================

.. automodule:: akida
    :members:

    .. attribute:: __version__

        Returns the current version of the akida module.

    Model
    =====
    .. autoclass:: Model
        :members:
        :inherited-members:

    Layer
    =====

    Layer
    -----
    .. autoclass:: Layer
        :members:

    Mapping
    -------
    .. autoclass:: akida.Layer.Mapping
        :members:

    Akida layers
    ============
    .. autoclass:: InputData
        :members:

    Akida V1 layers
    ===============
    .. autoclass:: InputConvolutional
        :members:
    .. autoclass:: FullyConnected
        :members:
    .. autoclass:: Convolutional
        :members:
    .. autoclass:: SeparableConvolutional
        :members:

    Akida V2 layers
    ===============
    .. autoclass:: InputConv2D
        :members:
    .. autoclass:: Stem
        :members:
    .. autoclass:: Conv2D
        :members:
    .. autoclass:: Conv2DTranspose
        :members:
    .. autoclass:: Dense1D
        :members:
    .. autoclass:: Dense2D
        :members:
    .. autoclass:: DepthwiseConv2D
        :members:
    .. autoclass:: DepthwiseConv2DTranspose
        :members:
    .. autoclass:: Attention
        :members:
    .. autoclass:: VitEncoderBlock
        :members:
    .. autoclass:: Add
        :members:
    .. autoclass:: Concatenate
        :members:
    .. autoclass:: ExtractToken
        :members:
    .. autoclass:: BatchNormalization
        :members:
    .. autoclass:: MadNorm
        :members:
    .. autoclass:: Shiftmax
        :members:
    .. autoclass:: Dequantizer
        :members:

    Layer parameters
    ================

    LayerType
    ---------
    .. autoclass:: LayerType

    Padding
    -------
    .. autoclass:: Padding

    PoolType
    --------
    .. autoclass:: PoolType

    Optimizers
    ==========

    .. autoclass:: akida.core.Optimizer
        :members:
    .. autoclass:: AkidaUnsupervised

    Sequence
    ========

    Sequence
    -----------
    .. autoclass:: Sequence
        :members:

    BackendType
    -----------
    .. autoclass:: BackendType

    Pass
    ----
    .. autoclass:: Pass
        :members:


    Device
    ======

    Device
    ------
    .. autoclass:: Device
        :members:
    .. autofunction:: akida.devices
    .. autofunction:: akida.AKD1000
    .. autofunction:: akida.TwoNodesIP

    HwVersion
    ---------
    .. autoclass:: HwVersion
        :members:


    HWDevice
    ========

    HWDevice
    --------
    .. autoclass:: HardwareDevice
        :members:

    SocDriver
    ---------
    .. autoclass:: akida.core.SocDriver
        :members:

    ClockMode
    ---------
    .. autoclass:: akida.core.soc.ClockMode


    PowerMeter
    ==========
    .. autoclass:: PowerMeter
         :members:
    .. autoclass:: PowerEvent
         :members:


    NP
    ==
    .. autoclass:: akida.NP.Mesh
        :members:
    .. autoclass:: akida.NP.Info
        :members:
    .. autoclass:: akida.NP.Ident
        :members:
    .. autoclass:: akida.NP.Type
    .. autoclass:: akida.NP.Mapping
        :members:

    Mapping
    =======
    .. autoclass:: akida.MapMode
        :members:
    .. autoclass:: akida.MapConstraints
        :members:

    Tools
    =====

    Sparsity
    --------
    .. autofunction:: akida.evaluate_sparsity
