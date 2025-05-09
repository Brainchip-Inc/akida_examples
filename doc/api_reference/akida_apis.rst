
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

    Akida layers
    ============

    Layer API
    ---------
    .. autoclass:: Layer
        :members:
    .. autoclass:: akida.Layer.Mapping
        :members:

    Common layer
    ------------
    .. autoclass:: InputData
        :members:

    Akida V1 layers
    ---------------
    .. autoclass:: InputConvolutional
        :members:
    .. autoclass:: FullyConnected
        :members:
    .. autoclass:: Convolutional
        :members:
    .. autoclass:: SeparableConvolutional
        :members:

    Akida V2 layers
    ---------------
    .. autoclass:: InputConv2D
        :members:
    .. autoclass:: Conv2D
        :members:
    .. autoclass:: Conv2DTranspose
        :members:
    .. autoclass:: Dense1D
        :members:
    .. autoclass:: DepthwiseConv2D
        :members:
    .. autoclass:: DepthwiseConv2DTranspose
        :members:
    .. autoclass:: Add
        :members:
    .. autoclass:: Concatenate
        :members:
    .. autoclass:: Dequantizer
        :members:

    Layer parameters
    ================

    LayerType
    ---------
    .. autoclass:: LayerType

    ActivationType
    --------------
    .. autoclass:: ActivationType

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
    .. autofunction:: akida.AKD1500
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
    .. autoclass:: akida.NP.NpSpace
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
