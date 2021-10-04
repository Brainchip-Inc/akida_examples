
Akida Execution Engine API
==========================

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
    .. autoclass:: Layer
        :members:

    LayerStatistics
    ===============
    .. autoclass:: LayerStatistics
        :members:

    InputData
    =========
    .. autoclass:: InputData
        :members:

    InputConvolutional
    ==================
    .. autoclass:: InputConvolutional
        :members:

    FullyConnected
    ==============
    .. autoclass:: FullyConnected
        :members:

    Convolutional
    =============
    .. autoclass:: Convolutional
        :members:

    SeparableConvolutional
    ======================
    .. autoclass:: SeparableConvolutional
        :members:

    Concat
    ======
    .. autoclass:: Concat
        :members:

    BackendType
    ===========
    .. autoclass:: BackendType

    ConvolutionMode
    ===============
    .. autoclass:: ConvolutionMode

    PoolingType
    ===========
    .. autoclass:: PoolingType

    LearningType
    ============
    .. autoclass:: LearningType

    HwVersion
    =========
    .. autoclass:: HwVersion
        :members:

    Compatibility
    =============
    .. autofunction:: akida.compatibility.create_from_model

    Device
    ======
    .. autoclass:: Device
        :members:
    .. autofunction:: akida.devices
    .. autofunction:: akida.AKD1000
    .. autofunction:: akida.TwoNodesIP

    Sequence
    ==============
    .. autoclass:: Sequence
        :members:

    Program
    ==============
    .. autoclass:: akida.core.Program
        :members:

    NP
    ==
    .. autoclass:: akida.NP.Mesh
        :members:
    .. autoclass:: akida.NP.Info
        :members:
    .. autoclass:: akida.NP.Ident
        :members:

    soc
    ===
    .. autoclass:: akida.core.soc.ClockMode
    .. autofunction:: akida.core.soc.get_clock_mode
    .. autofunction:: akida.core.soc.set_clock_mode
