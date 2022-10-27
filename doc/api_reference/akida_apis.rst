
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

    Dense2D
    =======
    .. autoclass:: Dense2D
        :members:

    Convolutional
    =============
    .. autoclass:: Convolutional
        :members:

    SeparableConvolutional
    ======================
    .. autoclass:: SeparableConvolutional
        :members:

    Shiftmax
    ========
    .. autoclass:: Shiftmax
        :members:

    Add
    ===
    .. autoclass:: Add
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

    Tools
    =====

    Sparsity
    --------
    .. autofunction:: akida.evaluate_sparsity

    Compatibility
    -------------
    .. autofunction:: akida.compatibility.create_from_model
    .. autofunction:: akida.compatibility.transpose
