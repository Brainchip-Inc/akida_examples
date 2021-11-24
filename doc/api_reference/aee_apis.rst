
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

    LearningType
    ------------
    .. autoclass:: LearningType


    Sequence
    ========

    Sequence
    -----------
    .. autoclass:: Sequence
        :members:

    BackendType
    -----------
    .. autoclass:: BackendType


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

    soc
    ---
    .. autoclass:: akida.core.soc.ClockMode
    .. autofunction:: akida.core.soc.get_clock_mode
    .. autofunction:: akida.core.soc.set_clock_mode


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


    Tools
    =====

    Sparsity
    --------
    .. autofunction:: akida.evaluate_sparsity

    Compatibility
    -------------
    .. autofunction:: akida.compatibility.create_from_model
