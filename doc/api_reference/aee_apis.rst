
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

        .. automethod:: __init__

    Layer
    =====
    .. autoclass:: Layer
        :members:

    LayerStatistics
    ===============
    .. autoclass:: LayerStatistics
        :members:

    Observer
    ========
    .. autoclass:: Observer
        :members:

    InputData
    =========
    .. autoclass:: InputData
        :members:

        .. automethod:: __init__

    InputConvolutional
    ==================
    .. autoclass:: InputConvolutional
        :members:

        .. automethod:: __init__

    FullyConnected
    ==============
    .. autoclass:: FullyConnected
        :members:

        .. automethod:: __init__

    Convolutional
    =============
    .. autoclass:: Convolutional
        :members:

        .. automethod:: __init__

    SeparableConvolutional
    ======================
    .. autoclass:: SeparableConvolutional
        :members:

        .. automethod:: __init__

    Dense
    ======
    .. autoclass:: Dense
        :members:
        :inherited-members:

    Sparse
    ======
    .. autoclass:: Sparse
        :members:
        :inherited-members:

    coords_to_sparse
    ================
    .. autofunction:: coords_to_sparse

    dense_to_sparse
    ===============
    .. autofunction:: dense_to_sparse

    packetize
    =========
    .. autofunction:: packetize

    Backend
    =======
    .. autoclass:: BackendType
    .. autofunction:: has_backend
    .. autofunction:: backends

    ConvolutionMode
    ===============
    .. autoclass:: ConvolutionMode

    PoolingType
    ===========
    .. autoclass:: PoolingType

    LearningType
    ============
    .. autoclass:: LearningType

    Compatibility
    =============
    .. autofunction:: akida.compatibility.model_hardware_incompatibilities
    .. autofunction:: akida.compatibility.create_from_model
    .. autoclass:: NsocVersion