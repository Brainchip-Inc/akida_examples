
Akida models API
================

.. automodule:: akida_models
    :members:

    Quantization blocks
    ===================

    conv_block
    ----------
    .. autofunction:: akida_models.quantization_blocks.conv_block

    separable_conv_block
    --------------------
    .. autofunction:: akida_models.quantization_blocks.separable_conv_block

    dense_block
    -----------
    .. autofunction:: akida_models.quantization_blocks.dense_block

    Model zoo
    =========

    Mobilenet
    ---------

    ImageNet
    ~~~~~~~~~~~~~~~~~~
    .. autofunction:: akida_models.mobilenet_imagenet

    Preprocessing
    *************
    .. autofunction:: akida_models.imagenet.preprocessing.process_record_dataset
    .. autofunction:: akida_models.imagenet.preprocessing.get_filenames
    .. autofunction:: akida_models.imagenet.preprocessing.parse_record
    .. autofunction:: akida_models.imagenet.preprocessing.input_fn
    .. autofunction:: akida_models.imagenet.preprocessing.preprocess_image
    .. autofunction:: akida_models.imagenet.preprocessing.index_to_label
    .. autofunction:: akida_models.imagenet.preprocessing.resize_and_crop

    DS-CNN
    ---------

    CIFAR-10
    ~~~~~~~~~~~~~~~~~
    .. autofunction:: akida_models.ds_cnn_cifar10

    KWS
    ~~~~~~~~~~~~~
    .. autofunction:: akida_models.ds_cnn_kws

    Preprocessing
    *************
    .. autofunction:: akida_models.kws.preprocessing.prepare_model_settings
    .. autofunction:: akida_models.kws.preprocessing.prepare_words_list
    .. autofunction:: akida_models.kws.preprocessing.which_set
    .. autoclass:: akida_models.kws.preprocessing.AudioProcessor
        :members:

    VGG
    ---------

    CIFAR-10
    ~~~~~~~~~~~
    .. autofunction:: akida_models.vgg_cifar10

    UTK Face
    ~~~~~~~~
    .. autofunction:: akida_models.vgg_utk_face

    Preprocessing
    *************
    .. autofunction:: akida_models.utk_face.preprocessing.load_data

    YOLO
    ---------

    .. autofunction:: akida_models.yolo_base
    .. autofunction:: akida_models.yolo_widerface_pretrained
    .. autofunction:: akida_models.yolo_voc_pretrained

    Processing
    ~~~~~~~~~~
    .. autofunction:: akida_models.detection.processing.load_image
    .. autofunction:: akida_models.detection.processing.preprocess_image
    .. autofunction:: akida_models.detection.processing.decode_output
    .. autoclass:: akida_models.detection.processing.BoundingBox
        :members: