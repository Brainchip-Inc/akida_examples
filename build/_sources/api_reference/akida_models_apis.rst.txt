
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

    CIFAR-10
    ~~~~~~~~~~~~~~~~~
    .. autofunction:: akida_models.mobilenet_cifar10

    ImageNet
    ~~~~~~~~~~~~~~~~~~
    .. autofunction:: akida_models.mobilenet_imagenet

    Preprocessing
    *************
    .. autofunction:: akida_models.mobilenet.imagenet.imagenet_preprocessing.process_record_dataset
    .. autofunction:: akida_models.mobilenet.imagenet.imagenet_preprocessing.get_filenames
    .. autofunction:: akida_models.mobilenet.imagenet.imagenet_preprocessing.parse_record
    .. autofunction:: akida_models.mobilenet.imagenet.imagenet_preprocessing.input_fn
    .. autofunction:: akida_models.mobilenet.imagenet.imagenet_preprocessing.preprocess_image
    .. autofunction:: akida_models.mobilenet.imagenet.imagenet_preprocessing.index_to_label

    KWS
    ~~~~~~~~~~~~~
    .. autofunction:: akida_models.mobilenet_kws

    Preprocessing
    *************
    .. autofunction:: akida_models.mobilenet.kws.kws_preprocessing.prepare_model_settings
    .. autofunction:: akida_models.mobilenet.kws.kws_preprocessing.prepare_words_list
    .. autofunction:: akida_models.mobilenet.kws.kws_preprocessing.which_set
    .. autoclass:: akida_models.mobilenet.kws.kws_preprocessing.AudioProcessor
        :members:

    VGG
    ---------

    CIFAR-10
    ~~~~~~~~~~~
    .. autofunction::  akida_models.vgg_cifar10
