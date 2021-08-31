
Akida models API
================

.. automodule:: akida_models
    :members:

    Layer blocks
    ============

    conv_block
    ----------
    .. autofunction:: akida_models.layer_blocks.conv_block

    separable_conv_block
    --------------------
    .. autofunction:: akida_models.layer_blocks.separable_conv_block

    dense_block
    -----------
    .. autofunction:: akida_models.layer_blocks.dense_block

    Helpers
    =======

    BatchNormalization gamma constraint
    -----------------------------------
    .. autofunction:: add_gamma_constraint

    Knowledge distillation
    ======================

    .. autoclass:: akida_models.distiller.Distiller
        :members:
    .. autofunction:: akida_models.distiller.KLDistillationLoss

    Pruning
    =======

    .. autofunction:: akida_models.delete_filters
    .. autofunction:: akida_models.prune_model


    Model zoo
    =========

    Mobilenet
    ---------

    ImageNet
    ~~~~~~~~
    .. autofunction:: akida_models.mobilenet_imagenet
    .. autofunction:: akida_models.mobilenet_imagenet_pretrained
    .. autofunction:: akida_models.mobilenet_imagenette_pretrained
    .. autofunction:: akida_models.mobilenet_cats_vs_dogs_pretrained
    .. autofunction:: akida_models.mobilenet_edge_imagenet
    .. autofunction:: akida_models.mobilenet_edge_imagenet_pretrained

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
    ------

    CIFAR-10
    ~~~~~~~~
    .. autofunction:: akida_models.ds_cnn_cifar10
    .. autofunction:: akida_models.ds_cnn_cifar10_pretrained

    KWS
    ~~~
    .. autofunction:: akida_models.ds_cnn_kws
    .. autofunction:: akida_models.ds_cnn_kws_pretrained

    Preprocessing
    *************
    .. autofunction:: akida_models.kws.preprocessing.prepare_model_settings
    .. autofunction:: akida_models.kws.preprocessing.prepare_words_list
    .. autofunction:: akida_models.kws.preprocessing.which_set
    .. autoclass:: akida_models.kws.preprocessing.AudioProcessor
        :members:

    VGG
    ---

    CIFAR-10
    ~~~~~~~~
    .. autofunction:: akida_models.vgg_cifar10
    .. autofunction:: akida_models.vgg_cifar10_pretrained

    ImageNet
    ~~~~~~~~
    .. autofunction:: akida_models.vgg_imagenet
    .. autofunction:: akida_models.vgg_imagenet_pretrained
    .. autofunction:: akida_models.vgg_melanoma_pretrained
    .. autofunction:: akida_models.vgg_odir5k_pretrained
    .. autofunction:: akida_models.vgg_retinal_oct_pretrained

    UTK Face
    ~~~~~~~~
    .. autofunction:: akida_models.vgg_utk_face
    .. autofunction:: akida_models.vgg_utk_face_pretrained

    Preprocessing
    *************
    .. autofunction:: akida_models.utk_face.preprocessing.load_data

    YOLO
    ----

    .. autofunction:: akida_models.yolo_base
    .. autofunction:: akida_models.yolo_widerface_pretrained
    .. autofunction:: akida_models.yolo_voc_pretrained

    YOLO Toolkit
    ~~~~~~~~~~~~

    Processing
    **********
    .. autofunction:: akida_models.detection.processing.load_image
    .. autofunction:: akida_models.detection.processing.preprocess_image
    .. autofunction:: akida_models.detection.processing.decode_output
    .. autofunction:: akida_models.detection.processing.parse_voc_annotations
    .. autofunction:: akida_models.detection.processing.parse_widerface_annotations
    .. autoclass:: akida_models.detection.processing.BoundingBox
        :members:

    Performances
    ************
    .. autoclass:: akida_models.detection.map_evaluation.MapEvaluation
        :members:

    Anchors
    *******
    .. autofunction:: akida_models.detection.generate_anchors.generate_anchors

    ConvTiny
    --------

    CWRU
    ~~~~
    .. autofunction:: akida_models.convtiny_cwru
    .. autofunction:: akida_models.convtiny_cwru_pretrained