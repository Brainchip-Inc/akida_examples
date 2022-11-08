
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

    mlp_block
    ---------
    .. autofunction:: akida_models.layer_blocks.mlp_block

    multi_head_attention
    --------------------
    .. autofunction:: akida_models.layer_blocks.multi_head_attention

    transformer_block
    -----------------
    .. autofunction:: akida_models.layer_blocks.transformer_block

    conv_transpose_block
    --------------------
    .. autofunction:: akida_models.layer_blocks.conv_transpose_block

    sepconv_transpose_block
    -----------------------
    .. autofunction:: akida_models.layer_blocks.sepconv_transpose_block

    tools
    -----
    .. autofunction:: akida_models.layer_blocks.act_to_layer
    .. autofunction:: akida_models.layer_blocks.norm_to_layer

    Helpers
    =======

    BatchNormalization gamma constraint
    -----------------------------------
    .. autofunction:: add_gamma_constraint

    Knowledge distillation
    ======================

    .. autoclass:: akida_models.distiller.Distiller
    .. autoclass:: akida_models.distiller.DeitDistiller
    .. autofunction:: akida_models.distiller.KLDistillationLoss

    Pruning
    =======

    .. autofunction:: akida_models.prune_model
    .. autofunction:: akida_models.delete_filters

    Training
    ========

    .. autofunction:: akida_models.training.freeze_model_before
    .. autofunction:: akida_models.training.evaluate_model
    .. autofunction:: akida_models.training.evaluate_akida_model
    .. autofunction:: akida_models.training.compile_model

    MACS
    ====
    .. autofunction:: akida_models.macs.get_flops
    .. autofunction:: akida_models.macs.display_macs

    Utils
    =====
    .. autofunction:: akida_models.utils.fetch_file

    Model zoo
    =========

    AkidaNet
    --------

    ImageNet
    ~~~~~~~~
    .. autofunction:: akida_models.akidanet_imagenet
    .. autofunction:: akida_models.akidanet_imagenet_pretrained
    .. autofunction:: akida_models.akidanet_edge_imagenet
    .. autofunction:: akida_models.akidanet_edge_imagenet_pretrained
    .. autofunction:: akida_models.akidanet_imagenette_pretrained
    .. autofunction:: akida_models.akidanet_cats_vs_dogs_pretrained
    .. autofunction:: akida_models.akidanet_faceidentification_pretrained
    .. autofunction:: akida_models.akidanet_faceidentification_edge_pretrained
    .. autofunction:: akida_models.akidanet_faceverification_pretrained
    .. autofunction:: akida_models.akidanet_melanoma_pretrained
    .. autofunction:: akida_models.akidanet_odir5k_pretrained
    .. autofunction:: akida_models.akidanet_retinal_oct_pretrained
    .. autofunction:: akida_models.akidanet_ecg_pretrained
    .. autofunction:: akida_models.akidanet_plantvillage_pretrained
    .. autofunction:: akida_models.akidanet_cifar10_pretrained
    .. autofunction:: akida_models.akidanet_vww_pretrained

    Preprocessing
    *************
    .. autofunction:: akida_models.imagenet.preprocessing.preprocess_image
    .. autofunction:: akida_models.imagenet.preprocessing.index_to_label

    Mobilenet
    ---------

    ImageNet
    ~~~~~~~~
    .. autofunction:: akida_models.mobilenet_imagenet
    .. autofunction:: akida_models.mobilenet_imagenet_pretrained
    .. autofunction:: akida_models.mobilenet_edge_imagenet
    .. autofunction:: akida_models.mobilenet_edge_imagenet_pretrained

    DS-CNN
    ------

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

    ImageNet
    ~~~~~~~~
    .. autofunction:: akida_models.vgg_imagenet
    .. autofunction:: akida_models.vgg_imagenet_pretrained

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

    PointNet++
    ----------

    ModelNet40
    ~~~~~~~~~~
    .. autofunction:: akida_models.pointnet_plus_modelnet40
    .. autofunction:: akida_models.pointnet_plus_modelnet40_pretrained

    Processing
    **********
    .. autofunction:: akida_models.modelnet40.preprocessing.get_modelnet_from_file
    .. autofunction:: akida_models.modelnet40.preprocessing.get_modelnet

    GXNOR
    -----

    MNIST
    ~~~~~
    .. autofunction:: akida_models.gxnor_mnist
    .. autofunction:: akida_models.gxnor_mnist_pretrained
