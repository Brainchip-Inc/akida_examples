
Akida models API
================

.. automodule:: akida_models
    :members:

    Layer blocks
    ============

    CNN blocks
    ----------
    .. autofunction:: akida_models.layer_blocks.conv_block
    .. autofunction:: akida_models.layer_blocks.separable_conv_block
    .. autofunction:: akida_models.layer_blocks.dense_block

    Transposed blocks
    -----------------
    .. autofunction:: akida_models.layer_blocks.conv_transpose_block
    .. autofunction:: akida_models.layer_blocks.sepconv_transpose_block

    Detection block
    ---------------
    .. autofunction:: akida_models.layer_blocks.yolo_head_block

    Spatiotemporal blocks
    ---------------------
    .. autofunction:: akida_models.layer_blocks.conv3d_block
    .. autofunction:: akida_models.layer_blocks.spatiotemporal_block

    Helpers
    =======

    Gamma constraint
    ----------------
    .. autofunction:: akida_models.gamma_constraint.add_gamma_constraint

    Unfusing SeparableConvolutional
    -------------------------------
    .. autofunction:: akida_models.unfuse_sepconv_layers.unfuse_sepconv2d

    Extract samples
    ---------------
    .. autofunction:: akida_models.extract.extract_samples


    Knowledge distillation
    ======================

    .. autoclass:: akida_models.distiller.Distiller
    .. autofunction:: akida_models.distiller.KLDistillationLoss

    Sparsity
    ========
    .. autofunction:: akida_models.sparsity.compute_sparsity

    Model I/O
    =========
    .. autofunction:: akida_models.model_io.load_model
    .. autofunction:: akida_models.model_io.load_weights
    .. autofunction:: akida_models.model_io.save_weights
    .. autofunction:: akida_models.model_io.get_model_path

    Utils
    =====
    .. autofunction:: akida_models.utils.fetch_file
    .. autofunction:: akida_models.utils.get_tensorboard_callback
    .. autofunction:: akida_models.utils.get_params_by_version

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
    .. autofunction:: akida_models.akidanet18_imagenet
    .. autofunction:: akida_models.akidanet18_imagenet_pretrained
    .. autofunction:: akida_models.akidanet_faceidentification_pretrained
    .. autofunction:: akida_models.akidanet_faceidentification_edge_pretrained
    .. autofunction:: akida_models.akidanet_plantvillage_pretrained
    .. autofunction:: akida_models.akidanet_vww_pretrained

    Preprocessing
    *************
    .. autofunction:: akida_models.imagenet.get_preprocessed_samples
    .. autofunction:: akida_models.imagenet.preprocessing.preprocess_image
    .. autofunction:: akida_models.imagenet.preprocessing.index_to_label

    Mobilenet
    ---------

    ImageNet
    ~~~~~~~~
    .. autofunction:: akida_models.mobilenet_imagenet
    .. autofunction:: akida_models.mobilenet_imagenet_pretrained

    DS-CNN
    ------

    KWS
    ~~~
    .. autofunction:: akida_models.ds_cnn_kws
    .. autofunction:: akida_models.ds_cnn_kws_pretrained

    Preprocessing
    *************
    .. autoclass:: akida_models.kws.preprocessing.AudioProcessor
        :members:

    VGG
    ---

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

    Detection data
    ~~~~~~~~~~~~~~

    .. autofunction:: akida_models.detection.data.get_detection_datasets
    .. autofunction:: akida_models.detection.voc.data.get_voc_dataset
    .. autofunction:: akida_models.detection.coco.data.get_coco_dataset
    .. autofunction:: akida_models.detection.widerface.data.get_widerface_dataset

    Preprocessing
    *************
    .. autofunction:: akida_models.detection.preprocess_data.preprocess_dataset

    Utils
    *****
    .. autofunction:: akida_models.detection.data_utils.remove_empty_objects
    .. autofunction:: akida_models.detection.data_utils.get_dataset_length
    .. autoclass:: akida_models.detection.data_utils.Coord
        :members:

    YOLO Toolkit
    ~~~~~~~~~~~~

    Processing
    **********
    .. autofunction:: akida_models.detection.processing.load_image
    .. autofunction:: akida_models.detection.processing.preprocess_image
    .. autofunction:: akida_models.detection.processing.get_affine_transform
    .. autofunction:: akida_models.detection.processing.apply_affine_transform_to_bboxes
    .. autofunction:: akida_models.detection.processing.desize_bboxes
    .. autofunction:: akida_models.detection.processing.decode_output
    .. autofunction:: akida_models.detection.processing.create_yolo_targets
    .. autoclass:: akida_models.detection.processing.BoundingBox
        :members:

    YOLO Data Augmentation
    **********************
    .. autofunction:: akida_models.detection.data_augmentation.augment_sample
    .. autofunction:: akida_models.detection.data_augmentation.build_yolo_aug_pipeline
    .. autofunction:: akida_models.detection.data_augmentation.init_random_vars

    Performance
    ***********
    .. autoclass:: akida_models.detection.map_evaluation.MapEvaluation
        :members:

    Anchors
    *******
    .. autofunction:: akida_models.detection.generate_anchors.generate_anchors

    Utils
    *****
    .. autofunction:: akida_models.detection.box_utils.xywh_to_xyxy
    .. autofunction:: akida_models.detection.box_utils.xyxy_to_xywh
    .. autofunction:: akida_models.detection.box_utils.compute_overlap
    .. autofunction:: akida_models.detection.box_utils.compute_center_xy
    .. autofunction:: akida_models.detection.box_utils.compute_center_wh

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

    CenterNet
    ---------
    .. autofunction:: akida_models.centernet_base
    .. autofunction:: akida_models.centernet_voc_pretrained
    .. autofunction:: akida_models.centernet.centernet_processing.decode_output
    .. autofunction:: akida_models.centernet.centernet_utils.create_centernet_targets
    .. autofunction:: akida_models.centernet.centernet_utils.build_centernet_aug_pipeline
    .. autoclass:: akida_models.centernet.centernet_loss.CenternetLoss

    AkidaUNet
    ---------
    .. autofunction:: akida_models.akida_unet_portrait128
    .. autofunction:: akida_models.akida_unet_portrait128_pretrained

    Spatiotemporal TENNs
    --------------------
    .. autofunction:: akida_models.tenn_spatiotemporal_dvs128
    .. autofunction:: akida_models.tenn_spatiotemporal_dvs128_pretrained
    .. autofunction:: akida_models.tenn_spatiotemporal_eye
    .. autofunction:: akida_models.tenn_spatiotemporal_eye_pretrained
    .. autofunction:: akida_models.tenn_spatiotemporal_jester
    .. autofunction:: akida_models.tenn_spatiotemporal_jester_pretrained

    Conversion
    ~~~~~~~~~~
    .. autofunction:: akida_models.tenn_spatiotemporal.convert_to_buffer
