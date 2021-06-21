Model zoo performances
======================

This page lets you discover all of Akida model zoo machine learning models with
their respective performances.

.. note::
    The download links provided point towards standard Tensorflow Keras models
    that must be converted to Akida model using
    `cnn2snn.convert <api_reference/cnn2snn_apis.html#convert>`_ with the
    given `input_scaling` value.


.. |image_icon_ref| image:: ./img/image_icon.png
   :scale: 5 %

|image_icon_ref| Image domain
-----------------------------

Classification
~~~~~~~~~~~~~~

.. |mb_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_2_mobilenet_imagenet.html

.. |mb_160_25_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/mobilenet_imagenet_160_alpha_25_iq8_wq4_aq4.h5

.. |mb_160_50_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/mobilenet_imagenet_160_alpha_50_iq8_wq4_aq4.h5

.. |mb_160_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/mobilenet_imagenet_160_iq8_wq4_aq4.h5

.. |mb_224_25_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/mobilenet_imagenet_224_alpha_25_iq8_wq4_aq4.h5

.. |mb_224_50_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/mobilenet_imagenet_224_alpha_50_iq8_wq4_aq4.h5

.. |mb_224_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/mobilenet_imagenet_224_iq8_wq4_aq4.h5

.. |mbe_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/edge/plot_0_edge_learning_vision.html#

.. |mbe_160_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet_edge/mobilenet_imagenet_160_alpha_50_edge_iq8_wq4_aq4.h5

.. |mbe_224_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet_edge/mobilenet_imagenet_224_alpha_50_edge_iq8_wq4_aq4.h5

.. |vgg11_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/vgg/vgg11_imagenet_224_iq8_wq4_aq4.h5

.. |ds_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_1_ds_cnn_cifar10.html

.. |ds_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/ds_cnn/ds_cnn_cifar10_iq4_wq4_aq4.h5

.. |vgg_c10_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/vgg/vgg_cifar10_iq2_wq2_aq2.h5

.. |mb_cvd_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_5_transfer_learning.html

.. |mb_cvd_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/mobilenet_cats_vs_dogs_iq8_wq4_aq4.h5

.. |mb_ite_25_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/mobilenet_imagenette_224_alpha_25_iq8_wq4_aq4.h5

.. |mb_ite_50_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/mobilenet_imagenette_224_alpha_50_iq8_wq4_aq4.h5

.. |mb_ite_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/mobilenet_imagenette_224_iq8_wq4_aq4.h5

.. |vgg_mel_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/vgg/vgg11_melanoma_iq8_wq4_aq4.h5

.. |vgg_odir_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/vgg/vgg11_odir5k_iq8_wq4_aq4.h5

.. |vgg_oct_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/vgg/vgg11_retinal_oct_iq8_wq4_aq4.h5

.. |gx_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_0_gxnor_mnist.html

.. |gx_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/gxnor/gxnor_mnist.fbz

+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| Architecture     | Resolution | Dataset            | Quantization | Top-1 accuracy | Example     | Input scaling  | Download       |
+==================+============+====================+==============+================+=============+================+================+
| MobileNetV1 0.25 | 160        | ImageNet           | 8/4/4        | 40.86%         | |mb_ex|     | (128, 128)     | |mb_160_25_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| MobileNetV1 0.5  | 160        | ImageNet           | 8/4/4        | 55.94%         | |mb_ex|     | (128, 128)     | |mb_160_50_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| MobileNetV1      | 160        | ImageNet           | 8/4/4        | 66.40%         | |mb_ex|     | (128, 128)     | |mb_160_dl|    |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| MobileNetV1 0.25 | 224        | ImageNet           | 8/4/4        | 45.12%         | |mb_ex|     | (128, 128)     | |mb_224_25_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| MobileNetV1 0.5  | 224        | ImageNet           | 8/4/4        | 59.76%         | |mb_ex|     | (128, 128)     | |mb_224_50_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| MobileNetV1      | 224        | ImageNet           | 8/4/4        | 69.53%         | |mb_ex|     | (128, 128)     | |mb_224_dl|    |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| MobileNetV1      | 160        | Cats vs dogs       | 8/4/4        | 98.11%         | |mb_cvd_ex| | (127.5, 127.5) | |mb_cvd_dl|    |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| MobileNetV1 0.25 | 224        | Imagenette         | 8/4/4        | 86.83%         |             | (128, 128)     | |mb_ite_25_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| MobileNetV1 0.5  | 224        | Imagenette         | 8/4/4        | 92.05%         |             | (128, 128)     | |mb_ite_50_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| MobileNetV1      | 224        | Imagenette         | 8/4/4        | 94.34%         |             | (128, 128)     | |mb_ite_dl|    |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| MobileNetV1 0.5  | 160        | ImageNet           | 8/4/4        | 49.69%         | |mbe_ex|    | (128, 128)     | |mbe_160_dl|   |
| edge             |            |                    |              |                |             |                |                |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| MobileNetV1 0.5  | 224        | ImageNet           | 8/4/4        | 51.83%         | |mbe_ex|    | (128, 128)     | |mbe_224_dl|   |
| edge             |            |                    |              |                |             |                |                |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| DS-CNN           | 224        | CIFAR10            | 4/4/4        | 93.04%         | |ds_ex|     | (255, 0)       | |ds_dl|        |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| VGG-like         | 224        | CIFAR10            | 2/2/2        | 90.67%         |             | (255, 0)       | |vgg_c10_dl|   |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| VGG11            | 224        | ImageNet           | 8/4/4        | 51.09%         |             | (128, 128)     | |vgg11_dl|     |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| VGG11            | 224        | SIIM-ISIC Melanoma | 8/4/4        | 98.31% -       |             | (255, 0)       | |vgg_mel_dl|   |
|                  |            | Classification     |              | AUROC 0,8020   |             |                |                |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| VGG11            | 224        | ODIR-5K Ocular     | 8/4/4        | 90.53% -       |             | (255, 0)       | |vgg_odir_dl|  |
|                  |            | disease recognition|              | AUROC 0,9473   |             |                |                |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| VGG11            | 224        | Retinal OCT ocular | 8/4/4        | 80.60% -       |             | (255, 0)       | |vgg_oct_dl|   |
|                  |            | disease recognition|              | AUROC 0,9768   |             |                |                |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+
| GXNOR            | 28         | MNIST              | 2/2/2        | 99.07%         | |gx_ex|     |                | |gx_dl|        |
+------------------+------------+--------------------+--------------+----------------+-------------+----------------+----------------+


Object detection
~~~~~~~~~~~~~~~~

.. |yl_voc_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_6_voc_yolo_detection.html

.. |yl_voc_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/yolo/yolo_voc_iq8_wq4_aq4.h5

.. |yl_wf_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/yolo/yolo_widerface_iq8_wq4_aq4.h5

+--------------+------------+--------------------------+--------------+--------+-------------+----------------+-------------+
| Architecture | Resolution | Dataset                  | Quantization | mAP    | Example     | Input scaling  | Download    |
+==============+============+==========================+==============+========+=============+================+=============+
| YOLOv2       | 224        | PASCAL-VOC 2007 -        | 8/4/4        | 29.39% | |yl_voc_ex| | (127.5, 127.5) | |yl_voc_dl| |
|              |            | person and car classes   |              |        |             |                |             |
+--------------+------------+--------------------------+--------------+--------+-------------+----------------+-------------+
| YOLOv2       | 224        | WIDER FACE               | 8/4/4        | 71.44% |             | (127.5, 127.5) | |yl_wf_dl|  |
+--------------+------------+--------------------------+--------------+--------+-------------+----------------+-------------+


Regression
~~~~~~~~~~

.. |reg_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_4_regression.html

.. |reg_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/vgg/vgg_utk_face_iq8_wq2_aq2.h5

+--------------+------------+--------------------------+--------------+--------+----------+---------------+----------+
| Architecture | Resolution | Dataset                  | Quantization | MAE    | Example  | Input scaling | Download |
+==============+============+==========================+==============+========+==========+===============+==========+
| VGG-like     | 32         | UTKFace (age estimation) | 8/2/2        | 6.1791 | |reg_ex| | (127, 127)    | |reg_dl| |
+--------------+------------+--------------------------+--------------+--------+----------+---------------+----------+


.. |audio_icon_ref| image:: ./img/headphones_icon.png
   :scale: 5 %

|audio_icon_ref| Audio domain
-----------------------------

Keyword spotting
~~~~~~~~~~~~~~~~

.. |kws_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_3_ds_cnn_kws.html

.. |kws_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/ds_cnn/ds_cnn_kws_iq8_wq4_aq4_laq1.h5

+--------------+-----------------------+--------------+----------------+----------+---------------+----------+
| Architecture | Dataset               | Quantization | Top-1 accuracy | Example  | Input scaling | Download |
+==============+=======================+==============+================+==========+===============+==========+
| DS-CNN       | Google speech command | 8/4/4        | 91.33%         | |kws_ex| | (225, 0)      | |kws_dl| |
+--------------+-----------------------+--------------+----------------+----------+---------------+----------+
