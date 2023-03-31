Model zoo performances
======================

This page lets you discover all of Akida model zoo machine learning models with
their respective performances.

.. note::
    The download links provided point towards standard Tensorflow Keras models
    that must be converted to Akida model using
    `cnn2snn.convert <api_reference/cnn2snn_apis.html#convert>`_.


.. |image_icon_ref| image:: ./img/image_icon.png
   :scale: 5 %

|image_icon_ref| Image domain
-----------------------------

Classification
~~~~~~~~~~~~~~

.. |an_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_1_akidanet_imagenet.html

.. |an_160_25_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/akidanet/akidanet_imagenet_160_alpha_25_iq8_wq4_aq4.h5

.. |an_160_50_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/akidanet/akidanet_imagenet_160_alpha_50_iq8_wq4_aq4.h5

.. |an_160_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/akidanet/akidanet_imagenet_160_iq8_wq4_aq4.h5

.. |an_224_25_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/akidanet/akidanet_imagenet_224_alpha_25_iq8_wq4_aq4.h5

.. |an_224_50_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/akidanet/akidanet_imagenet_224_alpha_50_iq8_wq4_aq4.h5

.. |an_224_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/akidanet/akidanet_imagenet_224_iq8_wq4_aq4.h5

.. |mb_160_25_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/stride2/mobilenet_imagenet_160_alpha_25_iq8_wq4_aq4.h5

.. |mb_160_50_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/stride2/mobilenet_imagenet_160_alpha_50_iq8_wq4_aq4.h5

.. |mb_160_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/stride2/mobilenet_imagenet_160_iq8_wq4_aq4.h5

.. |mb_224_25_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/stride2/mobilenet_imagenet_224_alpha_25_iq8_wq4_aq4.h5

.. |mb_224_50_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/stride2/mobilenet_imagenet_224_alpha_50_iq8_wq4_aq4.h5

.. |mb_224_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/mobilenet/stride2/mobilenet_imagenet_224_iq8_wq4_aq4.h5

.. |ane_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/edge/plot_0_edge_learning_vision.html#

.. |ane_160_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/akidanet_edge/akidanet_imagenet_160_alpha_50_edge_iq8_wq4_aq4.h5

.. |ane_224_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/akidanet_edge/akidanet_imagenet_224_alpha_50_edge_iq8_wq4_aq4.h5

.. |an_pv_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_4_transfer_learning.html

.. |gx_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_0_gxnor_mnist.html

.. |gx_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/gxnor/gxnor_mnist_iq2_wq2_aq1.h5

.. |an_pv_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/akidanet/akidanet_plantvillage_iq8_wq4_aq4.h5

.. |vww_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/akidanet/akidanet_vww_iq8_wq4_aq4.h5

+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| Architecture     | Resolution | Dataset            | Quantization | Top-1 accuracy | Example     | #Params | Size (KB) | NPs | Download       |
+==================+============+====================+==============+================+=============+=========+===========+=====+================+
| AkidaNet 0.25    | 160        | ImageNet           | 8/4/4        | 42.58%         | |an_ex|     | 480K    | 392.3     | 23  | |an_160_25_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| AkidaNet 0.5     | 160        | ImageNet           | 8/4/4        | 57.80%         | |an_ex|     | 1.4M    | 1099.4    | 30  | |an_160_50_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| AkidaNet         | 160        | ImageNet           | 8/4/4        | 66.94%         | |an_ex|     | 4.4M    | 4090.2    | 81  | |an_160_dl|    |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| AkidaNet 0.25    | 224        | ImageNet           | 8/4/4        | 46.71%         | |an_ex|     | 480K    | 398.1     | 25  | |an_224_25_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| AkidaNet 0.5     | 224        | ImageNet           | 8/4/4        | 61.30%         | |an_ex|     | 1.4M    | 1214.4    | 38  | |an_224_50_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| AkidaNet         | 224        | ImageNet           | 8/4/4        | 69.65%         | |an_ex|     | 4.4M    | 6322.6    | 129 | |an_224_dl|    |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| AkidaNet 0.5     | 160        | ImageNet           | 8/4/4        | 51.66%         | |ane_ex|    | 4.0M    | 2017.1    | 38  | |ane_160_dl|   |
| edge             |            |                    |              |                |             |         |           |     |                |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| AkidaNet 0.5     | 224        | ImageNet           | 8/4/4        | 54.03%         | |ane_ex|    | 4.0M    | 2130.1    | 46  | |ane_224_dl|   |
| edge             |            |                    |              |                |             |         |           |     |                |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| AkidaNet 0.5     | 224        | PlantVillage       | 8/4/4        | 97.92%         | |an_pv_ex|  | 1.1M    | 1018.8    | 33  | |an_pv_dl|     |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| AkidaNet 0.25    | 96         | Visual Wake Words  | 8/4/4        | 84.77%         |             | 229K    | 179.2     | 16  | |vww_dl|       |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| MobileNetV1 0.25 | 160        | ImageNet           | 8/4/4        | 36.05%         |             | 467K    | 365.4     | 23  | |mb_160_25_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| MobileNetV1 0.5  | 160        | ImageNet           | 8/4/4        | 54.59%         |             | 1.3M    | 1017.1    | 30  | |mb_160_50_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| MobileNetV1      | 160        | ImageNet           | 8/4/4        | 65.47%         |             | 4.2M    | 3554.5    | 78  | |mb_160_dl|    |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| MobileNetV1 0.25 | 224        | ImageNet           | 8/4/4        | 39.73%         |             | 467K    | 366.9     | 25  | |mb_224_25_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| MobileNetV1 0.5  | 224        | ImageNet           | 8/4/4        | 58.50%         |             | 1.3M    | 1075.4    | 38  | |mb_224_50_dl| |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| MobileNetV1      | 224        | ImageNet           | 8/4/4        | 68.76%         |             | 4.2M    | 5251.8    | 123 | |mb_224_dl|    |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+
| GXNOR            | 28         | MNIST              | 2/2/1        | 98.03%         | |gx_ex|     | 1.6M    | 412.6     | 3   | |gx_dl|        |
+------------------+------------+--------------------+--------------+----------------+-------------+---------+-----------+-----+----------------+


Object detection
~~~~~~~~~~~~~~~~

.. |yl_voc_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_5_voc_yolo_detection.html

.. |yl_voc_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/yolo/yolo_akidanet_voc_iq8_wq4_aq4.h5

.. |yl_wf_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/yolo/yolo_akidanet_widerface_iq8_wq4_aq4.h5

+--------------+------------+--------------------------+--------------+--------+-------------+---------+-----------+-----+-------------+
| Architecture | Resolution | Dataset                  | Quantization | mAP    | Example     | #Params | Size (KB) | NPs | Download    |
+==============+============+==========================+==============+========+=============+=========+===========+=====+=============+
| YOLOv2       | 224        | PASCAL-VOC 2007 -        | 8/4/4        | 41.51% | |yl_voc_ex| | 3.6M    | 3061.0    | 71  | |yl_voc_dl| |
|              |            | person and car classes   |              |        |             |         |           |     |             |
+--------------+------------+--------------------------+--------------+--------+-------------+---------+-----------+-----+-------------+
| YOLOv2       | 224        | WIDER FACE               | 8/4/4        | 77.63% |             | 3.5M    | 3052.7    | 71  | |yl_wf_dl|  |
+--------------+------------+--------------------------+--------------+--------+-------------+---------+-----------+-----+-------------+


Regression
~~~~~~~~~~

.. |reg_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_3_regression.html

.. |reg_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/vgg/vgg_utk_face_iq8_wq2_aq2.h5

+--------------+------------+--------------------------+--------------+--------+----------+---------+-----------+-----+----------+
| Architecture | Resolution | Dataset                  | Quantization | MAE    | Example  | #Params | Size (KB) | NPs | Download |
+==============+============+==========================+==============+========+==========+=========+===========+=====+==========+
| VGG-like     | 32         | UTKFace (age estimation) | 8/2/2        | 6.1791 | |reg_ex| | 458K    | 139.8     | 6   | |reg_dl| |
+--------------+------------+--------------------------+--------------+--------+----------+---------+-----------+-----+----------+


Face recognition
~~~~~~~~~~~~~~~~

.. |fid_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/akidanet/akidanet_faceidentification_iq8_wq4_aq4.h5

.. |fide_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/akidanet_edge/akidanet_faceidentification_edge_iq8_wq4_aq4.h5

+--------------+------------+----------------------+--------------+----------+---------+-----------+-----+-----------+
| Architecture | Resolution | Dataset              | Quantization | Accuracy | #Params | Size (KB) | NPs | Download  |
+==============+============+======================+==============+==========+=========+===========+=====+===========+
| AkidaNet 0.5 | 112x96     | CASIA Webface        | 8/4/4        | 70.18%   | 2.3M    | 1929.8    | 21  | |fid_dl|  |
|              |            | face identification  |              |          |         |           |     |           |
+--------------+------------+----------------------+--------------+----------+---------+-----------+-----+-----------+
| AkidaNet 0.5 | 112x96     | CASIA Webface        | 8/4/4        | 71.13%   | 23.6M   | 6979.6    | 35  | |fide_dl| |
| edge         |            | face identification  |              |          |         |           |     |           |
+--------------+------------+----------------------+--------------+----------+---------+-----------+-----+-----------+


.. |audio_icon_ref| image:: ./img/headphones_icon.png
   :scale: 5 %

|audio_icon_ref| Audio domain
-----------------------------

Keyword spotting
~~~~~~~~~~~~~~~~

.. |kws_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_2_ds_cnn_kws.html

.. |kws_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/ds_cnn/ds_cnn_kws_iq8_wq4_aq4_laq1.h5

+--------------+-----------------------+--------------+----------------+----------+---------+-----------+-----+----------+
| Architecture | Dataset               | Quantization | Top-1 accuracy | Example  | #Params | Size (KB) | NPs | Download |
+==============+=======================+==============+================+==========+=========+===========+=====+==========+
| DS-CNN       | Google speech command | 8/4/4        | 91.72%         | |kws_ex| | 22.7K   | 22.8      | 5   | |kws_dl| |
+--------------+-----------------------+--------------+----------------+----------+---------+-----------+-----+----------+

.. |pointcloud_icon_ref| image:: ./img/pointcloud_icon.png
   :scale: 5 %

|pointcloud_icon_ref| Point cloud
---------------------------------

Classification
~~~~~~~~~~~~~~

.. |p++_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: http://data.brainchip.com/models/pointnet_plus/pointnet_plus_modelnet40_iq8_wq4_aq4.h5

+--------------+--------------------+--------------+--------------+---------------+---------+-----------+-----+-----------+
| Architecture | Dataset            | Quantization | Accuracy     | Input scaling | #Params | Size (KB) | NPs | Download  |
+==============+====================+==============+==============+===============+=========+===========+=====+===========+
| PointNet++   | ModelNet40         | 8/4/4        | 84.76%       | (127, 127)    | 602K    | 528.5     | 17  | |p++_dl|  |
|              | 3D Point Cloud     |              |              |               |         |           |     |           |
+--------------+--------------------+--------------+--------------+---------------+---------+-----------+-----+-----------+
