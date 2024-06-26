Model zoo performance
=====================

| The Brainchip `akida_models <https://pypi.org/project/akida-models>`__ package offers a set of pre-built
  akida compatible models (e.g MobileNet, AkidaNet, ViT), pretrained weights for those models and training
  scripts. Please refer to the `model zoo API reference <./api_reference/akida_models_apis.html#model-zoo>`__
  for a complete list of the available models.

| This page lists the performance of all models from the zoo reported for both Akida 1.0 and Akida 2.0. Please
  refer to:

* `Akida 1.0 models`_ for models targetting the Akida Neuromorphic Processor IP 1.0 and the AKD1000 reference SoC,
* `Akida 2.0 models`_ for models targetting the Akida Neuromorphic Processor IP 2.0,
* `Upgrading to Akida 2.0 tutorial <./examples/quantization/plot_1_upgrading_to_2.0.html>`_ to understand the
  architectural differences between 1.0 and 2.0 models and their respective workflows.

.. note::
    The download links provided point towards standard Tensorflow Keras models
    that must be converted to Akida model using
    `cnn2snn.convert <api_reference/cnn2snn_apis.html#convert>`_.

.. |image_icon_ref| image:: ./img/image_icon.png
   :scale: 5 %

.. |audio_icon_ref| image:: ./img/headphones_icon.png
   :scale: 5 %

.. |pointcloud_icon_ref| image:: ./img/pointcloud_icon.png
   :scale: 5 %

Akida 1.0 models
----------------

For 1.0 models, 4-bit accuracy are provided and are always obtained through a QAT phase.

.. note::
    The "8/4/4" quantization scheme stands for 8-bit weights in the input layer, 4-bit weights in
    other layers and 4-bit activations.

.. note::
    The NPs column provides the minimal number of neural processors required for the model excecution
    on the Akida IP. The numbers given are the result of the
    `map <api_reference/akida_apis.html#akida.Model.map>`_ operation using the
    `Minimal MapMode <api_reference/akida_apis.html#akida.MapMode>`_ targetting AKD1000 reference SoC.

|image_icon_ref| Image domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classification
""""""""""""""

.. |an_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_1_akidanet_imagenet.html

.. |an_160_25_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_imagenet_160_alpha_25_iq8_wq4_aq4.h5

.. |an_160_50_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_imagenet_160_alpha_50_iq8_wq4_aq4.h5

.. |an_160_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_imagenet_160_iq8_wq4_aq4.h5

.. |an_224_25_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_imagenet_224_alpha_25_iq8_wq4_aq4.h5

.. |an_224_50_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_imagenet_224_alpha_50_iq8_wq4_aq4.h5

.. |an_224_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_imagenet_224_iq8_wq4_aq4.h5

.. |mb_160_25_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/mobilenet/mobilenet_imagenet_160_alpha_25_iq8_wq4_aq4.h5

.. |mb_160_50_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/mobilenet/mobilenet_imagenet_160_alpha_50_iq8_wq4_aq4.h5

.. |mb_160_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/mobilenet/mobilenet_imagenet_160_iq8_wq4_aq4.h5

.. |mb_224_25_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/mobilenet/mobilenet_imagenet_224_alpha_25_iq8_wq4_aq4.h5

.. |mb_224_50_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/mobilenet/mobilenet_imagenet_224_alpha_50_iq8_wq4_aq4.h5

.. |mb_224_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/mobilenet/mobilenet_imagenet_224_iq8_wq4_aq4.h5

.. |ane_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/edge/plot_0_edge_learning_vision.html#

.. |ane_160_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet_edge/akidanet_imagenet_160_alpha_50_edge_iq8_wq4_aq4.h5

.. |ane_224_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet_edge/akidanet_imagenet_224_alpha_50_edge_iq8_wq4_aq4.h5

.. |an_pv_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_4_transfer_learning.html

.. |gx_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/gxnor/gxnor_mnist_iq2_wq2_aq1.h5

.. |an_pv_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_plantvillage_iq8_wq4_aq4.h5

.. |vww_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_vww_iq8_wq4_aq4.h5

+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| Architecture     | Resolution | Dataset            | #Params | Quantization | Top-1 accuracy | Example     | Size (KB) | NPs | Download       |
+==================+============+====================+=========+==============+================+=============+===========+=====+================+
| AkidaNet 0.25    | 160        | ImageNet           | 480K    | 8/4/4        | 42.58%         | |an_ex|     | 403.3     | 20  | |an_160_25_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| AkidaNet 0.5     | 160        | ImageNet           | 1.4M    | 8/4/4        | 57.80%         | |an_ex|     | 1089.1    | 24  | |an_160_50_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| AkidaNet         | 160        | ImageNet           | 4.4M    | 8/4/4        | 66.94%         | |an_ex|     | 4061.1    | 68  | |an_160_dl|    |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| AkidaNet 0.25    | 224        | ImageNet           | 480K    | 8/4/4        | 46.71%         | |an_ex|     | 409.1     | 22  | |an_224_25_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| AkidaNet 0.5     | 224        | ImageNet           | 1.4M    | 8/4/4        | 61.30%         | |an_ex|     | 1202.2    | 32  | |an_224_50_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| AkidaNet         | 224        | ImageNet           | 4.4M    | 8/4/4        | 69.65%         | |an_ex|     | 6294.0    | 116 | |an_224_dl|    |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| AkidaNet 0.5     | 160        | ImageNet           | 4.0M    | 8/4/4        | 51.66%         | |ane_ex|    | 2017.4    | 38  | |ane_160_dl|   |
| edge             |            |                    |         |              |                |             |           |     |                |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| AkidaNet 0.5     | 224        | ImageNet           | 4.0M    | 8/4/4        | 54.03%         | |ane_ex|    | 2130.5    | 46  | |ane_224_dl|   |
| edge             |            |                    |         |              |                |             |           |     |                |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| AkidaNet 0.5     | 224        | PlantVillage       | 1.1M    | 8/4/4        | 97.92%         | |an_pv_ex|  | 1019.1    | 33  | |an_pv_dl|     |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| AkidaNet 0.25    | 96         | Visual Wake Words  | 229K    | 8/4/4        | 84.77%         |             | 179.6     | 16  | |vww_dl|       |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| MobileNetV1 0.25 | 160        | ImageNet           | 467K    | 8/4/4        | 36.05%         |             | 376.4     | 20  | |mb_160_25_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| MobileNetV1 0.5  | 160        | ImageNet           | 1.3M    | 8/4/4        | 54.59%         |             | 1007.0    | 24  | |mb_160_50_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| MobileNetV1      | 160        | ImageNet           | 4.2M    | 8/4/4        | 65.47%         |             | 3525.8    | 65  | |mb_160_dl|    |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| MobileNetV1 0.25 | 224        | ImageNet           | 467K    | 8/4/4        | 39.73%         |             | 377.9     | 22  | |mb_224_25_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| MobileNetV1 0.5  | 224        | ImageNet           | 1.3M    | 8/4/4        | 58.50%         |             | 1065.3    | 32  | |mb_224_50_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| MobileNetV1      | 224        | ImageNet           | 4.2M    | 8/4/4        | 68.76%         |             | 5223.3    | 110 | |mb_224_dl|    |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+
| GXNOR            | 28         | MNIST              | 1.6M    | 2/2/1        | 98.03%         |             | 412.8     | 3   | |gx_dl|        |
+------------------+------------+--------------------+---------+--------------+----------------+-------------+-----------+-----+----------------+


Object detection
""""""""""""""""

.. |yl_voc_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_5_voc_yolo_detection.html

.. |yl_voc_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/yolo/yolo_akidanet_voc_iq8_wq4_aq4.h5

.. |yl_wf_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/yolo/yolo_akidanet_widerface_iq8_wq4_aq4.h5

+--------------+------------+--------------------------+---------+--------------+--------+-------------+-----------+-----+-------------+
| Architecture | Resolution | Dataset                  | #Params | Quantization | mAP    | Example     | Size (KB) | NPs | Download    |
+==============+============+==========================+=========+==============+========+=============+===========+=====+=============+
| YOLOv2       | 224        | PASCAL-VOC 2007 -        | 3.6M    | 8/4/4        | 41.51% | |yl_voc_ex| | 3061.4    | 71  | |yl_voc_dl| |
|              |            | person and car classes   |         |              |        |             |           |     |             |
+--------------+------------+--------------------------+---------+--------------+--------+-------------+-----------+-----+-------------+
| YOLOv2       | 224        | WIDER FACE               | 3.5M    | 8/4/4        | 77.63% |             | 3053.1    | 71  | |yl_wf_dl|  |
+--------------+------------+--------------------------+---------+--------------+--------+-------------+-----------+-----+-------------+


Regression
""""""""""

.. |reg_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_3_regression.html

.. |reg_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/vgg/vgg_utk_face_iq8_wq2_aq2.h5

+--------------+------------+--------------------------+---------+--------------+--------+----------+-----------+-----+----------+
| Architecture | Resolution | Dataset                  | #Params | Quantization | MAE    | Example  | Size (KB) | NPs | Download |
+==============+============+==========================+=========+==============+========+==========+===========+=====+==========+
| VGG-like     | 32         | UTKFace (age estimation) | 458K    | 8/2/2        | 6.1791 | |reg_ex| | 138.6     | 6   | |reg_dl| |
+--------------+------------+--------------------------+---------+--------------+--------+----------+-----------+-----+----------+


Face recognition
""""""""""""""""

.. |fid_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_faceidentification_iq8_wq4_aq4.h5

.. |fide_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet_edge/akidanet_faceidentification_edge_iq8_wq4_aq4.h5

+--------------+------------+----------------------+---------+--------------+----------+-----------+-----+-----------+
| Architecture | Resolution | Dataset              | #Params | Quantization | Accuracy | Size (KB) | NPs | Download  |
+==============+============+======================+=========+==============+==========+===========+=====+===========+
| AkidaNet 0.5 | 112×96     | CASIA Webface        | 2.3M    | 8/4/4        | 70.18%   | 1930.1    | 21  | |fid_dl|  |
|              |            | face identification  |         |              |          |           |     |           |
+--------------+------------+----------------------+---------+--------------+----------+-----------+-----+-----------+
| AkidaNet 0.5 | 112×96     | CASIA Webface        | 23.6M   | 8/4/4        | 71.13%   | 6980.2    | 34  | |fide_dl| |
| edge         |            | face identification  |         |              |          |           |     |           |
+--------------+------------+----------------------+---------+--------------+----------+-----------+-----+-----------+



|audio_icon_ref| Audio domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keyword spotting
""""""""""""""""

.. |kws_ex| image:: ./img/link_icon.png
   :scale: 4 %
   :target: examples/general/plot_2_ds_cnn_kws.html

.. |kws_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/ds_cnn/ds_cnn_kws_iq8_wq4_aq4_laq1.h5

+--------------+-----------------------+---------+--------------+----------------+----------+-----------+-----+----------+
| Architecture | Dataset               | #Params | Quantization | Top-1 accuracy | Example  | Size (KB) | NPs | Download |
+==============+=======================+=========+==============+================+==========+===========+=====+==========+
| DS-CNN       | Google speech command | 22.7K   | 8/4/4        | 91.72%         | |kws_ex| | 23.1      | 5   | |kws_dl| |
+--------------+-----------------------+---------+--------------+----------------+----------+-----------+-----+----------+


|pointcloud_icon_ref| Point cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classification
""""""""""""""

.. |p++_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/pointnet_plus/pointnet_plus_modelnet40_iq8_wq4_aq4.h5

+--------------+--------------------+---------+--------------+--------------+-----------+-----+-----------+
| Architecture | Dataset            | #Params | Quantization | Accuracy     | Size (KB) | NPs | Download  |
+==============+====================+=========+==============+==============+===========+=====+===========+
| PointNet++   | ModelNet40         | 602K    | 8/4/4        | 79.78%       | 490.9     | 12  | |p++_dl|  |
|              | 3D Point Cloud     |         |              |              |           |     |           |
+--------------+--------------------+---------+--------------+--------------+-----------+-----+-----------+


Akida 2.0 models
----------------

For 2.0 models, both 8-bit PTQ and 4-bit QAT numbers are given. When not explicitely stated 8-bit PTQ
accuracy is given as is (ie no further tuning/training, only quantization and calibration). The 4-bit
QAT is the same as for 1.0.

.. note::
    The digit for quantization scheme stands for both weights and activations bitwidth. Weights in
    the first layer are always quantized to 8-bit. When given, 'edge' means that the model backbone
    output (before classification layer) is quantized to 1-bit to allow Akida edge learning.

|image_icon_ref| Image domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classification
""""""""""""""

.. |an_160_25_8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_160_alpha_0.25_i8_w8_a8.h5

.. |an_160_25_4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_160_alpha_0.25_i8_w4_a4.h5

.. |an_160_50_8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_160_alpha_0.5_i8_w8_a8.h5

.. |an_160_50_4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_160_alpha_0.5_i8_w4_a4.h5

.. |an_160_8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_160_alpha_1_i8_w8_a8.h5

.. |an_160_4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_160_alpha_1_i8_w4_a4.h5

.. |an_224_25_8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_224_alpha_0.25_i8_w8_a8.h5

.. |an_224_25_4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_224_alpha_0.25_i8_w4_a4.h5

.. |an_224_50_8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_224_alpha_0.5_i8_w8_a8.h5

.. |an_224_50_4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_224_alpha_0.5_i8_w4_a4.h5

.. |an_224_8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_224_alpha_1_i8_w8_a8.h5

.. |an_224_4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_224_alpha_1_i8_w4_a4.h5

.. |an_pv8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_plantvillage_i8_w8_a8.h5

.. |an_pv4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_plantvillage_i8_w4_a4.h5

.. |vww8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_vww_i8_w8_a8.h5

.. |vww4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_vww_i8_w4_a4.h5

.. |an18_160_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet18/akidanet18_imagenet_160_i8_w8_a8.h5

.. |an18_224_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet18/akidanet18_imagenet_224_i8_w8_a8.h5

.. |mb_160_25_8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/mobilenet/mobilenet_imagenet_160_alpha_0.25_i8_w8_a8.h5

.. |mb_160_25_4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/mobilenet/mobilenet_imagenet_160_alpha_0.25_i8_w4_a4.h5

.. |mb_160_50_8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/mobilenet/mobilenet_imagenet_160_alpha_0.5_i8_w8_a8.h5

.. |mb_160_50_4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/mobilenet/mobilenet_imagenet_160_alpha_0.5_i8_w4_a4.h5

.. |mb_160_8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/mobilenet/mobilenet_imagenet_160_alpha_1_i8_w8_a8.h5

.. |mb_160_4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/mobilenet/mobilenet_imagenet_160_alpha_1_i8_w4_a4.h5

.. |mb_224_25_8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/mobilenet/mobilenet_imagenet_224_alpha_0.25_i8_w8_a8.h5

.. |mb_224_25_4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/mobilenet/mobilenet_imagenet_224_alpha_0.25_i8_w4_a4.h5

.. |mb_224_50_8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/mobilenet/mobilenet_imagenet_224_alpha_0.5_i8_w8_a8.h5

.. |mb_224_50_4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/mobilenet/mobilenet_imagenet_224_alpha_0.5_i8_w4_a4.h5

.. |mb_224_8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/mobilenet/mobilenet_imagenet_224_alpha_1_i8_w8_a8.h5

.. |mb_224_4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/mobilenet/mobilenet_imagenet_224_alpha_1_i8_w4_a4.h5

.. |gx2_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/gxnor/gxnor_mnist_i4_w4_a4.h5

CNNs
++++

+------------------+------------+--------------------+---------+--------------+----------+------------------+
| Architecture     | Resolution | Dataset            | #Params | Quantization | Accuracy | Download         |
+==================+============+====================+=========+==============+==========+==================+
| AkidaNet 0.25    | 160        | ImageNet           | 483K    | 8            | 48.50%   | |an_160_25_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 41.60%   | |an_160_25_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet 0.5     | 160        | ImageNet           | 1.4M    | 8            | 61.86%   | |an_160_50_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 57.93%   | |an_160_50_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet         | 160        | ImageNet           | 4.4M    | 8            | 69.94%   | |an_160_8_dl|    |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 67.23%   | |an_160_4_dl|    |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet 0.25    | 224        | ImageNet           | 483K    | 8            | 52.39%   | |an_224_25_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 46.06%   | |an_224_25_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet 0.5     | 224        | ImageNet           | 1.4M    | 8            | 64.88%   | |an_224_50_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 61.47%   | |an_224_50_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet         | 224        | ImageNet           | 4.4M    | 8            | 72.16%   | |an_224_8_dl|    |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 70.11%   | |an_224_4_dl|    |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet 0.5     | 224        | PlantVillage       | 1.2M    | 8            | 99.61%   | |an_pv8_dl|      |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 98.25%   | |an_pv4_dl|      |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet 0.25    | 96         | Visual Wake Words  | 227K    | 8            | 87.03%   | |vww8_dl|        |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 85.80%   | |vww4_dl|        |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet18       | 160        | ImageNet           | 2.4M    | 8            | 64.77%   | |an18_160_dl|    |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet18       | 224        | ImageNet           | 2.4M    | 8            | 67.32%   | |an18_224_dl|    |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| MobileNetV1 0.25 | 160        | ImageNet           | 469K    | 8            | 45.72%   | |mb_160_25_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 37.51%   | |mb_160_25_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| MobileNetV1 0.5  | 160        | ImageNet           | 1.3M    | 8            | 60.27%   | |mb_160_50_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 54.81%   | |mb_160_50_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| MobileNetV1      | 160        | ImageNet           | 4.2M    | 8            | 69.02%   | |mb_160_8_dl|    |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 65.28%   | |mb_160_4_dl|    |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| MobileNetV1 0.25 | 224        | ImageNet           | 469K    | 8            | 49.63%   | |mb_224_25_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 42.08%   | |mb_224_25_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| MobileNetV1 0.5  | 224        | ImageNet           | 1.3M    | 8            | 63.65%   | |mb_224_50_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 59.20%   | |mb_224_50_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| MobileNetV1      | 224        | ImageNet           | 4.2M    | 8            | 71.18%   | |mb_224_8_dl|    |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 68.52%   | |mb_224_4_dl|    |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| GXNOR            | 28         | MNIST              | 1.6M    | 4            | 98.81%   | |gx2_dl|         |
+------------------+------------+--------------------+---------+--------------+----------+------------------+

Transformers
++++++++++++

.. |vit_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/vit/bc_vit_ti16_224_i8_w8_a8.h5

.. |deitd_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/deit/bc_deit_dist_ti16_224_i8_w8_a8.h5

+--------------+------------+----------+---------+--------------+-----------------+------------+
| Architecture | Resolution | Dataset  | #Params | Quantization | Accuracy        | Download   |
+==============+============+==========+=========+==============+=================+============+
| ViT          | 224        | ImageNet | 5.8M    | 8            | 73.79% [#fn-2]_ | |vit_dl|   |
+--------------+------------+----------+---------+--------------+-----------------+------------+
| DeiT-dist    | 224        | ImageNet | 6.0M    | 8            | 74.34% [#fn-2]_ | |deitd_dl| |
+--------------+------------+----------+---------+--------------+-----------------+------------+

.. [#fn-2] PTQ accuracy boosted with 1 epoch of QAT.

Object detection
""""""""""""""""

.. |yl_voc8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/yolo/yolo_akidanet_voc_i8_w8_a8.h5

.. |yl_voc4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/yolo/yolo_akidanet_voc_i8_w4_a4.h5

.. |ce_voc_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/centernet/centernet_akidanet18_voc_384_i8_w8_a8.h5

.. |yl_wf8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/yolo/yolo_akidanet_widerface_i8_w8_a8.h5

.. |yl_wf4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/yolo/yolo_akidanet_widerface_i8_w4_a4.h5

+------------------------------------+------------+--------------------------+---------+--------------+--------+--------+--------+--------------+
| Architecture                       | Resolution | Dataset                  | #Params | Quantization | mAP 50 | mAP 75 | mAP    | Download     |
+====================================+============+==========================+=========+==============+========+========+========+==============+
| YOLOv2 **(AkidaNet 0.5 backbone)** | 224        | PASCAL-VOC 2007          | 3.6M    | 8            | 50.96% | 27.40% | 27.79% | |yl_voc8_dl| |
|                                    |            |                          |         |              |        |        |        |              |
|                                    |            |                          |         | 4            | 47.21% | 23.21% | 24.66% | |yl_voc4_dl| |
+------------------------------------+------------+--------------------------+---------+--------------+--------+--------+--------+--------------+
| CenterNet **(AkidaNet18 backbone)**| 384        | PASCAL-VOC 2007          | 2.4M    | 8            | 70.32% | 47.30% | 43.88% | |ce_voc_dl|  |
|                                    |            |                          |         |              |        |        |        |              |
+------------------------------------+------------+--------------------------+---------+--------------+--------+--------+--------+--------------+
| YOLOv2 **(AkidaNet 0.5 backbone)** | 224        | WIDER FACE               | 3.6M    | 8            | 80.19% |        |        | |yl_wf8_dl|  |
|                                    |            |                          |         |              |        |        |        |              |
|                                    |            |                          |         | 4            | 78.60% |        |        | |yl_wf4_dl|  |
+------------------------------------+------------+--------------------------+---------+--------------+--------+--------+--------+--------------+


Regression
""""""""""

.. |reg8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/vgg/vgg_utk_face_i8_w8_a8.h5

.. |reg4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/vgg/vgg_utk_face_i8_w4_a4.h5

+--------------+------------+--------------------------+---------+--------------+--------+-----------+
| Architecture | Resolution | Dataset                  | #Params | Quantization | MAE    | Download  |
+==============+============+==========================+=========+==============+========+===========+
| VGG-like     | 32         | UTKFace (age estimation) | 458K    | 8            | 6.0304 | |reg8_dl| |
|              |            |                          |         |              |        |           |
|              |            |                          |         | 4            | 5.8858 | |reg4_dl| |
+--------------+------------+--------------------------+---------+--------------+--------+-----------+


Face recognition
""""""""""""""""

.. |fid8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_faceidentification_i8_w8_a8.h5

.. |fid4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_faceidentification_i8_w4_a4.h5

+--------------+------------+----------------------+---------+--------------+----------+-----------+
| Architecture | Resolution | Dataset              | #Params | Quantization | Accuracy | Download  |
+==============+============+======================+=========+==============+==========+===========+
| AkidaNet 0.5 | 112×96     | CASIA Webface        | 2.3M    | 8            | 72.83%   | |fid8_dl| |
|              |            | face identification  |         |              |          |           |
|              |            |                      |         | 4            | 69.79%   | |fid4_dl| |
+--------------+------------+----------------------+---------+--------------+----------+-----------+

Segmentation
""""""""""""

.. |unet_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akida_unet/akida_unet_portrait128_i8_w8_a8.h5

+---------------+------------+-------------+---------+--------------+-----------------+-----------+
| Architecture  | Resolution | Dataset     | #Params | Quantization | Binary IOU      | Download  |
+===============+============+=============+=========+==============+=================+===========+
| AkidaUNet 0.5 | 128        | Portrait128 | 1.1M    | 8            | 0.9057 [#fn-3]_ | |unet_dl| |
+---------------+------------+-------------+---------+--------------+-----------------+-----------+

.. [#fn-3] PTQ accuracy boosted with 1 epoch QAT.

|audio_icon_ref| Audio domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keyword spotting
""""""""""""""""

.. |kws8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/ds_cnn/ds_cnn_kws_i8_w8_a8.h5

.. |kws4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/ds_cnn/ds_cnn_kws_i8_w4_a4.h5

.. |kws4e_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/ds_cnn/ds_cnn_kws_edge_i8_w4_a4.h5

+--------------+-----------------------+---------+--------------+----------------+------------+
| Architecture | Dataset               | #Params | Quantization | Top-1 accuracy | Download   |
+==============+=======================+=========+==============+================+============+
| DS-CNN       | Google speech command | 23.8K   | 8            | 92.87%         | |kws8_dl|  |
|              |                       |         |              |                |            |
|              |                       |         | 4            | 92.67%         | |kws4_dl|  |
|              |                       |         |              |                |            |
|              |                       |         | 4 + edge     | 90.61%         | |kws4e_dl| |
+--------------+-----------------------+---------+--------------+----------------+------------+


Classification
""""""""""""""

.. |audio_vit_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/audio_vit/bc_vit_urbansound_i8_w8_a8.h5

+---------------+------------+--------------+---------+--------------+-----------------+----------------+
| Architecture  | Resolution | Dataset      | #Params | Quantization | Accuracy        | Download       |
+===============+============+==============+=========+==============+=================+================+
| ViT (1 block) | 224        | Urbansound8k | 539.9K  | 8            | 97.73% [#fn-6]_ | |audio_vit_dl| |
+---------------+------------+--------------+---------+--------------+-----------------+----------------+

.. [#fn-6] PTQ accuracy boosted with 5 epochs QAT.


|pointcloud_icon_ref| Point cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classification
""""""""""""""

.. |p++8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/pointnet_plus/pointnet_plus_modelnet40_i8_w8_a8.h5

.. |p++4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/pointnet_plus/pointnet_plus_modelnet40_i8_w4_a4.h5

+--------------+--------------------+---------+--------------+-----------------+-----------+
| Architecture | Dataset            | #Params | Quantization | Accuracy        | Download  |
+==============+====================+=========+==============+=================+===========+
| PointNet++   | ModelNet40         | 605K    | 8            | 80.88% [#fn-1]_ | |p++8_dl| |
|              | 3D Point Cloud     |         |              |                 |           |
|              |                    |         | 4            | 81.56%          | |p++4_dl| |
+--------------+--------------------+---------+--------------+-----------------+-----------+

.. [#fn-1] PTQ accuracy boosted with 5 epochs QAT.
