Model zoo performance
=====================

| The Brainchip `akida_models <https://pypi.org/project/akida-models>`__ package offers a set of pre-built
  Akida compatible models (e.g MobileNet, AkidaNet), pretrained weights for those models and training
  scripts. Please refer to the `model zoo API reference <./api_reference/akida_models_apis.html#model-zoo>`__
  for a complete list of the available models.

| This page lists the performance of all models from the zoo reported for both Akida 1.0 and Akida 2.0. Please
  refer to:

* `Akida 1.0 models`_ for models targeting the Akida Neuromorphic Processor IP 1.0 and the AKD1000 reference SoC,
* `Akida 2.0 models`_ for models targeting the Akida Neuromorphic Processor IP 2.0,
* `Upgrading to Akida 2.0 tutorial <./examples/quantization/plot_1_upgrading_to_2.0.html>`_ to understand the
  architectural differences between 1.0 and 2.0 models and their respective workflows.

.. note::
    The download links provided point towards standard Tensorflow Keras models
    that must be converted to Akida model using
    `cnn2snn.convert <./api_reference/cnn2snn_apis.html#convert>`_.

.. |image_icon_ref| image:: ./img/image_icon.png
   :scale: 5 %

.. |audio_icon_ref| image:: ./img/headphones_icon.png
   :scale: 5 %

.. |pointcloud_icon_ref| image:: ./img/pointcloud_icon.png
   :scale: 5 %

.. |tenns_icon_ref| image:: ./img/tenns_icon.png
   :scale: 12 %

Akida 1.0 models
----------------

For 1.0 models, 4-bit accuracy is provided and is always obtained through a QAT phase.

.. note::
    The "8/4/4" quantization scheme stands for 8-bit weights in the input layer, 4-bit weights in
    other layers and 4-bit activations.

.. note::
    The NPs column provides the minimal number of neural processors required for the model execution
    on the Akida IP. The numbers given are the result of the
    `map <./api_reference/akida_apis.html#akida.Model.map>`_ operation using the
    `Minimal MapMode <./api_reference/akida_apis.html#akida.MapMode>`_ targeting AKD1000 reference SoC.

|image_icon_ref| Image domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classification
""""""""""""""

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

.. |ane_160_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet_edge/akidanet_imagenet_160_alpha_50_edge_iq8_wq4_aq4.h5

.. |ane_224_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet_edge/akidanet_imagenet_224_alpha_50_edge_iq8_wq4_aq4.h5

.. |gx_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/gxnor/gxnor_mnist_iq2_wq2_aq1.h5

.. |an_pv_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_plantvillage_iq8_wq4_aq4.h5

.. |vww_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_vww_iq8_wq4_aq4.h5

+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| Architecture     | Resolution | Dataset            | #Params | Quantization | Top-1 accuracy | Size (KB) | NPs | Download       |
+==================+============+====================+=========+==============+================+===========+=====+================+
| AkidaNet 0.25    | 160        | ImageNet           | 480K    | 8/4/4        | 42.58%         | 403.3     | 20  | |an_160_25_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| AkidaNet 0.5     | 160        | ImageNet           | 1.4M    | 8/4/4        | 57.80%         | 1089.1    | 24  | |an_160_50_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| AkidaNet         | 160        | ImageNet           | 4.4M    | 8/4/4        | 66.94%         | 4061.1    | 68  | |an_160_dl|    |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| AkidaNet 0.25    | 224        | ImageNet           | 480K    | 8/4/4        | 46.71%         | 409.1     | 22  | |an_224_25_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| AkidaNet 0.5     | 224        | ImageNet           | 1.4M    | 8/4/4        | 61.30%         | 1202.2    | 32  | |an_224_50_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| AkidaNet         | 224        | ImageNet           | 4.4M    | 8/4/4        | 69.65%         | 6294.0    | 116 | |an_224_dl|    |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| AkidaNet 0.5     | 160        | ImageNet           | 4.0M    | 8/4/4        | 51.66%         | 2017.4    | 38  | |ane_160_dl|   |
| edge             |            |                    |         |              |                |           |     |                |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| AkidaNet 0.5     | 224        | ImageNet           | 4.0M    | 8/4/4        | 54.03%         | 2130.5    | 46  | |ane_224_dl|   |
| edge             |            |                    |         |              |                |           |     |                |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| AkidaNet 0.5     | 224        | PlantVillage       | 1.1M    | 8/4/4        | 97.92%         | 1019.1    | 33  | |an_pv_dl|     |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| AkidaNet 0.25    | 96         | Visual Wake Words  | 229K    | 8/4/4        | 84.77%         | 179.6     | 16  | |vww_dl|       |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| MobileNetV1 0.25 | 160        | ImageNet           | 467K    | 8/4/4        | 36.05%         | 376.4     | 20  | |mb_160_25_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| MobileNetV1 0.5  | 160        | ImageNet           | 1.3M    | 8/4/4        | 54.59%         | 1007.0    | 24  | |mb_160_50_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| MobileNetV1      | 160        | ImageNet           | 4.2M    | 8/4/4        | 65.47%         | 3525.8    | 65  | |mb_160_dl|    |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| MobileNetV1 0.25 | 224        | ImageNet           | 467K    | 8/4/4        | 39.73%         | 377.9     | 22  | |mb_224_25_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| MobileNetV1 0.5  | 224        | ImageNet           | 1.3M    | 8/4/4        | 58.50%         | 1065.3    | 32  | |mb_224_50_dl| |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| MobileNetV1      | 224        | ImageNet           | 4.2M    | 8/4/4        | 68.76%         | 5223.3    | 110 | |mb_224_dl|    |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+
| GXNOR            | 28         | MNIST              | 1.6M    | 2/2/1        | 98.03%         | 412.8     | 3   | |gx_dl|        |
+------------------+------------+--------------------+---------+--------------+----------------+-----------+-----+----------------+


Object detection
""""""""""""""""

.. |yl_voc_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/yolo/yolo_akidanet_voc_iq8_wq4_aq4.h5

.. |yl_wf_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/yolo/yolo_akidanet_widerface_iq8_wq4_aq4.h5

+--------------+------------+--------------------------+---------+--------------+--------+-----------+-----+-------------+
| Architecture | Resolution | Dataset                  | #Params | Quantization | mAP    | Size (KB) | NPs | Download    |
+==============+============+==========================+=========+==============+========+===========+=====+=============+
| YOLOv2       | 224        | PASCAL-VOC 2007 -        | 3.6M    | 8/4/4        | 41.51% | 3061.4    | 71  | |yl_voc_dl| |
|              |            | person and car classes   |         |              |        |           |     |             |
+--------------+------------+--------------------------+---------+--------------+--------+-----------+-----+-------------+
| YOLOv2       | 224        | WIDER FACE               | 3.5M    | 8/4/4        | 77.63% | 3053.1    | 71  | |yl_wf_dl|  |
+--------------+------------+--------------------------+---------+--------------+--------+-----------+-----+-------------+


Regression
""""""""""

.. |reg_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/vgg/vgg_utk_face_iq8_wq2_aq2.h5

+--------------+------------+--------------------------+---------+--------------+--------+-----------+-----+----------+
| Architecture | Resolution | Dataset                  | #Params | Quantization | MAE    | Size (KB) | NPs | Download |
+==============+============+==========================+=========+==============+========+===========+=====+==========+
| VGG-like     | 32         | UTKFace (age estimation) | 458K    | 8/2/2        | 6.1791 | 138.6     | 6   | |reg_dl| |
+--------------+------------+--------------------------+---------+--------------+--------+-----------+-----+----------+


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

.. |kws_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV1/ds_cnn/ds_cnn_kws_iq8_wq4_aq4_laq1.h5

+--------------+-----------------------+---------+--------------+----------------+-----------+-----+----------+
| Architecture | Dataset               | #Params | Quantization | Top-1 accuracy | Size (KB) | NPs | Download |
+==============+=======================+=========+==============+================+===========+=====+==========+
| DS-CNN       | Google speech command | 22.7K   | 8/4/4        | 91.72%         | 23.1      | 5   | |kws_dl| |
+--------------+-----------------------+---------+--------------+----------------+-----------+-----+----------+


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

For 2.0 models, both 8-bit PTQ and 4-bit QAT numbers are given. When not explicitly stated 8-bit PTQ
accuracy is given as is (ie no further tuning/training, only quantization and calibration). The 4-bit
QAT is the same as for 1.0.

.. note::
    The digit for quantization scheme stands for both weights and activations bitwidth. Weights in
    the first layer are always quantized to 8-bit.

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
   :target: https://data.brainchip.com/models/AkidaV2/gxnor/gxnor_mnist_i8_w4_a4.h5

+------------------+------------+--------------------+---------+--------------+----------+------------------+
| Architecture     | Resolution | Dataset            | #Params | Quantization | Accuracy | Download         |
+==================+============+====================+=========+==============+==========+==================+
| AkidaNet 0.25    | 160        | ImageNet           | 483K    | 8            | 48.61%   | |an_160_25_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 40.69%   | |an_160_25_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet 0.5     | 160        | ImageNet           | 1.4M    | 8            | 61.92%   | |an_160_50_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 57.42%   | |an_160_50_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet         | 160        | ImageNet           | 4.4M    | 8            | 69.96%   | |an_160_8_dl|    |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 66.80%   | |an_160_4_dl|    |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet 0.25    | 224        | ImageNet           | 483K    | 8            | 52.38%   | |an_224_25_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 44.48%   | |an_224_25_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet 0.5     | 224        | ImageNet           | 1.4M    | 8            | 64.85%   | |an_224_50_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 60.53%   | |an_224_50_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet         | 224        | ImageNet           | 4.4M    | 8            | 72.23%   | |an_224_8_dl|    |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 69.21%   | |an_224_4_dl|    |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet 0.5     | 224        | PlantVillage       | 1.2M    | 8            | 99.61%   | |an_pv8_dl|      |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 99.30%   | |an_pv4_dl|      |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet 0.25    | 96         | Visual Wake Words  | 227K    | 8            | 87.05%   | |vww8_dl|        |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 85.70%   | |vww4_dl|        |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet18       | 160        | ImageNet           | 2.4M    | 8            | 64.72%   | |an18_160_dl|    |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| AkidaNet18       | 224        | ImageNet           | 2.4M    | 8            | 67.32%   | |an18_224_dl|    |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| MobileNetV1 0.25 | 160        | ImageNet           | 469K    | 8            | 45.72%   | |mb_160_25_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 36.96%   | |mb_160_25_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| MobileNetV1 0.5  | 160        | ImageNet           | 1.3M    | 8            | 60.16%   | |mb_160_50_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 54.09%   | |mb_160_50_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| MobileNetV1      | 160        | ImageNet           | 4.2M    | 8            | 69.04%   | |mb_160_8_dl|    |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 64.92%   | |mb_160_4_dl|    |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| MobileNetV1 0.25 | 224        | ImageNet           | 469K    | 8            | 49.58%   | |mb_224_25_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 40.80%   | |mb_224_25_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| MobileNetV1 0.5  | 224        | ImageNet           | 1.3M    | 8            | 63.67%   | |mb_224_50_8_dl| |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 57.87%   | |mb_224_50_4_dl| |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| MobileNetV1      | 224        | ImageNet           | 4.2M    | 8            | 71.31%   | |mb_224_8_dl|    |
|                  |            |                    |         |              |          |                  |
|                  |            |                    |         | 4            | 67.72%   | |mb_224_4_dl|    |
+------------------+------------+--------------------+---------+--------------+----------+------------------+
| GXNOR            | 28         | MNIST              | 1.6M    | 4            | 98.57%   | |gx2_dl|         |
+------------------+------------+--------------------+---------+--------------+----------+------------------+

Object detection
""""""""""""""""

.. |yl_voc8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/yolo/yolo_akidanet_voc_i8_w8_a8.h5

.. |yl_voc4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/yolo/yolo_akidanet_voc_i8_w4_a4.h5

.. |ce_voc_dl_384| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/centernet/centernet_akidanet18_voc_384_i8_w8_a8.h5

.. |ce_voc_dl_224| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/centernet/centernet_akidanet18_voc_224_i8_w8_a8.h5

.. |yl_wf8_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/yolo/yolo_akidanet_widerface_i8_w8_a8.h5

.. |yl_wf4_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/yolo/yolo_akidanet_widerface_i8_w4_a4.h5

+------------------------------------+------------+--------------------------+---------+--------------+----------+------------------+
| Architecture                       | Resolution | Dataset                  | #Params | Quantization | mAP 50   | Download         |
+====================================+============+==========================+=========+==============+==========+==================+
| YOLOv2 **(AkidaNet 0.5 backbone)** | 224        | PASCAL-VOC 2007          | 3.6M    | 8            | 51.41%   | |yl_voc8_dl|     |
|                                    |            |                          |         |              |          |                  |
|                                    |            |                          |         | 4            | 46.74%   | |yl_voc4_dl|     |
+------------------------------------+------------+--------------------------+---------+--------------+----------+------------------+
| CenterNet **(AkidaNet18 backbone)**| 384        | PASCAL-VOC 2007          | 2.4M    | 8            | 72.77%   | |ce_voc_dl_384|  |
|                                    |            |                          |         |              | [#fn-2]_ |                  |
+------------------------------------+------------+--------------------------+---------+--------------+----------+------------------+
| CenterNet **(AkidaNet18 backbone)**| 224        | PASCAL-VOC 2007          | 2.4M    | 8            | 66.08%   | |ce_voc_dl_224|  |
|                                    |            |                          |         |              | [#fn-2]_ |                  |
+------------------------------------+------------+--------------------------+---------+--------------+----------+------------------+
| YOLOv2 **(AkidaNet 0.5 backbone)** | 224        | WIDER FACE               | 3.6M    | 8            | 80.51%   | |yl_wf8_dl|      |
|                                    |            |                          |         |              |          |                  |
|                                    |            |                          |         | 4            | 78.69%   | |yl_wf4_dl|      |
+------------------------------------+------------+--------------------------+---------+--------------+----------+------------------+

.. [#fn-2] PTQ accuracy boosted with 1 epoch QAT.

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
| VGG-like     | 32         | UTKFace (age estimation) | 458K    | 8            | 6.0299 | |reg8_dl| |
|              |            |                          |         |              |        |           |
|              |            |                          |         | 4            | 6.1421 | |reg4_dl| |
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
| AkidaNet 0.5 | 112×96     | CASIA Webface        | 2.3M    | 8            | 73.02%   | |fid8_dl| |
|              |            | face identification  |         |              |          |           |
|              |            |                      |         | 4            | 68.60%   | |fid4_dl| |
+--------------+------------+----------------------+---------+--------------+----------+-----------+

Segmentation
""""""""""""

.. |unet_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/akida_unet/akida_unet_portrait128_i8_w8_a8.h5

+---------------+------------+-------------+---------+--------------+-----------------+-----------+
| Architecture  | Resolution | Dataset     | #Params | Quantization | Binary IOU      | Download  |
+===============+============+=============+=========+==============+=================+===========+
| AkidaUNet 0.5 | 128        | Portrait128 | 1.1M    | 8            | 0.9076 [#fn-3]_ | |unet_dl| |
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

+--------------+-----------------------+---------+--------------+----------------+------------+
| Architecture | Dataset               | #Params | Quantization | Top-1 accuracy | Download   |
+==============+=======================+=========+==============+================+============+
| DS-CNN       | Google speech command | 23.8K   | 8            | 92.83%         | |kws8_dl|  |
|              |                       |         |              |                |            |
|              |                       |         | 4            | 92.58%         | |kws4_dl|  |
+--------------+-----------------------+---------+--------------+----------------+------------+

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
| PointNet++   | ModelNet40         | 277K    | 8            | 79.62% [#fn-1]_ | |p++8_dl| |
|              | 3D Point Cloud     |         |              |                 |           |
|              |                    |         | 4            | 79.50%          | |p++4_dl| |
+--------------+--------------------+---------+--------------+-----------------+-----------+

|tenns_icon_ref| TENNs
~~~~~~~~~~~~~~~~~~~~~~

Gesture recognition
"""""""""""""""""""

.. |tenns_dvs_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/tenn_spatiotemporal/tenn_spatiotemporal_dvs128_buffer_i8_w8_a8.h5

.. |tenns_jester_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/tenn_spatiotemporal/tenn_spatiotemporal_jester_buffer_i8_w8_a8.h5

+--------------------+---------+--------------+----------+-------------------+
| Dataset            | #Params | Quantization | Accuracy | Download          |
+====================+=========+==============+==========+===================+
| DVS128             | 165K    | 8            | 97.12%   | |tenns_dvs_dl|    |
+--------------------+---------+--------------+----------+-------------------+
| Jester             | 1.3M    | 8            | 95.04%   | |tenns_jester_dl| |
+--------------------+---------+--------------+----------+-------------------+

Eye tracking
""""""""""""

.. |tenns_eye_dl| image:: ./img/download_icon.png
   :scale: 4 %
   :target: https://data.brainchip.com/models/AkidaV2/tenn_spatiotemporal/tenn_spatiotemporal_eye_buffer_i8_w8_a8.h5

+--------------------+---------+--------------+---------------------+----------------+
| Dataset            | #Params | Quantization | Accuracy            | Download       |
+====================+=========+==============+=====================+================+
| Eye tracking       | 219K    | 8            | p10: 98.58%         | |tenns_eye_dl| |
| CVPR 2024          |         |              |                     |                |
|                    |         |              | mean_distance: 2.17 |                |
+--------------------+---------+--------------+---------------------+----------------+

.. [#fn-1] PTQ accuracy boosted with 5 epochs QAT.
