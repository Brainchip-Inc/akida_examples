:orphan:

Akida examples
==============
To learn how to use Akida, the QuantizeML and CNN2SNN toolkits and check the Akida accelerator
performance against some commonly used datasets please refer to the sections below.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

General examples
----------------



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Using the MNIST dataset, this example shows the definition and training of a TF-Keras floating point model, its quantization to 8-bit with the help of calibration, its quantization to 4-bit using QAT and its conversion to Akida. Notice that the performance of the original TF-Keras floating point model is maintained throughout the Akida flow. Please refer to the Akida user guide for further information.">

.. only:: html

  .. image:: /examples/general/images/thumb/sphx_glr_plot_0_global_workflow_thumb.png
    :alt:

  :ref:`sphx_glr_examples_general_plot_0_global_workflow.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Global Akida workflow</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial presents how to convert, map, and capture performance from AKD1000 Hardware using an AkidaNet model.">

.. only:: html

  .. image:: /examples/general/images/thumb/sphx_glr_plot_1_akidanet_imagenet_thumb.png
    :alt:

  :ref:`sphx_glr_examples_general_plot_1_akidanet_imagenet.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AkidaNet/ImageNet inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial illustrates the process of developing an Akida-compatible speech recognition model that can identify thirty-two different keywords.">

.. only:: html

  .. image:: /examples/general/images/thumb/sphx_glr_plot_2_ds_cnn_kws_thumb.png
    :alt:

  :ref:`sphx_glr_examples_general_plot_2_ds_cnn_kws.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DS-CNN/KWS inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial aims to demonstrate the comparable accuracy of the Akida-compatible model to the traditional TF-Keras model in performing an age estimation task.">

.. only:: html

  .. image:: /examples/general/images/thumb/sphx_glr_plot_3_regression_thumb.png
    :alt:

  :ref:`sphx_glr_examples_general_plot_3_regression.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age estimation (regression) example</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial presents how to perform transfer learning for quantized models targeting an Akida accelerator.">

.. only:: html

  .. image:: /examples/general/images/thumb/sphx_glr_plot_4_transfer_learning_thumb.png
    :alt:

  :ref:`sphx_glr_examples_general_plot_4_transfer_learning.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Transfer learning with AkidaNet for PlantVillage</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates that Akida can perform object detection. This is illustrated using a subset of the PASCAL-VOC 2007 dataset which contains 20 classes. The YOLOv2 architecture from Redmon et al (2016) has been chosen to tackle this object detection problem.">

.. only:: html

  .. image:: /examples/general/images/thumb/sphx_glr_plot_5_voc_yolo_detection_thumb.png
    :alt:

  :ref:`sphx_glr_examples_general_plot_5_voc_yolo_detection.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">YOLO/PASCAL-VOC detection tutorial</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates image segmentation with an Akida-compatible model as illustrated through person segmentation using the Portrait128 dataset.">

.. only:: html

  .. image:: /examples/general/images/thumb/sphx_glr_plot_6_segmentation_thumb.png
    :alt:

  :ref:`sphx_glr_examples_general_plot_6_segmentation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Segmentation tutorial</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Global Akida workflow guide describes the steps to prepare a model for Akida starting from a TF-Keras model. Here we will instead describe a workflow to go from a model trained in PyTorch.">

.. only:: html

  .. image:: /examples/general/images/thumb/sphx_glr_plot_7_global_pytorch_workflow_thumb.png
    :alt:

  :ref:`sphx_glr_examples_general_plot_7_global_pytorch_workflow.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PyTorch to Akida workflow</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/general/plot_0_global_workflow
   /examples/general/plot_1_akidanet_imagenet
   /examples/general/plot_2_ds_cnn_kws
   /examples/general/plot_3_regression
   /examples/general/plot_4_transfer_learning
   /examples/general/plot_5_voc_yolo_detection
   /examples/general/plot_6_segmentation
   /examples/general/plot_7_global_pytorch_workflow

Quantization
------------



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial provides a comprehensive understanding of quantization in QuantizeML python package. Refer to QuantizeML user guide  and Global Akida workflow tutorial for additional resources.">

.. only:: html

  .. image:: /examples/quantization/images/thumb/sphx_glr_plot_0_advanced_quantizeml_thumb.png
    :alt:

  :ref:`sphx_glr_examples_quantization_plot_0_advanced_quantizeml.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Advanced QuantizeML tutorial</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial targets Akida 1.0 users that are looking for advice on how to migrate their Akida 1.0 model towards Akida 2.0. It also lists the major differences in model architecture compatibilities between 1.0 and 2.0.">

.. only:: html

  .. image:: /examples/quantization/images/thumb/sphx_glr_plot_1_upgrading_to_2.0_thumb.png
    :alt:

  :ref:`sphx_glr_examples_quantization_plot_1_upgrading_to_2.0.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Upgrading to Akida 2.0</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="| The Global Akida workflow and the   PyTorch to Akida workflow guides   describe all the steps required to create, train, quantize and convert a model for Akida,   respectively using TF-Keras and PyTorch frameworks. | Here we will illustrate off-the-shelf/pretrained CNN models quantization for Akida using   MobileNet V2 from   the Hugging Face Hub.">

.. only:: html

  .. image:: /examples/quantization/images/thumb/sphx_glr_plot_2_off_the_shelf_quantization_thumb.png
    :alt:

  :ref:`sphx_glr_examples_quantization_plot_2_off_the_shelf_quantization.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Off-the-shelf models quantization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Akida, like any specialized hardware accelerator, sacrifices very generalized computational ability in favor of highly optimized implementations of a subset of key operations. While we strive to make sure that Akida directly supports the most important models, it isn&#x27;t feasible to support all possibilities. You may thus occasionally find yourself with a model which is very nearly compatible with Akida, but which fails to convert due to just a few incompatibilities. In this example, we will look at some simple workarounds and how to implement them. The goal is to successfully convert the model to Akida without having to retrain.">

.. only:: html

  .. image:: /examples/quantization/images/thumb/sphx_glr_plot_3_custom_patterns_thumb.png
    :alt:

  :ref:`sphx_glr_examples_quantization_plot_3_custom_patterns.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Advanced ONNX models quantization</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/quantization/plot_0_advanced_quantizeml
   /examples/quantization/plot_1_upgrading_to_2.0
   /examples/quantization/plot_2_off_the_shelf_quantization
   /examples/quantization/plot_3_custom_patterns

Spatiotemporal examples
-----------------------



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A tutorial on designing efficient models for streaming video tasks.">

.. only:: html

  .. image:: /examples/spatiotemporal/images/thumb/sphx_glr_plot_0_introduction_to_spatiotemporal_models_thumb.png
    :alt:

  :ref:`sphx_glr_examples_spatiotemporal_plot_0_introduction_to_spatiotemporal_models.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gesture recognition with spatiotemporal models</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Efficient online eye tracking with a lightweight spatiotemporal network and event cameras">

.. only:: html

  .. image:: /examples/spatiotemporal/images/thumb/sphx_glr_plot_1_eye_tracking_cvpr_thumb.png
    :alt:

  :ref:`sphx_glr_examples_spatiotemporal_plot_1_eye_tracking_cvpr.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Efficient online eye tracking with a lightweight spatiotemporal network and event cameras</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/spatiotemporal/plot_0_introduction_to_spatiotemporal_models
   /examples/spatiotemporal/plot_1_eye_tracking_cvpr

Edge examples (Akida 1.0 only)
------------------------------



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates the Akida NSoC edge learning capabilities using its built-in learning algorithm. It focuses on an image classification example, where an existing Akida network is re-trained to be able to classify images from 4 new classes.">

.. only:: html

  .. image:: /examples/edge/images/thumb/sphx_glr_plot_0_edge_learning_vision_thumb.png
    :alt:

  :ref:`sphx_glr_examples_edge_plot_0_edge_learning_vision.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Akida vision edge learning</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates the Akida NSoC edge learning capabilities using its built-in learning algorithm.">

.. only:: html

  .. image:: /examples/edge/images/thumb/sphx_glr_plot_1_edge_learning_kws_thumb.png
    :alt:

  :ref:`sphx_glr_examples_edge_plot_1_edge_learning_kws.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Akida edge learning for keyword spotting</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial gives details about the Akida learning parameters and tips to set their values in a first try in an edge learning application. The KWS dataset and the DS-CNN-edge model are used as a classification example to showcase the handy tips.">

.. only:: html

  .. image:: /examples/edge/images/thumb/sphx_glr_plot_2_edge_learning_parameters_thumb.png
    :alt:

  :ref:`sphx_glr_examples_edge_plot_2_edge_learning_parameters.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tips to set Akida edge learning parameters</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/edge/plot_0_edge_learning_vision
   /examples/edge/plot_1_edge_learning_kws
   /examples/edge/plot_2_edge_learning_parameters


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: examples_python.zip </examples/examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: examples_jupyter.zip </examples/examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
