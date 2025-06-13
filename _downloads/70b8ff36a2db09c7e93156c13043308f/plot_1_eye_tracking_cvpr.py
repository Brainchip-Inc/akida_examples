"""
Efficient online eye tracking with a lightweight spatiotemporal network and event cameras
=========================================================================================
"""

######################################################################
# 1. Introduction
# ---------------
#
# Event cameras are biologically inspired sensors that output asynchronous streams of per-pixel
# brightness changes, rather than fixed-rate frames. This modality is especially well suited for
# high-speed, low-power applications like real-time eye tracking on embedded hardware. Traditional
# deep learning models, however, are often ill-suited for exploiting the unique characteristics of
# event data — particularly they lack the tools to leverage their temporal precision and sparsity.
#
# This tutorial presents a lightweight spatiotemporal neural network architecture designed
# specifically for online inference on event camera data. The model is:
#
# * **Causal and streaming-capable**, using FIFO buffering for minimal-latency inference.
# * **Highly efficient**, with a small compute and memory footprint.
# * **Accurate**, achieving state-of-the-art results on a competitive eye tracking benchmark.
# * **Further optimizable** via activation sparsification, maintaining performance while reducing
#   computational load.
#
# The following sections outline the architecture, dataset characteristics, evaluation results,
# buffering mechanism, and advanced optimization strategies.

######################################################################
# 2. Network architecture
# -----------------------
#
# The proposed architecture is a stack of **spatiotemporal convolutional blocks**, each consisting
# of a **temporal convolution followed by a spatial convolution**. These are designed to extract
# both fine-grained temporal features and local spatial structure from event-based input tensors.
# The figure below shows the details of the model architecture.
#
# .. figure:: ../../img/eye_tracking_model_figure.png
#    :target: ../../_images/eye_tracking_model_figure.png
#    :alt: Model architecture overview
#    :scale: 70 %
#    :align: center

######################################################################
# 2.1 Key design features
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# 1. **Causal Temporal Convolutions**
#
#    Temporal convolutions are strictly causal—output at time *t* depends only on input at time ≤
#    *t*. This property is critical for real-time, online inference, allowing inference from the
#    first received frame from the sensor.
#
# 2. **Factorized 3D Convolution Scheme**
#
#    Our spatiotemporal blocks perform temporal convolutions first, followed by spatial
#    convolutions. Decomposing the 3D convolutions into temporal and spatial layers greatly
#    reduces computation (in much the same way that depthwise separable layers do for 2D
#    convolutions).
#
#    .. figure:: ../../img/eye_tracking_block_description.png
#       :target: ../../_images/eye_tracking_block_description.png
#       :alt: Factorization of conv3D into temporal and spatial convolutions
#       :scale: 80 %
#       :align: center
#
# 3. **Depthwise-Separable Convolutions (DWS)**
#
#    Both temporal and spatial layers can optionally be configured as depthwise-separable to
#    further reduce computation with minimal loss in accuracy.
#
# 4. **No Residual Connections**
#
#    To conserve memory and simplify deployment on edge devices, residual connections are omitted.
#    Since the model has a reduced number of layers, they are not critical to achieve SOTA
#    performance.
#
# 5. **Detection Head**
#
#    A lightweight head, inspired by CenterNet `Zhou et al. 2019
#    <https://arxiv.org/abs/1904.07850>`__, predicts a confidence score and local spatial offsets
#    for the pupil position over a coarse spatial grid. The predicted position of the pupil can
#    then be reconstructed.
#
#    .. figure:: ../../img/eye_tracking_post_processing.png
#       :target: ../../_images/eye_tracking_post_processing.png
#       :alt: Centernet head and post processing
#       :scale: 80 %
#       :align: center

######################################################################
# 2.2 Instantiating the spatiotemporal blocks
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# QuantizeML and Akida Models natively work with Tensorflow/Keras layers: akida_models has all the
# necessary functions to instantiate a network based on spatiotemporal layers as well as training
# pipelines available to train models on the jester dataset, the dvs128 dataset or this dataset.
#
# In this tutorial, we'll use PyTorch and introduce the
# `tenns_modules <https://pypi.org/project/tenns-modules/>`__ package which is available to create
# Akida compatible spatiotemporal blocks. The package contains a `spatio-temporal block
# <../../api_reference/tenns_modules_apis.html#tenns_modules.SpatioTemporalBlock>`__
# composed of a `spatial <../../api_reference/tenns_modules_apis.html#tenns_modules.SpatialBlock>`__
# and a `temporal <../../api_reference/tenns_modules_apis.html#tenns_modules.TemporalBlock>`__
# block.
#
# The code below shows how to instantiate the simple 10 layers architecture we used to track the
# pupil coordinates in time using the tenns_modules package.

# Show how to load and create the model
import torch
import torch.nn as nn

from tenns_modules import SpatioTemporalBlock
from torchinfo import summary

n_depthwise_layers = 4
channels = [2, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256]
t_kernel_size = 5  # can vary from 1 to 10
s_kernel_size = 3  # can vary in [1, 3, 5, 7] (1 only when depthwise is False)


class TennSt(nn.Module):
    def __init__(self, channels, t_kernel_size, s_kernel_size, n_depthwise_layers):
        super().__init__()

        depthwises = [False] * (10 - n_depthwise_layers) + [True] * n_depthwise_layers
        self.backbone = nn.Sequential()
        for i in range(0, len(depthwises), 2):
            in_channels, med_channels, out_channels = channels[i], channels[i + 1], channels[i + 2]
            t_depthwise, s_depthwise = depthwises[i], depthwises[i]

            self.backbone.append(
                SpatioTemporalBlock(in_channels=in_channels, med_channels=med_channels,
                                    out_channels=out_channels, t_kernel_size=t_kernel_size,
                                    s_kernel_size=s_kernel_size, s_stride=2, bias=False,
                                    t_depthwise=t_depthwise, s_depthwise=s_depthwise))

        self.head = nn.Sequential(
            SpatioTemporalBlock(channels[-1], channels[-1], channels[-1],
                                t_kernel_size=t_kernel_size, s_kernel_size=s_kernel_size,
                                t_depthwise=False, s_depthwise=False),
            nn.Conv3d(channels[-1], 3, 1)
        )

    def forward(self, input):
        return self.head((self.backbone(input)))


model = TennSt(channels, t_kernel_size, s_kernel_size, n_depthwise_layers)
summary(model, input_size=(1, 2, 50, 96, 128), depth=4, verbose=0)

######################################################################
# 3. Dataset and preprocessing
# ----------------------------
#
# The model is trained and evaluated on the
# `AIS 2024 Event-Based Eye Tracking Challenge Dataset
# <https://www.kaggle.com/competitions/event-based-eye-tracking-ais2024>`__, which contains
# recordings from 13 participants, captured using 480×640-resolution event camera. Each participant
# has between 2 and 6 recording sessions. The ground truth pupil (x- and y-) coordinates are
# provided at a resolution of 100Hz. The evaluation of the predictions is done at 20Hz at a
# resolution of 60x80 when the eyes are opened.
#
# The video below shows you an example of the reconstructed frames (note that the video has been
# sped up). The ground truth pupil location is represented by a cross: the cross is green when the
# eye is opened and it turns red when the eye closes.
#
# .. video:: ../../img/eye_tracking_valdata_gt_only_fast.mp4
#    :nocontrols:
#    :autoplay:
#    :playsinline:
#    :muted:
#    :loop:
#    :width: 50%
#    :align: center

######################################################################
# 3.1 Preprocessing
# ^^^^^^^^^^^^^^^^^
#
# The following preprocessing is applied to the event data:
#
# - temporal augmentations (for training only)
# - spatial downsampling (by 5) and event binning to create segments with fixed temporal length
# - spatial affine transforms
# - frames where the eye is labeled as closed are ignored during training

######################################################################
# 3.1.1 Event binning
# """""""""""""""""""
#
# Events are represented as 4-tuples: *(polarity, x, y, timestamp)*. These are converted into
# tensors of shape **(P=2, T, H, W)** using **causal event volume binning**, a method that preserves
# temporal fidelity while avoiding future context. Binning uses a causal triangle kernel to
# approximate each event’s influence over space and time, as you can see from the graph below.
#
# .. figure:: ../../img/eye_tracking_causal_event_binning.png
#    :target: ../../_images/eye_tracking_causal_event_binning.png
#    :alt: Example of causal event binning
#    :scale: 80 %
#    :align: center

######################################################################
# 3.1.2 Augmentation
# """"""""""""""""""
#
# To improve generalization in a data-limited regime, the following transforms are applied to the
# events (and the corresponding pupil coordinates) during training only:
#
# * **Spatial affine transforms** are applied such as scaling, rotation, translation.
# * **Temporal augmentations** including random time scaling and flipping (with polarity inversion).
# * **Random temporal flip** with probability 0.5 is applied to the time and polarity dimension.
#
# These transforms are applied to each segment independently (but not varied within a segment).
# For better legibility, the dataset was preprocessed offline and made available for evaluation
# purposes only.

######################################################################
# 3.2 Evaluation metric
# ^^^^^^^^^^^^^^^^^^^^^
#
# For the competition, the primary metric for model evaluation was the “p10” accuracy: the
# percentage of predictions falling within 10 pixels of the ground truth (i.e. if the predicted
# pupil center falls within the blue dashed circle in the figure below). We can also consider more
# stringent measures, such as a p3 accuracy (3 pixels); or simpler linear measures, such as the
# Euclidean distance (L2).
#
# .. figure:: ../../img/eye_tracking_pixel_accuracy_euclidean.png
#    :target: ../../_images/eye_tracking_pixel_accuracy_euclidean.png
#    :alt: Metrics used in the competition
#    :scale: 80 %
#    :align: center

######################################################################
# 4. Model training & evaluation
# ------------------------------

######################################################################
# 4.1 Training details
# ^^^^^^^^^^^^^^^^^^^^
#
# The following hyperparameters were used for training:
#
# - batch size of 32
# - 50 event frames per segment
# - 200 epochs
# - AdamW optimizer with base LR of 0.002 and weight decay of 0.005
# - learning rate scheduler with linear warm up (for 2.5% of total epochs) and a cosine decay
#
# .. Note::
#   We don't train the model here as it requires access to a GPU but rather load a pre-trained model
#   for convenience.

# Load the pretrained weights in our model
from akida_models import fetch_file

ckpt_file = fetch_file(
    fname="tenn_spatiotemporal_eye.ckpt",
    origin="https://data.brainchip.com/models/AkidaV2/tenn_spatiotemporal/tenn_spatiotemporal_eye.ckpt",
    cache_subdir='models')

checkpoint = torch.load(ckpt_file, map_location="cpu")
new_state_dict = {k.replace('model._orig_mod.', ''): v for k, v in checkpoint["state_dict"].items()}
model.load_state_dict(new_state_dict)
_ = model.eval().cpu()


######################################################################
# 4.2 Evaluation
# ^^^^^^^^^^^^^^
#
# The preprocessed validation data have been set aside and can be loaded from the archive available
# online.
#
# .. Note::
#   To optimize storage and reduce processing time, only the first 500 frames from each validation
#   file have been mirrored on the dataset server. This subset is representative and sufficient for
#   validation purposes in this tutorial.

import numpy as np

samples = fetch_file("https://data.brainchip.com/dataset-mirror/eye_tracking_ais2024_cvpr/eye_tracking_preprocessed_500frames_val.npz",
                     fname="eye_tracking_preprocessed_500frames_val.npz")
data = np.load(samples, allow_pickle=True)
events, centers = data["events"], data["centers"]


######################################################################
# To evaluate the model, we pass the data through our spatiotemporal model. Once we have the output,
# we need to post process the model's output to reconstruct the predicted pupil coordinates in the
# prediction space (60, 80).

def process_detector_prediction(pred):
    """Post-processing of model predictions to extract the predicted pupil coordinates for a model that has a
    centernet like head.

    Args:
        preds (torch.Tensor): shape (B, 3, H, W)

    Returns:
        torch matrice of (B, 2) containing the x and y predicted coordinates
    """
    torch_device = pred.device
    batch_size, _, frames, height, width = pred.shape
    # Extract the center heatmap, and the x and y offset maps
    pred_pupil, pred_x_mod, pred_y_mod = pred.moveaxis(1, 0)
    pred_x_mod = torch.sigmoid(pred_x_mod)
    pred_y_mod = torch.sigmoid(pred_y_mod)

    # Find the stronger peak in the center heatmap and it's coordinates
    pupil_ind = pred_pupil.flatten(-2, -1).argmax(-1)  # (batch, frames)
    pupil_ind_x = pupil_ind % width
    pupil_ind_y = pupil_ind // width

    # Reconstruct the predicted offset
    batch_range = torch.arange(batch_size, device=torch_device).repeat_interleave(frames)
    frames_range = torch.arange(frames, device=torch_device).repeat(batch_size)
    pred_x_mod = pred_x_mod[batch_range, frames_range, pupil_ind_y.flatten(), pupil_ind_x.flatten()]
    pred_y_mod = pred_y_mod[batch_range, frames_range, pupil_ind_y.flatten(), pupil_ind_x.flatten()]

    # Express the coordinates in size agnostic terms (between 0 and 1)
    x = (pupil_ind_x + pred_x_mod.view(batch_size, frames)) / width
    y = (pupil_ind_y + pred_y_mod.view(batch_size, frames)) / height
    return torch.stack([x, y], dim=1)


def compute_distance(pred, center):
    """Computes the L2 distance for a prediction and center matrice

    Args:
        pred: torch tensor of shape (2, T)
        center: torch tensor of shape (2, T)
    """
    height, width = 60, 80
    pred = pred.detach().clone()
    center = center.detach().clone()
    pred[0, :] *= width
    pred[1, :] *= height
    center[0, :] *= width
    center[1, :] *= height
    l2_distances = torch.norm(center - pred, dim=0)
    return l2_distances


def pretty_print_results(collected_distances):
    """Prints the distance and accuracy within different pixel tolerance.

    By default, only the results at 20Hz will be printed (to be compatible with the
    metrics of the challenge). To print the results computed on the whole trial,
    use downsample=False. In practice, this changes very little to the final performance
    of the model.
    """
    for t in [10, 5, 3, 1]:
        p_acc = (collected_distances < t).sum() / collected_distances.size
        print(f'- p{t}: {p_acc:.3f}')
    print(f'- Euc. Dist: {collected_distances.mean():.3f} ')


######################################################################

# Get the model device to propagate the events properly
torch_device = next(model.parameters()).device

# Compute the distances across all 9 trials
collected_l2_distances = np.zeros((0,))
for trial_idx, event in enumerate(events):
    center = torch.from_numpy(centers[trial_idx]).float().to(torch_device)
    event = torch.from_numpy(event).unsqueeze(0).float().to(torch_device)
    pred = model(event)
    pred = process_detector_prediction(pred).squeeze(0)
    l2_distances = compute_distance(pred, center)
    collected_l2_distances = np.concatenate((collected_l2_distances, l2_distances), axis=0)

pretty_print_results(collected_l2_distances)

######################################################################
# 5. Official competition results
# -------------------------------
#
# The results for the competition are on the test set (labels are not available). The main metric in
# the challenge was the p10. Using this metric, our model ranked 3rd (see table below copied from
# the `original challenge survey paper
# <https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/Wang_Event-Based_Eye_Tracking._AIS_2024_Challenge_Survey_CVPRW_2024_paper.pdf>`__).
#
# However, other metrics were reported in the original challenge survey: the accuracy within 5 (p5),
# 3 (p3) or 1 pixel (p1), as well as metrics directly measuring the distance between ground truth
# and predicted pupil location (L2 and L1, i.e. smaller values are better). On these more stringent
# metrics, our model outperforms the other models on all the other metrics.
#
# .. list-table::
#    :header-rows: 1
#
#    * - **Team**
#      - **Rank**
#      - p10 private (primary)
#      - p10 🡑
#      - p5 🡑
#      - p3 🡑
#      - p1 🡑
#      - *L2* 🡓
#      - *L1* 🡓
#    * - USTCEventGroup
#      - 1
#      - **99.58**
#      - **99.42**
#      - 97.05
#      - 90.73
#      - 33.75
#      - 1.67
#      - 2.11
#    * - FreeEvs
#      - 2
#      - 99.27
#      - 99.26
#      - 94.31
#      - 83.83
#      - 23.91
#      - 2.03
#      - 2.56
#    * - **Brainchip**
#      - 3
#      - 99.16
#      - 99.00
#      - **97.79**
#      - **94.58**
#      - **45.50**
#      - **1.44**
#      - **1.82**
#    * - Go Sparse
#      - 4
#      - 98.74
#      - 99.00
#      - 77.20
#      - 47.97
#      - 7.32
#      - 3.51
#      - 4.63
#    * - MeMo
#      - 4
#      - 98.74
#      - 99.05
#      - 89.36
#      - 50.87
#      - 6.53
#      - 3.2
#      - 4.04
#
# The best metric in class is highlighted in bold, 🡑 means higher values are best, 🡓 means lower
# values are best.
#
# The code below shows an inference on the model using the *validation* dataset. Note that the
# results below differ from the challenge metrics reported above because:
#
# 1. As mentionned, the metrics reported above are on the *test* set (no label available).
# 2. Our submission model was trained on both the train and validation data to achieve the best
#    possible performance (as allowed by the rules), but the model below that was used for the
#    ablation studies was trained on the train set only.
# 3. the validation dataset, it turns out, is much harder than the test dataset

######################################################################
# 6. Ablation studies and efficiency optimization
# -----------------------------------------------
#
# Figure reproduced from the original paper.
#
# .. figure:: ../../img/paper_figure3.png
#    :target: ../../_images/paper_figure3.png
#    :alt: Figure 3 from the original paper
#    :scale: 80 %
#    :align: center

######################################################################
# 6.1 Ablation studies
# ^^^^^^^^^^^^^^^^^^^^
#
# To test the robustness of our design choices, we performed a series of ablation studies. To
# provide a baseline model for the ablation study, we trained a model on the 'train' split only and
# tested it on the validation dataset. This model gets a p10 of 0.963 and an l2 distance of 2.79.
#
# This showed that:
#
# 1. Removing spatial affine augmentation reduces performance dramatically (from 0.963 → 0.588).
# 2. Causal event binning performs equivalently to other methods while enabling streaming inference.
# 3. Larger temporal kernels (e.g., size 5 vs. 3) offer small but consistent improvements in
#    accuracy.
# 4. Using only batch normalization (BN) layers gave a small improvement over group norm (GN) only
#    or a mix of BN/GN(96.9 vs 96.0 or 96.3).
#
# For more details you can refer to the paper (links below).

######################################################################
# 6.2 Efficiency-accuracy trade-offs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In certain environments, such as edge or low-power devices, the balance between model size and
# computational demand often matters more than achieving state-of-the-art accuracy. This section
# explores the trade-off between maximizing accuracy and maintaining model efficiency along 3 axis.

######################################################################
# 6.2.1 Spatial resolution
# """"""""""""""""""""""""
#
# We looked at how reducing input image size affects model performance (see figure 3.A). Even with
# an input size of 60 x 80 (downsampling by a factor of 8), the model still performs almost as well
# as with our default setting (downsampling by a factor of 5), while requiring only a third of the
# computation.

######################################################################
# 6.2.2 Depthwise separable convolutions
# """"""""""""""""""""""""""""""""""""""
#
# From the outset, we decided to further decompose our factorized convolutions into depthwise and
# pointwise convolutions (similar to depthwise separable convolutions introduced in MobileNet V1).
# We explored howthese impacted model performance (see figure 3.B): as the number of separable
# convolutions used increases,the MACs of the model decrease, with a relatively small impact on the
# validation distance. When no separablelayers are used, the final validation distance is 2.6 vs.
# 3.1 when all layers are separable. Our baselinemodel had the last 4 layers configured as
# separable. Changing just 2 more to separable could lead to areduction of almost 30% in compute,
# with almost no impact on performance (compare the turquoise vers greenlines on the figure 3.B).
#
# The combination of these techniques results in a highly efficient model with a computational cost
# of just 55M MACs/frame, and even less when sparsity is exploited.

######################################################################
# 6.2.3 Activity regularization
# """""""""""""""""""""""""""""
#
# Event camera data is inherently sparse. However, intermediate layers in a neural network may still
# produce dense activations unless explicitly regularized. When measuring the baseline sparsity in
# the network, we found it to be on average 50% (about what one would expect given ReLU activation
# functions), much of which may not be informative given the very high spatial sparsity of the input
# to the network. By applying L1 regularization to ReLU activations during training, the model is
# encouraged to silence unnecessary activations. We applied 5 different levels of regularization to
# our model: figure 3.C shows how the average distance varies depending on the regularization
# strength while figure 3.D shows how the sparse aware MACs (i.e. MACs multiplied by the model's
# mean sparsity per layer) is affected by regularization. We can see that over 90% activation
# sparsity is achievable with a negligible performance degradation (p10 remains >0.96).
#
# This is especially interesting because Akida is an event based hardware: it is capable of skipping
# zero operations. In such hardware, high level of activation sparsity can translate into ~5× speedups.
#
# .. warning::
#   Based on these ablation studies, the model made available through the
#   `model zoo <../../model_zoo_performance.html#eye-tracking>`__ has been optimized for the inference
#   on Akida Hardware (downsampling by a factor of 6, use of depthwise separable convolutions), so the
#   number of parameters and accuracy reported differ.

######################################################################
# 7. FIFO buffering for streaming inference
# -----------------------------------------

######################################################################
# 7.1 Key mechanism
# ^^^^^^^^^^^^^^^^^
#
# Each temporal convolutional layer maintains a fixed-length FIFO buffer of its input history (equal
# to the kernel size). At each time step:
#
# - The buffer is updated with the newest frame.
# - A dot product is computed between the buffer contents and the kernel weights.
# - The result is passed through normalization and spatial convolution.
#
# This approach mimics the operation of a sliding temporal convolution but avoids recomputation and memory
# redundancy, ensuring minimal latency and efficient real-time processing.
# For more details of this approach, see the tutorial that `introduced spatiotemporal models
# <./plot_0_introduction_to_spatiotemporal_models.html#streaming-inference-making-real-time-predictions>`__.
#
# .. figure:: ../../img/fifo_buffer.png
#    :target: ../../_images/fifo_buffer.png
#    :alt: Fifo buffer
#    :scale: 80 %
#    :align: center

######################################################################
# 7.2 Exporting to ONNX
# ^^^^^^^^^^^^^^^^^^^^^
#
# The transformation to buffer mode is done during quantization step (see dedicated section below).
# The first step is to export the model to ONNX format. This is made very easy using the
# `tenns_modules <https://pypi.org/project/tenns-modules/>`__ package and the `export_to_onnx
# <../../api_reference/tenns_modules_apis.html#tenns_modules.export_to_onnx>`__ function.

from tenns_modules import export_to_onnx

# Using a batch size of 10 to export with a dynamic batch size
onnx_checkpoint_path = "tenns_modules_onnx.onnx"
export_to_onnx(model, (10, 2, 50, 96, 128), out_path=onnx_checkpoint_path)

######################################################################
# Load the ONNX model that was automatically saved
import onnx

model = onnx.load(onnx_checkpoint_path)

######################################################################
# 8. Quantization and conversion to Akida
# ---------------------------------------

######################################################################
# 8.1 Quantization
# ^^^^^^^^^^^^^^^^
#
# To be deployable on Akida, the model needs to be quantized. This can easily be done using the
# QuantizeML package. For more details on the quantization scheme with the ONNX package see this
# example on `off-the-shelf model quantization
# <../quantization/plot_2_off_the_shelf_quantization.html>`__.

from quantizeml.models import quantize
from quantizeml.layers import QuantizationParams

# Retrieve calibration samples:
samples = fetch_file("https://data.brainchip.com/dataset-mirror/samples/eye_tracking/eye_tracking_onnx_samples_bs100.npz",
                     fname="eye_tracking_onnx_samples_bs100.npz")

# Define quantization parameters and load quantization samples
qparams = QuantizationParams(per_tensor_activations=True, input_dtype='int8')
data = np.load(samples)
samples = np.concatenate([data[item] for item in data.files])

# Quantize the model
model_quant = quantize(model, qparams=qparams, epochs=1, batch_size=100, samples=samples)

######################################################################
# .. Note::
#   During this step, the model is also bufferized, meaning that the FIFOs of the temporal
#   convolutions are automatically created and initialized from the 3D convolutions.

######################################################################
# 8.2 ONNX model evaluation
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This model can be evaluated using the same process as before with a few differences:
#
# - We need to pass each frame to the model independently (i.e. the model now has a 4-D input shape
#   (B, C, H,  W) - batch, channels, height, width).
# - The post processing function needs to be modified to use numpy functions (instead of torch)
# - Once all frames from a given trial have been passed through, the FIFO buffers of the temporal
#   convolutions need to be reset using the `reset_buffers
#   <../../api_reference/quantizeml_apis.html#quantizeml.onnx_support.layers.buffertempconv.reset_buffers>`__
#   available from QuantizeML.


def custom_process_detector_prediction(pred):
    """ Post-processing of the model's output heatmap.

    Reconstructs the predicted x- and y- center location using numpy functions to post-process
    the output of a ONNX model.
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # Pred shape is (batch, channels, height, width)
    batch_size, _, height, width = pred.shape

    # Split channels - reshape to move frames dimension after batch
    # Now (batch, height, width, channels)
    pred = np.moveaxis(pred, 1, -1)
    pred_pupil = pred[..., 0]
    pred_x_mod = sigmoid(pred[..., 1])
    pred_y_mod = sigmoid(pred[..., 2])

    # Find pupil location
    pred_pupil_flat = pred_pupil.reshape(batch_size, -1)
    pupil_ind = np.argmax(pred_pupil_flat, axis=-1)
    pupil_ind_x = pupil_ind % width
    pupil_ind_y = pupil_ind // width

    # Get the learned x- y- offset
    batch_idx = np.repeat(np.arange(batch_size)[:, None], 1, axis=1)
    x_mods = pred_x_mod[batch_idx, pupil_ind_y, pupil_ind_x]
    y_mods = pred_y_mod[batch_idx, pupil_ind_y, pupil_ind_x]

    # Calculate final coordinates
    x = (pupil_ind_x + x_mods) / width
    y = (pupil_ind_y + y_mods) / height

    return np.stack([x, y], axis=1)


######################################################################
# Create the inference session for the ONNX model and evaluate
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path
from quantizeml.onnx_support.quantization import ONNXModel

sess_options = SessionOptions()
sess_options.register_custom_ops_library(get_library_path())
model_quant = ONNXModel(model_quant)
session = InferenceSession(model_quant.serialized, sess_options=sess_options,
                           providers=['CPUExecutionProvider'])

######################################################################
from quantizeml.onnx_support.layers.buffertempconv import reset_buffers
from tqdm import tqdm

# And then evaluate the model
collected_l2_distances = []
for trial_idx, event in enumerate(events):
    center = centers[trial_idx]
    for frame_idx in tqdm(range(event.shape[1])):
        frame = event[:, frame_idx, ...][None, ...].astype(np.float32)
        pred = session.run(None, {model_quant.input[0].name: frame})[0]
        pred = custom_process_detector_prediction(pred).squeeze()
        y_pred_x = pred[0] * 80
        y_pred_y = pred[1] * 60
        center_x = center[0, frame_idx] * 80
        center_y = center[1, frame_idx] * 60
        collected_l2_distances.append(np.sqrt(np.square(
            center_x - y_pred_x) + np.square(center_y - y_pred_y)))
    # Reset FIFOs between each file
    reset_buffers(model_quant)

######################################################################
pretty_print_results(np.array(collected_l2_distances))

######################################################################
# 8.3 Conversion to Akida
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The quantized model can be easily converted to Akida using the cnn2snn package.
# The `convert <../../api_reference/cnn2snn_apis.html#cnn2snn.convert>`__ function
# returns a model in Akida format ready for inference.

from cnn2snn import convert

akida_model = convert(model_quant.model)
akida_model.summary()


######################################################################
# .. Note::
#   - For more information you can refer to the paper available `here
#     <https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/Pei_A_Lightweight_Spatiotemporal_Network_for_Online_Eye_Tracking_with_Event_CVPRW_2024_paper.pdf>`__.
#   - There is also a full training pipeline available in tensorflow/Keras from the akida_models
#     package that reproduces the performance presented in the paper available with the
#     `akida_models.tenn_spatiotemporal
#     <../../api_reference/akida_models_apis.html#akida_models.tenn_spatiotemporal_eye>`__ function.
