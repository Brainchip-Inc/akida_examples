"""
PyTorch to Akida workflow
=========================

The `Global Akida workflow <../general/plot_0_global_workflow.html>`__ guide
describes the steps to prepare a model for Akida starting from a TensorFlow/Keras model.
Here we will instead describe a workflow to go from a model trained in PyTorch.

.. Note::
   | This example targets those users who already have a PyTorch training pipeline
     in place, and a trained model: this workflow will allow you to rapidly convert
     your model to Akida 2.0.
   | Note however that this pathway offers slightly less flexibility than our default,
     TensorFlow-based pathway - specifically, fine tuning of the quantized model is
     not possible when starting from PyTorch.
   | In most cases, that won't matter, there should be almost no performance drop when
     quantizing to 8-bit anyway.
   | However, advanced users interested in further optimization of the original model
     (going to 4-bit quantization for example) or those users who don't yet have a
     training pipeline in place may prefer the extra options afforded by our default,
     TensorFlow-based `Global Akida workflow <../general/plot_0_global_workflow.html>`__.


QuantizeML allows to quantize and fine-tune TensorFlow models natively. While it does
not support PyTorch quantization natively, it allows to quantize float models stored in
the `Open Neural Network eXchange (ONNX) <https://onnx.ai>`__ format. Export from
PyTorch to ONNX is well supported, and so this provides a straightforward pathway to
prepare your PyTorch model for Akida.

As a concrete example, we will prepare a PyTorch model on a simple classification task
(MNIST). This model will then be exported to ONNX, from where it will be quantized to
8-bit using QuantizeML. The quantized model is then converted to Akida, and performance
evaluated to show that there has been no loss in accuracy.

Please refer to the `Akida user guide <../../user_guide/akida.html>`__ for further information.

.. Note::
   | This example is loosely based on the PyTorch `Training a Classifier
     <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`__ tutorial and
     does not aim to describe PyTorch training in detail. We assume that if you are following
     this example, it's because you already have a trained PyTorch model.
   | `PyTorch 2.0.1 <https://github.com/pytorch/pytorch/releases/tag/v2.0.1>`__ is used
     for this example.

     .. code-block::

        pip install torch==2.0.1 torchvision

.. Warning::
   | The MNIST example below is light enough to train on the CPU only.
   | However, where GPU acceleration is desirable for the PyTorch training step, you may find
     it simpler to use separate virtual environments for the PyTorch-dependent sections
     (`1. Create and train`_ and `2. Export`_ ) vs the TensorFlow-dependent sections
     (`3. Quantize`_ and `4. Convert`_ ).


.. figure:: ../../img/overall_onnx_flow.png
   :target: ../../_images/overall_onnx_flow.png
   :alt: Overall pytorch flow
   :scale: 60 %
   :align: center

   PyTorch Akida workflow
"""

######################################################################
# 1. Create and train
# ~~~~~~~~~~~~~~~~~~~
#

######################################################################
# 1.1. Load and normalize MNIST dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

batch_size = 128


def get_dataloader(train, batch_size, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])
    dataset = torchvision.datasets.MNIST(root='datasets/mnist',
                                         train=train,
                                         download=True,
                                         transform=transform)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=train,
                                       num_workers=num_workers)


# Load MNIST dataset and normalize between [-1, 1]
trainloader = get_dataloader(train=True, batch_size=batch_size)
testloader = get_dataloader(train=False, batch_size=batch_size)


def imshow(img):
    # Unnormalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(npimg.transpose((1, 2, 0)))
    plt.show()


# Get some random training images
images, labels = next(iter(trainloader))

# Show images and labels
imshow(torchvision.utils.make_grid(images, nrow=8))
print("[INFO] labels:\n", labels.reshape((-1, 8)))

######################################################################
# 1.2. Model definition
# ^^^^^^^^^^^^^^^^^^^^^
#
# Note that at this stage, there is nothing specific to the Akida IP.
# The model constructed below uses the `torch.nn.Sequential
# <https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html#nn-sequential>`__
# module to define a standard CNN.
#

model_torch = torch.nn.Sequential(torch.nn.Conv2d(1, 32, 5, padding=(2, 2)),
                                  torch.nn.ReLU6(),
                                  torch.nn.MaxPool2d(kernel_size=2),
                                  torch.nn.Conv2d(32, 64, 3, stride=2),
                                  torch.nn.ReLU(),
                                  torch.nn.Dropout(0.25),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(2304, 512),
                                  torch.nn.ReLU(),
                                  torch.nn.Dropout(0.5),
                                  torch.nn.Linear(512, 10))
print(model_torch)

######################################################################
# 1.3. Model training
# ^^^^^^^^^^^^^^^^^^^
#

# Define training rules
device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = torch.optim.Adam(model_torch.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
epochs = 10

# Loop over the dataset multiple times
model_torch = model_torch.to(device)
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs and labels
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = model_torch(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.detach().item()
        if (i + 1) % 100 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


######################################################################
# 1.4. Model testing
# ^^^^^^^^^^^^^^^^^^
#
# Evaluate the model performance on the test set. It should achieve an accuracy over 98%.
#

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Calculate outputs by running images through the network
        outputs = model_torch(inputs)
        # The class with the highest score is the prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

assert correct / total >= 0.98
print(f'[INFO] Test accuracy: {100 * correct // total} %')

######################################################################
# 2. Export
# ~~~~~~~~~
#
# PyTorch models are not directly compatible with the `QuantizeML quantization
# tool <../../api_reference/quantizeml_apis.html>`__, it is therefore necessary
# to use an intermediate format. Like many other machine learning frameworks,
# PyTorch has tools to export modules in the `ONNX <https://onnx.ai>`__ format.
#
# Note that several versions are available to `map the operation set in ONNX
# <https://onnx.ai/onnx/intro/concepts.html#what-is-an-opset-version>`__.
# QuantizeML uses the most recent opset version, provided by the installed onnx package.
# This can be verified as follows:
#

import onnx

opset_version = onnx.defs.onnx_opset_version()
print("[INFO] Current opset version:", onnx.defs.onnx_opset_version())

######################################################################
# Then, the model is exported by the following code:
#

model_torch = model_torch.cpu()
sample, _ = next(iter(trainloader))
torch.onnx.export(model_torch,
                  sample.cpu(),
                  f="mnist_cnn.onnx",
                  opset_version=opset_version,
                  input_names=["inputs"],
                  output_names=["outputs"],
                  dynamic_axes={'inputs': {0: 'batch_size'}, 'outputs': {0: 'batch_size'}})

######################################################################
# .. Note::
#  Find more information about how to export PyTorch models in ONNX at
#  `<https://pytorch.org/docs/stable/onnx.html>`_.
#

######################################################################
# 3. Quantize
# ~~~~~~~~~~~
#
# An Akida accelerator processes integer activations and weights. Therefore, the floating
# point model must be quantized in preparation to run on an Akida accelerator.
#
# The `QuantizeML quantize() <../../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__
# function recognizes `ModelProto <https://onnx.ai/onnx/api/classes.html#modelproto>`__ objects
# and can quantize them for Akida. The result is another ``ModelProto``, compatible with the
# `CNN2SNN Toolkit <../../user_guide/cnn2snn.html>`__.
#
# .. Warning::
#  ONNX and PyTorch offer their own quantization methods. You should not use those when preparing
#  your model for Akida. Only the `QuantizeML quantize()
#  <../../api_reference/quantizeml_apis.html#quantizeml.models.quantize>`__ function
#  can be used to generate a quantized model ready for conversion to Akida.
#
# .. Note::
#  For this simple model, using random samples for calibration is sufficient, as
#  shown in the following steps.
#

from quantizeml.models import quantize

# Read the exported ONNX model
model_onnx = onnx.load_model("mnist_cnn.onnx")

# Quantize
model_quantized = quantize(model_onnx, num_samples=128)
print(onnx.helper.printable_graph(model_quantized.graph))

######################################################################
# 4. Convert
# ~~~~~~~~~~
#

######################################################################
# 4.1 Convert to Akida model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The quantized model can now be converted to the native Akida format.
# The `convert() <../../api_reference/cnn2snn_apis.html#cnn2snn.convert>`__
# function returns a model in Akida format ready for inference.
#

from cnn2snn import convert

model_akida = convert(model_quantized)
model_akida.summary()

######################################################################
# 4.2. Check performance
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Native PyTorch data must be presented in a different format to perform
# the evaluation in Akida models. Specifically:
#
# 1. images must be numpy-raw, with an 8-bit unsigned integer data type and
# 2. the channel dimension must be in the last dimension.
#

# Read raw data and convert it into numpy
x_test = testloader.dataset.data.numpy()
y_test = testloader.dataset.targets.numpy()

# Add a channels dimension to the image sets as Akida expects 4-D inputs corresponding to
# (num_samples, width, height, channels). Note: MNIST is a grayscale dataset and is unusual
# in this respect - most image data already includes a channel dimension, and this step will
# not be necessary.
x_test = x_test[..., None]
y_test = y_test[..., None]

accuracy = model_akida.evaluate(x_test, y_test)
print('Test accuracy after conversion:', accuracy)

# For non-regression purposes
assert accuracy > 0.96

######################################################################
# 4.3 Show predictions for a single image
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Display one of the test images, such as the first image in the dataset from above, to visualize
# the output of the model.
#

# Test a single example
sample_image = 0
image = x_test[sample_image]
outputs = model_akida.predict(image.reshape(1, 28, 28, 1))

plt.imshow(x_test[sample_image].reshape((28, 28)), cmap="Greys")
print('[INFO] Input Label:', y_test[sample_image].item())
print('[INFO] Prediction Label:', outputs.squeeze().argmax())
