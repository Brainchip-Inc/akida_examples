Installation
============

.. important::
    MetaTF 2.16 was the last release supporting Tensorflow 2.15.0 (Keras 2) and python 3.9.

    Starting with MetaTF 2.17, releases will support
    `TF-Keras <https://github.com/keras-team/tf-keras>`__ 2.19 (Tensorflow 2.19) and python 3.10 to
    3.12.


Supported configurations
------------------------

* **Operating systems:**
    * Windows 10, Windows 11
    * Any Linux variant compatible with `manylinux 2.28 <https://github.com/pypa/manylinux>`_ (Ubuntu 22.04, Ubuntu 24.04, ...)
* **Python versions:** 3.10 to 3.12
* **TF-Keras versions:** 2.19
* **pytorch version:** While not an explicit MetaTF/ONNX requirement, Pytorch versions >= 2.6 are supported. GPU support and Tensorflow dependency management are left to the user discretion.

.. warning::
    Using Windows, the latest Visual C++ redistributable package is required.
    Please refer to `this link
    <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist>`__
    for installation.

Quick installation
------------------

The akida, cnn2snn, quantizeml and akida_models python packages can
be setup with Python's pip package manager by installing the akida-models
package only:

.. code-block:: bash

    pip install akida-models=={MODELS_VERSION}

.. note::
    Installing akida-models automatically pulls the other MetaTF packages as
    dependencies: `cnn2snn <https://pypi.org/project/cnn2snn>`_, `akida
    <https://pypi.org/project/akida>`_ and `quantizeml
    <https://pypi.org/project/quantizeml>`_, along with `TensorFlow
    <https://www.tensorflow.org/>`_ and `TF-Keras
    <https://github.com/keras-team/tf-keras>`__ — there is no need to install
    TensorFlow separately. For a GPU-enabled TensorFlow setup, please refer to
    `Install TensorFlow with pip <https://www.tensorflow.org/install/pip>`_.

.. note::
    We recommend using virtual environment such as `Conda <https://conda.io/docs/>`_.
    Please note that the python version must be explicitly specified when creating a
    conda environment. The specification must be for one of the supported python
    versions listed above.

    .. code-block:: bash

      conda create --name akida_env python=3.11
      conda activate akida_env

Verify the installation
------------------------

Once the packages are installed, check that everything is in place:

.. code-block:: bash

    python -c "import akida; print(akida.__version__); print(akida.devices())"

The expected output is:

.. code-block:: bash

    {AKIDA_VERSION}
    []

An empty device list ``[]`` is the correct result on a machine without Akida
hardware: no hardware is required, and everything on this site (user guide and
examples) runs on the software simulator that comes with the akida package.

.. note::
    If the import fails with ``ModuleNotFoundError``, the virtual environment
    where the packages were installed is probably not active. If the printed
    version is not ``{AKIDA_VERSION}``, update the package with
    ``pip install --upgrade akida=={AKIDA_VERSION}``.

Running examples
----------------

The Akida tutorials can be downloaded from the `examples <./examples/index.html>`_
section as python scripts or Jupyter Notebooks. Dependencies needed to replay
the examples can be installed using the :download:`requirements.txt <../requirements.txt>`
file:

.. code-block:: bash

    pip install -r requirements.txt

.. note::
    Please refer to `this link <https://jupyter.org/>`__ for Jupyter Notebook installation
    and configuration.
