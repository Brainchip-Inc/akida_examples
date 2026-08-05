Installation
============

.. important::
    MetaTF 2.16 was the last release supporting TensorFlow 2.15.0 (Keras 2) and Python 3.9.

    Starting with MetaTF 2.17, releases will support
    `TF-Keras <https://github.com/keras-team/tf-keras>`__ 2.19 (TensorFlow 2.19) and Python 3.10 to
    3.12.


Supported configurations
------------------------

* **Operating systems:**
    * Windows 10, Windows 11
    * Any Linux variant compatible with `manylinux 2.28 <https://github.com/pypa/manylinux>`_ (Ubuntu 22.04, Ubuntu 24.04, ...)
* **Python versions:** 3.10 to 3.12
* **TF-Keras versions:** 2.19
* **PyTorch versions:** While not an explicit MetaTF/ONNX requirement, PyTorch versions >= 2.6 are supported. GPU support and TensorFlow dependency management are left to the user's discretion.

.. warning::
    On Windows, the latest Visual C++ redistributable package is required.
    Please refer to `this link
    <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist>`__
    for installation.

Quick installation
------------------

The complete MetaTF framework can be set up with Python's pip package manager
by installing the `metatf <https://pypi.org/project/metatf>`_ package only:

.. code-block:: bash

    pip install metatf=={METATF_VERSION}

.. note::
    metatf is a meta-package that pulls the complete, validated set of MetaTF
    packages as dependencies: `akida-models
    <https://pypi.org/project/akida-models>`_, `cnn2snn
    <https://pypi.org/project/cnn2snn>`_, `akida
    <https://pypi.org/project/akida>`_ and `quantizeml
    <https://pypi.org/project/quantizeml>`_, along with `TensorFlow
    <https://www.tensorflow.org/>`_ and `TF-Keras
    <https://github.com/keras-team/tf-keras>`__ — there is no need to install
    TensorFlow separately. metatf itself contains no code: it provides no
    ``metatf`` module or command line, the four packages above are imported
    and used directly. For a GPU-enabled TensorFlow setup, please refer to
    `Install TensorFlow with pip <https://www.tensorflow.org/install/pip>`_.

.. note::
    We recommend using a virtual environment such as `Conda <https://conda.io/docs/>`_.
    Please note that the Python version must be explicitly specified when creating a
    conda environment. The specification must be for one of the supported Python
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
    version is not ``{AKIDA_VERSION}``, update the framework with
    ``pip install --upgrade metatf=={METATF_VERSION}``.

Running examples
----------------

The Akida tutorials can be downloaded from the `examples <./examples/index.html>`_
section as Python scripts or Jupyter Notebooks. Dependencies needed to replay
the examples can be installed using the :download:`requirements.txt <../requirements.txt>`
file:

.. code-block:: bash

    pip install -r requirements.txt

.. note::
    Please refer to `this link <https://jupyter.org/>`__ for Jupyter Notebook installation
    and configuration.
