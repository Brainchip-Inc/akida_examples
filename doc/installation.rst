Installation
============

Requirements
------------

* **Supported operating systems:** Windows 10, Ubuntu 16.04, 18.04 and 20.04
* **Python version:** python 3.6 to 3.8
* **TensorFlow version:** >= 2.4.0

.. warning::
    Using Windows, the latest Visual C++ redistributable package is required.
    Please refer to `this link
    <https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads>`_
    for installation.

Quick installation
------------------

.. warning::
    TensorFlow package is required to use the `CNN2SNN tool
    <https://pypi.org/project/cnn2snn>`_, the `Akida model zoo
    <https://pypi.org/project/akida-models>`_ and to run the `examples
    <examples/index.html>`_. Please refer to
    `Install TensorFlow with pip <https://www.tensorflow.org/install/pip>`_
    for installation.

The Akida Execution Engine, the CNN2SNN tool, and the Akida models packages can
be setup with Python's pip package manager:

.. code-block:: bash

    pip install akida
    pip install cnn2snn
    pip install akida-models

.. note::
    We recommend using virtual environment such as `Conda <https://conda.io/docs/>`_:

    .. code-block:: bash

      conda create --name akida_env python=3.6
      conda activate akida_env

Running examples
----------------

The Akida tutorials can be downloaded from the `examples <examples/index.html>`_
section. Dependencies needed to replay the examples can be installed using the
:download:`requirements.txt <../requirements.txt>` file:

.. code-block:: bash

    pip install -r requirements.txt
