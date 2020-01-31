Installation
============

Requirements
------------

* **Supported operating systems:** Windows 10, Ubuntu 16.04 and 18.04
* **Python version:** python 3.6 to 3.7
* **TensorFlow version:** 2.0.0

.. note::
    Using Windows, you may need to install the latest
    `Visual C++ redistributable package <https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads>`_.

Quick installation
------------------

.. warning::
    TensorFlow package is not automatically installed. Please refer to
    `Install TensorFlow with pip <https://www.tensorflow.org/install/pip>`_.

The Akida Execution Engine, the CNN2SNN tool, and the Akida models packages can
be setup with Python's pip package manager:

.. code-block:: bash

    pip install akida
    pip install cnn2snn
    pip install akida-models

.. note::
    We recommend using virtual environment such as `Conda <https://conda.io/docs/>`_

    .. code-block:: bash

      ``conda create --name akida_env python=3.6``

Running examples
----------------

The Akida tutorials can be downloaded from the `examples <examples/index.html>`_
section. Dependencies needed to replay the examples can be installed using the
:download:`requirements.txt <../requirements.txt>` file:

.. code-block:: bash

    pip install -r requirements.txt