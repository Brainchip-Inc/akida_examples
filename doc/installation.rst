Installation
============

.. warning::
    As of July 7, 2025, the latest
    `opencv-python <https://pypi.org/project/opencv-python/#history>`__ package introduces
    conflicting dependencies.

    **Before installing any version of MetaTF, you must run:**

    .. code-block:: bash

        pip install opencv-python==4.11.0.86

    From MetaTF 2.15 onward, this issue will no longer occur.


Supported configurations
------------------------

* **Operating systems:**
    * Windows 10, Windows 11
    * Any Linux variant compatible with `manylinux 2.28 <https://github.com/pypa/manylinux>`_ (Ubuntu 20.04, Ubuntu 22.04, ...)
* **Python versions:** 3.9 to 3.11
* **TensorFlow versions:** 2.15

.. warning::
    Using Windows, the latest Visual C++ redistributable package is required.
    Please refer to `this link
    <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist>`__
    for installation.

Quick installation
------------------

.. warning::
    TensorFlow package is required to use the `CNN2SNN tool
    <https://pypi.org/project/cnn2snn>`_, the `Akida model zoo
    <https://pypi.org/project/akida-models>`_ and to run the `examples
    <./examples/index.html>`_. Please refer to
    `Install TensorFlow with pip <https://www.tensorflow.org/install/pip>`_
    for installation.

The akida, CNN2SNN and akida_models python packages can
be setup with Python's pip package manager:

.. code-block:: bash

    pip install akida=={AKIDA_VERSION}
    pip install cnn2snn=={CNN2SNN_VERSION}
    pip install akida-models=={MODELS_VERSION}

.. note::
    We recommend using virtual environment such as `Conda <https://conda.io/docs/>`_.
    Please note that the python version must be explicitly specified when creating a
    conda environment. The specification must be for one of the supported python
    versions listed above.

    .. code-block:: bash

      conda create --name akida_env python=3.11
      conda activate akida_env

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
