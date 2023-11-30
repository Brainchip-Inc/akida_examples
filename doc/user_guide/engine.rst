
Akida Engine
===============

Overview
--------

Engine is a C++ library that is part of the Akida software and is responsible for
programming different Akida chips, and performing inference on them.


It is a "lightweight" library targeting Micro Controller Unit (MCU). Models that have been
trained/converted (offline) can be passed for inference.

The program generation part must be done on a host PC, using the whole Akida
library containing what is called Hardware Backend. Refer to the
`model hardware mapping <akida.html#model-hardware-mapping>`__ section for
details on how to achieve this.

The engine library is deployed as source files. The deployment can be done using the
python package CLI:

.. code-block:: sh

    akida engine deploy --dest-path .

This will populate a directory containing headers, C++ sources, cmake files to
retrieve dependencies (Flatbuffer) and a README file.

To build the library, you can do:

.. code-block:: sh

    mkdir build
    cmake . -B build
    make -C build


.. note::
    The deployed ``README.md`` file will give additional information on how to use the
    library and interact with it. Few examples are also provided to show how
    the library can be used.


Engine directory structure
--------------------------

The Engine contains the following directories:

- ``api`` contains the headers that can be used as entry point when working with the Akida engine. It contains these subdirectories:

    - ``akida``, containing the most commonly used API when working with the engine,
    - ``infra``, with generic utility functions, and the `system.h` header containing a list of functions that should be implemented to have a working engine library,
    - ``akd1500`` and ``akd1000``, with headers that can be used to build device drivers for drivers for the engineering-sample chips that Brainchip designed and delivers as part of its development kits,
- ``cmake`` contains a Cmake file to fetch dependencies and build the engine library,
- ``devices`` provides driver instances for akd1000 and akd1500 in embedded environments.
- ``inc``, containing internal headers used by the engine sources,
- ``src`` holds main engine source files,
- ``test`` providing device-specific tests.

Engine API overview
-------------------

HardwareDriver
^^^^^^^^^^^^^^

The ``HardwareDriver`` class (whose definition is in ``infra/hardware_device.h``), is a pure virtual class that provides a means of communication with an Akida device.
Few example implementations of subclasses to communicate with the AKD1000 and AKD1500 are provided.

The methods that need to be implemented to have a working ``HardwareDevice`` are:

- ``read`` and ``write`` operations, to read and write data to registers and memory addressable by Akida,
- ``desc``, providing a null-terminated string with driver description,
- ``scratch_memory`` and ``scratch_memory_size``, that will return the address and size of the scratch memory. This is a memory area addressable by Akida that can be used to store temporary data to achieve programming and inference,
- ``akida_visible_memory`` and ``akida_visible_memory_size``, that return the address and size of memory that are accessible by Akida.

HardwareDevice
^^^^^^^^^^^^^^

To interact with an Akida device, the main entry point is the ``HardwareDevice`` (definition in ``akida/hardware_device.h``). To obtain a shared pointer of an instance of this abstract class it is possible to use the ``HardwareDevice::create`` static function. This takes a pointer to a ``HardwareDriver`` instance as parameter. Once a ``HardwareDevice`` instance has been successfully created, it is possible to interact with it using its methods. Here's a list of the most useful methods:

- ``version``, to obtain the version of this device,
- ``create``, a static function, will create a ``shared_ptr`` to a ``HardwareDevice`` instance as described,
- ``program``, used to provide a program previously generated from a model mapped on a device with the same hardware version,
- ``set_batch_size``, that indicates the number of inputs that will be processed by Akida during inference. If the driver's `akida_visible_memory` method returns 0, it is possible to require the device to allocate memory for inputs in the scratch memory area,
- ``enqueue``, to trigger an inference on the device,
- ``fetch``, to poll for an output when the inference is completed,
- ``dequantize``, to apply a scale and bias to the output to obtain a float tensor.


.. note::
    The ``HardwareDevice`` API in the engine is different from the Python API for the `HardwareDevice <../api_reference/akida_apis.html#hwdevice>`__. This is because the Python API is intended to be a higher-level, easy to use API that introduces to the concepts of Akida's hardware device programming. However, it can be observed that there are several similarities, as the Python API will end up calling the C++ instance.

Dense
^^^^^

In akida, all input and output buffers are wrapped in an abstract ``Dense`` class (defined in ``akida/dense.h``). This class is used to describe multidimensional dense arrays with a given type. Here's a list of the most useful methods:

- ``create``, a static function to allocate a buffer of a given ``TensorType``, ``Shape`` and ``Dense::Layout``, and create a ``Dense`` unique pointer that holds the buffer.
- ``create_view``, another static function, similar to the previous one, but whose data buffer is not allocated, but rather provided by the calling function. This function can be used to create a ``Dense`` instance to use as inference input coming from a user-provided buffer.
- ``split``, to obtain a vector of 3D ``Dense`` inputs that might have been prepared in four dimensions. The inference functions in the ``HardwareDevice`` require a vector of 3D inputs to be provided.
- ``buffer`` to obtain a pointer to the underlying ``Buffer`` object, that will provide a ``size`` and ``data`` methods. These could be used to read the output values.
- ``dimensions``, returning the shape of the Dense object.
- ``operator==``, that can be used to compare with another ``Dense`` object.

Shape
^^^^^

``Shape`` is a utility class defined in ``akida/shape.h`` that holds the shape dimensions, used by ``Dense`` class object. It can represent up to 4 dimensions. The methods are similar to the ``std::vector``, but they do not require dynamic allocation.

HwVersion
^^^^^^^^^

Defined in header ``akida/hw_version.h``, it is a structure that identifies uniquely a device version, with provided fields: ``vendor_id``, ``product_id``, ``major_rev`` and ``minor_rev``.

Sparse and Input conversion functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some models, akida will require inputs to be provided as sparse tensors, or it might provide sparse outputs. For these situations, an ``api/input_conversion.h`` header provides a collection of functions that allow conversion from dense to sparse and viceversa.

Other headers in the API
^^^^^^^^^^^^^^^^^^^^^^^^

Other headers in the engine API are there mostly to support the model library used by the python package. These are not usually necessary to develop C++ applications using the engine library.
