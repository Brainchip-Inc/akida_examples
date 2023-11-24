
Akida Engine
===============

Overview
--------

Engine is a C++ library that is part of the Akida software and is responsible to
program different Akida chips, and perform inference on it.

It can be run from a host PC, but it is also meant to run embedded on a
microcontroller (MCU), so it has limited capabilities: it contains only code
related to hardware programming, and has no knowledge of Model, or how to map
layers on its mesh. It works with what is called a "program", but does not know
how to generate one.

The program generation part must be done on a host PC, using the whole Akida
library containing what is called Hardware Backend.

The engine library is deployed in source. The deployment can be done using the
python package CLI:

.. code-block:: sh

    akida engine deploy --dest_path .

This will populate a directory containing headers, C++ sources, cmake files to
retrieve dependencies (Flatbuffer) and a README file that walks through the building
of the library and shows how to use it. If you want to test it on a linux host PC,
it is possible to use the ``--with-host-examples`` switch to obtain examples that will
run using the host library.

Engine directory structure
--------------------------

The Engine contains the following directories:

* ``api`` contains the headers that can be used as entry point when working with the akida engine. It contains these subdirectories:
    * ``akida``, containing the most commonly used API when working with the engine,
    * ``infra``, with generic utility functions, and the `system.h` header containing a list of the functions that should be implemented to have a working engine library,
    * ``akd500`` and ``akd1000``, with headers that can be used to build device drivers for the available devices containing Akida produced by Brainchip,
* ``cmake`` contain a Cmake file to fetch dependencies and build the engine library,
* ``devices`` provides driver instances that can be used with the available devices containing Akida produced by Brainchip,
* ``inc``, containing internal headers used by the engine sources,
* ``src`` hold main engine source files,
* ``test`` providing device-specific tests.

Engine API overview
-------------------

HardwareDriver
^^^^^^^^^^^^^^

The ``HardwareDriver`` class (whose definition is in ``infra/hardware_device.h``), is a pure virtual class that provides a mean to communicate with an Akida device.
Few example implementations of subclasses are provided to communicate with the AKD1000 and AKD1500.

The methods that need to be implemented to have a working ``HardwareDevice`` are:

* ``read`` and ``write`` operations, to read and write data to registers and memory addressable by Akida,
* ``desc``, providing a null-terminated string with driver description,
* ``scratch_memory`` and ``scratch_memory_size``, that will return the address and size of the scratch memory. This is a memory area addressable by Akida that it can be used to store temporary data to achieve programming and inference,
* ``akida_visible_memory`` and ``akida_visible_memory_size``, that return the address and size of memory that are accessible by Akida.

HardwareDevice
^^^^^^^^^^^^^^

To interact with an Akida device, the main entry point is the ``HardwareDevice`` (definition in ``akida/hardware_device.h``). To obtain a shared pointer of an instance of this abstract class it is possible to use the ``HardwareDevice::create`` static function. This takes a pointer to a``HardwareDriver`` instance as parameter. Once a ``HardwareDevice`` instance has been successfully created, it is possible to interact with it using its methods. Here's a list of the most useful methods:

* ``version``, to obtain the version of this device,
* ``create``, a static function, will create a ``shared_ptr`` to a ``HardwareDevice`` instance as described,
* ``program``, used to provide a program previously generated from a model mapped on a device with the same hardware version,
* ``set_batch_size``, that indicates the number of inputs that will be processed by Akida during inference. If the driver's `akida_visible_memory` method returns 0, it is possible to require the device to allocate memory for inputs in the scratch memory area,
* ``enqueue``, to trigger an inference on the device,
* ``fetch``, to poll for an output when the inference is completed,
* ``dequantize``, to apply a scale and bias to the output to obtain a float tensor.

Dense
^^^^^

In akida, all input and output buffers are wrapped in an abstract ``Dense`` class (defined in ``akida/dense.h``). This class is used to describe multidimensional dense arrays with a given type. Here's a list of the most useful methods:

* ``create``, a static function to allocate a buffer of a given ``TensorType``, ``Shape`` and ``Dense::Layout``, and create a ``Dense`` unique pointer that holds the buffer.
* ``create_view``, another static function, similar to the previous one, but whose data buffer is not allocated, but rather provided by the calling function. This function can be used to create a ``Dense`` instance to use as inference input coming from a user-provided buffer.
* ``split``, to obtain a vector of 3D ``Dense`` inputs that might have been prepared in four dimensions. The inference functions in the ``HardwareDevice`` require a vector of 3D inputs to be provided.
* ``buffer`` to obtain a pointer to the underlying ``Buffer`` object, that will provide a ``size`` and ``data`` methods. These could be used to read the output values.
* ``dimensions``, returrning the shape of the Dense object.
* ``operator==``, that can be used to compare with another ``Dense`` object.

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
