
Hardware constraints
====================

While working with CNN2SNN and the Akida simulator, only few limitations are
imposed. When mapping a model to the FPGA emulator or to the Akida hardware,
more constraints have to be taken into account.

The tables below summarizes the hardware limitations for Akida layers.

Input layer
-----------

+------------------+--------------+-----------+-------------+--------+-----------+
|Type              |input channels|weightsBits|kernelSize   |stride  |type       |
+==================+==============+===========+=============+========+===========+
|InputConvolutional|1, 3          |8          |3×3, 5×5, 7×7|1, 2, 3 |Same, Valid|
+------------------+--------------+-----------+-------------+--------+-----------+

InputConvolutional embeds a pooling step with hardware constraints:

+------------+-------------+------+-----------+
|Pooling type|Kernel size  |Stride|Type       |
+============+=============+======+===========+
|Max         |1×2, 2×1, 2×2|2     |Same, Valid|
+------------+-------------+------+-----------+

Data-Processing layers
----------------------

Convolutional layer
^^^^^^^^^^^^^^^^^^^

+----------------------+-----------+------------------+------+------+
|Type                  |weightsBits|kernelSize        |stride|type  |
+======================+===========+==================+======+======+
|Convolutional         |1, 2       |1×1, 3×3, 5×5, 7×7|1     |Same  |
+----------------------+-----------+------------------+------+------+
|SeparableConvolutional|2, 4       |3×3, 5×5, 7×7     |1     |Same  |
+----------------------+-----------+------------------+------+------+

All convolutional layers embed a pooling step. Those steps can be active or
inactive and have got some limitations.

+---------------+---------------+-----------------+------+
|Pooling type   |kernelSize     |stride           |type  |
+===============+===============+=================+======+
|Max            |2×2            |1, 2             |Same  |
|               |               |                 |      |
|               |3x3            |1, 2, 3          |      |
+---------------+---------------+-----------------+------+
|Global Average |               |                 |Same  |
+---------------+---------------+-----------------+------+

.. note::
       Global average pooling kernel size mustn't exceed 20 bits.

Fully connected layer
^^^^^^^^^^^^^^^^^^^^^

+--------------+------------+-----------+
|Layer type    |input spikes|weightsBits|
+==============+============+===========+
|FullyConnected|1, 2        |1, 2, 3, 4 |
+--------------+------------+-----------+

.. note::
       The layer placed before a FullyConnected layer must have a
       ``threshold_fire_bits`` set to 1 or 2.

.. note::
       FullyConnected layers cannot be placed after a layer with max pooling
       enabled.
