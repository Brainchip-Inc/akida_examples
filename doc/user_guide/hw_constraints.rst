
Hardware constraints
====================

While working with CNN2SNN and the Akida simulator, only few limitations are
imposed. When mapping a model to the Akida hardware, not all Model and Layer
configurations are supported.

Akida NSoC (Pre-production)
---------------------------

InputConvolutional
^^^^^^^^^^^^^^^^^^

The InputConvolutional layer type is not supported.

Convolutional
^^^^^^^^^^^^^

+----------------+------------------+----------+--------+
|**Convolutions**|**Kernel Size**   |**Stride**|**Type**|
+----------------+------------------+----------+--------+
|**Parameters**  |1x1, 3×3, 5×5, 7×7|1         |Same    |
+----------------+------------------+----------+--------+

+---------------+-------------+----------+
|**Max Pooling**|**Size**     |**Stride**|
+---------------+-------------+----------+
|**Parameters** |1x1, 2×2, 3x3|1, 2, 3   |
+---------------+-------------+----------+

.. note::
       * pooling stride cannot be greater than pooling size.
       * pooling size cannot be greater than input dimensions.
       * with max pooling, kernel size cannot be greater than input dimensions.
       * a layer with max pooling must be followed by another Convolutional or
         SeparableConvolutional layer.

+-------------------+-------------+-------------+
|**Average Pooling**|**Width**    |**Height**   |
+-------------------+-------------+-------------+
|**Dimensions**     |8, 16, 24, 32|multiple of 8|
+-------------------+-------------+-------------+

+----------------+---------+-----------+--------------+
|**Quantization**|**Input**|**Weights**|**Activation**|
+----------------+---------+-----------+--------------+
|**Bitwidth**    |1, 2, 4  |1, 2       |1, 2, 4       |
+----------------+---------+-----------+--------------+

SeparableConvolutional
^^^^^^^^^^^^^^^^^^^^^^

+----------------+---------------+----------+--------+
|**Convolutions**|**Kernel Size**|**Stride**|**Type**|
+----------------+---------------+----------+--------+
|**Parameters**  |3×3, 5×5, 7×7  |1         |Same    |
+----------------+---------------+----------+--------+

+---------------+-------------+----------+
|**Max Pooling**|**Size**     |**Stride**|
+---------------+-------------+----------+
|**Parameters** |1x1, 2×2, 3x3|1, 2, 3   |
+---------------+-------------+----------+

.. note::
       * pooling stride cannot be greater than pooling size.
       * pooling size cannot be greater than input dimensions.
       * with max pooling, kernel size cannot be greater than input dimensions.
       * a layer with max pooling must be followed by another Convolutional or
         SeparableConvolutional layer.

+-------------------+-------------+-------------+
|**Average Pooling**|**Width**    |**Height**   |
+-------------------+-------------+-------------+
|**Dimensions**     |8, 16, 24, 32|multiple of 8|
+-------------------+-------------+-------------+

+----------------+---------+-----------+--------------+
|**Quantization**|**Input**|**Weights**|**Activation**|
+----------------+---------+-----------+--------------+
|**Bitwidth**    |1, 2, 4  |2, 4       |1, 2, 4       |
+----------------+---------+-----------+--------------+

FullyConnected
^^^^^^^^^^^^^^

+--------------+---------+----------+------------+
|**Input**     |**Width**|**Height**|**Channels**|
+--------------+---------+----------+------------+
|**Dimensions**|1        |1         |81920       |
+--------------+---------+----------+------------+

+----------------+---------+-----------+--------------+
|**Quantization**|**Input**|**Weights**|**Activation**|
+----------------+---------+-----------+--------------+
|**Bitwidth**    |1, 2     |1, 2       |1, 2, 4       |
+----------------+---------+-----------+--------------+

Akida NSoC (Production)
-----------------------

InputConvolutional
^^^^^^^^^^^^^^^^^^

+--------------+---------+----------+------------+
|**Input**     |**Width**|**Height**|**Channels**|
+--------------+---------+----------+------------+
|**Dimensions**|[5:256]  |>= 5      |1, 3        |
+--------------+---------+----------+------------+

+----------------+---------------+----------+-----------+
|**Convolutions**|**Kernel Size**|**Stride**|**Type**   |
+----------------+---------------+----------+-----------+
|**Parameters**  |3×3, 5×5, 7×7  |1, 2, 3   |Same, Valid|
+----------------+---------------+----------+-----------+

+-------------+-------+-------+-------+
|**Filters**  |**3x3**|**5x5**|**7x7**|
+-------------+-------+-------+-------+
|**Max(1 ch)**|512    |192    |96     +
+-------------+-------+-------+-------+
|**Max(3 ch)**|192    |64     |32     +
+-------------+-------+-------+-------+

+---------------+------------------+----------+
|**Max Pooling**|**Size**          |**Stride**|
+---------------+------------------+----------+
|**Parameters** |1x1, 1×2, 2×1, 2×2|2         |
+---------------+------------------+----------+

+----------------+---------+-----------+--------------+
|**Quantization**|**Input**|**Weights**|**Activation**|
+----------------+---------+-----------+--------------+
|**Bitwidth**    |8        |[1:8]      |1, 2, 4       |
+----------------+---------+-----------+--------------+

Convolutional
^^^^^^^^^^^^^

+----------------+------------------+----------+--------+
|**Convolutions**|**Kernel Size**   |**Stride**|**Type**|
+----------------+------------------+----------+--------+
|**Parameters**  |1x1, 3×3, 5×5, 7×7|1, 2      |Same    |
+----------------+------------------+----------+--------+

.. note::
       * stride 2 is only supported with 3x3 kernels

+---------------+-------------+----------+
|**Max Pooling**|**Size**     |**Stride**|
+---------------+-------------+----------+
|**Parameters** |1x1, 2×2, 3x3|1, 2, 3   |
+---------------+-------------+----------+

.. note::
       * pooling stride cannot be greater than pooling size
       * a layer with max pooling must be followed by another Convolutional or
         SeparableConvolutional layer.

+-------------------+---------+
|**Average Pooling**|**Width**|
+-------------------+---------+
|**Dimensions**     |[1:32]   |
+-------------------+---------+

+----------------+---------+-----------+--------------+
|**Quantization**|**Input**|**Weights**|**Activation**|
+----------------+---------+-----------+--------------+
|**Bitwidth**    |1, 2, 4  |1, 2, 4    |1, 2, 4       |
+----------------+---------+-----------+--------------+

SeparableConvolutional
^^^^^^^^^^^^^^^^^^^^^^

+----------------+---------------+----------+--------+
|**Convolutions**|**Kernel Size**|**Stride**|**Type**|
+----------------+---------------+----------+--------+
|**Parameters**  |3×3, 5×5, 7×7  |1, 2      |Same    |
+----------------+---------------+----------+--------+

.. note::
       * stride 2 is only supported with 3x3 kernels

+---------------+-------------+----------+
|**Max Pooling**|**Size**     |**Stride**|
+---------------+-------------+----------+
|**Parameters** |1x1, 2×2, 3x3|1, 2, 3   |
+---------------+-------------+----------+

.. note::
       * pooling stride cannot be greater than pooling size.
       * a layer with max pooling must be followed by another Convolutional or
         SeparableConvolutional layer.

+-------------------+---------+
|**Average Pooling**|**Width**|
+-------------------+---------+
|**Dimensions**     |[1:32]   |
+-------------------+---------+

+----------------+---------+-----------+--------------+
|**Quantization**|**Input**|**Weights**|**Activation**|
+----------------+---------+-----------+--------------+
|**Bitwidth**    |1, 2, 4  |2, 4       |1, 2, 4       |
+----------------+---------+-----------+--------------+

FullyConnected
^^^^^^^^^^^^^^

+--------------+---------+----------+---------+
|**Input**     |**Width**|**Height**|**WxHxC**|
+--------------+---------+----------+---------+
|**Dimensions**|1        |1         |81920    |
+--------------+---------+----------+---------+

+----------------+---------+-----------+--------------+
|**Quantization**|**Input**|**Weights**|**Activation**|
+----------------+---------+-----------+--------------+
|**Bitwidth**    |1, 2, 4  |1, 2, 4    |1, 2, 4       |
+----------------+---------+-----------+--------------+

