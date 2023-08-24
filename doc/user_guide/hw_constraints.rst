
Hardware constraints
====================

.. warning::
       The following constraints concern only Akida 1.0 IP based solutions
       and the AKD1000 reference SoC.

While working with CNN2SNN and the Akida simulator, only few limitations are
imposed. When mapping a model to the Akida hardware, not all Model and Layer
configurations are supported.

.. tab-set::

    .. tab-item:: InputConvolutional

        .. card::

            **Input dimensions**
            ^^^^^^^^^^^^^^^^^^^^
            +---------+----------+------------+
            |**Width**|**Height**|**Channels**|
            +---------+----------+------------+
            |[5:256]  |>= 5      |1, 3        |
            +---------+----------+------------+

        .. card::

            **Convolution parameters**
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
            +---------------+----------+-----------+
            |**Kernel Size**|**Stride**|**Type**   |
            +---------------+----------+-----------+
            |3×3, 5×5, 7×7  |1, 2, 3   |Same, Valid|
            +---------------+----------+-----------+

        .. card::

            **Maximum number of filters**
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            +-----------------+-------+-------+-------+
            |**Kernel Sise**  |**3×3**|**5×5**|**7×7**|
            +-----------------+-------+-------+-------+
            |**Max(1 ch)**    |512    |192    |96     +
            +-----------------+-------+-------+-------+
            |**Max(3 ch)**    |192    |64     |32     +
            +-----------------+-------+-------+-------+

        .. card::

            **Max Pooling size**
            ^^^^^^^^^^^^^^^^^^^^
            1×1, 1×2, 2×1, 2×2
            +++++++++
            :octicon:`report;1em;sd-text-warning` Pool stride is equal to pool size for
            `InputConvolutional <../api_reference/akida_apis.html#akida.InputConvolutional>`__.

        .. card::

            **Quantization bitwidth**
            ^^^^^^^^^^^^^^^^^^^^^^^^^
            +---------+------------+--------------+
            |**Input**|**Weights** |**Activation**|
            +---------+------------+--------------+
            |8        | 1, 2, 4, 8 |1, 2, 4       |
            +---------+------------+--------------+
            +++++++++
            :octicon:`report;1em;sd-text-warning` While minimum weights bitwidth supported is 1
            for native learning, CNN2SNN quantization only allows quantization with bitwidth >=2
            because float weights are signed while 1-bit integers are unsigned by definition.


    .. tab-item:: Convolutional

        .. card::

            **Convolution parameters**
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
            +------------------+----------+--------+
            |**Kernel Size**   |**Stride**|**Type**|
            +------------------+----------+--------+
            |1×1, 3×3, 5×5, 7×7|1, 2      |Same    |
            +------------------+----------+--------+
            ++++++++
            :octicon:`report;1em;sd-text-warning` Stride 2 is only supported with 3×3 kernels.

        .. card::

            **Max Pooling parameters**
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
            +-------------+----------+
            |**Size**     |**Stride**|
            +-------------+----------+
            |1×1, 2×2     |1, 2      |
            +-------------+----------+
            ++++++++
            :octicon:`report;1em;sd-text-warning` Pooling stride cannot be greater than pooling size,
            layer with max pooling must be followed by another `Convolutional
            <../api_reference/akida_apis.html#akida.Convolutional>`__ or `SeparableConvolutional
            <../api_reference/akida_apis.html#akida.SeparableConvolutional>`__ layer.

        .. card::

            **Global Average Pooling width**
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            [1:32]
            ++++++++
            :octicon:`report;1em;sd-text-warning` The output of the convolution must have at least 3 rows.

        .. card::

            **Quantization bitwidth**
            ^^^^^^^^^^^^^^^^^^^^^^^^^
            +---------+-----------+--------------+
            |**Input**|**Weights**|**Activation**|
            +---------+-----------+--------------+
            |1, 2, 4  |1, 2, 4    |1, 2, 4       |
            +---------+-----------+--------------+
            +++++++++
            :octicon:`report;1em;sd-text-warning` While minimum weights bitwidth supported is 1
            for native learning, CNN2SNN quantization only allows quantization with bitwidth >=2
            because float weights are signed while 1-bit integers are unsigned by definition.


    .. tab-item:: SeparableConvolutional

        .. card::

            **Convolution parameters**
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
            +---------------+----------+--------+
            |**Kernel Size**|**Stride**|**Type**|
            +---------------+----------+--------+
            |3×3, 5×5, 7×7  |1, 2      |Same    |
            +---------------+----------+--------+
            +++++++++
            :octicon:`report;1em;sd-text-warning` Stride 2 is only supported with 3×3 kernels.

        .. card::

            **Max Pooling parameters**
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
            +-------------+----------+
            |**Size**     |**Stride**|
            +-------------+----------+
            |1×1, 2×2     |1, 2      |
            +-------------+----------+
            ++++++++
            :octicon:`report;1em;sd-text-warning` Pooling stride cannot be greater than pooling size,
            layer with max pooling must be followed by another `Convolutional
            <../api_reference/akida_apis.html#akida.Convolutional>`__ or `SeparableConvolutional
            <../api_reference/akida_apis.html#akida.SeparableConvolutional>`__ layer.

        .. card::

            **Global Average Pooling width**
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            [1:32]
            ++++++++
            :octicon:`report;1em;sd-text-warning` The output of the convolution must have at least 3 rows,
            1×1 inputs are not supported.

        .. card::

            **Quantization bitwidth**
            ^^^^^^^^^^^^^^^^^^^^^^^^^
            +---------+-----------+--------------+
            |**Input**|**Weights**|**Activation**|
            +---------+-----------+--------------+
            |1, 2, 4  |2, 4       |1, 2, 4       |
            +---------+-----------+--------------+
            ++++++++
            :octicon:`report;1em;sd-text-warning` While minimum weights bitwidth supported is 1
            for native learning, CNN2SNN quantization only allows quantization with bitwidth >=2
            because float weights are signed while 1-bit integers are unsigned by definition.

    .. tab-item:: FullyConnected

        .. card::

            **Input dimensions**
            ^^^^^^^^^^^^^^^^^^^^
            +---------+----------+---------+
            |**Width**|**Height**|**WxHxC**|
            +---------+----------+---------+
            |1        |1         |<= 57334 |
            +---------+----------+---------+

        .. card::

            **Quantization bitwidth**
            ^^^^^^^^^^^^^^^^^^^^^^^^^
            +---------+-----------+--------------+
            |**Input**|**Weights**|**Activation**|
            +---------+-----------+--------------+
            |1, 2, 4  |1, 2, 4    |1, 2, 4       |
            +---------+-----------+--------------+
            ++++++++
            :octicon:`report;1em;sd-text-warning` While minimum weights bitwidth supported is 1
            for native learning, CNN2SNN quantization only allows quantization with bitwidth >=2
            because float weights are signed while 1-bit integers are unsigned by definition.
