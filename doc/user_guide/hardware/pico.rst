Akida Pico capabilities
=======================

.. note::
       These details are relevant to Akida Pico IP-based solutions.

For compatibility with multiple possible hardware backends downstream, CNN2SNN and
the Akida simulator impose few constraints on layer dimensions. However, hardware
does have limits in this respect, which will be checked at the stage of mapping a
model to a specific device (real or virtual). This page details the limits for the
Akida Pico IP.

Please refer to `Akida Pico layers <../../api_reference/akida_apis.html#akida-pico-layers>`__
for layers description.

.. tab-set::

    .. tab-item:: StatefulRecurrent

        .. card::

            **Model structure**
            ^^^^^^^^^^^^^^^^^^^
            | :octicon:`report;1em;sd-text-warning` A model cannot have more than 8
              StatefulRecurrent layers.
            | :octicon:`report;1em;sd-text-warning` Only a Dequantizer or a PicoPostProcessing
              layer is allowed at the end of a series of StatefulRecurrent layers, and it must be
              the last layer of the model.
            | :octicon:`report;1em;sd-text-warning` When the ReLU activation is present, it must be
              unbounded.
        .. card::

            **Input**
            ^^^^^^^^^
            +--------------------------+---------------------------+----------------------------+
            |**Layer**                 |**Input bitwidth**         |**Channels**                |
            +--------------------------+---------------------------+----------------------------+
            |First                     |8                          |<=256                       |
            +--------------------------+---------------------------+----------------------------+
            |First                     |16                         |<=128                       |
            +--------------------------+---------------------------+----------------------------+
            |Intermediate              |8                          |<=256                       |
            +--------------------------+---------------------------+----------------------------+

            | :octicon:`report;1em;sd-text-warning` For the first layer, the packed input row
              (channels × input bits) must occupy a power-of-2 number of 32-bit words and cannot
              exceed 512 bytes.

            For example:

            +-----------------------+---------------+----------------+----------------------------------------------+
            |**Input configuration**|**Packed size**|**32-bit words**|**Status**                                    |
            +-----------------------+---------------+----------------+----------------------------------------------+
            |3 × 8-bit              |24 bits        |1               |Valid                                         |
            +-----------------------+---------------+----------------+----------------------------------------------+
            |9 × 8-bit              |72 bits        |3               |Mapping error, pad channels to 13-16 (4 words)|
            +-----------------------+---------------+----------------+----------------------------------------------+
            |3 × 16-bit             |48 bits        |2               |Valid                                         |
            +-----------------------+---------------+----------------+----------------------------------------------+
            |6 × 16-bit             |96 bits        |3               |Mapping error, pad channels to 7-8 (4 words)  |
            +-----------------------+---------------+----------------+----------------------------------------------+

            | :octicon:`report;1em;sd-text-warning` Padding is not performed automatically by MetaTF
              tools: if the packed input row does not occupy a power-of-2 number of
              32-bit words, mapping fails and the user must pad the input channels to reach the
              next power-of-2 word count.

        .. card::

            **Parameters**
            ^^^^^^^^^^^^^^
            +---------------------+-------------------+-------------+
            |**Stateful channels**|**Output channels**|**Subsample**|
            +---------------------+-------------------+-------------+
            |<=256, power of 2    |<=256              |<=4          |
            +---------------------+-------------------+-------------+
            ++++++++
            | :octicon:`report;1em;sd-text-warning` Stateful channels must be the same for all
              StatefulRecurrent layers of the model.
            | :octicon:`report;1em;sd-text-warning` The last StatefulRecurrent layer of the model
              must have 8-bit, 28-bit or 32-bit outputs.

    .. tab-item:: PicoPostProcessing

        .. card::

            **Input dimensions**
            ^^^^^^^^^^^^^^^^^^^^
            +------------------+-------------+
            |**Channels**      |**Timesteps**|
            +------------------+-------------+
            |1                 |[3:128]      |
            +------------------+-------------+

.. note::
      In addition to the limits above, the memory required by each layer (events, states,
      filters and time constants) is checked against the SRAM sizes of the target device
      at mapping time.
