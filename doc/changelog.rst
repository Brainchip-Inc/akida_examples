
Changelog
==========

| The MetaTF documentation is generated from the `akida_examples repository <https://github.com/Brainchip-Inc/akida_examples>`_.
| It relies on `Sphinx <https://www.sphinx-doc.org>`_ python documentation
  generator and `GitHub Pages <https://docs.github.com/pages>`_ documentation
  for rendering.

| Please refer to the repository `release area <https://github.com/Brainchip-Inc/akida_examples/releases>`_
  for the full changelog.

.. dropdown:: For reference, this documentation was generated using the following packages
   :animate: fade-in

      {PIP_FREEZE}

MetaTF 2.19.2
-------------

Released in July 2026. Package set: **akida/cnn2snn 2.19.2**, **quantizeml 1.2.4**,
**akida-models 1.14.1**. Aligned with FPGA versions 1764 (2-nodes), 1765 (6-nodes),
1766 (6-nodes bittware) and 905 (Pico).

**New features:**

* Added 4-node and 12-node virtual devices
* Added AKD1500 files to the engine deploy CLI
* Added InputQuantizer support for 1.0 models
* Relaxed TNP-R (Pico) mapping constraints to match hardware capabilities
* Enabled dynamic shape support on Conv2D for TensorFlow evaluation (quantizeml)

**Bug fixes:**

* InputConvolutional: reject ``act_bits=8``, fixed MaxPooling padding mismatch and
  activation equalization errors
* Improved TNP-R layer error messages
* Resolved importlib_resources and tensorflow-metadata dependency conflicts, and
  updated the matplotlib requirement to >= 3.11 (quantizeml)
* Pico fault classification model is now hardware compatible (akida-models)

Previous versions
-----------------

.. dropdown:: MetaTF documentation previous versions
   :animate: fade-in

   * `2.18.2-doc-1 <https://brainchip-inc.github.io/akida_examples_2.18.2-doc-1/>`_
   * `2.17.0-doc-1 <https://brainchip-inc.github.io/akida_examples_2.17.0-doc-1/>`_
   * `2.16.1-doc-1 <https://brainchip-inc.github.io/akida_examples_2.16.1-doc-1/>`_
   * `2.15.0-doc-1 <https://brainchip-inc.github.io/akida_examples_2.15.0-doc-1/>`_
   * `2.14.0-doc-1 <https://brainchip-inc.github.io/akida_examples_2.14.0-doc-1/>`_
   * `2.13.0-doc-1 <https://brainchip-inc.github.io/akida_examples_2.13.0-doc-1/>`_
   * `2.12.0-doc-1 <https://brainchip-inc.github.io/akida_examples_2.12.0-doc-1/>`_
   * `2.11.0-doc-1 <https://brainchip-inc.github.io/akida_examples_2.11.0-doc-1/>`_
   * `2.10.0-doc-1 <https://brainchip-inc.github.io/akida_examples_2.10.0-doc-1/>`_
   * `2.9.0-doc-1 <https://brainchip-inc.github.io/akida_examples_2.9.0-doc-1/>`_
   * `2.8.1-doc-1 <https://brainchip-inc.github.io/akida_examples_2.8.1-doc-1/>`_
   * `2.7.2-doc-1 <https://brainchip-inc.github.io/akida_examples_2.7.2-doc-1/>`_
   * `2.6.0-doc-1 <https://brainchip-inc.github.io/akida_examples_2.6.0-doc-1/>`_
   * `2.4.0-doc-1 <https://brainchip-inc.github.io/akida_examples_2.4.0-doc-1/>`_
   * `2.3.0-doc-1 <https://brainchip-inc.github.io/akida_examples_2.3.0-doc-1/>`_

MetaTF Beta
-----------

Beta releases are not intended for production use and may be unstable. They are provided to allow
users to test new features and provide feedback.

- Beta releases are available at https://doc.brainchipinc.com/beta
- Related packages are to be downloaded from https://data.brainchip.com/metatf_beta/
- Please fill-in the `feedback survey
  <https://docs.google.com/forms/d/e/1FAIpQLSd9gzZROr-CHdY5jipGdIB8VtNNa5vPL4UvLZ5GOBXOZv2MGw/viewform>`_
  to help us improve future releases.