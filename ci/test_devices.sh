#!/bin/sh

set -e
python -c "import akida; assert len(akida.devices()), 'Device not found.'"
python -c "import tensorflow as tf; assert len(tf.config.list_physical_devices('GPU'))"
