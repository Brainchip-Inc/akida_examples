#!/bin/sh

set -e
python -c "import tensorflow as tf; assert len(tf.config.list_physical_devices('GPU'))"
