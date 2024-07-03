#!/bin/sh

set -e
python -c "import akida; assert len(akida.devices()), 'Device not found.'"
