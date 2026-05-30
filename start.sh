#!/usr/bin/env bash
# Launch the reader server. Chatterbox uses PyTorch which manages CUDA itself.
exec python "$(dirname "$0")/server.py"
