#!/bin/bash
# watch_gpu.sh - Refreshes nvidia-smi every 30 seconds

echo "[INFO] Starting GPU monitoring (refresh every 30s)"
watch -n 30 nvidia-smi
