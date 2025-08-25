#!/bin/bash
# watch_htop.sh - Refreshes htop every 30 seconds

echo "[INFO] Starting htop monitoring (refresh every 30s)"
watch -n 30 htop
