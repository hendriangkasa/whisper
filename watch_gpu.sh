#!/bin/bash
# watch_gpu.sh - Log GPU status using watch

log_base_dir="logs/monitoring"
current_month=$(date +"%Y%m")
log_dir="$log_base_dir/$current_month"

mkdir -p "$log_dir"

log_file="$log_dir/gpu_$(date +"%Y%m%d").log"

echo "[INFO] Logging GPU monitoring to: $log_file"
echo "[INFO] Press Ctrl+C to stop."

# Run watch, append timestamp + nvidia-smi every 30s
watch -n 30 "echo '===== \$(date +\"%Y-%m-%d %H:%M:%S\") =====' >> $log_file; nvidia-smi >> $log_file; echo '' >> $log_file"
