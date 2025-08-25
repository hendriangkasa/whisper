#!/bin/bash
# watch_cpu.sh - Log CPU/memory status using top (htop alternative)

log_base_dir="logs/monitoring"
current_month=$(date +"%Y%m")
log_dir="$log_base_dir/$current_month"

mkdir -p "$log_dir"

log_file="$log_dir/cpu_$(date +"%Y%m%d").log"

echo "[INFO] Logging CPU monitoring to: $log_file"
echo "[INFO] Press Ctrl+C to stop."

# Run top in batch mode (-b), capture a snapshot every 30s
watch -n 30 "echo '===== \$(date +\"%Y-%m-%d %H:%M:%S\") =====' >> $log_file; top -b -n 1 >> $log_file; echo '' >> $log_file"
