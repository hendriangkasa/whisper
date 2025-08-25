#!/bin/bash
# watch_cpu.sh - Log CPU/memory status every 30s

log_base_dir="logs/monitoring"
current_month=$(date +"%Y%m")
log_dir="$log_base_dir/$current_month"

mkdir -p "$log_dir"

log_file="$log_dir/cpu_$(date +"%Y%m%d").log"

echo "[INFO] Logging CPU monitoring to: $log_file"
echo "[INFO] Press Ctrl+C to stop."

# Loop every 30s
while true; do
    echo "===== $(date +"%Y-%m-%d %H:%M:%S") =====" >> "$log_file"
    top -b -n 1 >> "$log_file"
    echo "" >> "$log_file"
    sleep 30
done
