#!/bin/bash

log_base_dir="logs/monitoring"
log_dir="$log_base_dir/$(date +"%Y%m")"
mkdir -p "$log_dir"
log_file="$log_dir/cpu_$(date +"%Y%m%d").log"

echo "[INFO] Writing htop-like logs to: $log_file"

while true; do
    echo "===== $(date +"%Y-%m-%d %H:%M:%S") =====" >> "$log_file"
    
    # CPU usage per core
    mpstat -P ALL 1 1 | grep -v "CPU" >> "$log_file"
    
    # Memory usage
    free -h >> "$log_file"
    
    echo "" >> "$log_file"
    sleep 5
done
