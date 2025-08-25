#!/bin/bash
log_base_dir="logs/monitoring"
log_dir="$log_base_dir/$(date +"%Y%m")"
mkdir -p "$log_dir"
log_file="$log_dir/cpu_$(date +"%Y%m%d").log"
echo "[INFO] Writing htop-like logs to: $log_file"

while true; do
    echo "===== $(date +"%Y-%m-%d %H:%M:%S") =====" >> "$log_file"
    
    # CPU usage per core using htop in batch mode
    htop -d 10 -n 1 -b | head -20 >> "$log_file"
    
    # Memory usage
    free -h >> "$log_file"
    
    # Load average
    echo "Load: $(cat /proc/loadavg)" >> "$log_file"
    
    echo "" >> "$log_file"
    sleep 5
done
