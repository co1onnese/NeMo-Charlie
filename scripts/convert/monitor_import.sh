#!/bin/bash
# Monitor GPU usage during NeMo import

echo "Monitoring GPU usage every 30 seconds for 5 minutes..."
echo "======================================================"
echo ""

for i in {1..10}; do
    echo "--- Check $i/10 ($(date +%H:%M:%S)) ---"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits
    
    # Check if import process is still running
    if ps aux | grep -q "[i]mport_to_nemo.py"; then
        echo "Status: import_to_nemo.py is RUNNING"
    else
        echo "Status: import_to_nemo.py is NOT running (finished or crashed)"
    fi
    
    # Check disk I/O to /data
    echo "Disk I/O to /data:"
    iostat -x 1 2 | grep -A1 "Device" | tail -1 || echo "iostat not available"
    
    echo ""
    sleep 30
done

echo "======================================================"
echo "5-minute monitoring complete"
echo "Final GPU state:"
nvidia-smi
