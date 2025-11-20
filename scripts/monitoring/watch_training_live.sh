#!/bin/bash

# Live Training Monitor - Auto-refreshes every 10 seconds
# Press Ctrl+C to exit

REFRESH_INTERVAL=10

while true; do
    clear
    ./monitor_training_detailed.sh
    echo ""
    echo "Auto-refreshing every ${REFRESH_INTERVAL} seconds... (Press Ctrl+C to exit)"
    sleep $REFRESH_INTERVAL
done
