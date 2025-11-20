#!/bin/bash
# Stop the running training process

cd /home/anik/code/Triton

echo "=================================================="
echo "Triton Training Stopper"
echo "=================================================="
echo ""

# Check if PID file exists
if [ ! -f logs/training.pid ]; then
    echo "❌ No training process found."
    echo "   No PID file at logs/training.pid"
    exit 1
fi

PID=$(cat logs/training.pid)

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo "Found training process (PID: $PID)"
    echo ""
    read -p "Are you sure you want to stop training? (y/n): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping training process..."
        kill $PID
        sleep 2

        # Check if it stopped
        if ps -p $PID > /dev/null 2>&1; then
            echo "Process still running, forcing stop..."
            kill -9 $PID
        fi

        echo "✓ Training stopped."
        rm -f logs/training.pid

        # Stop monitor if running
        if [ -f logs/monitor.pid ]; then
            MONITOR_PID=$(cat logs/monitor.pid)
            if ps -p $MONITOR_PID > /dev/null 2>&1; then
                echo "Stopping monitor process..."
                kill $MONITOR_PID 2>/dev/null
                echo "✓ Monitor stopped."
            fi
            rm -f logs/monitor.pid
        fi
    else
        echo "Cancelled. Training continues."
    fi
else
    echo "❌ Process $PID is not running."
    echo "   Cleaning up PID file..."
    rm -f logs/training.pid
fi
