#!/bin/bash
# Launch script for robust Optuna hyperparameter search

cd /astro/users/lindajin/WL_ML

echo "=========================================="
echo "Starting Optuna Hyperparameter Search"
echo "Time: $(date)"
echo "=========================================="

# Run with nohup for true background execution
nohup /astro/users/lindajin/miniforge3/envs/WL_ML_Challenge/bin/python run_optuna_robust.py > optuna_log_$(date +%Y%m%d_%H%M%S).txt 2>&1 &

# Get the process ID
PID=$!

echo ""
echo "Process started with PID: $PID"
echo "Log file: optuna_log_$(date +%Y%m%d_%H%M%S).txt"
echo ""
echo "To monitor progress:"
echo "  tail -f optuna_log_*.txt"
echo ""
echo "To check if running:"
echo "  ps aux | grep $PID"
echo ""
echo "To stop:"
echo "  kill $PID"
echo ""
echo "=========================================="
