#!/usr/bin/env sh

set -ex

set -- 4 6 8 10 12 14

for window; do
    echo
    echo "Running MPC with $window hour window"
    echo

    start=$(date +%s)
    uv run main.py run-mpc --horizon-hours "$window" --num-steps 12 --mode forecast
    end=$(date +%s)
    elapsed=$((end - start))

    echo
    echo "Took $elapsed s"
    echo

    window_dir=$(printf "data/results/forecast-no-buffer-12-steps-slsqp-ftol+10/%02d-hour" "$window")
    mkdir -p "$window_dir"
    cp results/optimal_control.json "$window_dir"
done
