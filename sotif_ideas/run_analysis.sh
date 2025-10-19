#!/bin/bash

echo "--- Starting SOTIF Analysis ---"

# Define the output directory for results
OUTPUT_DIR="sotif_results"
mkdir -p $OUTPUT_DIR

# --- TEST 1: Baseline (No Degradation) ---
echo "Running Baseline Test..."
python loop_over_sotif.py > "${OUTPUT_DIR}/baseline_log.txt"

# --- TEST 2: Glare Degradation ---
echo "Running Glare Tests..."
for intensity in 0.2 0.4 0.6 0.8; do
    echo "  - Glare Intensity: $intensity"
    python loop_over_sotif.py --sotif_mode glare --sotif_intensity $intensity > "${OUTPUT_DIR}/glare_${intensity}_log.txt"
done

# --- TEST 3: Rain Degradation ---
echo "Running Rain Tests..."
for intensity in 0.2 0.4 0.6 0.8; do
    echo "  - Rain Intensity: $intensity"
    python loop_over_sotif.py --sotif_mode rain --sotif_intensity $intensity > "${OUTPUT_DIR}/rain_${intensity}_log.txt"
done

echo "--- SOTIF Analysis Complete. Results saved in ${OUTPUT_DIR} ---"
