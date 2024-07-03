#!/bin/bash

# Set variables for screen session
SCREEN_SESSION_NAME="optimization_analysis"
# LOG_FILE="run_jobs.log"

# Start a new screen session if it doesn't exist, or attach to existing one
if ! screen -ls | grep -q "$SCREEN_SESSION_NAME"; then
    screen -dmS $SCREEN_SESSION_NAME /bin/bash -c "exec bash -c './$SCREEN_SESSION_NAME.sh'"
    echo "Started new screen session: $SCREEN_SESSION_NAME"
else
    echo "Attaching to existing screen session: $SCREEN_SESSION_NAME"
fi

# You can continue with your main script logic here
# List of Python scripts to run
scripts=("optimization_analysis.py")

# Number of repetitions
num_repeats=10

# Loop through scripts and run them
for script in "${scripts[@]}"; do
    for ((i=1; i<=num_repeats; i++)); do
        echo "Running $script iteration $i"
        python "$script"
        if [ $? -ne 0 ]; then
            echo "$script iteration $i encountered an error, exiting"
            exit 1
        fi
        echo "$script iteration $i finished successfully"
    done
done
