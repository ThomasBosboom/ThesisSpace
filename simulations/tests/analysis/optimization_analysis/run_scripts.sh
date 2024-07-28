#!/bin/bash

# Array of script names
# scripts=("optimization_analysis_nelder_mead.py" "optimization_analysis_particle_swarm.py")
scripts=("optimization_analysis_particle_swarm.py")
# scripts=("optimization_analysis_particle_swarm_test1.py" "optimization_analysis_particle_swarm_test2.py")


# Number of times to run each script
runs=20

# Function to run a script a specified number of times with error handling
run_script() {
    script=$1
    for ((i=1; i<=runs; i++)); do
        echo "Running $script, iteration $i"
        if python "$script"; then
            echo "$script iteration $i completed successfully."
        else
            echo "Error running $script iteration $i. Exiting."
            exit 1
        fi
    done
}

# Run each script in parallel
for script in "${scripts[@]}"; do
    run_script "$script" &
done

# Wait for all background jobs to finish
wait

echo "All scripts have completed."
