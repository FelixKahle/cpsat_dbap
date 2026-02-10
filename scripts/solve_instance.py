# Copyright (c) 2026 Felix Kahle.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Explicitly add the project root to sys.path to allow execution 
# without installing the package (e.g., when running from the 'scripts' folder).
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from cpsat_dbap.instance import parse_instance
from cpsat_dbap.solver import solve, SolverConfig
from cpsat_dbap.plotting import plot_schedule

def main() -> None:
    """
    Main execution entry point for solving a specific DBAP instance file.
    
    This workflow:
    1. Loads a text-based instance file from the 'data' directory.
    2. Parses the layout into a DBAPInstance object.
    3. Configures the CP-SAT solver (limits, workers, hints).
    4. Solves the problem to minimize weighted turnaround time.
    5. Reports metrics and displays a Gantt chart.
    """
    
    # --- 1. Path Configuration ---
    
    # Name of the instance file to solve
    instance_filename = "f200x15-01.txt"
    
    # Construct the absolute path to the data file
    data_path = project_root / "data" / instance_filename

    if not data_path.exists():
        print(f"Error: Data file not found at: {data_path}")
        print("Please ensure the 'data' folder exists in the project root.")
        return

    # --- 2. Instance Parsing ---
    
    print(f"Loading instance: {instance_filename}...")
    try:
        with open(data_path, "r") as f:
            instance = parse_instance(f)
    except Exception as e:
        print(f"Failed to parse instance: {e}")
        return

    print(f"Successfully loaded {instance.num_vessels} vessels and {instance.num_berths} berths.")

    # --- 3. Solver Configuration & Execution ---
    
    config = SolverConfig(
        time_limit_seconds=60.0,
        log_search_progress=True,         # Enable logging to see solver convergence
        num_workers=os.cpu_count() or 0,  # Utilize all available CPU cores
        use_hints=True                    # Warm-start with the greedy heuristic
    )

    print(f"\nStarting solver (Time Limit: {config.time_limit_seconds}s)...")
    solution = solve(instance, config)

    # --- 4. Reporting & Visualization ---
    
    if solution:
        print("\n" + "="*50)
        print("  SOLUTION FOUND")
        print("="*50)
        print(f"Makespan:                {solution.makespan}")
        print(f"Total Turnaround Time:   {solution.total_turnaround_time}")
        print(f"Mean Turnaround Time:    {solution.mean_turnaround_time:.2f}")
        print("-" * 50)
        
        # Display a preview table of the first 10 assignments
        print(f"{'Vessel':<8} | {'Berth':<6} | {'Arrival':<8} | {'Start':<8} | {'End':<8}")
        print("-" * 50)
        
        preview_limit = 10
        for v in range(min(preview_limit, instance.num_vessels)):
            print(f"V{v:<7} | B{solution.vessel_berths[v]:<5} | "
                  f"{instance.arrival_times[v]:<8} | "
                  f"{solution.vessel_start_times[v]:<8} | "
                  f"{solution.vessel_end_times[v]:<8}")
        
        if instance.num_vessels > preview_limit:
            print(f"... and {instance.num_vessels - preview_limit} more.")

        # Generate and show the Gantt chart
        print("\nGenerating Gantt Chart...")
        plot_schedule(
            solution, 
            instance=instance, 
            title=f"Schedule for {instance_filename} (Makespan: {solution.makespan})"
        )
        
        # Block execution until the plot window is closed
        plt.show()

    else:
        print("\nNo feasible solution found within the time limit.")

if __name__ == "__main__":
    main()