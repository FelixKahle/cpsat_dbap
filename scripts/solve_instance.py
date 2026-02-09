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

# Explicitly add 'src' to path to allow running without installing (just in case)
project_root = Path(__file__).resolve().parent.parent

from cpsat_dbap.instance import parse_instance
from cpsat_dbap.solver import solve, SolverConfig
from cpsat_dbap.plotting import plot_schedule

def main():
    # ------------------------------------------------------------------
    # 1. Setup Paths
    # ------------------------------------------------------------------
    # Adjust this filename if you want to run a different instance
    instance_filename = "f200x15-03.txt"
    
    data_path = project_root / "data" / instance_filename

    if not data_path.exists():
        print(f"Error: Data file not found at: {data_path}")
        print("Please ensure the 'data' folder exists in the project root.")
        return

    # ------------------------------------------------------------------
    # 2. Parse Instance
    # ------------------------------------------------------------------
    print(f"Loading instance: {instance_filename}...")
    try:
        with open(data_path, "r") as f:
            instance = parse_instance(f)
    except Exception as e:
        print(f"Failed to parse instance: {e}")
        return

    print(f"Successfully loaded {instance.num_vessels} vessels and {instance.num_berths} berths.")

    # ------------------------------------------------------------------
    # 3. Configure & Solve
    # ------------------------------------------------------------------
    # 20 seconds time limit as requested
    config = SolverConfig(
        time_limit_seconds=40.0,
        log_search_progress=False,
        num_workers=os.cpu_count() or 0,  # Use all available cores
        use_hints=True
    )

    print(f"\nStarting solver (Time Limit: {config.time_limit_seconds}s)...")
    solution = solve(instance, config)

    # ------------------------------------------------------------------
    # 4. Report & Plot
    # ------------------------------------------------------------------
    if solution:
        print("\n" + "="*50)
        print("  SOLUTION FOUND")
        print("="*50)
        print(f"Makespan:                {solution.makespan}")
        print(f"Total Turnaround Time:   {solution.total_turnaround_time}")
        print(f"Mean Turnaround Time:    {solution.mean_turnaround_time:.2f}")
        print("-" * 50)
        
        # Print first 10 vessels as a preview
        print(f"{'Vessel':<8} | {'Berth':<6} | {'Arrival':<8} | {'Start':<8} | {'End':<8}")
        print("-" * 50)
        for v in range(min(10, instance.num_vessels)):
            print(f"V{v:<7} | B{solution.vessel_berths[v]:<5} | "
                  f"{instance.arrival_times[v]:<8} | "
                  f"{solution.vessel_start_times[v]:<8} | "
                  f"{solution.vessel_end_times[v]:<8}")
        
        if instance.num_vessels > 10:
            print(f"... and {instance.num_vessels - 10} more.")

        # Plot
        print("\nGenerating Gantt Chart...")
        plot_schedule(
            solution, 
            instance=instance, 
            title=f"Schedule for {instance_filename} (Makespan: {solution.makespan})"
        )
        
        # This keeps the window open until you close it
        plt.show()

    else:
        print("\nNo solution found within the time limit.")

if __name__ == "__main__":
    main()