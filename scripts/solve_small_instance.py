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

import matplotlib.pyplot as plt

from cpsat_dbap.instance import (
    DBAPInstance, 
    ProcessingTime, 
    HalfOpenInterval, 
    INVALID_PROCESSING_TIME
)
from cpsat_dbap.solver import solve, SolverConfig
from cpsat_dbap.plotting import plot_schedule

def run_small_test() -> None:
    """
    Constructs a manual, small-scale instance of the Berth Allocation Problem,
    solves it using the CP-SAT solver, and visualizes the result.

    This test case involves:
    - 3 Vessels (V0, V1, V2)
    - 2 Berths (B0, B1)
    - Specific processing times and constraints (e.g., V1 forbidden on B1).
    - Varying berth availability windows.
    """
    print("Constructing small manual instance...")

    # Abbreviations for readability
    pt = ProcessingTime
    inv = INVALID_PROCESSING_TIME

    # Define the Processing Time Matrix [Vessel][Berth]
    # V0: Can use B0 (10h) or B1 (12h)
    # V1: Can use B0 (8h), but CANNOT use B1
    # V2: Can use B0 (6h) or B1 (5h)
    matrix = [
        [pt(10), pt(12)], 
        [pt(8),  inv],    
        [pt(6),  pt(5)]   
    ]

    # Define Berth Availability Windows [Start, End)
    # B0: Available from 0 to 50
    # B1: Available from 5 to 60
    berth_windows = [
        HalfOpenInterval(0, 50),
        HalfOpenInterval(5, 60)
    ]

    # Initialize the Instance Container
    instance = DBAPInstance(
        num_vessels=3,
        num_berths=2,
        vessel_weights=[1, 1, 1],       # All vessels have equal priority
        arrival_times=[0, 5, 10],       # V0 arrives at 0, V1 at 5, V2 at 10
        latest_departure_times=[100, 100, 100], # Relaxed deadlines
        processing_times=matrix,
        berth_opening_times=berth_windows
    )

    # Configure and Execute Solver
    config = SolverConfig(
        time_limit_seconds=10.0, 
        log_search_progress=False
    )
    solution = solve(instance, config)

    # Process and Display Results
    if solution:
        print("\n" + "="*50)
        print(f"  OPTIMAL SCHEDULE FOUND (Makespan: {solution.makespan})")
        print("="*50)
        
        # Table Header
        print(f"{'Vessel':<8} | {'Berth':<8} | {'Start':<8} | {'End':<8}")
        print("-" * 50)
        
        # Table Rows
        for v in range(instance.num_vessels):
            b_id = solution.vessel_berths[v]
            start = solution.vessel_start_times[v]
            end = solution.vessel_end_times[v]
            
            print(f"V{v:<7} | B{b_id:<7} | {start:<8} | {end:<8}")

        # Visualization
        print("\nGenerating Gantt Chart...")
        plot_schedule(solution, instance=instance, title="Small Instance Test")
        plt.show()
        
    else:
        print("No feasible solution found for the given constraints.")

if __name__ == "__main__":
    run_small_test()