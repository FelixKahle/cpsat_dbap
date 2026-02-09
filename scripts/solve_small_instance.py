# scripts/solve_small_instance.py

import matplotlib.pyplot as plt
from cpsat_dbap.instance import (
    DBAPInstance, 
    ProcessingTime, 
    HalfOpenInterval, 
    INVALID_PROCESSING_TIME
)
from cpsat_dbap.solver import solve, SolverConfig
from cpsat_dbap.plotting import plot_schedule  # <-- Import the plotting function

def run_small_test():
    print("Constructing small manual instance...")

    # --- Data Setup ---
    pt = ProcessingTime
    inv = INVALID_PROCESSING_TIME

    # 3 Vessels, 2 Berths
    # V0: 10 on B0, 12 on B1
    # V1: 8 on B0, Forbidden on B1
    # V2: 6 on B0, 5 on B1
    matrix = [
        [pt(10), pt(12)], 
        [pt(8),  inv],    
        [pt(6),  pt(5)]   
    ]

    berth_windows = [
        HalfOpenInterval(0, 50),
        HalfOpenInterval(5, 60)
    ]

    instance = DBAPInstance(
        num_vessels=3,
        num_berths=2,
        vessel_weights=[1, 1, 1],
        arrival_times=[0, 5, 10],
        latest_departure_times=[100, 100, 100],
        processing_times=matrix,
        berth_opening_times=berth_windows
    )

    # --- Solve ---
    config = SolverConfig(time_limit_seconds=10.0, log_search_progress=False)
    solution = solve(instance, config)

    # --- Output ---
    if solution:
        print("\n" + "="*40)
        print(f"  OPTIMAL SCHEDULE (Makespan: {solution.makespan})")
        print("="*40)
        print(f"{'Vessel':<6} | {'Berth':<6} | {'Start':<6} | {'End':<6}")
        print("-" * 40)
        
        for v in range(instance.num_vessels):
            print(f"V{v:<5} | B{solution.vessel_berths[v]:<5} | "
                  f"{solution.vessel_start_times[v]:<6} | {solution.vessel_end_times[v]:<6}")

        # --- Plotting ---
        print("\nGenerating Gantt Chart...")
        plot_schedule(solution, instance=instance, title="Small Instance Test")
        plt.show()  # Display the plot window
        
    else:
        print("No solution found.")

if __name__ == "__main__":
    run_small_test()