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

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Any

from ortools.sat.python import cp_model

from .instance import DBAPInstance
from .solution import Solution

# ============================================================
# Configuration
# ============================================================

@dataclass
class SolverConfig:
    """
    Configuration parameters for the CP-SAT solver.

    Attributes:
        time_limit_seconds: Maximum time allowed for the solver search phase.
        num_workers: Number of parallel workers (defaults to CPU count).
        log_search_progress: If True, prints solver logs to stdout.
        random_seed: Seed for the solver's random number generator (for reproducibility).
        use_hints: If True, uses the heuristic solution to warm-start the solver.
    """
    time_limit_seconds: float = 60.0
    num_workers: int = os.cpu_count() or 0
    log_search_progress: bool = True
    random_seed: int = 42
    use_hints: bool = True


# ============================================================
# Heuristic (Earliest Deadline First)
# ============================================================

def greedy_heuristic(instance: DBAPInstance) -> Optional[Solution]:
    """
    Constructs a feasible initial solution using a greedy Earliest-Deadline-First (EDF) rule.

    This function attempts to schedule vessels by prioritizing those with the tightest 
    deadlines. If deadlines are equal, it prioritizes earlier arrival times. It assigns
    the first available berth that fits the vessel's timing constraints.

    Returns:
        A valid Solution object if a feasible schedule is found, or None if the
        heuristic fails (which does not imply the instance is impossible, only 
        that this greedy approach failed).
    """
    if instance.num_vessels == 0:
        return Solution([], [], [], [], [])

    # Sort vessels by priority: Primary = Deadline, Secondary = Arrival Time
    sorted_vessels = sorted(
        range(instance.num_vessels), 
        key=lambda v: (instance.latest_departure_times[v], instance.arrival_times[v])
    )
    
    # Track the time when each berth becomes free. Initially, this is the berth opening time.
    berth_free_times = [
        instance.berth_opening_times[b].start 
        for b in range(instance.num_berths)
    ]
    
    v_berths = [0] * instance.num_vessels
    v_starts = [0] * instance.num_vessels
    v_ends = [0] * instance.num_vessels
    
    for v_id in sorted_vessels:
        arrival = instance.arrival_times[v_id]
        deadline = instance.latest_departure_times[v_id]
        
        best_finish = float('inf')
        best_start = -1
        best_b_id = -1
        
        # Iterate over all berths to find the best fit for this specific vessel
        for b_id in range(instance.num_berths):
            pt = instance.processing_times[v_id][b_id]
            if pt.is_invalid:
                continue
            
            duration = pt.value()
            berth_window = instance.berth_opening_times[b_id]
            
            # Determine earliest possible start time
            # Must be after: Vessel Arrival AND Previous Vessel Finish AND Berth Opening
            possible_start = max(arrival, berth_free_times[b_id], berth_window.start)
            possible_end = possible_start + duration
            
            # Check validity against Berth Closing Time and Vessel Deadline
            if (possible_end <= berth_window.finish) and (possible_end <= deadline):
                # Greedily minimize finish time
                if possible_end < best_finish:
                    best_finish = possible_end
                    best_start = possible_start
                    best_b_id = b_id
        
        # If no valid berth is found for a vessel, the heuristic fails
        if best_b_id == -1:
            return None

        # Assign values for the successful vessel
        v_berths[v_id] = best_b_id
        v_starts[v_id] = best_start
        v_ends[v_id] = int(best_finish)
        
        # Update availability for the chosen berth
        berth_free_times[best_b_id] = int(best_finish)
        
    try:
        return Solution(
            vessel_berths=v_berths, 
            vessel_start_times=v_starts, 
            vessel_end_times=v_ends, 
            weights=instance.vessel_weights,
            arrival_times=instance.arrival_times
        )
    except ValueError:
        return None


# ============================================================
# Main Solver Logic (CP-SAT)
# ============================================================

def solve(instance: DBAPInstance, config: SolverConfig = SolverConfig()) -> Optional[Solution]:
    """
    Solves the Discrete Berth Allocation Problem (DBAP) using Google OR-Tools CP-SAT solver.
    
    Objective:
        Minimize Total Weighted Turnaround Time.
        Turnaround = Completion Time - Arrival Time.
        
    Strategy:
        The solver creates a 'Master' start variable for each vessel and 'Local' interval 
        variables for every possible berth assignment. Boolean presence literals link 
        local intervals to the master variables.
    
    Args:
        instance: The problem instance data.
        config: Solver configuration options.

    Returns:
        An optimal (or best found) Solution object, or None if no feasible solution exists.
    """
    if instance.num_vessels == 0:
        return Solution([], [], [], [], [])

    # --- Pre-calculation and Horizon Analysis ---

    # Run heuristic to get hints for warm-starting the search
    greedy_sol = greedy_heuristic(instance)
    if greedy_sol is not None and config.use_hints:
        print(f"Found heuristic solution with total weighted turnaround time: {greedy_sol.total_weighted_turnaround_time}")
    
    # Calculate a safe time horizon
    # The horizon is the maximum possible time any event could occur.
    max_arrival = max(instance.arrival_times)
    sum_max_durations = 0
    for v in range(instance.num_vessels):
        valid_pts = [pt.value() for pt in instance.processing_times[v] if pt.is_valid]
        if valid_pts:
            sum_max_durations += max(valid_pts)
        else:
            # If a vessel has no valid processing times, the instance is unsolvable
            return None 

    horizon = max_arrival + sum_max_durations
    global_deadline_max = max(instance.latest_departure_times)
    
    # Ensure horizon covers the latest allowed departure
    horizon = max(horizon, global_deadline_max)

    # --- Model Construction ---

    model: Any = cp_model.CpModel()
    
    # Storage for solution extraction
    vessel_vars = {}
    
    # Storage for "No Overlap" constraints (grouped by berth)
    intervals_per_berth = [[] for _ in range(instance.num_berths)]
    
    # Collection of all boolean presence variables for search strategy hints
    all_presence_literals = []
    
    # Objective function terms
    # We minimize Sum(Weight * (End - Arrival))
    # This is equivalent to: Sum(Weight * End) - Sum(Weight * Arrival)
    # We accumulate the variable parts in `obj_terms` and the constant parts in `constant_offset`.
    obj_terms = []
    constant_offset = 0

    for v in range(instance.num_vessels):
        arrival = instance.arrival_times[v]
        deadline = instance.latest_departure_times[v]
        weight = instance.vessel_weights[v]
        
        # Add the constant offset for this vessel: - (Weight * Arrival)
        constant_offset -= (weight * arrival)

        # Master Start Variable: The actual start time of the vessel regardless of berth
        v_start = model.NewIntVar(arrival, deadline, f'v{v}_start')
        
        # Master Berth Variable: Which berth serves the vessel
        v_berth = model.NewIntVar(0, instance.num_berths - 1, f'v{v}_berth')
        
        vessel_vars[v] = {'start': v_start, 'berth': v_berth}
        
        # Add (Weight * StartTime) to objective. 
        # Since End = Start + Duration, we add the Duration part later inside the berth loop.
        obj_terms.append(weight * v_start)

        # Apply heuristic hints if available
        if config.use_hints and greedy_sol:
            model.AddHint(v_start, greedy_sol.vessel_start_times[v])
            model.AddHint(v_berth, greedy_sol.vessel_berths[v])

        possible_berths_literals = []

        # Create local optional intervals for each berth
        for b in range(instance.num_berths):
            pt = instance.processing_times[v][b]
            if pt.is_invalid: continue

            duration = pt.value()
            berth_interval = instance.berth_opening_times[b]
            
            # Pruning 1: If earliest possible finish exceeds berth availability
            earliest_finish = max(arrival, berth_interval.start) + duration
            if earliest_finish > berth_interval.finish:
                continue
            
            # Pruning 2: If earliest possible finish exceeds vessel deadline
            if earliest_finish > deadline:
                continue

            # Presence Boolean: True if vessel 'v' is assigned to berth 'b'
            is_present = model.NewBoolVar(f'pres_v{v}_b{b}')
            possible_berths_literals.append(is_present)
            all_presence_literals.append(is_present)
            
            # Add (Weight * Duration) to objective for this specific berth assignment
            # This completes the term: Weight * (Start + Duration) = Weight * End
            obj_terms.append(is_present * (weight * duration))

            # Define the Local Interval variable
            # Enforce: Local Start >= Arrival AND Local Start >= Berth Opening
            safe_min_start = max(arrival, berth_interval.start)
            
            local_start = model.NewIntVar(safe_min_start, deadline, f's_v{v}_b{b}')
            local_end = model.NewIntVar(safe_min_start + duration, deadline, f'e_v{v}_b{b}')
            
            # Create the optional interval used by the NoOverlap constraint
            interval = model.NewOptionalIntervalVar(
                local_start, duration, local_end, is_present, f'int_v{v}_b{b}'
            )
            intervals_per_berth[b].append(interval)
            
            # Link Local variables to Master variables
            # If is_present is True, Master Start must match Local Start
            model.Add(v_start == local_start).OnlyEnforceIf(is_present)
            
            # If is_present is True, Master Berth must be 'b'
            model.Add(v_berth == b).OnlyEnforceIf(is_present)
            
            # Enforce Berth Availability: Vessel must finish before berth closes
            model.Add(local_end <= berth_interval.finish).OnlyEnforceIf(is_present)

            # Apply heuristic hint for berth choice
            if config.use_hints and greedy_sol:
                was_chosen = (greedy_sol.vessel_berths[v] == b)
                model.AddHint(is_present, 1 if was_chosen else 0)

        # Constraint: Each vessel must be assigned to exactly one valid berth
        if not possible_berths_literals:
            return None # Feasibility impossible
        model.Add(sum(possible_berths_literals) == 1)

    # --- Disjunctive Constraints ---
    
    # Ensure no two vessels overlap in time on the same berth
    for b in range(instance.num_berths):
        if intervals_per_berth[b]:
            model.AddNoOverlap(intervals_per_berth[b])

    # --- Objective Definition ---
    
    # Minimize the total weighted turnaround time
    model.Minimize(sum(obj_terms) + constant_offset)
    
    # --- Search Strategy ---
    
    # Tell the solver to prioritize deciding which berth a vessel goes to (presence literals)
    # before deciding exact timestamps. This usually prunes the tree faster.
    model.AddDecisionStrategy(
        all_presence_literals, 
        cp_model.CHOOSE_FIRST, 
        cp_model.SELECT_MAX_VALUE
    )

    # --- Solver Execution ---
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.time_limit_seconds
    solver.parameters.num_search_workers = config.num_workers if config.num_workers > 0 else (os.cpu_count() or 1)
    solver.parameters.log_search_progress = config.log_search_progress
    solver.parameters.random_seed = config.random_seed
    
    status = solver.Solve(model)

    # --- Result Extraction ---
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        res_berths = []
        res_starts = []
        res_ends = []

        for v in range(instance.num_vessels):
            # Retrieve basic values from solver
            b_val = solver.Value(vessel_vars[v]['berth'])
            s_val = solver.Value(vessel_vars[v]['start'])
            
            # Re-calculate End Time based on the chosen berth's processing duration
            duration = instance.processing_times[v][b_val].value()
            e_val = s_val + duration
            
            res_berths.append(b_val)
            res_starts.append(s_val)
            res_ends.append(e_val)
        
        return Solution(
            vessel_berths=res_berths,
            vessel_start_times=res_starts,
            vessel_end_times=res_ends,
            weights=instance.vessel_weights,
            arrival_times=instance.arrival_times
        )
    
    return None