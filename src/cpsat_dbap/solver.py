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
    Constructs a feasible solution using an Earliest-Deadline-First (EDF) rule.
    For each vessel, it selects the berth that minimizes the completion time.
    
    Returns None if the greedy approach fails to respect strict windows.
    """
    if instance.num_vessels == 0:
        return Solution([], [], [], [], [])

    # Sort vessels by Deadline (primary) and Arrival (secondary)
    # This acts like a sweep-line prioritizing urgency.
    sorted_vessels = sorted(
        range(instance.num_vessels), 
        key=lambda v: (instance.latest_departure_times[v], instance.arrival_times[v])
    )
    
    # Track when each berth becomes free. 
    # Initially, this is the berth's opening time.
    berth_free_times = [
        instance.berth_opening_times[b].start 
        for b in range(instance.num_berths)
    ]
    
    # Storage for the solution
    v_berths = [0] * instance.num_vessels
    v_starts = [0] * instance.num_vessels
    v_ends = [0] * instance.num_vessels
    
    for v_id in sorted_vessels:
        arrival = instance.arrival_times[v_id]
        deadline = instance.latest_departure_times[v_id]
        
        best_finish = float('inf')
        best_start = -1
        best_b_id = -1
        
        # Scan ALL berths to find the best fit for this urgent vessel
        for b_id in range(instance.num_berths):
            # 1. Check processing validity
            pt = instance.processing_times[v_id][b_id]
            if pt.is_invalid:
                continue
            
            duration = pt.value()
            berth_window = instance.berth_opening_times[b_id]
            
            # 2. Calculate Timing
            # Start = Max(Vessel Arrival, Berth Free Time, Berth Open Time)
            possible_start = max(arrival, berth_free_times[b_id], berth_window.start)
            possible_end = possible_start + duration
            
            # 3. Check Constraints
            # - Must finish before berth closes
            # - Must finish before vessel deadline
            if (possible_end <= berth_window.finish) and (possible_end <= deadline):
                # We greedily minimize finish time to keep the horizon compact
                if possible_end < best_finish:
                    best_finish = possible_end
                    best_start = possible_start
                    best_b_id = b_id
        
        # If no valid berth found for this vessel, the heuristic fails
        if best_b_id == -1:
            return None

        # Assign
        v_berths[v_id] = best_b_id
        v_starts[v_id] = best_start
        v_ends[v_id] = int(best_finish)
        
        # Update berth availability
        berth_free_times[best_b_id] = int(best_finish)
        
    try:
        return Solution(
            vessel_berths=v_berths, 
            vessel_start_times=v_starts, 
            vessel_end_times=v_ends, 
            weights=instance.vessel_weights,
            arrival_times=instance.arrival_times # <--- Pass this
        )
    except ValueError:
        return None

# ============================================================
# Main Solver Logic
# ============================================================

def solve(instance: DBAPInstance, config: SolverConfig = SolverConfig()) -> Optional[Solution]:
    """
    Solves the Discrete Berth Allocation Problem using CP-SAT.
    """

    # 0. Edge Cases
    if instance.num_vessels == 0:
        return Solution([], [], [], [], [])

    # 1. Run Heuristic for Horizon & Hints
    greedy_sol = greedy_heuristic(instance)
    
    # Calculate a safe horizon
    max_arrival = max(instance.arrival_times)
    sum_max_durations = 0
    for v in range(instance.num_vessels):
        valid_pts = [pt.value() for pt in instance.processing_times[v] if pt.is_valid]
        if valid_pts:
            sum_max_durations += max(valid_pts)
        else:
            return None # Impossible instance

    horizon = max_arrival + sum_max_durations
    global_deadline_max = max(instance.latest_departure_times)
    if global_deadline_max < horizon:
        horizon = global_deadline_max

    # --- Build Model ---
    model: Any = cp_model.CpModel()
    
    # vessel_vars[v] = { 'start': var, 'berth': var }  <-- Removed 'end'
    vessel_vars = {}
    
    # intervals_per_berth[b] = [interval_var, ...]
    intervals_per_berth = [[] for _ in range(instance.num_berths)]
    
    # Store literals for search strategy
    all_presence_literals = []
    
    # Accumulate terms for the objective function here
    objective_terms = []

    for v in range(instance.num_vessels):
        arrival = instance.arrival_times[v]
        latest_departure = instance.latest_departure_times[v]
        upper_bound = min(horizon, latest_departure)
        
        # Master variables (v_end is REMOVED)
        v_start = model.NewIntVar(arrival, upper_bound, f'v{v}_start')
        v_berth = model.NewIntVar(0, instance.num_berths - 1, f'v{v}_berth')
        
        vessel_vars[v] = {'start': v_start, 'berth': v_berth}
        
        # Add Start Time to objective (End = Start + Duration)
        objective_terms.append(v_start)
        
        # Hints
        if config.use_hints and greedy_sol:
            model.AddHint(v_start, greedy_sol.vessel_start_times[v])
            model.AddHint(v_berth, greedy_sol.vessel_berths[v])

        possible_berths = []
        
        for b in range(instance.num_berths):
            pt = instance.processing_times[v][b]
            if pt.is_invalid: continue

            duration = pt.value()
            berth_interval = instance.berth_opening_times[b]
            
            if berth_interval.finish < arrival + duration:
                continue
            
            # Presence Boolean
            is_present = model.NewBoolVar(f'pres_v{v}_b{b}')
            possible_berths.append(is_present)
            all_presence_literals.append(is_present)
            
            # Add (is_present * duration) to objective
            # This accounts for the variable duration depending on berth choice
            objective_terms.append(is_present * duration)
            
            # Hint presence
            if config.use_hints and greedy_sol:
                was_chosen = (greedy_sol.vessel_berths[v] == b)
                model.AddHint(is_present, 1 if was_chosen else 0)

            # Local Interval
            safe_start_min = max(arrival, berth_interval.start)
            
            local_start = model.NewIntVar(safe_start_min, upper_bound, f's_v{v}_b{b}')
            # We still need local_end for the interval definition, but we don't link it to a master var
            local_end = model.NewIntVar(safe_start_min + duration, upper_bound, f'e_v{v}_b{b}')
            
            interval = model.NewOptionalIntervalVar(
                local_start, duration, local_end, is_present, f'int_v{v}_b{b}'
            )
            intervals_per_berth[b].append(interval)
            
            # Link to Master Start & Berth
            model.Add(v_start == local_start).OnlyEnforceIf(is_present)
            model.Add(v_berth == b).OnlyEnforceIf(is_present)

        if not possible_berths:
            return None
            
        model.Add(sum(possible_berths) == 1)

    # Disjunctive Constraints
    for b in range(instance.num_berths):
        if intervals_per_berth[b]:
            model.AddNoOverlap(intervals_per_berth[b])

    # Objective: Minimize Sum of (Start + Duration)
    model.Minimize(sum(objective_terms))
    
    # --- Search Strategy ---
    model.AddDecisionStrategy(
        all_presence_literals, 
        cp_model.CHOOSE_FIRST, 
        cp_model.SELECT_MAX_VALUE
    )

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.time_limit_seconds
    solver.parameters.num_search_workers = config.num_workers if config.num_workers > 0 else (os.cpu_count() or 1)
    solver.parameters.log_search_progress = config.log_search_progress
    solver.parameters.random_seed = config.random_seed
    
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        res_berths = []
        res_starts = []
        res_ends = []

        for v in range(instance.num_vessels):
            # Retrieve solver values
            b_val = solver.Value(vessel_vars[v]['berth'])
            s_val = solver.Value(vessel_vars[v]['start'])
            
            # Re-calculate End Time: Start + Duration of chosen berth
            # We must look up the processing time from the instance data again
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