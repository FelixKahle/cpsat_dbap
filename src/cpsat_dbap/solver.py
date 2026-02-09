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
import heapq
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple

from ortools.sat.python import cp_model

from .instance import DBAPInstance, ProcessingTime  # <-- Add the dot
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
    num_workers: int = 8  # 0 means all available cores
    log_search_progress: bool = True
    random_seed: int = 42
    use_hints: bool = True


# ============================================================
# Horizon Calculation
# ============================================================

def determine_horizon_bound(instance: DBAPInstance) -> int:
    """
    Calculates a mathematically proven upper bound for the horizon.
    Logic: The optimal Total Flow Cost cannot exceed the Total Flow Cost of 
    a simple greedy schedule. 
    """
    # 1. Sort vessels by Arrival Time (Standard Greedy approach)
    # We store (arrival_time, vessel_index)
    sorted_vessels = sorted(
        range(instance.num_vessels), 
        key=lambda v: instance.arrival_times[v]
    )
    
    # 2. Priority Queue for Berths: stores (available_time, berth_id)
    # Initialize all berths at their opening times
    berth_heap: List[Tuple[int, int]] = []
    for b_id in range(instance.num_berths):
        interval = instance.berth_opening_times[b_id]
        heapq.heappush(berth_heap, (interval.start, b_id))
    
    greedy_max_finish = 0
    
    # 3. Simulate Schedule
    for v_id in sorted_vessels:
        arrival = instance.arrival_times[v_id]
        
        # Try to find the earliest available berth that can process this vessel
        temp_popped = []
        chosen_finish = float('inf')
        chosen_b_id = -1
        
        # Pop berths until we find a valid one or run out
        while berth_heap:
            avail_time, b_id = heapq.heappop(berth_heap)
            
            # Check if this berth can handle this vessel
            pt = instance.processing_times[v_id][b_id]
            
            if pt.is_valid:
                # Found valid berth!
                duration = pt.value()
                start_time = max(arrival, avail_time)
                chosen_finish = start_time + duration
                chosen_b_id = b_id
                
                # Push back the chosen berth with new availability
                heapq.heappush(berth_heap, (chosen_finish, chosen_b_id))
                break
            else:
                # Store incompatible berth to put back later
                temp_popped.append((avail_time, b_id))
        
        # Put back the incompatible berths we popped
        for item in temp_popped:
            heapq.heappush(berth_heap, item)
            
        if chosen_b_id == -1:
            # Fallback: If heuristic fails (e.g. strict forbidden windows), 
            # use a conservative bound: Max Arrival + Sum of Max Durations
            total_max_duration = 0
            for v in range(instance.num_vessels):
                valid_durations = [
                    p.value() for p in instance.processing_times[v] if p.is_valid
                ]
                if valid_durations:
                    total_max_duration += max(valid_durations)
            
            max_arr = max(instance.arrival_times)
            return max_arr + total_max_duration

        greedy_max_finish = max(greedy_max_finish, chosen_finish)

    # 4. Calculate The Rigorous Bound
    # Generally, specific horizon = Greedy Makespan + Sum of Processing Times is extremely loose.
    # A tighter bound for CP is often: Greedy Makespan * 1.5 or just a large safe integer.
    # However, to be safe and allow the solver room to optimize "waiting", 
    # we usually set it to (Max Arrival + Sum of all processing times).
    
    # Let's use the logic from the original script:
    # proven_horizon = greedy_total_flow + max_arrival (This was in the original script)
    # But strictly speaking for CP domains, we just need a valid upper bound for the *latest completion time*.
    
    # We will use: Sum of all maximum processing times + Max Arrival. 
    # This guarantees that even if every ship waits for every other ship, it fits.
    
    total_max_proc = 0
    for v in range(instance.num_vessels):
        valid_ps = [p.value() for p in instance.processing_times[v] if p.is_valid]
        if valid_ps:
            total_max_proc += max(valid_ps)
            
    max_arrival = max(instance.arrival_times) if instance.arrival_times else 0
    
    # Respect the hard deadlines from input if they exist and are binding
    # (The instance has 'latest_departure_times', but those might be infinite/large)
    
    calculated_horizon = max_arrival + total_max_proc
    
    # If the instance has tighter deadlines, we can clamp, but usually
    # we want the horizon large enough to prove infeasibility if deadlines can't be met.
    return calculated_horizon


# ============================================================
# Main Solver Logic
# ============================================================

def solve(instance: DBAPInstance, config: SolverConfig = SolverConfig()) -> Optional[Solution]:
    """
    Solves the Discrete Berth Allocation Problem using CP-SAT.
    """
    model: Any = cp_model.CpModel()
    
    # 1. Horizon
    horizon = determine_horizon_bound(instance)
    
    # --- Variables ---
    
    # vessel_vars[v] = { 'start': var, 'end': var, 'berth': var }
    vessel_vars = {}
    
    # intervals_per_berth[b] = [interval_var, ...]
    intervals_per_berth = [[] for _ in range(instance.num_berths)]
    
    # We need to map which berth is chosen for the solution reconstruction
    # (vessel, berth) -> boolean_presence_var
    allocation_bools = {} 

    # 2. Build Model
    for v in range(instance.num_vessels):
        arrival = instance.arrival_times[v]
        latest_departure = instance.latest_departure_times[v]
        
        # Master variables for this vessel
        # Constrain domain by arrival and hard deadline (if < horizon)
        upper_bound = min(horizon, latest_departure)
        
        v_start = model.NewIntVar(arrival, upper_bound, f'v{v}_start')
        v_end = model.NewIntVar(arrival, upper_bound, f'v{v}_end')
        v_berth = model.NewIntVar(0, instance.num_berths - 1, f'v{v}_berth')
        
        vessel_vars[v] = {'start': v_start, 'end': v_end, 'berth': v_berth}

        # Create optional intervals for each berth
        possible_berths = []
        
        for b in range(instance.num_berths):
            pt = instance.processing_times[v][b]
            
            # Skip invalid (forbidden) assignments
            if pt.is_invalid:
                continue

            duration = pt.value()
            berth_interval = instance.berth_opening_times[b]
            
            # Sanity check: if berth closes before arrival, skip
            if berth_interval.finish < arrival:
                continue
            
            # Create presence boolean
            is_present = model.NewBoolVar(f'pres_v{v}_b{b}')
            allocation_bools[(v, b)] = is_present
            possible_berths.append(is_present)
            
            # Local start/end for this specific berth assignment
            # Must be within berth opening times
            local_start = model.NewIntVar(
                max(arrival, berth_interval.start), 
                min(upper_bound, berth_interval.finish), 
                f'start_v{v}_b{b}'
            )
            
            local_end = model.NewIntVar(
                max(arrival, berth_interval.start) + duration,
                min(upper_bound, berth_interval.finish),
                f'end_v{v}_b{b}'
            )
            
            # Optional Interval for NoOverlap
            interval = model.NewOptionalIntervalVar(
                local_start, duration, local_end, is_present, f'interval_v{v}_b{b}'
            )
            intervals_per_berth[b].append(interval)
            
            # Link Local -> Master variables
            # If present on berth b:
            # 1. master_start == local_start
            # 2. master_end   == local_end
            # 3. master_berth == b
            model.Add(v_start == local_start).OnlyEnforceIf(is_present)
            model.Add(v_end == local_end).OnlyEnforceIf(is_present)
            model.Add(v_berth == b).OnlyEnforceIf(is_present)

        # Constraint: Vessel must be assigned to exactly one valid berth
        if not possible_berths:
            # Infeasible for this vessel
            return None
            
        model.Add(sum(possible_berths) == 1)

    # 3. Disjunctive Constraints (No Overlap per berth)
    for b in range(instance.num_berths):
        if intervals_per_berth[b]:
            model.AddNoOverlap(intervals_per_berth[b])

    # 4. Objective: Minimize Total Flow Time (equivalent to minimizing Sum of End Times)
    # Objective = Sum(End_v - Arrival_v)
    # Since Arrival_v is constant, we just Minimize(Sum(End_v))
    total_completion_time = sum(vessel_vars[v]['end'] for v in range(instance.num_vessels))
    model.Minimize(total_completion_time)
    
    # 5. Hints (Optional)
    if config.use_hints:
        # Simple heuristic: sort by arrival
        sorted_vessels = sorted(range(instance.num_vessels), key=lambda x: instance.arrival_times[x])
        for v in sorted_vessels:
            model.AddHint(vessel_vars[v]['start'], instance.arrival_times[v])

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.time_limit_seconds
    solver.parameters.num_search_workers = config.num_workers if config.num_workers > 0 else (os.cpu_count() or 1)
    solver.parameters.log_search_progress = config.log_search_progress
    solver.parameters.random_seed = config.random_seed
    
    status = solver.Solve(model)

    # --- Extract Solution ---
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        
        res_berths = [0] * instance.num_vessels
        res_starts = [0] * instance.num_vessels
        res_ends = [0] * instance.num_vessels
        
        for v in range(instance.num_vessels):
            res_starts[v] = solver.Value(vessel_vars[v]['start'])
            res_ends[v] = solver.Value(vessel_vars[v]['end'])
            res_berths[v] = solver.Value(vessel_vars[v]['berth'])
            
            # Double check against allocation bools (sanity check)
            # for b in range(instance.num_berths):
            #     if (v, b) in allocation_bools and solver.Value(allocation_bools[(v, b)]):
            #         assert res_berths[v] == b
        
        return Solution(
            vessel_berths=res_berths,
            vessel_start_times=res_starts,
            vessel_end_times=res_ends,
            weights=instance.vessel_weights
        )
    
    return None