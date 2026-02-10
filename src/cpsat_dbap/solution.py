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

from dataclasses import dataclass, field, InitVar
from typing import List

@dataclass(frozen=True)
class Solution:
    """
    Represents a specific allocation and scheduling solution for the DBAP.
    
    This class is immutable. Derived metrics (turnaround times, weighted objectives)
    are calculated immediately upon initialization to ensure consistency and 
    prevent recalculation overhead.

    Attributes:
        vessel_berths: A list of berth indices assigned to each vessel.
        vessel_start_times: A list of start times for each vessel.
        vessel_end_times: A list of completion times for each vessel.
        vessel_turnaround_times: Calculated flow time (End - Arrival) for each vessel.
        vessel_weighted_turnaround_times: Weighted flow time for each vessel.
        total_turnaround_time: The sum of all vessel turnaround times.
        total_weighted_turnaround_time: The objective value (sum of weighted turnaround times).
    """
    vessel_berths: List[int]
    vessel_start_times: List[int]
    vessel_end_times: List[int]
    
    # Initialization arguments (required for calculation but not stored in the final instance)
    weights: InitVar[List[int]]
    arrival_times: InitVar[List[int]]

    # Derived fields (populated during __post_init__)
    vessel_turnaround_times: List[int] = field(init=False)
    vessel_weighted_turnaround_times: List[int] = field(init=False)
    total_turnaround_time: int = field(init=False)
    total_weighted_turnaround_time: int = field(init=False)

    def __post_init__(self, weights: List[int], arrival_times: List[int]) -> None:
        """
        Validates input consistency and calculates derived performance metrics.
        """
        n = len(self.vessel_berths)

        # Validate that all input vectors have matching lengths
        if len(self.vessel_start_times) != n:
            raise ValueError("Length mismatch: vessel_start_times does not match vessel count")
        if len(self.vessel_end_times) != n:
            raise ValueError("Length mismatch: vessel_end_times does not match vessel count")
        if len(weights) != n:
            raise ValueError("Length mismatch: weights vector does not match vessel count")
        if len(arrival_times) != n:
            raise ValueError("Length mismatch: arrival_times vector does not match vessel count")

        turnaround = []
        weighted = []
        total_ta = 0
        total_wta = 0

        # Calculate per-vessel metrics
        for i in range(n):
            start = self.vessel_start_times[i]
            end = self.vessel_end_times[i]
            arrival = arrival_times[i]
            w = weights[i]

            # Ensure basic temporal causality
            if end < start:
                raise ValueError(f"Temporal paradox: end time {end} < start time {start} for vessel {i}")

            # Turnaround Time = Completion Time - Arrival Time
            # This represents the total time the vessel spent in the system (waiting + processing)
            ta = end - arrival
            
            wta = ta * w

            turnaround.append(ta)
            weighted.append(wta)
            total_ta += ta
            total_wta += wta

        # Bypass frozen dataclass restrictions to set derived attributes
        object.__setattr__(self, 'vessel_turnaround_times', turnaround)
        object.__setattr__(self, 'vessel_weighted_turnaround_times', weighted)
        object.__setattr__(self, 'total_turnaround_time', total_ta)
        object.__setattr__(self, 'total_weighted_turnaround_time', total_wta)

    @property
    def num_vessels(self) -> int:
        """Returns the number of vessels in this solution."""
        return len(self.vessel_berths)

    @property
    def makespan(self) -> int:
        """
        Returns the makespan of the schedule.
        
        The makespan is defined as the completion time of the last vessel to finish.
        Returns 0 if the solution is empty.
        """
        if not self.vessel_end_times:
            return 0
        return max(self.vessel_end_times)
    
    @property
    def mean_turnaround_time(self) -> float:
        """Returns the average turnaround time across all vessels."""
        if self.num_vessels == 0:
            return 0.0
        return self.total_turnaround_time / self.num_vessels

    def validate(self) -> bool:
        """
        Performs a deep check of internal consistency.

        While __post_init__ handles basic length checks and metric calculation,
        this method can be used for explicit verification of the solution state.
        
        Returns:
            True if valid.
            
        Raises:
            ValueError: If inconsistencies are found in vector lengths or timing logic.
        """
        n = self.num_vessels

        # Verify all vector lengths align
        arrays = [
            self.vessel_start_times,
            self.vessel_end_times,
            self.vessel_turnaround_times,
            self.vessel_weighted_turnaround_times
        ]
        
        for vec in arrays:
            if len(vec) != n:
                raise ValueError("Vector length mismatch detected in solution state")

        # Verify temporal logic for every vessel
        for i in range(n):
            if self.vessel_end_times[i] < self.vessel_start_times[i]:
                raise ValueError(f"Invalid timing at vessel {i}: End time precedes Start time")

        return True

    def __str__(self) -> str:
        return (
            f"Solution({self.num_vessels} vessels, "
            f"total_turnaround={self.total_turnaround_time}, "
            f"weighted_objective={self.total_weighted_turnaround_time})"
        )