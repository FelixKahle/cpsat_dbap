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
from dataclasses import dataclass
from typing import List, Optional, TextIO, Union, Iterator, Callable

# ============================================================
# ProcessingTime
# ============================================================

@dataclass(frozen=True, order=True)
class ProcessingTime:
    """
    Represents a discrete processing time unit.

    This class encapsulates time values while providing a sentinel mechanism
    for invalid or forbidden processing times (e.g., when a specific vessel
    cannot be serviced at a specific berth).

    Attributes:
        time: The integer value of time. Non-negative values are valid;
              negative values indicate an invalid state.
    """
    time: int

    @property
    def is_valid(self) -> bool:
        """Checks if the processing time represents a valid duration."""
        return self.time >= 0

    @property
    def is_invalid(self) -> bool:
        """Checks if the processing time represents a forbidden or invalid state."""
        return self.time < 0

    def value(self) -> int:
        """
        Returns the raw integer time value.

        Raises:
            ValueError: If the processing time is invalid.
        """
        if self.is_valid:
            return self.time
        raise ValueError("Invalid ProcessingTime has no usable value")

    def __str__(self) -> str:
        return f"ProcessingTime({self.time})" if self.is_valid else "ProcessingTime(INVALID)"

    def __int__(self) -> int:
        return self.value()

    # --- Arithmetic Operations ---

    def _combine(self, other: Union[ProcessingTime, int], op: Callable[[int, int], int]) -> ProcessingTime:
        """Helper to combine two time values using a given operator."""
        if isinstance(other, int):
            other = ProcessingTime(other)
        
        # Propagation of invalidity: if either operand is invalid, the result is invalid.
        if self.is_valid and other.is_valid:
            return ProcessingTime(op(self.time, other.time))
        return INVALID_PROCESSING_TIME

    def __add__(self, other: Union[ProcessingTime, int]) -> ProcessingTime:
        return self._combine(other, lambda a, b: a + b)

    def __radd__(self, other: int) -> ProcessingTime:
        return self + other

    def __sub__(self, other: Union[ProcessingTime, int]) -> ProcessingTime:
        return self._combine(other, lambda a, b: a - b)

    def __rsub__(self, other: int) -> ProcessingTime:
        # Subtraction is non-commutative; we must explicitly wrap the integer
        # and check validity order.
        other_pt = ProcessingTime(other)
        if other_pt.is_valid and self.is_valid:
            return ProcessingTime(other - self.time)
        return INVALID_PROCESSING_TIME

    def __mul__(self, other: Union[ProcessingTime, int]) -> ProcessingTime:
        return self._combine(other, lambda a, b: a * b)

    def __rmul__(self, other: int) -> ProcessingTime:
        return self * other

    def __floordiv__(self, other: Union[ProcessingTime, int]) -> ProcessingTime:
        if isinstance(other, int):
            other = ProcessingTime(other)
        
        # Check for division by zero and validity
        if self.is_valid and other.is_valid and other.time != 0:
            return ProcessingTime(self.time // other.time)
        return INVALID_PROCESSING_TIME

# Sentinel constant for invalid time
INVALID_PROCESSING_TIME = ProcessingTime(-1)


# ============================================================
# HalfOpenInterval
# ============================================================

@dataclass(frozen=True, order=True)
class HalfOpenInterval:
    """
    Represents a time interval [start, end).
    
    The interval includes the start point but excludes the end point.
    This structure is often preferred for discrete scheduling to avoid
    off-by-one errors during adjacency checks.
    """
    start_inclusive: int
    end_exclusive: int

    def __post_init__(self) -> None:
        """Validates that the interval is well-formed (start <= end)."""
        if self.end_exclusive < self.start_inclusive:
            raise ValueError(f"Interval end ({self.end_exclusive}) must be >= start ({self.start_inclusive})")

    @property
    def start(self) -> int:
        """Alias for the inclusive start time."""
        return self.start_inclusive
    
    @property
    def finish(self) -> int:
        """Alias for the exclusive end time."""
        return self.end_exclusive

    def __len__(self) -> int:
        """Returns the duration of the interval."""
        return self.end_exclusive - self.start_inclusive

    def is_empty(self) -> bool:
        """True if the duration is zero."""
        return len(self) == 0

    def contains(self, t: int) -> bool:
        """Checks if time t lies within [start, end)."""
        return self.start_inclusive <= t < self.end_exclusive
    
    def __contains__(self, item: int) -> bool:
        return self.contains(item)

    def overlaps(self, other: HalfOpenInterval) -> bool:
        """
        Checks if this interval overlaps with another.
        
        Logic: Start of one must be strictly less than the End of the other.
        """
        return (self.start_inclusive < other.end_exclusive) and \
               (other.start_inclusive < self.end_exclusive)

    def adjacent(self, other: HalfOpenInterval) -> bool:
        """
        Checks if two intervals touch without overlapping.
        e.g., [0, 5) and [5, 10).
        """
        return (self.end_exclusive == other.start_inclusive) or \
               (other.end_exclusive == self.start_inclusive)

    def intersection(self, other: HalfOpenInterval) -> Optional[HalfOpenInterval]:
        """
        Returns the intersection of two intervals, or None if disjoint.
        """
        if self.overlaps(other):
            return HalfOpenInterval(
                max(self.start_inclusive, other.start_inclusive),
                min(self.end_exclusive, other.end_exclusive)
            )
        return None

    def __str__(self) -> str:
        return f"[{self.start_inclusive}, {self.end_exclusive})"


# ============================================================
# DBAPInstance
# ============================================================

@dataclass(frozen=True)
class DBAPInstance:
    """
    Immutable container for the Discrete Berth Allocation Problem (DBAP) data.

    Attributes:
        num_vessels: Total number of vessels to be serviced.
        num_berths: Total number of available berths.
        vessel_weights: Priority weights for each vessel (default 1).
        arrival_times: The earliest time each vessel arrives at the port.
        latest_departure_times: The hard deadline for each vessel.
        processing_times: A matrix [vessel][berth] representing service duration.
                          Invalid durations are marked with sentinel values.
        berth_opening_times: Availability windows for each berth.
    """
    num_vessels: int
    num_berths: int
    vessel_weights: List[int]
    arrival_times: List[int]
    latest_departure_times: List[int]
    # Matrix represented as list of lists (row-major: [vessel][berth])
    processing_times: List[List[ProcessingTime]]
    berth_opening_times: List[HalfOpenInterval]

    def __post_init__(self) -> None:
        """Validates array dimensions match vessel and berth counts."""
        if len(self.vessel_weights) != self.num_vessels:
            raise ValueError("vessel_weights length mismatch")
        if len(self.arrival_times) != self.num_vessels:
            raise ValueError("arrival_times length mismatch")
        if len(self.latest_departure_times) != self.num_vessels:
            raise ValueError("latest_departure_times length mismatch")
        if len(self.processing_times) != self.num_vessels:
            raise ValueError("processing matrix rows mismatch")
        if any(len(row) != self.num_berths for row in self.processing_times):
            raise ValueError("processing matrix columns mismatch")
        if len(self.berth_opening_times) != self.num_berths:
            raise ValueError("berth intervals length mismatch")

    def get_processing_time(self, v: int, b: int) -> ProcessingTime:
        """Retrieves processing time for vessel v at berth b."""
        return self.processing_times[v][b]

    def get_berth_interval(self, b: int) -> HalfOpenInterval:
        """Retrieves the availability interval for berth b."""
        return self.berth_opening_times[b]

    def __str__(self) -> str:
        return f"DBAPInstance({self.num_vessels} vessels, {self.num_berths} berths)"


# ============================================================
# Parsing Logic
# ============================================================

def parse_instance(source: Union[str, TextIO], forbidden_threshold: int = 99999) -> DBAPInstance:
    """
    Parses a DBAP instance from a string or file-like object.

    The format is whitespace-delimited integers:
      N M             (Vessels, Berths)
      [Arrivals]      (N integers)
      [Openings]      (M integers)
      [Proc Matrix]   (N rows * M cols integers)
      [Endings]       (M integers)
      [Deadlines]     (N integers)

    Args:
        source: A raw string containing the data or a file object.
        forbidden_threshold: Integer value in the input above which a 
                             processing time is considered invalid/forbidden.

    Returns:
        A validated DBAPInstance object.

    Raises:
        ValueError: If data is malformed or logical constraints (start < end) are violated.
        EOFError: If the input stream ends prematurely.
    """

    # Helper generator to lazily yield whitespace-separated tokens
    def token_generator(src: Union[str, TextIO]) -> Iterator[str]:
        if isinstance(src, str):
            yield from src.split()
        else:
            # Read the entire content at once for standard compliance with
            # whitespace splitting, suitable for typical instance sizes.
            content = src.read()
            yield from content.split()

    tokens = token_generator(source)

    def read_int() -> int:
        try:
            return int(next(tokens))
        except StopIteration:
            raise EOFError("Unexpected end of input while reading instance data")
        except ValueError:
            raise ValueError("Encountered non-integer token in input")

    # Read problem dimensions
    n = read_int()
    m = read_int()

    if n <= 0: raise ValueError("Number of vessels must be positive")
    if m <= 0: raise ValueError("Number of berths must be positive")

    # Read vessel arrival times
    arrivals = [read_int() for _ in range(n)]

    # Read berth opening times (start of availability)
    openings = [read_int() for _ in range(m)]

    # Read processing time matrix (N rows, M columns)
    # Values >= forbidden_threshold are converted to INVALID_PROCESSING_TIME
    proc_matrix: List[List[ProcessingTime]] = []
    for _ in range(n):
        row: List[ProcessingTime] = []
        for _ in range(m):
            h = read_int()
            if h >= forbidden_threshold:
                row.append(INVALID_PROCESSING_TIME)
            else:
                row.append(ProcessingTime(h))
        proc_matrix.append(row)

    # Read berth closing times (end of availability)
    endings = [read_int() for _ in range(m)]

    # Immediate validation of berth windows
    for i in range(m):
        if openings[i] > endings[i]:
            raise ValueError(f"Invalid berth interval at index {i}: start {openings[i]} > end {endings[i]}")

    # Read latest departure times (deadlines)
    latest = [read_int() for _ in range(n)]

    # Logical validation: Arrival cannot be later than Deadline
    for i in range(n):
        if arrivals[i] > latest[i]:
            raise ValueError(f"Arrival exceeds latest departure for vessel {i}")

    # Convert closed input intervals [start, end] to internal HalfOpenIntervals [start, end + 1)
    intervals = [
        HalfOpenInterval(openings[i], endings[i] + 1)
        for i in range(m)
    ]

    # Initialize default weights (all 1)
    weights = [1] * n

    return DBAPInstance(
        num_vessels=n,
        num_berths=m,
        vessel_weights=weights,
        arrival_times=arrivals,
        latest_departure_times=latest,
        processing_times=proc_matrix,
        berth_opening_times=intervals
    )