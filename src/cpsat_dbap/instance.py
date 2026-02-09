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
from dataclasses import dataclass, field
from typing import List, Optional, TextIO, Union, Generator, Iterator, Callable

# ============================================================
# ProcessingTime
# ============================================================

@dataclass(frozen=True, order=True)
class ProcessingTime:
    """
    Represents a processing time in integer units.
    
    - time >= 0 -> valid
    - time < 0  -> invalid sentinel
    """
    time: int

    @property
    def is_valid(self) -> bool:
        return self.time >= 0

    @property
    def is_invalid(self) -> bool:
        return self.time < 0

    def value(self) -> int:
        if self.is_valid:
            return self.time
        raise ValueError("Invalid ProcessingTime has no usable value")

    def __str__(self) -> str:
        return f"ProcessingTime({self.time})" if self.is_valid else "ProcessingTime(INVALID)"

    def __int__(self) -> int:
        return self.value()

    # --- Arithmetic ---

    def _combine(self, other: Union[ProcessingTime, int], op: Callable[[int, int], int]) -> ProcessingTime:
        if isinstance(other, int):
            other = ProcessingTime(other)
        
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
        # Check explicitly because subtraction is non-commutative
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
        
        if self.is_valid and other.is_valid and other.time != 0:
            return ProcessingTime(self.time // other.time)
        return INVALID_PROCESSING_TIME

INVALID_PROCESSING_TIME = ProcessingTime(-1)


# ============================================================
# HalfOpenInterval
# ============================================================

@dataclass(frozen=True, order=True)
class HalfOpenInterval:
    """
    Represents interval [start_inclusive, end_exclusive).
    """
    start_inclusive: int
    end_exclusive: int

    def __post_init__(self) -> None:
        if self.end_exclusive < self.start_inclusive:
            raise ValueError(f"Interval end ({self.end_exclusive}) must be >= start ({self.start_inclusive})")

    @property
    def start(self) -> int:
        return self.start_inclusive
    
    @property
    def finish(self) -> int:
        return self.end_exclusive

    def __len__(self) -> int:
        return self.end_exclusive - self.start_inclusive

    def is_empty(self) -> bool:
        return len(self) == 0

    def contains(self, t: int) -> bool:
        return self.start_inclusive <= t < self.end_exclusive
    
    def __contains__(self, item: int) -> bool:
        return self.contains(item)

    def overlaps(self, other: HalfOpenInterval) -> bool:
        return (self.start_inclusive < other.end_exclusive) and \
               (other.start_inclusive < self.end_exclusive)

    def adjacent(self, other: HalfOpenInterval) -> bool:
        return (self.end_exclusive == other.start_inclusive) or \
               (other.end_exclusive == self.start_inclusive)

    def intersection(self, other: HalfOpenInterval) -> Optional[HalfOpenInterval]:
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
    Container describing a berth allocation instance.
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
        # Validations
        if len(self.vessel_weights) != self.num_vessels:
            raise ValueError("vessel_weights mismatch")
        if len(self.arrival_times) != self.num_vessels:
            raise ValueError("arrival_times mismatch")
        if len(self.latest_departure_times) != self.num_vessels:
            raise ValueError("latest_departure_times mismatch")
        if len(self.processing_times) != self.num_vessels:
            raise ValueError("processing matrix rows mismatch")
        if any(len(row) != self.num_berths for row in self.processing_times):
            raise ValueError("processing matrix columns mismatch")
        if len(self.berth_opening_times) != self.num_berths:
            raise ValueError("berth intervals mismatch")

    def get_processing_time(self, v: int, b: int) -> ProcessingTime:
        # Adjust for 0-based indexing if input is 0-based, or assumes caller handles it.
        # Python uses 0-based indexing.
        return self.processing_times[v][b]

    def get_berth_interval(self, b: int) -> HalfOpenInterval:
        return self.berth_opening_times[b]

    def __str__(self) -> str:
        return f"DBAPInstance({self.num_vessels} vessels, {self.num_berths} berths)"


# ============================================================
# Parsing Logic
# ============================================================

def parse_instance(source: Union[str, TextIO], forbidden_threshold: int = 99999) -> DBAPInstance:
    """
    Parse a DBAP instance from a string or IO stream.

    Input Format (Whitespace separated):
    N (vessels)
    M (berths)
    ta_1 ... ta_|N|         (arrivals)
    s_1 ... s_|M|           (opening times)
    h_1_1 ... h_1_|M|       (processing times row 1)
    ...
    h_|N|_1 ... h_|N|_|M|   (processing times row N)
    e_1 ... e_|M|           (ending times)
    t'_1 ... t'_|N|         (max departure times)
    """

    # Create a token generator
    def token_generator(src: Union[str, TextIO]) -> Iterator[str]:
        if isinstance(src, str):
            yield from src.split()
        else:
            # Read chunks or lines to avoid loading massive files into RAM if unnecessary
            # For simplicity and standard compliance with whitespace skipping:
            content = src.read()
            yield from content.split()

    tokens = token_generator(source)

    def read_int() -> int:
        try:
            return int(next(tokens))
        except StopIteration:
            raise EOFError("Unexpected end of input")
        except ValueError:
            raise ValueError("Invalid integer token")

    # 1. Header
    n = read_int()
    m = read_int()

    if n <= 0: raise ValueError("Number of vessels must be positive")
    if m <= 0: raise ValueError("Number of berths must be positive")

    # 2. Arrivals
    arrivals = [read_int() for _ in range(n)]

    # 3. Openings (Berth start times)
    openings = [read_int() for _ in range(m)]

    # 4. Handling Matrix (Processing Times)
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

    # 5. Endings (Berth end times)
    endings = [read_int() for _ in range(m)]

    # Validate intervals immediately (closed form check)
    for i in range(m):
        if openings[i] > endings[i]:
            raise ValueError(f"Invalid berth interval at index {i}: start {openings[i]} > end {endings[i]}")

    # 6. Latest departures
    latest = [read_int() for _ in range(n)]

    # Logical sanity: arrival <= latest
    for i in range(n):
        if arrivals[i] > latest[i]:
            raise ValueError(f"Arrival exceeds latest departure for vessel {i}")

    # 7. Interval Construction (Closed [s, e] -> Half-Open [s, e+1))
    intervals = [
        HalfOpenInterval(openings[i], endings[i] + 1)
        for i in range(m)
    ]

    # Default weights (all 1)
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