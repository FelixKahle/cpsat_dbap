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
import re
import time
from pathlib import Path
from typing import List, Tuple

# Explicitly add the project root to sys.path to allow execution 
# without installing the package.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from cpsat_dbap.instance import parse_instance, DBAPInstance
from cpsat_dbap.solver import solve, SolverConfig

def get_matching_files(data_dir: Path, pattern: str) -> List[Path]:
    """
    Scans the given directory and returns a sorted list of Path objects
    that match the provided regex pattern.
    """
    if not data_dir.exists():
        print(f"Error: Data directory not found at: {data_dir}")
        return []

    regex = re.compile(pattern)
    matched_files = []

    for entry in data_dir.iterdir():
        if entry.is_file() and regex.match(entry.name):
            matched_files.append(entry)

    # Sort files naturally (e.g., f10 before f2) or alphabetically
    matched_files.sort(key=lambda p: p.name)
    return matched_files

def main() -> None:
    """
    Iterates through all instance files matching the specific regex pattern,
    solves them, and prints a summary table of the results.
    """
    
    data_dir = project_root / "data"
    file_pattern = r"^f\d+x\d+-\d+\.txt$"
    
    # Solver settings (adjust time limit as needed for batch processing)
    time_limit = 10.0 
    config = SolverConfig(
        time_limit_seconds=time_limit,
        log_search_progress=False,  # Keep logs quiet for clean table output
        num_workers=os.cpu_count() or 1,
        use_hints=True
    )
    
    files = get_matching_files(data_dir, file_pattern)
    
    if not files:
        print(f"No files found matching pattern '{file_pattern}' in {data_dir}")
        return

    print(f"Found {len(files)} instances. Starting batch processing...")
    print(f"Solver Time Limit: {time_limit}s per instance\n")
    
    # Define column widths
    col_inst = 25
    col_ves = 12
    col_ber = 12
    col_time = 12
    col_obj = 15

    header = (
        f"{'Instance':<{col_inst}} | "
        f"{'Vessels':<{col_ves}} | "
        f"{'Berths':<{col_ber}} | "
        f"{'Solve Time':<{col_time}} | "
        f"{'Objective':<{col_obj}}"
    )
    separator = "-" * len(header)

    print(header)
    print(separator)
    
    for file_path in files:
        instance_name = file_path.name
        
        try:
            # Parse
            with open(file_path, "r") as f:
                instance = parse_instance(f)
            
            # Solve and measure pure wall time
            start_time = time.perf_counter()
            solution = solve(instance, config)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            
            # Extract metrics
            num_vessels = instance.num_vessels
            num_berths = instance.num_berths
            
            if solution:
                objective_val = solution.total_weighted_turnaround_time
                obj_str = str(objective_val)
            else:
                obj_str = "No Solution"

            # Print Row
            print(
                f"{instance_name:<{col_inst}} | "
                f"{num_vessels:<{col_ves}} | "
                f"{num_berths:<{col_ber}} | "
                f"{elapsed:<{col_time}.4f} | "
                f"{obj_str:<{col_obj}}"
            )
            
            # Optional: Flush stdout to see progress immediately
            sys.stdout.flush()

        except Exception as e:
            # Handle parsing errors or other unexpected crashes gracefully
            print(
                f"{instance_name:<{col_inst}} | "
                f"{'ERROR':<{col_ves}} | "
                f"{'-':<{col_ber}} | "
                f"{'-':<{col_time}} | "
                f"{str(e):<{col_obj}}"
            )

    print(separator)
    print("Batch processing complete.")

if __name__ == "__main__":
    main()