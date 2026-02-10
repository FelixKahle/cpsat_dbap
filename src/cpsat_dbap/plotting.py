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

from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.artist import Artist

from .solution import Solution
from .instance import DBAPInstance

def plot_schedule(
    solution: Solution, 
    instance: Optional[DBAPInstance] = None,
    title: str = "Berth Allocation Schedule"
) -> Tuple[Figure, Axes]:
    """
    Generates a Gantt chart visualization for a Berth Allocation Solution.

    The chart displays berths on the Y-axis and time on the X-axis. 
    It visualizes:
    1. Vessel processing blocks (colored bars).
    2. Berth unavailability/closure windows (gray hatched areas), if instance data is provided.
    3. Waiting times (red dotted lines from arrival to start), if instance data is provided.

    Args:
        solution: The solution object containing the calculated schedule.
        instance: (Optional) The original problem instance. providing this enables
                  visualization of berth limits and vessel arrival delays.
        title: The title text for the chart.

    Returns:
        A tuple (Figure, Axes) containing the generated matplotlib plot.
        The caller is responsible for displaying (plt.show) or saving the figure.
    """
    if solution.num_vessels == 0:
        print("Warning: Empty solution provided to plot_schedule.")
        # Return an empty figure to prevent runtime errors in the caller
        return plt.subplots()

    # Initialize the figure layout
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # --- Axis Configuration ---
    
    # Determine the number of berths to display
    if instance:
        num_berths = instance.num_berths
    else:
        # Fallback if instance is missing: infer from solution
        num_berths = max(solution.vessel_berths) + 1 if solution.vessel_berths else 1
        
    berth_indices = range(num_berths)
    
    # --- Berth Availability Visualization ---
    
    if instance:
        # Calculate the visual horizon for the plot
        # We find the latest closing time across all berths to define the x-axis limit
        max_end = 0
        for interval in instance.berth_opening_times:
            # Check for finite finish times to avoid infinite loops/plots
            if interval.finish < float('inf'):
                 max_end = max(max_end, interval.finish)
        
        # Extend the plot limit slightly beyond the schedule makespan or berth closing
        plot_limit = max(solution.makespan, max_end) if max_end > 0 else solution.makespan + 50
        
        for b in berth_indices:
            interval = instance.berth_opening_times[b]
            
            # Visualize the "Closed" period before the berth opens
            if interval.start > 0:
                ax.broken_barh(
                    xranges=[(0, interval.start)], 
                    yrange=(b - 0.4, 0.8), 
                    facecolors='gray', 
                    alpha=0.3, 
                    hatch='///'
                )
                
            # Visualize the "Closed" period after the berth closes
            if interval.finish < plot_limit:
                 ax.broken_barh(
                    xranges=[(interval.finish, plot_limit - interval.finish)], 
                    yrange=(b - 0.4, 0.8), 
                    facecolors='gray', 
                    alpha=0.3, 
                    hatch='///'
                )

    # --- Vessel Schedule Visualization ---
    
    # Use Tableau colors for distinct, professional coloring of vessels
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i in range(solution.num_vessels):
        b = solution.vessel_berths[i]
        start = solution.vessel_start_times[i]
        end = solution.vessel_end_times[i]
        duration = end - start
        
        # Cycle through colors based on vessel index
        col = colors[i % len(colors)]
        
        # Draw the main Gantt bar representing processing time
        ax.barh(
            y=b, 
            width=duration, 
            left=start, 
            height=0.6, 
            color=col, 
            edgecolor='black', 
            alpha=0.9,
            align='center',
            zorder=3  # Ensure bars sit above grid lines
        )
        
        # Place label inside the bar
        ax.text(
            x=start + duration / 2, 
            y=b, 
            s=f"V{i}", 
            ha='center', 
            va='center', 
            color='white', 
            fontweight='bold', 
            fontsize=9,
            zorder=4
        )

        # --- Waiting Time Visualization ---
        
        if instance:
            arrival = instance.arrival_times[i]
            
            # If start time is later than arrival, visualize the wait
            if start > arrival:
                # Dotted line connecting Arrival Time to Start Time
                ax.plot(
                    [arrival, start], 
                    [b, b], 
                    linestyle=':', 
                    color='red', 
                    linewidth=1.5,
                    zorder=2
                )
                # Vertical tick mark indicating the Arrival Time
                ax.plot(
                    [arrival], 
                    [b], 
                    marker='|', 
                    color='red', 
                    markersize=8, 
                    markeredgewidth=1.5,
                    zorder=2
                )

    # --- Final Plot Formatting ---
    
    # Configure Y-axis with Berth labels
    ax.set_yticks(list(berth_indices))
    ax.set_yticklabels([f"Berth {b}" for b in berth_indices])
    
    ax.set_xlabel("Time")
    ax.set_title(title)
    
    # Add vertical grid lines for easier time reading
    ax.grid(True, axis='x', linestyle='--', alpha=0.5, zorder=0)
    
    # Construct a custom legend
    legend_patches: List[Artist] = [
        mpatches.Patch(color='gray', alpha=0.3, hatch='///', label='Berth Closed'),
        mpatches.Patch(color=colors[0], label='Vessel Processing')
    ]
    
    if instance:
        legend_patches.append(
            Line2D([0], [0], color='red', linestyle=':', marker='|', label='Waiting Time')
        )
    
    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()
    
    return fig, ax