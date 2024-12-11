import matplotlib.pyplot as plt
import numpy as np

# Data from your results
prefetch_ahead = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
speedup = [1.00556, 1.11786, 1.21132, 1.27877, 1.32668, 1.32967, 1.35421, 
           1.33614, 1.31543, 1.36956, 1.37477, 1.34274, 1.3134, 1.29274, 
           1.32218, 1.37349]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(prefetch_ahead, speedup, 'bo-', linewidth=2, markersize=8)

# Customize the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Prefetch Ahead Distance', fontsize=12)
plt.ylabel('Speedup Factor', fontsize=12)
plt.title('Prefetch Distance Speedup', fontsize=14)

# Add horizontal line at y=1 to show baseline
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)

# Annotate the maximum speedup
max_speedup = max(speedup)
max_speedup_idx = speedup.index(max_speedup)
plt.annotate(f'Max Speedup: {max_speedup:.2f}x',
             xy=(prefetch_ahead[max_speedup_idx], max_speedup),
             xytext=(5, 10), textcoords='offset points',
             ha='left', va='bottom')

# Set y-axis limits to start from 0.9 for better visualization
plt.ylim(0.9, max(speedup) * 1.05)

# Set x-axis to show integer ticks
plt.xticks(prefetch_ahead)

# Save the plot
plt.savefig('prefetch_speedup.png', dpi=300, bbox_inches='tight')
plt.show()