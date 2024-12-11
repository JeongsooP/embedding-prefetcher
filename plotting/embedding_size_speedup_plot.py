import matplotlib.pyplot as plt
import numpy as np

# Data for embedding sizes and corresponding speedups
embedding_sizes = [25, 50, 100, 200, 300]
speedup = [1.37828, 1.38488, 1.28126, 1.19913, 1.1111]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(embedding_sizes, speedup, 'gs-', linewidth=2, markersize=8)

# Customize the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Embedding Size (d)', fontsize=12)
plt.ylabel('Speedup Factor', fontsize=12)
plt.title('Prefetching Speedup vs. Embedding Size', fontsize=14)

# Add horizontal line at y=1 to show baseline
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)

# Annotate the speedup values
for size, sp in zip(embedding_sizes, speedup):
    plt.annotate(f'{sp:.2f}x', xy=(size, sp),
                 xytext=(0, 5), textcoords='offset points',
                 ha='center', va='bottom', fontsize=10)

# Set y-axis limits for better visualization
plt.ylim(1.0, max(speedup) + 0.1)

# Set x-axis to show integer ticks
plt.xticks(embedding_sizes)

# Save the plot
plt.savefig('embedding_size_speedup.png', dpi=300, bbox_inches='tight')
plt.show() 