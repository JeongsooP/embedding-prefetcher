Prefetching Embedding Layer\
EECS 573 Final Project, FA 2024  
Yiming Dou, Gaurav Kaul, Jeongsoo Park
==============================

Main Components:
---------------
- main script: prefetch_embedding_layer.cpp
  * Implements optimized embedding layer access with hardware prefetching
  * Uses GloVe embeddings for word representation
  * Configurable prefetch-ahead distance (currently set to 11, optimal for 25 d embeddings)
  * Measures access time and performance 

Data:
--------------
data/
1. get_data.sh:
   * Downloads GloVe Twitter word embeddings (27B.zip)
   * Downloads Sherlock Holmes text as input data
   * Automatically extracts required files

Experiments/Analysis:
------------------
plotting/
1. embedding_size_speedup_plot.py:
   * Visualizes performance across different embedding sizes

2. prefetch_ahead_hp_plot.py:
   * Plots speedup factors against prefetch-ahead distances
   * Identifies optimal prefetch distance configuration
   * Generates prefetch_speedup.png with detailed analysis

Scripting Utilities:
------------------
scripting/
1. compile_and_run_prefetcher.sh:
   * Compiles C++ files with aggressive optimizations (-O3, -march=native, etc.)
   * Runs performance analysis using perf stat
   * Tracks key metrics: cycles, instructions, cache stats, branch misses

2. run_benchmarks.sh:
   * Comprehensive benchmark suite comparing baseline vs prefetch implementations
   * Measures and compares:
     - Cache miss rates
     - Instructions Per Cycle (IPC)
     - Overall performance metrics
   * Generates detailed performance comparison summary

Embedding Layer Implementations:
-----------------------------
embedding_layers/

1. baseline_embedding_layer.cpp
   - Basic implementation of embedding layer access
   - Serves as a reference point for performance comparisons
   - Loads word embeddings and performs simple row access operations

2. next_word_prefetching.cpp
   - Implements a learnable prefetching strategy
   - Uses a simple transition probability model to predict the next word
   - Demonstrates basic machine learning-based prefetching technique
   - Measures prediction accuracy and access time improvements

3. ngram_prefetching.cpp
   - Implements n-gram based prefetching strategy
   - Uses variable-order n-gram models for next word prediction
   - Supports different n-gram orders (configurable via NGRAM_ORDER)
   - Measures prediction accuracy and access time improvements

4. optimal_prefetching.cpp
   - Explores different prefetch-ahead distances
   - Conducts multiple runs to statistically analyze performance
   - Finds the optimal prefetch-ahead configuration
   - Provides detailed timing and speedup measurements

5. ngram.hpp
   - Header-only implementation of n-gram model
   - Provides generic n-gram model building and prediction functions
   - Used by ngram_prefetching.cpp for next word prediction
