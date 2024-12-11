    #include <iostream>
    #include <vector>
    #include <random>
    #include <chrono>
    #include <x86intrin.h> // For _mm_prefetch
    #include <fstream>
    #include <sstream>
    #include <unordered_map>
    #include <string>
    #include <numeric> // For std::accumulate
    #include <cmath> // For std::sqrt

    // Global constants
    const std::string GLOVE_PATH = "data/glove.840B.300d.txt";
    const std::string INPUT_PATH = "data/input.txt";
    const size_t NUM_COLS = 300;        // GloVe embedding dimension
    const size_t PREFETCH_AHEAD_START = 1;  // Start of prefetch ahead search range
    const size_t PREFETCH_AHEAD_END = 20;   // End of prefetch ahead search range
    const size_t NUM_RUNS = 10;         // Number of times to run each test

    // Function to perform row operations without prefetching
    double regularAccess(const std::vector<std::vector<double>>& matrix, 
                        const std::vector<size_t>& accessPattern) {
        double result = 0.0;
        
        for (size_t i = 0; i < accessPattern.size(); i++) {
            //if (i % (accessPattern.size() / 10) == 0) {
            //    std::cout << "Regular access: " << (i * 100 / accessPattern.size()) << "% complete" << std::endl;
            //}
            
            const auto& row = matrix[accessPattern[i]];
            // Compute dot product NUM_COLS times
            double row_sum = 0.0;
            for (size_t n = 0; n < 2; n++) {
                double dot_product = 0.0;
                for (size_t j = 0; j < NUM_COLS; j++) {
                    dot_product += row[j] * row[j];
                }
                row_sum += dot_product;
            }
            result += row_sum / NUM_COLS;
        }
        return result / accessPattern.size();
    }
    
    // Function to perform row operations with prefetching
    double prefetchedAccess(const std::vector<std::vector<double>>& matrix, 
                           const std::vector<size_t>& accessPattern,
                           size_t prefetch_ahead) {
        double result = 0.0;
        
        // Handle first chunk with prefetching
        size_t i = 0;
        for (; i < accessPattern.size() - prefetch_ahead; i++) {
            const char* next_row = reinterpret_cast<const char*>(&matrix[accessPattern[i + prefetch_ahead]][0]);
            _mm_prefetch(next_row, _MM_HINT_T0);
            //_MM_HINT_T0: Prefetch into all levels of the cache.
            //_MM_HINT_T1: Prefetch into L2 cache only.
            //_MM_HINT_T2: Prefetch into L1 cache only.
            //_MM_HINT_NTA: Do not prefetch.
            
            const auto& row = matrix[accessPattern[i]];
            double row_sum = 0.0;
            for (size_t n = 0; n < 2; n++) {
                double dot_product = 0.0;
                for (size_t j = 0; j < NUM_COLS; j++) {
                    dot_product += row[j] * row[j];
                }
                row_sum += dot_product;
            }
            result += row_sum / NUM_COLS;
        }
        
        // Handle remaining elements without prefetching
        for (; i < accessPattern.size(); i++) {
            const auto& row = matrix[accessPattern[i]];
            double row_sum = 0.0;
            for (size_t n = 0; n < NUM_COLS; n++) {
                double dot_product = 0.0;
                for (size_t j = 0; j < NUM_COLS; j++) {
                    dot_product += row[j] * row[j];
                }
                row_sum += dot_product;
            }
            result += row_sum / NUM_COLS;
        }
        
        return result / accessPattern.size();
    }
    
    int main() {
        // Load GloVe embeddings
        std::cout << "Loading GloVe embeddings..." << std::endl;
        std::unordered_map<std::string, size_t> word_to_idx;
        std::vector<std::vector<double>> matrix;
        
        std::ifstream glove_file(GLOVE_PATH);
        std::string line;
        size_t idx = 0;
        
        while (std::getline(glove_file, line)) {
            std::istringstream iss(line);
            std::string word;
            iss >> word;
            
            std::vector<double> embedding(NUM_COLS);
            for (size_t i = 0; i < NUM_COLS; i++) {
                iss >> embedding[i];
            }
            
            word_to_idx[word] = idx;
            matrix.push_back(embedding);
            idx++;
        }
        
        // Load input words and create access pattern
        std::cout << "Loading input words..." << std::endl;
        std::vector<size_t> accessPattern;
        std::ifstream input_file(INPUT_PATH);
        std::string word;
        
        while (input_file >> word) {
            if (word_to_idx.find(word) != word_to_idx.end()) {
                accessPattern.push_back(word_to_idx[word]);
            }
        }

        if (accessPattern.empty()) {
            std::cerr << "No valid words found in input file!" << std::endl;
            return 1;
        }

        double best_speedup = 0.0;
        size_t best_prefetch_ahead = 0;
        double best_reg_mean = 0.0;
        double best_pref_mean = 0.0;
        double best_reg_var = 0.0;
        double best_pref_var = 0.0;

        // Try different PREFETCH_AHEAD values
        for (size_t prefetch_ahead = PREFETCH_AHEAD_START; prefetch_ahead <= PREFETCH_AHEAD_END; prefetch_ahead++) {
            std::cout << "\nTesting PREFETCH_AHEAD = " << prefetch_ahead << std::endl;
            
            // Vectors to store timing results
            std::vector<double> regular_times;
            std::vector<double> prefetch_times;
            double result1, result2;

            // Run tests multiple times
            for (size_t run = 0; run < NUM_RUNS; run++) {
                std::cout << "Run " << run + 1 << "/" << NUM_RUNS << std::endl;
                
                // Test regular access
                auto start = std::chrono::steady_clock::now();
                result1 = regularAccess(matrix, accessPattern);
                auto end = std::chrono::steady_clock::now();
                auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                regular_times.push_back(duration1 / 1e6); // Convert to ms
            
                // Test prefetched access
                start = std::chrono::steady_clock::now();
                result2 = prefetchedAccess(matrix, accessPattern, prefetch_ahead);
                end = std::chrono::steady_clock::now();
                auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                prefetch_times.push_back(duration2 / 1e6); // Convert to ms
            }

            // Calculate statistics
            double reg_mean = std::accumulate(regular_times.begin(), regular_times.end(), 0.0) / NUM_RUNS;
            double pref_mean = std::accumulate(prefetch_times.begin(), prefetch_times.end(), 0.0) / NUM_RUNS;

            double reg_var = 0.0, pref_var = 0.0;
            for (size_t i = 0; i < NUM_RUNS; i++) {
                reg_var += std::pow(regular_times[i] - reg_mean, 2);
                pref_var += std::pow(prefetch_times[i] - pref_mean, 2);
            }
            reg_var /= NUM_RUNS;
            pref_var /= NUM_RUNS;

            double mean_speedup = reg_mean / pref_mean;
            
            // Print intermediate results
            std::cout << "Regular access time: " << reg_mean << " ± " << std::sqrt(reg_var) << " ms" << std::endl;
            std::cout << "Prefetched access time: " << pref_mean << " ± " << std::sqrt(pref_var) << " ms" << std::endl;
            std::cout << "Speedup: " << mean_speedup << "x" << std::endl;
            std::cout << "Results match: " << (std::abs(result1 - result2) < 1e-10) << std::endl;

            // Update best results if current speedup is better
            if (mean_speedup > best_speedup) {
                best_speedup = mean_speedup;
                best_prefetch_ahead = prefetch_ahead;
                best_reg_mean = reg_mean;
                best_pref_mean = pref_mean;
                best_reg_var = reg_var;
                best_pref_var = pref_var;
            }
        }

        // Print final results with best configuration
        std::cout << "\nBest configuration found:" << std::endl;
        std::cout << "PREFETCH_AHEAD: " << best_prefetch_ahead << std::endl;
        std::cout << "Regular access time: " << best_reg_mean << " ± " << std::sqrt(best_reg_var) << " ms" << std::endl;
        std::cout << "Prefetched access time: " << best_pref_mean << " ± " << std::sqrt(best_pref_var) << " ms" << std::endl;
        std::cout << "Best speedup achieved: " << best_speedup << "x" << std::endl;
    
        return 0;
    }
