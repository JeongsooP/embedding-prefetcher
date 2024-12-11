#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <x86intrin.h> // For _mm_prefetch
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>

// Global constants
const std::string GLOVE_PATH = "data/glove.twitter.27B.25d.txt";
const std::string INPUT_PATH = "data/input.txt";
const size_t NUM_COLS = 25;        // GloVe embedding dimension

// Function to perform row operations without prefetching
double regularAccess(const std::vector<std::vector<double>>& matrix, 
                    const std::vector<size_t>& accessPattern) {
    double result = 0.0;
    
    for (size_t i = 0; i < accessPattern.size(); i++) {
        //if (i % (accessPattern.size() / 10) == 0) {
       //     std::cout << "Regular access: " << (i * 100 / accessPattern.size()) << "% complete" << std::endl;
       // }
        
        const auto& row = matrix[accessPattern[i]];
        // Compute average of squared values in row
        double row_sum = 0.0;
        for (size_t j = 0; j < NUM_COLS; j++) {
            row_sum += row[j] * row[j];
        }
        result += row_sum / NUM_COLS;
    }
    return result / accessPattern.size();
}

// Function to perform row operations with learnable prefetching
double learnableAccess(const std::vector<std::vector<double>>& matrix, 
                      const std::vector<size_t>& accessPattern,
                      const std::unordered_map<size_t, size_t>& mostLikelyNext) {
    double result = 0.0;
    
    for (size_t i = 0; i < accessPattern.size(); i++) {
        //if (i % (accessPattern.size() / 10) == 0) {
       //     std::cout << "Learnable access: " << (i * 100 / accessPattern.size()) << "% complete" << std::endl;
       // }
        
        // Determine the next index to prefetch based on the most likely next word
        if (i < accessPattern.size() - 1) {
            size_t current_word = accessPattern[i];
            auto it = mostLikelyNext.find(current_word);
            if (it != mostLikelyNext.end()) {
                size_t next_word = it->second;
                if (next_word < matrix.size()) {
                    const char* next_row = reinterpret_cast<const char*>(&matrix[next_word][0]);
                    _mm_prefetch(next_row, _MM_HINT_T0);
                }
            }
        }
        
        const auto& row = matrix[accessPattern[i]];
        // Compute average of squared values in row
        double row_sum = 0.0;
        for (size_t j = 0; j < NUM_COLS; j++) {
            row_sum += row[j] * row[j];
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
    
    // Build the most likely next word mapping
    std::cout << "Building most likely next word mapping..." << std::endl;
    std::unordered_map<size_t, std::unordered_map<size_t, size_t>> transition_counts;
    for (size_t i = 0; i + 1 < accessPattern.size(); i++) {
        size_t current = accessPattern[i];
        size_t next = accessPattern[i + 1];
        transition_counts[current][next]++;
    }

    std::unordered_map<size_t, size_t> mostLikelyNext;
    for (const auto& pair : transition_counts) {
        size_t current = pair.first;
        size_t max_count = 0;
        size_t likely_next = 0;
        for (const auto& inner_pair : pair.second) {
            if (inner_pair.second > max_count) {
                max_count = inner_pair.second;
                likely_next = inner_pair.first;
            }
        }
        mostLikelyNext[current] = likely_next;
    }

    // Calculate prediction accuracy
    size_t correct_predictions = 0;
    size_t total_predictions = 0;
    for (size_t i = 0; i < accessPattern.size() - 1; i++) {
        size_t current = accessPattern[i];
        size_t actual_next = accessPattern[i + 1];
        auto it = mostLikelyNext.find(current);
        if (it != mostLikelyNext.end()) {
            total_predictions++;
            if (it->second == actual_next) {
                correct_predictions++;
            }
        }
    }
    
    double accuracy = total_predictions > 0 ? 
        (static_cast<double>(correct_predictions) / total_predictions) * 100.0 : 0.0;
    std::cout << "Next word prediction accuracy: " << accuracy << "%" << std::endl;

    // Test regular access
    std::cout << "Testing regular access..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    double result1 = regularAccess(matrix, accessPattern);
    auto end = std::chrono::steady_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Test learnable access
    std::cout << "Testing learnable access..." << std::endl;
    start = std::chrono::steady_clock::now();
    double result2 = learnableAccess(matrix, accessPattern, mostLikelyNext);
    end = std::chrono::steady_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Print results
    std::cout << "\nResults:" << std::endl;
    std::cout << "Regular access time: " << duration1 / 1e6 << "ms" << std::endl;
    std::cout << "Learnable access time: " << duration2 / 1e6 << "ms" << std::endl;
    
    // Fix division by zero error when duration2 is 0
    double speedup = (duration2 > 0) ? static_cast<double>(duration1) / duration2 : 0.0;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // Print results to verify correctness
    std::cout << "Results match: " << (std::abs(result1 - result2) < 1e-10) << std::endl;

    return 0;
}
