#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <x86intrin.h> // For _mm_prefetch
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <cmath> // For std::abs
#include "ngram.hpp" // Include the n-gram model header

// Global constants
const std::string GLOVE_PATH = "data/glove.twitter.27B.25d.txt";
const std::string INPUT_PATH = "data/input.txt";
const size_t NUM_COLS = 25;        // GloVe embedding dimension
const int NGRAM_ORDER = 3;         // Order of the n-gram model

// Function to perform row operations without prefetching
double regularAccess(const std::vector<std::vector<double>>& matrix, 
                    const std::vector<size_t>& accessPattern) {
    double result = 0.0;
    
    for (size_t i = 0; i < accessPattern.size(); i++) {
        if (i % (accessPattern.size() / 10) == 0) {
            std::cout << "Regular access: " << (i * 100 / accessPattern.size()) << "% complete" << std::endl;
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

// Function to perform row operations with embedding-based prefetching
double ngram_prefetch(const std::vector<std::vector<double>>& matrix, 
                        const std::vector<size_t>& accessPattern,
                        const std::vector<NGram>& ngramModels,
                        const std::vector<size_t>& tokens) {
    double result = 0.0;
    
    // Convert accessPattern indices back to tokens for n-gram usage
    // Assuming 'tokens' vector maps indices to actual tokens
    std::vector<size_t> context;
    
    for (size_t i = 0; i < accessPattern.size(); i++) {
        if (i % (accessPattern.size() / 10) == 0) {
            std::cout << "Embedding prefetch access: " << (i * 100 / accessPattern.size()) << "% complete" << std::endl;
        }
        
        // Update context
        if (context.size() >= NGRAM_ORDER - 1) {
            context.erase(context.begin());
        }
        context.push_back(accessPattern[i]);
        
        // Predict next word using n-gram model
        size_t predicted_next = predictNextWord(ngramModels, context);
        
        // Prefetch the predicted next row if valid
        if (predicted_next < matrix.size()) {
            const char* next_row = reinterpret_cast<const char*>(&matrix[predicted_next][0]);
            _mm_prefetch(next_row, _MM_HINT_T0);
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
    std::vector<size_t> tokens; // To store the sequence of tokens (indices)
    std::ifstream input_file(INPUT_PATH);
    std::string word;
    
    while (input_file >> word) {
        if (word_to_idx.find(word) != word_to_idx.end()) {
            size_t token_idx = word_to_idx[word];
            accessPattern.push_back(token_idx);
            tokens.push_back(token_idx);
        }
    }

    if (accessPattern.empty()) {
        std::cerr << "No valid words found in input file!" << std::endl;
        return 1;
    }
    
    // Build the n-gram model
    std::cout << "Building " << NGRAM_ORDER << "-gram model..." << std::endl;
    std::vector<NGram> ngramModels(NGRAM_ORDER);
    buildKGramModels(ngramModels, tokens, NGRAM_ORDER);

    // Calculate prediction accuracy
    size_t correct_predictions = 0;
    size_t total_predictions = 0;
    std::vector<size_t> context;

    for (size_t i = 0; i < accessPattern.size() - 1; i++) {
        if (context.size() >= NGRAM_ORDER - 1) {
            context.erase(context.begin());
        }
        context.push_back(accessPattern[i]);
        
        if (context.size() == NGRAM_ORDER - 1) {
            size_t predicted = predictNextWord(ngramModels, context);
            size_t actual = accessPattern[i + 1];
            total_predictions++;
            if (predicted == actual) {
                correct_predictions++;
            }
        }
    }

    double accuracy = total_predictions > 0 ? 
        (static_cast<double>(correct_predictions) / total_predictions) * 100.0 : 0.0;
    std::cout << "N-gram prediction accuracy: " << accuracy << "%" << std::endl;
    
    // Test regular access
    std::cout << "Testing regular access..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    double result1 = regularAccess(matrix, accessPattern);
    auto end = std::chrono::steady_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    // Test embedding-based prefetch access
    std::cout << "Testing ngram prefetch access..." << std::endl;
    start = std::chrono::steady_clock::now();
    double result2 = ngram_prefetch(matrix, accessPattern, ngramModels, tokens);
    end = std::chrono::steady_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    // Print results
    std::cout << "\nResults:" << std::endl;
    std::cout << "Regular access time: " << duration1 / 1e6 << "ms" << std::endl;
    std::cout << "Embedding prefetch access time: " << duration2 / 1e6 << "ms" << std::endl;
    
    // Fix division by zero error when duration2 is 0
    double speedup = (duration2 > 0) ? static_cast<double>(duration1) / duration2 : 0.0;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    // Print results to verify correctness
    std::cout << "Results match: " << (std::abs(result1 - result2) < 1e-10) << std::endl;
    
    return 0;
} 