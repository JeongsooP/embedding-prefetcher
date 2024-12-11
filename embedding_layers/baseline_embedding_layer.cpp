#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <numeric> // For std::accumulate
#include <cmath> // For std::sqrt

// Global constants
const std::string GLOVE_PATH = "data/glove.twitter.27B.25d.txt";
const std::string INPUT_PATH = "data/input.txt";
const size_t NUM_COLS = 25;        // GloVe embedding dimension

// Function to perform row operations without prefetching
double regularAccess(const std::vector<std::vector<double>>& matrix, 
                    const std::vector<size_t>& accessPattern) {
    double result = 0.0;
    
    for (size_t i = 0; i < accessPattern.size(); i++) {
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

    // Perform regular access
    std::cout << "Performing regular access..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    double result = regularAccess(matrix, accessPattern);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Regular access result: " << result << std::endl;
    std::cout << "Time taken: " << duration / 1e6 << " ms" << std::endl;

    return 0;
} 