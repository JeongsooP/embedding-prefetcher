#ifndef NGRAM_HPP
#define NGRAM_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <iterator>

using namespace std;

// Custom hash function for vector<size_t>
struct VectorHash {
    size_t operator()(const vector<size_t>& vec) const {
        size_t hash = 0;
        for (size_t v : vec) {
            hash ^= std::hash<size_t>()(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

// Define n-gram type using indices with custom hash function
using NGram = unordered_map<vector<size_t>, unordered_map<size_t, int>, VectorHash>;

// Function to build k-gram models using indices
inline void buildKGramModels(vector<NGram> &models, const vector<size_t> &tokens, int n)
{
    for (int k = 1; k <= n; ++k)
    {
        models[k - 1].reserve(tokens.size()); // Reserve space based on expected size
        for (size_t i = 0; i + k <= tokens.size(); ++i)
        {
            vector<size_t> prefix(tokens.begin() + i, tokens.begin() + i + k - 1);
            size_t nextWord = tokens[i + k - 1];
            models[k - 1][prefix][nextWord]++;
        }
    }
}

// Function to predict the next word using backoff with indices
inline size_t predictNextWord(const vector<NGram> &models, const vector<size_t> &queryTokens)
{
    for (int k = models.size(); k > 0; --k)
    {
        if (queryTokens.size() < k - 1)
            continue; // Skip if not enough tokens
        vector<size_t> prefix(queryTokens.end() - (k - 1), queryTokens.end());
        const auto &model = models[k - 1];
        auto it = model.find(prefix);
        if (it != model.end())
        {
            const auto &nextWords = it->second;
            return max_element(nextWords.begin(), nextWords.end(),
                               [](const pair<size_t, int> &a, const pair<size_t, int> &b)
                               {
                                   return a.second < b.second;
                               })
                ->first;
        }
    }
    return 0; // Return a default index if no prediction is possible
}

#endif // NGRAM_HPP