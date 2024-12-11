#!/bin/bash

# Download GloVe word embeddings trained on Twitter data
echo "Downloading GloVe Twitter word embeddings..."
wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip

# Download Sherlock Holmes text as input data
echo "Downloading Sherlock Holmes text for testing..."
wget -O input.txt https://assets.datacamp.com/production/repositories/3937/datasets/213ca262bf6af12428d42842848464565f3d5504/sherlock.txt

unzip glove.twitter.27B.zip

echo "Downloads complete!"