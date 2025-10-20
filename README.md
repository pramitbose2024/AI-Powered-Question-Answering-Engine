Question-Answer Prediction System

Duration: September 2025 - October 2025
Technologies: Python, PyTorch, pandas, NumPy

Overview:
This project implements a question-answer prediction system using a custom dataset of 100 unique question-answer pairs. The system leverages a SimpleRNN model to map questions to their corresponding answers, handling unknown words and providing probabilistic predictions with confidence thresholds.

Features:
1. Preprocesses text data with tokenization, lowercasing, and vocabulary building.
2. Converts questions and answers into numerical indices for model input
3. Implements a SimpleRNN architecture with embedding and fully connected layers.
4. Provides probabilistic predictions and a threshold-based response mechanism, returning “I don’t know” for low-confidence queries.
5. Achieved a training loss of 11.03 over 20 epochs.
