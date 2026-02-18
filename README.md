

# 🤖 Question–Answer Prediction System (RNN-Based)

**Duration:** September 2025 – October 2025
**Tech Stack:** Python, PyTorch, NumPy, pandas

---

## 📌 Project Overview

This project implements a **Recurrent Neural Network (RNN)** for predicting answers to questions using a custom QA dataset. The system demonstrates a complete pipeline—from text preprocessing to model training and answer prediction—designed to handle unseen questions gracefully.

The goal was to create an end-to-end **question-answering system** capable of learning patterns from a small dataset and providing reliable predictions with a confidence-based mechanism.

---

## 🎯 Objectives

* Build an RNN-based QA prediction model
* Preprocess text data for clean and structured inputs
* Develop a SimpleRNN architecture with embedding and fully connected layers
* Integrate a **threshold-based confidence mechanism** to improve answer reliability
* Evaluate model performance on a custom QA dataset

---

## 🧠 Model Architecture

* **Architecture:** SimpleRNN with Embedding + RNN + Fully Connected Layer
* **Input:** Tokenized question sequences
* **Output:** Predicted answer token
* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** Adam
* **Training:** 20 epochs
* **Achieved Training Loss:** 1.72

---

## ⚙️ Implementation Details

### 1️⃣ Data Preprocessing

* Lowercased all text and removed punctuation
* Tokenized questions and answers into word sequences
* Built a vocabulary dictionary assigning unique indices to each token
* Converted text to numerical indices for model input

### 2️⃣ Dataset & DataLoader

* Created a custom PyTorch `Dataset` class (`QADataset`)
* Used `DataLoader` for batching question-answer pairs

### 3️⃣ RNN Model

* **Embedding Layer:** Converts each token index into a 50-dimensional vector
* **RNN Layer:** Processes sequences to capture temporal dependencies
* **Fully Connected Layer:** Maps final hidden state to vocabulary space for predicted answer

### 4️⃣ Training Pipeline

* Forward pass → loss computation → backpropagation → parameter update
* Iterated over 20 epochs, achieving a stable training loss

### 5️⃣ Answer Prediction

* Predicted answers for given questions using trained model
* Implemented **threshold mechanism**:

  * If the model is not confident (probability < 0.5), it outputs *“I don’t know”*
  * Improves reliability when encountering unseen questions

---

## 🚀 Example Predictions

| Question                                           | Model Prediction |
| -------------------------------------------------- | ---------------- |
| What is the capital of France?                     | Paris            |
| Which month has 28 days in a common year?          | February         |
| Who was the first female Prime Minister of the UK? | Thatcher         |
| Which superhero is also known as the Dark Knight?  | Batman           |
| Who wrote *To Kill a Mockingbird*?                 | Harper Lee       |
| Who developed the theory of relativity?            | Einstein         |
| Which city is known as the Big Apple?              | New York         |

*Unseen or low-confidence questions are answered with “I don’t know.”*

---

## 📊 Key Learnings

* Text preprocessing is crucial for small datasets
* Embedding layers effectively represent tokens in vector space
* Simple RNNs can learn sequence dependencies for QA tasks
* Confidence thresholds improve user experience and reduce unreliable predictions
* End-to-end QA systems require careful design of both model and pipeline

