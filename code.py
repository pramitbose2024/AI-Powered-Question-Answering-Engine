# Load the Dataset
# Import the pandas library and load the CSV dataset into a DataFrame
import pandas as pd
df = pd.read_csv('/content/100_Unique_QA_Dataset.csv')

# Tokenize the dataset
# Tokenization means splitting each sentence in our dataset into words based on spaces.

def tokenize(text):
  text = text.lower() # convert all characters in the dataset to lowercase
  text = text.replace('?', '') # Remove all question marks from the text of our dataset
  text = text.replace("'", "") # Remove all single quotes (') from the text
  return text.split() # Split the text into words based on spaces and return the list of words.

# Call the tokenize function on a sample sentence to see the tokenized output
tokenize('What is the capital of France?') 

# Build our vocabulary
# Create a vocabulary (to find how many unique words are there in the dataset and assign an index to each unique word)
# Initialize a vocabulary dictionary to store unique words and their corresponding indices

vocab = {'<UNK>' : 0} # Initialize the vocabulary with a special token '<UNK>' for unknown words, assigned index 0

# Build Vocabulary from Dataset by Tokenizing Questions and Answers
def build_vocab(row):
  print(row['question'], row['answer'])

  # pick each question & answer and tokenize it
  tokenized_question = tokenize(row['question'])
  tokenized_answer = tokenize(row['answer'])

  # Merge the tokenized question and answer into a single list of tokens
  merged_tokens = tokenized_question + tokenized_answer

  # run a loop over the list
  for token in merged_tokens:

    # Check each token:
    # If it's not in our vocabulary, add it
    # Give a unique index to the new token based on the current vocabulary size
    if token not in vocab:
      vocab[token] = len(vocab)

df.apply(build_vocab, axis=1) # Apply the build_vocab function to each row of the DataFrame

# Display the list of words (tokens) and the index assigned to each
vocab

# Get the total count of unique words in the vocabulary
len(vocab) # our vocabulary dictionary has 324 unique words (tokens)

# Convert the dataset into numbers by changing each English word into a numeric value

# convert words to numerical indices
# Map each word in the dataset to its corresponding index in the vocabulary

# Function that tokenizes the input text and maps each word to its corresponding index in the vocabulary dictionary (vocab), replacing unknown words with '<UNK>'
def text_to_indices(text, vocab):

  indexed_text = [] # to store the indexed version of the text

  # Call the tokenize function on the text
  # This returns a list of tokens, which we will iterate over
  for token in tokenize(text):

    # For each word (token) in every sentence:
    # - Check if the word exists in the vocabulary dictionary.
    # - If it does, retrieve its corresponding index.
    # - Append that index to the 'indexed_text' list.
    if token in vocab:
      indexed_text.append(vocab[token])
    # If the word is not found in the vocabulary dictionary:
    # - Replace it with the index of the unknown token ('<UNK>').
    else:
      indexed_text.append(vocab['<UNK>']) # <UNK> → unknown token

  # Return the final list containing the indices of all tokens in the text
  return indexed_text

# Convert the sample text "What is Pramit" into a list of numerical indices using the vocabulary
text_to_indices("What is Pramit", vocab)

# Custom PyTorch Dataset Class for Question-Answer Pairs

import torch
from torch.utils.data import Dataset, DataLoader

# our class QADataset will inherit from the 'Dataset' class
class QADataset(Dataset):

  # Constructor method to initialize a new instance of the class
  def __init__(self, df, vocab):
    # Store the input DataFrame and vocabulary as instance attributes
    self.df = df
    self.vocab = vocab

  # Returns the number of elements/items in the dataset
  def __len__(self):
    return self.df.shape[0] # returns the number of rows in the dataset

  # Retrieves the item at the specified key or index from the object
  def __getitem__(self, index):
    numerical_question = text_to_indices(self.df.iloc[index]['question'], self.vocab) # to access the question of the given index
    numerical_answer = text_to_indices(self.df.iloc[index]['answer'], self.vocab) # to access the answer of the given index

    # convert the lists to tensor
    # Return the question and answer as tensors
    return torch.tensor(numerical_question), torch.tensor(numerical_answer)
  
# create the dataset object
# 'dataset' is an object of 'QADataset' class
dataset = QADataset(df, vocab)

dataset[0] # 1st question & answer as numerical indexes

# create the dataloader object
# 'dataloader' is an object of the 'DataLoader' class
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Create a DataLoader to iterate over the dataset with batches of 1 (question-answer pair) in random order
                                                              # each batch contains a pair of question & answer

# Iterate through the DataLoader and print each question-answer pair 
for question, answer in dataloader:
  print(question, answer[0])

# Design the architecture of our RNN (Recurrent Neural Network)

import torch.nn as nn

# the class 'SimpleRNN' will inherit from 'nn.Module' class
class SimpleRNN(nn.Module):

  # constructor
  def __init__(self, vocab_size): # constructor only need the size of our vocabulary (the number of words that are present in our vocabulary)

    # call the constructor of the parent class
    super().__init__()

    # embadding layer
    self.embedding = nn.Embedding(vocab_size, embedding_dim=50) # embedding_dim -> dimension of our embedding vector

    # RNN layer
    # Defines a simple RNN with input layer size 50 (from the embedding layer) and hidden layer size 64 (neurons)
    self.rnn = nn.RNN(50, 64, batch_first=True) # When batch_first=True, the first dimension of input tensors represents the batch size

    # output layer
    self.fc = nn.Linear(64, vocab_size) # output layer: Input -> 64 conncettion for the hidden layer,Output -> size of the vocabulary (324)

  # forward pass
  def forward(self, question):
    embedded_question = self.embedding(question) # send the question ot the embedding layer to get the embedding og each word in the question
    hidden, final = self.rnn(embedded_question) # Pass the embedded question through the RNN to get hidden states and the final output
    output = self.fc(final.squeeze(0)) # Pass the final hidden state through the fully connected layer to get the output (answer of the question)
                                       # squeeze(0) -> removes the 0th (first) dimension from tensor 'final' | shape changes from torch.Size([1, 1, 64]) to torch.Size([1, 64])

    return output # return the output (answer of the question)

# RNN-Based Question-to-Answer Prediction Pipeline
x = nn.Embedding(324, embedding_dim=50) # embedding layer that maps 324 unique tokens (words) to 50-dimensional vectors
y = nn.RNN(50, 64, batch_first=True) # RNN layer with input size 50 and hidden size 64 | # When batch_first=True, the first dimension of input tensors represents the batch size
z = nn.Linear(64, 324) # fully connected layer mapping from 64 → 324 dimensions

# Extract the first question from our question–answer pair dataset and reshape it to (1, 6) | (1, 6) -> 1 question with 6 words
a = dataset[0][0].reshape(1, 6)
print("shape of a: ", a.shape)

# Pass the input 'a' through the embedding layer to get word embeddings
b = x(a)
print("shape of b: ", b.shape)

# Pass the embeddings through the RNN to get output (c) and final hidden state (d)
c, d = y(b)
print("shape of c: ", c.shape)
print("shape of d: ", d.shape)

# Pass the final hidden state 'd' through the linear layer to predict the answer
e = z(d.squeeze(0)) # d.squeeze(0) -> removes the 0th (first) dimension from tensor d | shape changes from torch.Size([1, 1, 64]) to torch.Size([1, 64])
print("shape of e: ", e.shape) # We obtained the desired tensor shape: [1, 324]

# Set important variables

learning_rate = 0.001
epochs = 20

# Model initialization

# Create an instance of the SimpleRNN model, passing the vocabulary size as input dimension
model = SimpleRNN(len(vocab))

# Set Loss & Optimizer

criterion = nn.CrossEntropyLoss() # Define the loss function as Cross-Entropy Loss for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Initialize the Adam optimizer to update the model's parameters with the specified learning rate (0.001)

# Training Loop

# training loop

# Loop over the number of training epochs
for epoch in range(epochs):

  # to measure the loss
  total_loss = 0

  # Iterate over dataloader
  for question, answer in dataloader: # at a time, we will process one batch of question-answer pairs

    # clear gradients
    optimizer.zero_grad()

    # forward pass
    output = model(question) # pass the question to the model & we will get our output (prediction)

    # Loss calculate
    # Compute the loss based on the model's predictions (the answer we got) and true labels (the corrcet answer in the dataset)
    loss = criterion(output, answer[0]) # compute the loss between predicted output and true answer

    # Perform backpropagation to calculate gradients
    loss.backward()

    # Apply the optimizer to update the model's weights and biases
    optimizer.step()

    # update loss
    # Add the loss for this batch
    total_loss = total_loss + loss.item() # total_loss -> total loss so far & loss.item() -> current loss value

  # print the total loss
  print(f'Epoch: {epoch+1}, Loss: {total_loss}') 

# Perform the predictions

# How the prediction works:
# If the model recognizes the question from its training data and knows the answer, it will provide the answer.
# If the model encounters a question it has never seen before, it will respond with "I don't know."

# Function to Predict Answers for a Given Question Using the Model

# Define a function to predict the answer for a given question using the model
def predict(model, question, threshold=0.5):

  # Convert the input question into a sequence of numerical indices using the vocabulary with 'text_to_indices' function
  numerical_question = text_to_indices(question, vocab) 

  # Convert numerical_question into a PyTorch tensor and use unsqueeze(0) to add an extra dimension for batch processing
  question_tensor = torch.tensor(numerical_question).unsqueeze(0) # unsqueeze(0) -> adds a new dimension at the 0-th position (batch dimension) | Example: if numerical_question.shape = (3,), after unsqueeze(0) -> shape = (1, 3)

  # Pass the question tensor into the model to generate the predicted answer
  output = model(question_tensor)

  # Convert the model's output logits into probabilities using softmax
  probabilities = torch.nn.functional.softmax(output, dim=1)

  # Find both the highest probability value and its corresponding index from the 'probabilities' tensor
  value, index = torch.max(probabilities, dim=1)

  # Compare the maximum probability value with the threshold (0.5)
  if value < threshold: # Check if the highest probability value is less than the threshold
    print("I don't know")

  # Check if the highest probability value is greater than the threshold (0.5)
  print(list(vocab.keys())[index]) # Access the 'vocab' dictionary, get all keys as a list, retrieve the key at the specified index, and print the answer

# Use the model to predict answers for the given questions

# Ask the model a question and get its answer
predict(model, "what is the capital of france")

# Ask the model a question and get its answer
predict(model, "Which month has 28 days in a common year?")

# Ask the model a question and get its answer
predict(model, "Who was the first female Prime Minister of the UK?") 

# Ask the model a question and get its answer
predict(model, "Which superhero is also known as the Dark Knight?") 

# Ask the model a question and get its answer
predict(model, "Who directed the movie 'Titanic'?") 

# Ask the model a question and get its answer
predict(model, "Who wrote 'To Kill a Mockingbird'?") 

# Ask the model a question and get its answer
predict(model, "Who developed the theory of relativity?") 

# Ask the model a question and get its answer
predict(model, "Who is the author of '1984'?") 

# Ask the model a question and get its answer
predict(model, "Which city is known as the Big Apple?") 

# Ask the model a question and get its answer
predict(model, "What is the longest-running animated TV show?") 

# Ask the model a question and get its answer
predict(model, "Who wrote 'Romeo and Juliet'?") 