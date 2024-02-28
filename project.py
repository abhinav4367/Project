import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import torch

# Load your labeled dataset for Manglish hate speech detection
# Replace 'your_dataset.csv' with your actual dataset file
data = pd.read_csv('your_dataset.csv')

# Assuming your dataset has 'text' column for input and 'label' column for the target variable
X = data['text']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

X_train_tokens = []
X_test_tokens = []

for text in tqdm(X_train, desc="Tokenizing training data"):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    X_train_tokens.append(tokens)

for text in tqdm(X_test, desc="Tokenizing testing data"):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    X_test_tokens.append(tokens)

# Convert tokens to tensors
X_train_tensors = torch.LongTensor(X_train_tokens)
X_test_tensors = torch.LongTensor(X_test_tokens)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Extract BERT embeddings for training data
with torch.no_grad():
    train_outputs = model(X_train_tensors)

# Extract BERT embeddings for testing data
with torch.no_grad():
    test_outputs = model(X_test_tensors)

# Use BERT embeddings as features for Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(train_outputs['last_hidden_state'][:, 0, :].numpy(), y_train)

# Make predictions on the test set
predictions = nb_classifier.predict(test_outputs['last_hidden_state'][:, 0, :].numpy())

# Evaluate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
