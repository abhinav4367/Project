import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer for Malayalam
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Assuming binary classification (hate speech or not)

# Function to classify text
def classify_text(text):
    # Tokenize the input text
    input_ids = tokenizer.encode(text, truncation=True, padding=True, return_tensors='pt')

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Get prediction probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]

    # Get the predicted label
    predicted_label = torch.argmax(logits, dim=1).item()

    return predicted_label, probabilities

# Function to check hate speech
def check_hate_speech(text):
    # Classify the text
    predicted_label, probabilities = classify_text(text)

    # Interpret the results
    labels = ['Not Hate Speech', 'Hate Speech']
    result = {
        'label': labels[predicted_label],
        'probabilities': {labels[i]: round(prob, 3) for i, prob in enumerate(probabilities)}
    }

    return result

# User input
user_input = input("Enter a Malayalam word or sentence: ")

# Perform hate speech classification
result = check_hate_speech(user_input)

# Display the result
print(f"Result: {result['label']}")
print("Probabilities:")
for label, prob in result['probabilities'].items():
    print(f"{label}: {prob}")
