import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load your Malayalam language hate speech dataset
# Replace 'your_dataset_path' with the actual path to your dataset
dataset_path = 'test.csv'
df = pd.read_csv(dataset_path, encoding='utf-8')

# Assuming your dataset has 'text' as the column containing text data and 'label' as the column containing labels
X = df['text']
y = df['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text using CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_count, y_train)

# Function to classify text using Random Forest
def classify_text_rf(text):
    text_vectorized = vectorizer.transform([text])
    prediction = rf_model.predict(text_vectorized)[0]
    probabilities = rf_model.predict_proba(text_vectorized)[0]

    return prediction, probabilities

# Function to check hate speech
def check_hate_speech_rf(text):
    # Classify the text using Random Forest
    predicted_label, probabilities = classify_text_rf(text)

    # Interpret the results
    result = {
        'label': predicted_label,
        'probabilities': {class_label: round(prob, 3) for class_label, prob in zip(rf_model.classes_, probabilities)}
    }

    return result

# User input
user_input = input("Enter a Malayalam word or sentence: ")

# Perform hate speech classification using Random Forest
result = check_hate_speech_rf(user_input)

# Display the result
print(f"Result: {result['label']}")
print("Probabilities:")
for class_label, prob in result['probabilities'].items():
    print(f"{class_label}: {prob}")
