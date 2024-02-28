from transformers import BertTokenizer, BertForSequenceClassification, pipeline

def fine_tune_hate_speech_detection(train_data):
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
    return model, tokenizer

def predict_hate_speech(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    return predicted_class

def main():
    print("Malayalam Hate Speech Detection")
    print("-------------------------------")

    # Replace this with your fine-tuned model and tokenizer
    fine_tuned_model, fine_tuned_tokenizer = fine_tune_hate_speech_detection("test.csv")

    while True:
        user_input = input("Enter a Malayalam word or sentence (type 'exit' to end): ")

        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break

        predicted_class = predict_hate_speech(user_input, fine_tuned_model, fine_tuned_tokenizer)

        print(f"\nPredicted Class: {predicted_class}\n")

if __name__ == "__main__":
    main()
