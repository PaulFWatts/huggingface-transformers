from transformers import pipeline

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_classifier = pipeline(model=model_name)

text_input = "I'm really excited about using HuggingFace to run AI models!"
print(sentiment_classifier(text_input))

text_input = "I'm having a horrible day today."
print(sentiment_classifier(text_input))

text_input = "Most of the Earth is covered in water."
print(sentiment_classifier(text_input))

text_inputs = [
    "What a great time to be alive!",
    "How are you doing today?",
    "I'm in a horrible mood.",
]

print(sentiment_classifier(text_inputs))