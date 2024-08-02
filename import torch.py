from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Accessing the weights of the first layer
first_layer_weights = model.roberta.embeddings.word_embeddings.weight

print("First layer weights:", first_layer_weights)
