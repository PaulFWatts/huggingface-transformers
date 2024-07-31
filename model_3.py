from transformers import pipeline

image_classifier = pipeline(task="image-classification")

preds = image_classifier(["llamas.png"])
print(len(preds[0]))

print(preds[0][0])

print(preds[0][1])

print(preds[0][2])
