from transformers import pipeline

model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
zs_text_classifier = pipeline(model=model_name)

candidate_labels = [
    "Billing Issues",
    "Technical Support",
    "Account Information",
    "General Inquiry",
]

hypothesis_template = "This text is about {}"

customer_text = "My account was charged twice for a single order."
print(
    zs_text_classifier(
        customer_text,
        candidate_labels,
        hypothesis_template=hypothesis_template,
        multi_label=True,
    )
)
