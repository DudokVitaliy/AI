from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch

# Ініціалізація класифікатора правильним способом
classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=0 if torch.cuda.is_available() else -1
)

texts = [
    "Це чудовий фільм!",
    "Мені не подобається цей продукт.",
    "Цілком нормально, нічого особливого."
]

# Обробка текстів разом (більш ефективно)
results = classifier(texts)

# Виведення результатів
print("=" * 60)
for text, result in zip(texts, results):
    label = result['label']  # Тональність (POSITIVE, NEGATIVE, NEUTRAL)
    score = result['score']  # Впевненість (0-1)

    print(f"Текст: {text}")
    print(f"Тональність: {label}")
    print(f"Впевненість: {score:.2%}")
    print("-" * 60)