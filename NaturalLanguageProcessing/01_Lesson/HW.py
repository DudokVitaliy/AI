import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from nltk.util import bigrams

nltk.download('punkt')
nltk.download('stopwords')

#N1
df = pd.read_csv("feedback.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['cleaned'] = df['review'].apply(clean_text)

num_rows = len(df)
avg_length = df['cleaned'].apply(lambda x: len(x.split())).mean()

print(f"Кількість рядків: {num_rows}")
print(f"Середня довжина відгуку: {avg_length:.2f} слів")

#N2
stop_words = set(stopwords.words('english'))

all_tokens = []
clean_tokens = []

for text in df['cleaned']:
    tokens = word_tokenize(text)
    all_tokens.extend(tokens)

    filtered = [
        word for word in tokens
        if word not in stop_words and len(word) >= 3
    ]
    clean_tokens.extend(filtered)

print(f"Токенів до очищення: {len(all_tokens)}")
print(f"Токенів після очищення: {len(clean_tokens)}")

#N3
word_freq = Counter(clean_tokens)
top_words = word_freq.most_common(15)

print("\nТОП-15 слів:")
for word, freq in top_words:
    print(word, freq)

words = [w[0] for w in top_words]
freqs = [w[1] for w in top_words]

plt.figure()
plt.barh(words, freqs)
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.title("Top 15 Frequent Words")
plt.savefig("feedback_word_freq.png")
plt.show()

#N4
bigram_list = list(bigrams(clean_tokens))
bigram_freq = Counter(bigram_list)

top_bigrams = bigram_freq.most_common(10)

bigram_df = pd.DataFrame(
    [(f"{w1} {w2}", freq) for (w1, w2), freq in top_bigrams],
    columns=["Bigram", "Frequency"]
)

print("\nТОП-10 біграм:")
print(bigram_df)

bigram_df.to_csv("feedback_bigrams.csv", index=False)