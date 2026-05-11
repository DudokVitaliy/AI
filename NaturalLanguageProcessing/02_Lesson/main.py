import re
from collections import Counter

import torch
import torch.nn as nn

from transformers import pipeline

from support_data import pairs

# Завдання 1

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text


clean_pairs = []

for q, a in pairs:

    q = clean_text(q)

    a = clean_text(a)

    clean_pairs.append((q, a))


tokenized = []

for q, a in clean_pairs:

    tokenized.append(q.split())

    tokenized.append(a.split())


vocab = Counter()

for sentence in tokenized:

    vocab.update(sentence)


token2idx = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3
}


for word in vocab:

    token2idx[word] = len(token2idx)


idx2token = {v: k for k, v in token2idx.items()}


def encode(sentence):

    tokens = sentence.split()

    ids = [
        token2idx.get(token, token2idx["<UNK>"])
        for token in tokens
    ]

    ids = [token2idx["<SOS>"]] + ids + [token2idx["<EOS>"]]

    return ids


encoded_pairs = []

for q, a in clean_pairs:

    encoded_pairs.append((encode(q), encode(a)))


max_len = max(
    max(len(q), len(a))
    for q, a in encoded_pairs
)


def pad(sequence, max_len):

    return sequence + [0] * (max_len - len(sequence))


dataset = []

for q, a in encoded_pairs:

    dataset.append((
        pad(q, max_len),
        pad(a, max_len)
    ))


X = []
Y = []

for q, a in dataset:

    X.append(q)

    Y.append(a)


X = torch.tensor(X)

Y = torch.tensor(Y)

# Завдання 2

class SupportLSTM(nn.Module):

    def __init__(self, vocab_size,
                 embed_size=64,
                 hidden_size=128):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        output, _ = self.lstm(x)

        output = self.fc(output)

        return output


model = SupportLSTM(len(token2idx))


criterion = nn.CrossEntropyLoss(
    ignore_index=0
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 15

print("\nTRAINING LSTM MODEL...\n")

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(
        outputs.view(-1, len(token2idx)),
        Y.view(-1)
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {loss.item():.4f}"
    )

# Завдання 3

def generate_answer(question):

    question = clean_text(question)

    encoded = encode(question)

    encoded = pad(encoded, max_len)

    x = torch.tensor([encoded])

    with torch.no_grad():

        output = model(x)

    predicted = output.argmax(dim=-1)

    words = []

    for idx in predicted[0]:

        word = idx2token[idx.item()]

        if word in ["<PAD>", "<SOS>", "<EOS>"]:

            continue

        words.append(word)

    return " ".join(words)


tests = [
    "How can I reset my password?",
    "My order is delayed",
    "How do I contact support?",
    "Payment failed",
    "Can I return an item?"
]


print("\nLSTM RESPONSES:\n")

for test in tests:

    response = generate_answer(test)

    print("QUESTION:", test)

    print("ANSWER:", response)

    print("-" * 50)


torch.save(
    model.state_dict(),
    "supportflow_lstm.pth"
)

print("\nLSTM model saved!")

# Завдання 4

print("\nLOADING DISTILGPT2...\n")

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)


def gpt_answer(prompt):

    result = generator(
        prompt,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        truncation=True,
        pad_token_id=50256
    )

    return result[0]["generated_text"]


print("\nTRANSFORMER RESPONSES:\n")

for test in tests:

    response = gpt_answer(test)

    print("QUESTION:", test)

    print("GPT2 ANSWER:", response)

    print("-" * 50)

print("\nMODEL COMPARISON\n")

print("LSTM:")
print("- Faster")
print("- Simpler architecture")
print("- Weaker context understanding")
print("- Sometimes repetitive")

print("\nTransformer (DistilGPT2):")
print("- Better text quality")
print("- Better context understanding")
print("- More natural responses")
print("- Slower generation")

print("\nDONE.")