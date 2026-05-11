import stanza

stanza.download("uk")

nlp = stanza.Pipeline("uk")
# text = "Привіт друже! Ковбаси, сало. Салоїди."
text = "Ще не вмерла Україна, і слава, і воля,"

doc = nlp(text)

for sent in doc.sentences:
    for word in sent.words:
        print(f"{word.text} -> {word.lemma}")
