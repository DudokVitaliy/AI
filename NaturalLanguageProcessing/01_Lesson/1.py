import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import  word_tokenize, sent_tokenize

text = "Привіт! Як справи? NLTK це класно."

sentences = sent_tokenize(text)
print(sentences)

words = word_tokenize(text)
print(words)