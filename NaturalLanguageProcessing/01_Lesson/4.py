from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["running", "runner", "studies", "studying"]

for word in words:
    print(word, "->", stemmer.stem(word))