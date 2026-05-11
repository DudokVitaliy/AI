from nltk.corpus import gutenberg
import nltk
nltk.download('gutenberg')

print (gutenberg.fileids())

text = gutenberg.raw("austen-persuasion.txt")

print (text[:500])