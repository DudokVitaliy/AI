import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

text = ["Ukraine's glory has not yet perished, nor her freedom,Upon us, young brothers, fate shall yet smile.Our enemies will vanish, like dew in the sun,And we, too, brothers, shall live happily in our land."]

words = word_tokenize(text)

stop_words = set(stopwords.words('english'))

filtered_words = [w for w in words if not w in stop_words]