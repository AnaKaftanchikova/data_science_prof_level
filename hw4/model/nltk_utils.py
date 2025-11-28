import nltk
import numpy as np
import string

# при первом запуске раскомментируй:
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return word_tokenize(sentence.lower())

def lemmatize(word):
    return lemmatizer.lemmatize(word)

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [lemmatize(w) for w in tokenized_sentence if w not in string.punctuation]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
