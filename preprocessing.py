import re
from nltk.corpus import stopwords
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk



TAG_re = re.compile(r'<[^>]+>')


def remove_tags(text):
    '''Remove HTML tags: replace anything between opening  and closing <> with empty spaces'''
    return TAG_re.sub('', text)


def preprocess_text(sen):
    sentence = sen.lower()
    # Remove html tags
    sentence = remove_tags(sentence)

    # Remove punctuation and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+",  ' ', sentence)

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ',sentence)

    # Remove multiple stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english'))+r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence


def tokenize(X_train, x_test, maxlength):
    # tokenizing
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(X_train)

    X_train = word_tokenizer.texts_to_sequences(X_train)
    x_test = word_tokenizer.texts_to_sequences(x_test)

    # padding
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlength)
    x_test = pad_sequences(x_test, padding='post', maxlen=maxlength)

    vocab_length = len(word_tokenizer.word_index) + 1
    return X_train, x_test, vocab_length,

