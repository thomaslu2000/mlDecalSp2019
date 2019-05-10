import numpy as np
# import nltk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.stem.snowball import SnowballStemmer

# nltk.download('punkt')
stemmer = SnowballStemmer("english")


def select_one(big_list, special_idx):
    total = []
    for i in range(len(big_list)):
        if i != special_idx:
            total.extend(big_list[i])
    return np.array(total), np.array(big_list[special_idx])


def split_array(arr, pieces):
    result = []
    step = int(len(arr) / pieces)
    for i in range(0, len(arr), step):
        result.append(arr[i:i+step])
    return result


def process_data(data, length):
    processed = np.zeros(len(data) * length).reshape((len(data), length))
    for i, seq in enumerate(data):
        if len(seq) < length:
            processed[i] = np.pad(seq, (0, length-len(seq)), 'constant', constant_values=0)
        else:
            processed[i] = np.array(seq)[:length]
    return processed


def make_word_to_int_mapping(texts, dict_size):
    i = 1
    word_map = {}
    for text in texts:
        for word in word_tokenize(text):
            word = stemmer.stem(word.lower())
            if word not in word_map and word not in punctuation:
                word_map[word] = i
                i += 1
            if i == dict_size:
                return word_map
    return word_map


def text_to_ints(texts, word_map):
    int_texts = []
    for text in texts:
        words = []
        for word in text:
            word = stemmer.stem(word.lower())
            words.append(word_map[word] if word in word_map else 0)
        int_texts.append(np.array(words))
    return np.array(int_texts)


def tokenize_text(texts):
    return np.array([word_tokenize(text) for text in texts])


def extract(reviews, rated):
    def valid_review(r):
        return len(r['title']) > 0 and len(r['review_text']) > 0
    reviews = [r for r in reviews if valid_review(r)]
    titles = [r['title'].lower() for r in reviews]
    texts = [r['review_text'].lower() for r in reviews]
    if rated:
        ratings = np.array([int(float(r['rating']) > 3) for r in reviews])
        return ratings, titles, texts
    else:
        return titles, texts
