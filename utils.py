import torch
import os
import numpy as np
from csv import reader
from collections import Counter
from nltk.tokenize import word_tokenize

def one_hot_encode(label_list, nClass):
    return np.eye(nClass)[label_list]

def n_gram(text, ngram):
    unigram = text.split(" ")
    ngrams = list(zip(*[unigram[i:] for i in range(ngram)]))
    ngrams = [' '.join(grams) for grams in ngrams]
    unigram.extend(ngrams)
    return unigram

def get_data(path):
    word2idx = {}
    idx2word = {}
    idx = 0
    
    category = []
    text = []
    with open(path, 'r') as f:
        for line in reader(f):
            line[1] = line[1].replace("\\"," ")
            line[2] = line[2].replace("\\"," ")
            cls = int(line[0])-1
            title = line[1].lower()
            description = line[2].lower()
            category.append(cls)
            text.append(title + " " + description)

    for content in text:
        bigram = n_gram(content, ngram=2)
        for word in bigram:
            if not word in word2idx:
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1

    return word2idx, category, text


