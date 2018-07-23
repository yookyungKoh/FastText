import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

import os
import argparse

from utils import *

# model parameters
#parser = argparse.ArgumentParser(description='Bag of Tricks for Efficient Text Classification')
#
#parser.add_argument('--data_path', type=str, default='./data/ag_news_csv/', help='data path') 
#parser.add_argument('--hidden_dim', type=int, default=10, help='hidden dimension') 
#parser.add_argument('--num_class', type=int, default=4, help='number of classes') 
#parser.add_argument('--batch_size', type=int, default=20, help='batch size') 
#parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate') 
#parser.add_argument('--num_epochs', type=int, default=5, help='total number of epochs') 
#parser.add_argument('--hash_size', type=int, default=1e9, help='size of hash array') 
#
#args = parser.parse_args()
# -------------------------------------------------- #

data_path = './data/ag_news_csv/'
hidden_dim = 10
num_class = 4
batch_size = 20
learning_rate = 0.1
num_epochs = 5

class Model(nn.Module):
    ''' create model '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__() 
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.hidden_layer(x)
        out = self.output_layer(out)
        return out

# get data
word2idx, train_target, train_text = get_data(data_path + 'train.csv')
_, test_target, test_text = get_data(data_path + 'test.csv')

# vectorizer
bigram_vectorizer = CountVectorizer(ngram_range=(1,2), min_df=0, tokenizer=word_tokenize, vocabulary=word2idx)

# model
model = Model(len(word2idx), hidden_dim, num_class)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train
iteration = len(train_text) // batch_size
for epoch in range(num_epochs):

    running_loss = 0.
    correct = 0.
    total = 0.
    # get mini-batch inputs and targets
    for i in range(len(train_text), batch_size):
        inputs = bigram_vectorizer.fit_transform(train_text[i:i+batch_size]).toarray()
        targets = one_hot_encode(train_target[i:i+batch_size], num_class).toarray()

        inputs = Variable(torch.from_numpy(inputs))
        targets = Variable(torch.from_numpy(targets))

        print('input shape:',inputs.size)
        print('target shape:',target.size)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # backward
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() / iteration
        total += targets.size(0)
        correct += (torch.max(outputs.data, 1) == torch.max(targets.data, 1)).sum().item()
        accuracy = correct / total * 100

        # log
        print('Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f | Accuracy: %.4f'
        % (epoch+1, num_epochs, i+1, iteration, running_loss, accuracy))

