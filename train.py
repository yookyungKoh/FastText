import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import os
import argparse
import time

from utils import *

# model parameters
#parser = argparse.ArgumentParser(description='Bag of Tricks for Efficient Text Classification')
#
#parser.add_argument('--data_path', type=str, default='./data/ag_news_csv/', help='data path') 
#parser.add_argument('--hidden_dim', type=int, default=10, help='hidden dimension') 
#parser.add_argument('--num_class', type=int, default=4, help='number of classes') 
#parser.add_argument('--batch_size', type=int, default=100, help='batch size') 
#parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate') 
#parser.add_argument('--num_epochs', type=int, default=5, help='total number of epochs') 
#parser.add_argument('--hash_size', type=int, default=1e9, help='size of hash array') 
#
#args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = './data/ag_news_csv/'
hidden_dim = 10
num_class = 4
batch_size = 20
learning_rate = 0.05 # {0.05, 0.1, 0.25, 0.5}
num_epochs = 5

# get data
word2idx, train_target, train_text = get_data(data_path + 'train.csv')
vocab_size = len(word2idx)
print('vocab size:', vocab_size)

class Model(nn.Module):
    ''' create model '''
    def __init__(self, vocab_size, hidden_dim, output_dim):
        super(Model, self).__init__()
#        self.lookup = Variable(torch.randn(vocab_size, hidden_dim)).to(device)
        self.hidden_layer = nn.Linear(vocab_size, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, inputs):
        # inputs: BOW tensor (N, V)
#        out = torch.matmul(inputs, self.lookup)
#        print('matmul out size:',out.size())
        out = self.hidden_layer(inputs)
        out = self.output_layer(out) # (N, output_dim)

        return out

# model
model = Model(len(word2idx), hidden_dim, num_class).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# vectorizer
bigram_vectorizer = CountVectorizer(ngram_range=(1,2), min_df=0, tokenizer=word_tokenize, vocabulary=word2idx)

# train
start_time = time.time()
iteration = len(train_text) // batch_size
for epoch in range(num_epochs):

    running_loss = 0.
    correct = 0.
    total = 0.
    # get mini-batch inputs and targets
    for i in range(iteration):
        inputs = bigram_vectorizer.fit_transform(train_text[i*batch_size:(i+1)*batch_size]).toarray()
        targets = np.asarray(train_target[i*batch_size:(i+1)*batch_size])

        # normalize 
#        inputs = inputs / np.sum(inputs)
        inputs = Variable(torch.from_numpy(inputs).type(torch.FloatTensor)).to(device)
        targets = Variable(torch.from_numpy(targets).type(torch.LongTensor)).to(device)
        
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # backward
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() / iteration
        total += targets.size(0)

        correct += (torch.max(outputs.data, 1)[1] == targets.data).sum().item()
        accuracy = correct / total * 100

        # log
        if (i+1) % 50 == 0:
                print('Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f | Accuracy: %.4f | time: %.2f'
                % (epoch+1, num_epochs, i+1, iteration, running_loss, accuracy, time.time()-start_time))
    
    print('----------Epoch %d: %ss' % (epoch+1, time.time()-start_time))
    # save model per epoch
    torch.save(model.state_dict(), './checkpoints/model_{}.pth'.format(epoch+1))


# load model
model.load_state_dict(torch.load('./checkpoints/model_5.pth'))
model.eval()

# test
_, test_target, test_text = get_data(data_path + 'test.csv')
iters = len(test_text) // batch_size

corr = 0.
tot = 0.
acc = 0.
for i in range(iters):
    test_inputs = bigram_vectorizer.fit_transform(test_text[i*batch_size:(i+1)*batch_size]).toarray()
    test_targets = np.asarray(test_target[i*batch_size:(i+1)*batch_size])

#    test_inputs = test_inputs / np.sum(test_inputs)
    test_inputs = Variable(torch.from_numpy(test_inputs).type(torch.FloatTensor)).to(device)
    test_targets = Variable(torch.from_numpy(test_targets).type(torch.LongTensor)).to(device)
    
    # forward
    test_outputs = model(test_inputs)

    # check
    tot += test_targets.size(0)
    corr += (torch.max(test_outputs.data, 1)[1] == test_targets.data).sum().item()
    acc += corr / tot * 100
    
    print('Step [%d/%d] | running test acc: %.2f' %(i+1, iters, acc/(i+1)))



