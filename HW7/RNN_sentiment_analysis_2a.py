import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io


class StatefulLSTM(nn.Module):
    def __init__(self, in_size, out_size):
        super(StatefulLSTM, self).__init__()

        self.lstm = nn.LSTMCell(in_size, out_size)
        self.out_size = out_size

        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):

        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.c = Variable(torch.zeros(state_size)).cuda()
            self.h = Variable(torch.zeros(state_size)).cuda()
        self.h, self.c = self.lstm(x, (self.h, self.c))

        return self.h


# When processing sequence data, we will need to apply dropout after every timestep.
# It has been shown to be more effective to use the same dropout mask for an entire sequence
# as opposed to a different dropout mask each time.

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if train == False:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)

        return mask * x


# Note that if this module is called with train set to False, it will simply return the exact same input.
# If train is True, it checks to see if it already has a dropout mask self.m. If it does, it uses this same mask on the data.
# If it doesn’t, it creates a new mask and stores it in self.m.
# As long as we reset our LockedDropout() layer at the beginning of each batch, we can have a single mask applied to the entire sequence.


class RNN_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(RNN_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, no_of_hidden_units)  # ,padding_idx=0)

        self.lstm1 = StatefulLSTM(no_of_hidden_units, no_of_hidden_units)
        self.bn_lstm1 = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout()  # torch.nn.Dropout(p=0.5)

        #self.lstm2 = StatefulLSTM(no_of_hidden_units, no_of_hidden_units)
        #self.bn_lstm2 = nn.BatchNorm1d(no_of_hidden_units)
        # self.dropout2 = LockedDropout()  # torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)

        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss()

    def reset_state(self):
        # The reset_state() function is used so we can easily reset the state of any layer in our model that needs it.
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        # self.lstm2.reset_state()
        # self.dropout2.reset_state()

    def forward(self, x, t, train=True):
        '''
        Notice we have an additional input variable train. 
        We can’t rely on model.eval() to handle dropout appropriately for us anymore 
        so we will need to do it ourselves by passing this variable to the LockedDropout() layer.
        '''

        embed = self.embedding(x)  # batch_size, time_steps, features

        no_of_timesteps = embed.shape[1]

        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps):

            h = self.lstm1(embed[:, i, :])
            h = self.bn_lstm1(h)
            h = self.dropout1(h, dropout=0.5, train=train)

            #h = self.lstm2(h)
            #h = self.bn_lstm2(h)
            #h = self.dropout2(h, dropout=0.3, train=train)

            outputs.append(h)

        outputs = torch.stack(outputs)  # (time_steps,batch_size,features)
        outputs = outputs.permute(1, 2, 0)  # (batch_size,features,time_steps)

        pool = nn.MaxPool1d(no_of_timesteps)
        h = pool(outputs)
        h = h.view(h.size(0), -1)
        #h = self.dropout(h)

        h = self.fc_output(h)

        return self.loss(h[:, 0], t), h[:, 0]  # F.softmax(h, dim=1)


# RNN_sentiment_analysis
vocab_size = 8000

x_train = []
with io.open('preprocessed_data/imdb_train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)

    line[line > vocab_size] = 0

    x_train.append(line)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1


x_test = []
with io.open('preprocessed_data/imdb_test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)

    line[line > vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

model = RNN_model(vocab_size, 500)
model.cuda()


opt = 'adam'
LR = 0.001
#opt = 'adam'
#LR = 0.001
if(opt == 'adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt == 'sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)


batch_size = 200
no_of_epochs = 20
L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

train_loss = []
train_accu = []


for epoch in range(no_of_epochs):

    # training
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()

    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = 100
        x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j, 0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j, :] = x[start_index:(start_index+sequence_length)]
        y_input = y_train[I_permutation[i:i+batch_size]]

        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data, target, train=True)
        loss.backward()

        optimizer.step()   # update weights

        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print('Train: ', epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

# save the trained model
torch.save(model, 'rnn_2a.model')

# begin testing
no_of_epochs = 8
test_accu = []

for epoch in range(no_of_epochs):

    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()

    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):
        x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = (epoch+1)*50
        x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j, 0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j, :] = x[start_index:(start_index+sequence_length)]
        y_input = np.asarray([y_test[j] for j in I_permutation[i:i+batch_size]], dtype=np.int)
        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()
        with torch.no_grad():
            loss, pred = model(data, target, train=False)
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)
    test_accu.append(epoch_acc)
    time2 = time.time()
    time_elapsed = time2 - time1

    print('Test: ', epoch, sequence_length, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time_elapsed))


data = [train_loss, train_accu, test_accu]
data = np.asarray(data)
np.save('data_rnn_2a.npy', data)




