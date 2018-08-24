import json
import math
import time

import torch
from torch import nn
from torch.utils import data

from DataManager import DataManager, MyDataset
from Model import MyModel

__author__ = "Li Xi"

# --------- parameters -------------------
n_head = 8
d_model = 300
d_k = 50
d_v = 50
n_block = 6
dropout = 0.1
num_class = 5
batch_size = 25
lr = 0.01
epoches = 8

# -------- load dataset ------------------
vector_json_file = 'data/word2vector.json'
index_json = open(vector_json_file)
words_vector = json.load(index_json)
print("vector dictionary length: ", len(words_vector))

my_data = DataManager('./data')
word_list = my_data.gen_word()
train_set, dev_set, test_set = my_data.gen_data()

train_dataset = MyDataset(train_set, words_vector)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)
print("train set length: ", len(train_dataset))

dev_dataset = MyDataset(dev_set, words_vector)
dev_loader = data.DataLoader(dataset=dev_dataset,
                             batch_size=batch_size,
                             shuffle=True)
print("dev set length: ", len(dev_dataset))

test_dataset = MyDataset(test_set, words_vector)
test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=True)
print("test set length: ", len(test_dataset))

print("finish load dataset ...")

# ---------------- model init ------------------------

model = MyModel(n_head, d_model, d_k, d_v, n_block, num_class, dropout=0.1)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dev_accus = []
for epoch in range(epoches):
    print('[ Epoch', epoch + 1, ']')

    start = time.time()
    train_loss = 0.
    train_acc = 0.

    # trianing
    for i, (x, y) in enumerate(train_loader):
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)
        y = torch.max(y, 1)[1]

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(x)

        loss = criterion(outputs, y)

        train_loss += loss.data[0]
        pred = torch.max(outputs, 1)[1]
        train_correct = (pred == y).sum()
        train_acc += train_correct.data[0]

        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_dataset)
    train_acc = float(train_acc) / len(train_dataset)

    print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
          'elapse: {elapse:3.3f} min'.format(
        ppl=math.exp(min(train_loss, 100)), accu=100 * train_acc,
        elapse=(time.time() - start) / 60))

    # validation
    start = time.time()
    dev_loss = 0.
    dev_acc = 0.
    for i, (x, y) in enumerate(dev_loader):
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)
        y = torch.max(y, 1)[1]

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(x)

        loss = criterion(outputs, y)

        dev_loss += loss.data[0]
        pred = torch.max(outputs, 1)[1]
        dev_correct = (pred == y).sum()
        dev_acc += dev_correct.data[0]

        loss.backward()
        optimizer.step()

    dev_loss = dev_loss / len(dev_dataset)
    dev_acc = float(dev_acc) / len(dev_dataset)

    print('  - (Validation)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
          'elapse: {elapse:3.3f} min'.format(
        ppl=math.exp(min(dev_loss, 100)), accu=100 * dev_acc,
        elapse=(time.time() - start) / 60))

    dev_accus += [dev_acc]

    model_state_dict = model.state_dict()
    model_name = 'best_model.chkpt'

    if dev_acc >= max(dev_accus):
        torch.save(model_state_dict, model_name)
        print('    - [Info] The checkpoint file has been updated.')

# test
test_acc = 0.
for i, (x, y) in enumerate(test_loader):
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)
    y = torch.max(y, 1)[1]

    outputs = model(x)
    pred = torch.max(outputs, 1)[1]
    test_correct = (pred == y).sum()
    test_acc += test_correct.data[0]


test_acc = float(test_acc) / len(test_dataset)
print('  - (Training)   accuracy: {accu:3.3f} %'.
      format(accu=100 * test_acc))


