import json
import math
import time

import torch
from torch import nn
from torch.utils import data
from torchsummary import summary

from DataManager import DataManager, MyDataset
from Model import MyModel

__author__ = "Li Xi"

# --------- parameters -------------------
d_model = 300
n_head = 6
d_k = 50
d_v = 50
n_block = 8
dropout = 0.5  #
num_class = 3
batch_size = 256  #
lr = 0.01  #
l2_norm = 0.1
epoches = 10
max_len = 80
lable_type = 'sentiment'


# To test the model:
# 1. Uncomment the follwing line
# 2. In Model.py, uncomment the first two line in forward function (iminate the aspect input)
# 3. The output show the model srtucture, including the shape of model and number of parameter

#model = MyModel(n_head, d_model, d_k, d_v, n_block, num_class, max_len, dropout=0.1)
#summary(model, [max_len, 300])
#exit(0)


# --------------- load dataset ------------------
# -----------------------------------------------

vector_json_file = 'data/restaurant/word2vector.json'
index_json = open(vector_json_file)
words_vector = json.load(index_json)
print("vector dictionary length: ", len(words_vector))

my_data = DataManager('./data/restaurant') #
word_list = my_data.gen_word()
train_set, dev_set, test_set = my_data.gen_data()

# train set init
train_dataset = MyDataset(train_set, words_vector, max_len, lable_type)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)
print("train set length: ", len(train_dataset))

# dev set init
dev_dataset = MyDataset(dev_set, words_vector, max_len, lable_type)
dev_loader = data.DataLoader(dataset=dev_dataset,
                             batch_size=batch_size,
                             shuffle=True)
print("dev set length: ", len(dev_dataset))

# test set init
test_dataset = MyDataset(test_set, words_vector, max_len, lable_type)
test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=True)
print("test set length: ", len(test_dataset))

print("finish loading dataset ...")


# ---------------- model init ------------------------
# ----------------------------------------------------

model = MyModel(n_head, d_model, d_k, d_v, n_block, num_class, max_len, dropout=0.1)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_norm)

dev_accus = []
for epoch in range(epoches):
    print('[ Epoch', epoch + 1, ']')

    start = time.time()
    train_loss = 0.
    train_acc = 0.

    # trianing
    # x - sentence embeding
    # y - sentiment class
    # z - aspect embeding (may more than one aspect)
    for i, (x, y, z) in enumerate(train_loader):
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)
        y = torch.max(y, 1)[1]
        z = torch.FloatTensor(z)


        optimizer.zero_grad()
        outputs = model(x, z)

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
    for i, (x, y, z) in enumerate(dev_loader):
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)
        y = torch.max(y, 1)[1]

        # Forward
        outputs = model(x, z)
        loss = criterion(outputs, y)

        dev_loss += loss.data[0]
        pred = torch.max(outputs, 1)[1]
        dev_correct = (pred == y).sum()
        dev_acc += dev_correct.data[0]

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
for i, (x, y, z) in enumerate(test_loader):
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)
    y = torch.max(y, 1)[1]

    outputs = model(x, z)
    pred = torch.max(outputs, 1)[1]
    test_correct = (pred == y).sum()
    test_acc += test_correct.data[0]

test_acc = float(test_acc) / len(test_dataset)
print('  - (Testing)   accuracy: {accu:3.3f} %'.
      format(accu=100 * test_acc))
