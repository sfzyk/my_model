import json
from collections import OrderedDict
from operator import itemgetter

import gensim
import torch
from torch import nn
from torch.utils import data
from torchsummary import summary

from DataManager import DataManager,  MyDataset
from Model import MyModel

__author__ = "Li Xi"

'''

data = DataManager('./data')
word_list = data.gen_word()
train_set, _, _ = data.gen_data()

len_dict = {}
for item in train_set:
    sentence_length = len(item['sentence'])
    if sentence_length in len_dict.keys():
        len_dict[sentence_length] += 1
    else:
        len_dict[sentence_length] = 1
len_dict = OrderedDict(sorted(len_dict.items(), key=itemgetter(1)))
print(len_dict)
exit(0)

gensim_file = './data/glove_model.txt'
vector_json_file = './data/word2vector.json'

model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)
print("finish loading model ...")

vector_dict = {}
cnt = 0
for word in word_list.keys():
    index = word_list[word]
    try:
        word_vector = model[word].tolist()
        vector_dict[index] = word_vector
    except:
        print("do not have: ", word)
        pass
    cnt += 1
    if cnt % 50 == 0:
        print("finish solving %d words ..." % cnt)
print("finish solving %d the words !!!" % cnt)

with open(vector_json_file, "w") as f:
    json.dump(vector_dict, f)
    print('vector dictionary done!')

vector_json = open(vector_json_file)
vector_json = json.load(vector_json)
print("finish loading vector json ...")

print(len(vector_json))
print(type(vector_json))

'''
n_head = 8
d_model = 300
d_k = 50
d_v = 50
n_block = 6
dropout = 0.1
num_class = 5
batch_size = 10

#model = MyModel(n_head, d_model, d_k, d_v, n_block, num_class, dropout=0.1)
#summary(model, input_size=(10, 300))


vector_json_file = 'data/word2vector.json'
index_json = open(vector_json_file)
words_vector = json.load(index_json)
print("vector dictionary length: ", len(words_vector))

my_data = DataManager('./data')
word_list = my_data.gen_word()
train_set, _, _ = my_data.gen_data()

train_dataset = MyDataset(train_set, words_vector)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)

print("finish load dataset ...")

model = MyModel(n_head, d_model, d_k, d_v, n_block, num_class, dropout=0.1)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training
for epoch in range(1000):
    train_loss = 0.
    train_acc = 0.
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

    print(train_acc)
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_dataset)), train_acc / (len(train_dataset))))

torch.save(model.state_dict(), 'params.pkl')
