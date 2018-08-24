# -*- coding: UTF-8 -*-
import collections
import json
import xml.dom.minidom
from operator import itemgetter

import gensim
import h5py
import jieba
import numpy as np

from DataManager import DataManager

gensim_file = 'glove_model.txt'

'''
print('store train dataset ...')
data_file = 'SemEval2014/Restaurants_Train_v2.xml'
index_json_file = 'SemEval2014/SemEval2014_train_index.json'
vector_json_file = 'SemEval2014/SemEval2014_train_vector.json'
dataset_file = 'SemEval2014/dataset_train.h5'
# max length:  80
# sentences shape: (3518, 80)
# categories shape: (3518, 5)
# polarities shape: (3518, 3)
'''

print('store test dataset ...')
data_file = 'SemEval2014/restaurants-trial.xml'
index_json_file = 'SemEval2014/SemEval2014_test_index.json'
vector_json_file = 'SemEval2014/SemEval2014_test_vector.json'
dataset_file = 'SemEval2014/dataset_test.h5'
# max length:  40
# sentences shape: (113, 40)
# categories shape: (113, 5)
# polarities shape: (113, 3)

aspect_class = dict()
aspect_class['ambience'] = 0
aspect_class['anecdotes'] = 1
aspect_class['food'] = 2
aspect_class['price'] = 3
aspect_class['service'] = 4

sentiment_class = dict()
sentiment_class['neutral'] = 0
sentiment_class['positive'] = 1
sentiment_class['negative'] = 2

word_cnt = dict()
sentences = []
categories = []
polarities = []

# -------------------- read data from xml file -------------------------------
dom = xml.dom.minidom.parse(data_file)
root = dom.documentElement
sentences_root = root.getElementsByTagName('sentence')

for sentences_single in sentences_root:
    category_root = sentences_single.getElementsByTagName('aspectCategory')

    for category_single in category_root:

        sentence = sentences_single.getElementsByTagName('text')[0].firstChild.data
        category = category_single.getAttribute('category').split('/')[0]
        polarity = category_single.getAttribute('polarity')

        if polarity == 'conflict':
            continue
        sentences.append(sentence)
        categories.append(aspect_class[category])
        polarities.append(sentiment_class[polarity])

        word_cnt[category] = 1


# ------------- prepare word dict and store in json file ---------------------------

def count_words(input_sentence):
    global word_cnt, max_len
    input_sentence = list(jieba.cut(input_sentence))
    for word in input_sentence:
        if word != "":
            orig_stem = word  # .replace('\n', '')
            word_cnt[orig_stem] = 1


for line in sentences:
    count_words(line)

word_cnt = collections.OrderedDict(sorted(word_cnt.items(), key=lambda x: (x[1], x[0]), reverse=True))
word_list = list(word_cnt.keys())
word_dict = dict()

for i in range(len(word_list)):
    word_dict[word_list[i]] = i + 1

with open(index_json_file, "w") as f:
    json.dump(word_dict, f)
    print("word dict done!")

# -------------- make word vector dictionary --------------------

# load glove model

model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)  # GloVe Model
print("finish loading model ...")

# load word index

index_json = open(index_json_file)
word_index = json.load(index_json)
print("finish loading index json ...")

# make a vector dictionary
vector_dict = dict()
cnt = 0
for word in word_index.keys():
    index = word_index[word]
    try:
        # Object of type 'ndarray' is not JSON serializable
        # here convert 'ndarrary' to 'list'
        # 'lambc' is not in the
        word_vector = model[word].tolist()
        vector_dict[index] = word_vector
    except:
        print("do not have: ", word)
        pass
    cnt += 1
    if cnt % 50 == 0:
        print("finish solving %d words ..." % cnt)
print("finish solving %d the words !!!" % cnt)

# save vector dictionary (345 keys except 'lambc' )

with open(vector_json_file, "w") as f:
    json.dump(vector_dict, f)
    print('vector dictionary done!')

# ----------------------- sentence to index -----------------------------

# load word index
index_json = open(index_json_file)
word_index = json.load(index_json)
print("finish loading index json ...")

sentence_index = []

# get max length of sentence indexs
max_len = 0
for sentence in sentences:
    words = list(jieba.cut(sentence))
    word_indexs = []
    cnt_words = 0
    for w in words:
        if w != ' ':
            cnt_words += 1
    if max_len < cnt_words:
        max_len = cnt_words

print("max length: ", max_len)

for sentence in sentences:
    words = list(jieba.cut(sentence))
    word_indexs = []
    for w in words:
        if w != ' ':
            try:
                word_indexs.append(word_index[w])
            except:
                word_indexs.append(0)
                pass
    for i in range(max_len - len(word_indexs)):
        word_indexs.append(0)
    sentence_index.append(word_indexs)

sentences = np.array(sentence_index)
categories = np.array(categories)
polarities = np.array(polarities)

print('sentences shape:', sentences.shape)
print('categories shape:', categories.shape)
print('polarities shape:', polarities.shape)

# ------------------ store data -------------------

print('change to one hot code ... ')


def lable2onehot(lables, class_nums):
    one_hot_codes = np.eye(class_nums)
    output_lables = None
    for lable in lables:
        output_lable = one_hot_codes[int(lable)]
        output_lable.reshape(1, -1)
        if output_lables is None:
            output_lables = output_lable
        else:
            output_lables = np.concatenate((output_lables,
                                            output_lable), axis=0)

    return output_lables


categories = lable2onehot(categories, len(aspect_class)).reshape(len(sentences), -1)
polarities = lable2onehot(polarities, len(sentiment_class)).reshape(len(sentences), -1)

print('sentences shape:', sentences.shape)
print('categories shape:', categories.shape)
print('polarities shape:', polarities.shape)

f = h5py.File(dataset_file, "w")
f.create_dataset("sentences", shape=sentences.shape, data=sentences)
f.create_dataset("categories", shape=categories.shape, data=categories)
f.create_dataset("polarities", shape=polarities.shape, data=polarities)
f.close()

print("finish store files !!!")

vector_json_file = 'SemEval2014/SemEval2014_test_vector.json'
dataset_file = 'SemEval2014/dataset_test.h5'
f = h5py.File(dataset_file, "r")

sentences = f['sentences'].value
categories = f['categories'].value
polarities = f['polarities'].value

print('sentences shape:', sentences.shape)
print('categories shape:', categories.shape)
print('polarities shape:', polarities.shape)


# -----------------------------prepare dataset vector dict-----------------------------

data = DataManager('./')
word_list = data.gen_word()
train_set, _, _ = data.gen_data()

len_dict = {}
for item in train_set:
    sentence_length = len(item['sentence'])
    if sentence_length in len_dict.keys():
        len_dict[sentence_length] += 1
    else:
        len_dict[sentence_length] = 1
len_dict = collections.OrderedDict(sorted(len_dict.items(), key=itemgetter(1)))
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