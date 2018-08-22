import numpy as np
import torch
from torch.utils import data


class Sentence(object):

    def __init__(self, content, aspect, sentiment):
        self.content = content.lower()
        self.aspect = aspect.lower()
        self.sentiment = sentiment
        self.sentence_length = len(self.content.split(' '))

    def stat(self, aspect_dict, word_list):
        data, data_aspect, data_sentiment, i = [], [], [], 0

        for word in self.content.split(' '):
            data.append(word_list[word])

        for word in self.aspect.split(' '):
            data_aspect.append(word_list[word])

        data_sentiment.append(self.sentiment)

        return {'sentence': data,
                'aspect': data_aspect,
                'sentiment': data_sentiment,
                'aspect_index': self.get_aspect(aspect_dict)}

    def get_aspect(self, aspect_dict):
        return aspect_dict[self.aspect]


class DataManager(object):

    def __init__(self, dataset):
        self.file_list = ['train', 'test', 'dev']
        self.origin = {}

        for fname in self.file_list:
            data = []
            with open('%s/%s.cor' % (dataset, fname)) as f:
                sentences = f.readlines()
                for i in range(int(len(sentences) / 3)):
                    content, aspect, sentiment = sentences[i * 3].strip(), sentences[i * 3 + 1].strip(), sentences[
                        i * 3 + 2].strip()
                    sentence = Sentence(content, aspect, sentiment)
                    data.append(sentence)
                self.origin[fname] = data

        self.gen_aspect()

    def gen_word(self):
        wordcount = {}

        def sta(sentence):
            for word in sentence.content.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except:
                    wordcount[word] = 1

            for word in sentence.aspect.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except:
                    wordcount[word] = 1

        for fname in self.file_list:
            for sent in self.origin[fname]:
                sta(sent)

        words = wordcount.items()
        sorted(words, key=lambda x: x[1], reverse=True)
        self.wordlist = {item[0]: index + 1 for index, item in enumerate(words)}

        return self.wordlist

    def gen_aspect(self, threshold=5):
        self.dict_aspect = {}
        for fname in self.file_list:
            for sent in self.origin[fname]:
                if sent.aspect in self.dict_aspect.keys():
                    self.dict_aspect[sent.aspect] = self.dict_aspect[sent.aspect] + 1
                else:
                    self.dict_aspect[sent.aspect] = 1
        i = 0
        for (key, val) in self.dict_aspect.items():
            if val < threshold:
                self.dict_aspect[key] = 0
            else:
                self.dict_aspect[key] = i
                i = i + 1

        return self.dict_aspect

    def gen_data(self):
        self.data = {}
        for fname in self.file_list:
            self.data[fname] = []
            for sent in self.origin[fname]:
                self.data[fname].append(sent.stat(self.dict_aspect, self.wordlist))
        return self.data['train'], self.data['dev'], self.data['test']


class MyDataset(data.Dataset):
    def __init__(self, data, words_vector):
        '''
        root is the filepath of dataset
        x is the input, such as sentence
        y is the lable, such as aspect class or sentiment class
        '''
        self.data = data
        self.word_vector = words_vector
        self.train_data = torch.FloatTensor(self.load_vector())
        self.train_lable = torch.FloatTensor(self.get_items('aspect'))

    def get_max_sentence_length(self, sentences):
        max_length = 0
        for sentence in sentences:
            sentence_length = len(sentence)
            if sentence_length > max_length:
                max_length = sentence_length

        return max_length

    def index2vector(self, sentence, vector_size=300):
        '''sentence indexs to vectors, return a vector sentence'''

        vector_sentence = None
        for i in range(len(sentence)):
            word = str(sentence[i])  # pay attention to the type of key od dict

            try:
                word_vector = np.array(self.word_vector[word])
            except:
                word_vector = np.zeros(vector_size)
                pass
            word_vector = word_vector.reshape(1, -1)
            if vector_sentence is None:
                vector_sentence = word_vector
            else:
                vector_sentence = np.concatenate(
                    (vector_sentence, word_vector), axis=0)
        return vector_sentence

    def get_items(self, item_name):

        items = []
        for item in self.data:
            items.append(item[item_name])

        return items

    def load_vector(self):
        vector_sentences = None
        sentence_num = len(self.data)
        sentences = self.get_items('sentence')
        sentence_length = self.get_max_sentence_length(sentences)
        for sentence in sentences:
            l = sentence_length - len(sentence)
            for i in range(l):
                sentence.append(0)
            vector_sentence = self.index2vector(sentence)
            if vector_sentences is None:
                vector_sentences = vector_sentence
            else:
                vector_sentences = np.concatenate(
                    (vector_sentences, vector_sentence), axis=0)

        sentences = vector_sentences.reshape(sentence_num, sentence_length, 300)

        return sentences

    def __getitem__(self, index):
        x = self.train_data[index]
        y = self.train_lable[index]

        return x, y

    def __len__(self):
        return len(self.train_lable)
