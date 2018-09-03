import xml.dom.minidom

import jieba

gensim_file = 'glove_model.txt'
#data_file = 'SemEval2014/Laptops train data.xml'
data_file = 'SemEval2014/Laptops trial data.xml'

dom = xml.dom.minidom.parse(data_file)
root = dom.documentElement
sentences_root = root.getElementsByTagName('sentence')

sentences = []
categories = []
polarities = []

sentiment_class = dict()
sentiment_class['neutral'] = 0
sentiment_class['positive'] = 1
sentiment_class['negative'] = -1

for sentences_single in sentences_root:
    category_root = sentences_single.getElementsByTagName('aspectTerm')

    for category_single in category_root:

        sentence = sentences_single.getElementsByTagName('text')[0].firstChild.data
        category = category_single.getAttribute('term')
        polarity = category_single.getAttribute('polarity')

        if polarity == 'conflict':
            continue
        sentences.append(sentence.lower())
        categories.append(category.lower())
        polarities.append(sentiment_class[polarity])

print(len(sentences))
print(len(categories))
print(len(polarities))

print(sentences[:10])
print(categories[:10])
print(polarities[:10])

words = list(jieba.cut(sentences[0]))
print(words)




