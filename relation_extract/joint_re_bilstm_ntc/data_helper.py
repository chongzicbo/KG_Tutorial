import re
import json
import numpy as np
from tensorflow.keras.preprocessing import sequence
from config import *


def get_data(train_path, test_path):
    """
    从json中提取数据
    :param train_path:
    :param test_path:
    :return:
    """
    train_file = open(train_path).readlines()
    x_train = []
    y_train = []
    for i in train_file:
        data = json.loads(i)
        x_data, y_data = data_decoding(data)
        x_train += x_data
        y_train += y_data
    test_file = open(test_path).readlines()
    x_test = []
    y_test = []
    for j in test_file:
        data = json.loads(j)
        x_data, y_data = data_decoding(data)
        x_test += x_data
        y_test += y_data
    return x_train, y_train, x_test, y_test


def data_decoding(data):
    '''
      decode the json file
      sentText is the sentence
      each sentence may have multiple types of relations
      for every single data, it contains: (sentence-splited, labels)
    '''
    sentence = data["sentText"]
    relations = data["relationMentions"]
    x_data = []
    y_data = []
    for i in relations:
        entity_1 = i["em1Text"].split(" ")
        entity_2 = i["em2Text"].split(" ")
        relation = i["label"]
        relation_label_1 = entity_label_construction(entity_1)
        relation_label_2 = entity_label_construction(entity_2)
        output_list = sentence_label_construction(sentence, relation_label_1, relation_label_2, relation)
        x_data.append(sentence.split(" "))
        y_data.append(output_list)
    return x_data, y_data


def entity_label_construction(entity):
    '''
        give each word in an entity the label
        for entity with multiple words, it should follow the BIES rule
    '''
    relation_label = {}
    for i in range(len(entity)):
        if i == 0 and len(entity) >= 1:
            relation_label[entity[i]] = "B"
        if i != 0 and len(entity) >= 1 and i != len(entity) - 1:
            relation_label[entity[i]] = "I"

        if i == len(entity) - 1 and len(entity) >= 1:
            relation_label[entity[i]] = "E"

        if i == 0 and len(entity) == 1:
            relation_label[entity[i]] = "S"
    return relation_label


def sentence_label_construction(sentence, relation_label_1, relation_label_2, relation):
    '''
       combine the label for each word in each entity with the relation
       and then combine the relation-entity label with the position of the entity in the triple
    '''
    element_list = sentence.split(" ")
    dlist_1 = list(relation_label_1)
    dlist_2 = list(relation_label_2)
    output_list = []
    for i in element_list:
        if i in dlist_1:
            output_list.append(relation + "-" + relation_label_1[i] + "-1")
        elif i in dlist_2:
            output_list.append(relation + "-" + relation_label_2[i] + "-1")
        else:
            output_list.append("O")
    return output_list


def format_control(string):
    str1 = re.sub(r"\r", "", string)
    str2 = re.sub(r"\n", "", str1)
    str3 = re.sub(r"\s*", "", str2)
    return str3


def get_index(word_dict, tag_dict, x_data, y_data):
    x_out = [word_dict[str(k)] for k in x_data]
    y_out = [tag_dict.get(str(l), tag_dict["O"]) for l in y_data]
    return [x_out, y_out]


def word_tag_dict(word_dict_path, tag_dict_path):
    word_dict = {}
    f = open(word_dict_path, "r").readlines()
    for i, j in enumerate(f):
        word = re.sub(r"\n", "", str(j))
        word_dict[word] = i + 1
    tag_dict = {}
    f = open(tag_dict_path, "r").readlines()
    for m, n in enumerate(f):
        tag = re.sub(r"\n", "", str(n))
        tag_dict[tag] = m
    return word_dict, tag_dict


class DataGenerator:
    def __init__(self, word_dict, tag_dict, x_train, y_train, batch_size, max_len, is_test=False):
        self.max_len = max_len
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.is_test = is_test
        self.steps = len(self.x_train) // self.batch_size
        if len(self.x_train) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.x_train)))
            if not self.is_test:
                np.random.shuffle(idxs)
            x_data, y_data = [], []
            for i in idxs:
                x = self.x_train[i]
                y = self.y_train[i]
                x_out, y_out = get_index(self.word_dict, self.tag_dict, x, y)
                x_data.append(x_out)
                y_data.append(y_out)
                if len(x_data) == self.batch_size or i == idxs[-1]:
                    x_data = sequence.pad_sequences(x_data, maxlen=self.max_len, padding="post", truncating="post")
                    y_data = sequence.pad_sequences(y_data, maxlen=self.max_len, padding="post", truncating="post", value=self.tag_dict["O"])
                    yield np.array(x_data), np.array(y_data)
                    x_data, y_data = [], []


if __name__ == '__main__':
    sentence_train, seq_train, sentence_test, seq_test = get_data(TRAIN_PATH, TEST_PATH)
    max_len = max([len(s) for s in sentence_train])
    word_dict, tag_dict = word_tag_dict(WORD_DICT, TAG_DICT)
    dataGenerator = DataGenerator(word_dict, tag_dict, sentence_train, seq_train, 16, max_len)
    for x, y in dataGenerator.__iter__():
        print(x.shape, y.shape)
        print(x[0])
        print(y[0])
        break
