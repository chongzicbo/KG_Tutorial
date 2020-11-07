import codecs
import pandas as pd
import numpy as np
import collections
from config import *


def get_relation2id(file_path):
    relation2id = {}
    with codecs.open(file_path, "r", "utf-8") as f:
        for line in f.readlines():
            relation2id[line.split()[0]] = int(line.split()[1])
        f.close()
    return relation2id


def get_sentence_label_positionE(file_path, relation2id):
    datas = []
    labels = []
    positionE1 = []
    positionE2 = []
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    with codecs.open(file_path, "r", "utf-8") as f:
        for line in f:
            line_split = line.split("\t")
            # if count[relation2id.get(line[2], 0)] < 1500:
            sentence = []
            index1 = line_split[3].index(line_split[0])  # 实体1在语料中的索引位置
            position1 = []
            index2 = line_split[3].index(line_split[1])  # 实体2在语料中的索引位置
            position2 = []
            for i, word in enumerate(line_split[3]):
                sentence.append(word)
                position1.append(i - index1)  # 字符与实体1的相对位置
                position2.append(i - index2)  # 字符与实体2的相对位置
                i += 1
            datas.append(sentence)
            labels.append(relation2id[line_split[2]])  # 语料对应的标签
            positionE1.append(position1)
            positionE2.append(position2)
            count[relation2id[line_split[2]]] += 1
    return datas, labels, positionE1, positionE2


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def get_word2id(datas):
    all_words = flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words) + 1)
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    word2id["BLANK"] = len(word2id) + 1
    word2id["UNKNOWN"] = len(word2id + 1)
    id2word[len(id2word) + 1] = "BLANK"
    id2word[len(id2word) + 1] = "UNKNOWN"
    return word2id, id2word


def get_data_array(word2id, datas, labels, positionE1, positionE2, max_len=50):
    def X_padding(words):
        ids = []
        for i in words:
            if i in word2id:
                ids.append(word2id[i])
            else:
                ids.append(word2id["UNKNOWN"])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([word2id["BLANK"]] * (max_len - len(ids)))
        return ids

    def pos(num):
        if num < -40:
            return 0
        if num >= -40 and num <= 40:
            return num + 40
        if num > 40:
            return 80

    def position_padding(words):
        words = [pos(i) for i in words]
        if len(words) >= max_len:
            return words[:max_len]
        words.extend([81] * (max_len - len(words)))
        return words

    df_data = pd.DataFrame({'words': datas, 'tags': labels, 'positionE1': positionE1, 'positionE2': positionE2}, index=range(len(datas)))  # if __name__ == '__main__':
    df_data["words"] = df_data["words"].apply(X_padding)
    df_data["tags"] = df_data["tags"]

    df_data["positionE1"] = df_data["positionE1"].apply(position_padding)
    df_data["positionE2"] = df_data["positionE2"].apply(position_padding)
    datas = np.asarray(list(df_data["words"].values))
    labels = np.asarray(list(df_data["tags"].values))
    positionE1 = np.asarray(list(df_data["positionE1"].values))
    positionE2 = np.asarray(list(df_data["positionE2"].values))
    return datas, labels, positionE1, positionE2


if __name__ == '__main__':
    relation_file_path = relation_file_path
    train_data_file_path = train_file_path
    relation2id = get_relation2id(relation_file_path)
    print(relation2id)

    datas, labels, positionE1, positionE2 = get_sentence_label_positionE(train_data_file_path, relation2id)
    word2id, id2word = get_word2id(datas)

    datas, labels, positionE1, positionE2 = get_data_array(word2id, datas, labels, positionE1, positionE2)
    print(datas.shape, labels.shape, positionE1.shape, positionE2.shape)
    print(datas[0], labels[0])
    print(positionE1[0])
    print(positionE2[0])
