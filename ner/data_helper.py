import numpy as np


def get_w2i(vocab_path):
    """
    获取word to index 词典
    :param vocab_path: 使用的是bert模型中的vocab.txt文件
    :return: dict{word:id}
    """
    w2i = {}
    with open(vocab_path, encoding="utf-8") as f:
        while True:
            text = f.readline()
            if not text:
                break
            text = text.strip()
            if text and len(text) > 0:
                w2i[text] = len(w2i) + 1
    i2w = dict(zip(w2i.values(), w2i.keys()))
    return w2i, i2w


def get_tag2index(train_path):
    """
        读取训练文件，获取标签集合
    :param train_path:训练文件路径
    :return:
    """
    sequences = []
    sequence = []
    tags_list = []
    tags = []
    tag_set = set()
    with open(train_path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        word_tag = line.strip().split(" ")
        if len(word_tag[-1]) != 0:
            tag_set.add(word_tag[-1])
            tags.append(word_tag[-1])
            sequence.append(word_tag[0])
        else:
            sequences.append("".join(sequence))
            tags_list.append(tags)
            sequence = []
            tags = []
    index2tag = dict(enumerate(sorted(tag_set)))
    tag2index = dict(zip(index2tag.values(), index2tag.keys()))
    max_len = max([len(str(sequence)) for sequence in sequences])
    return index2tag, tag2index, sequences, tags_list, max_len


def get_data_array(sequences: list, tags_list: list, w2i: dict, tag2index: dict, max_len):
    """
        将字符串转为数值序列,并进行padding
    :param sequences:list(str)
    :param w2i:word->index
    :return:
    """

    sequences_ids = []
    tags_ids = []
    padding_sequences_ids = []
    padding_tags_ids = []
    for (sequence, tags) in zip(sequences, tags_list):
        sequence_ids = [w2i.get(word, 100) for word in sequence]  # 100对应字符[UNK]
        tag_ids = [tag2index.get(tag, tag2index["O"]) for tag in tags]

        padding_sequence_ids = sequence_ids + (max_len - len(sequence_ids)) * [0]  # 0 表示字符[PAD]
        padding_tag_ids = tag_ids + (max_len - len(tag_ids)) * [tag2index["O"]]

        sequences_ids.append(sequence_ids)
        tags_ids.append(tag_ids)
        padding_sequences_ids.append(padding_sequence_ids)
        padding_tags_ids.append(padding_tag_ids)

    return sequences_ids, padding_sequences_ids, tags_ids, padding_tags_ids


def get_data_array_bert(padding_sequences_ids, padding_tags_ids, tag2index):
    """
    生成bert所需的数据格式
    :param padding_sequences_ids:
    :param padding_tags_ids:
    :param tag2index:
    :return:
    """
    bert_sequence_ids = [[101] + sequence + [102] for sequence in padding_sequences_ids]
    bert_datatype_ids = [[0] * len(bert_sequence_ids[0]) for i in range(len(bert_sequence_ids))]
    bert_label_ids = [[tag2index["O"]] + tag_ids + [tag2index["O"]] for tag_ids in padding_tags_ids]
    return bert_sequence_ids, bert_datatype_ids, bert_label_ids


if __name__ == '__main__':
    train_path = "/python_tutorial/KG_Tutorial\\data\\ner\\train.txt"
    vocab_path = "/python_tutorial/KG_Tutorial\\data\\vocab.txt"
    w2i, i2w = get_w2i(vocab_path)
    index2tag, tag2index, sequences, tags_list, max_len = get_tag2index(train_path)
    print(tag2index, index2tag)
    print(len(sequences), len(tags_list))
    sequences_ids, padding_sequences_ids, tags_ids, padding_tags_ids = get_data_array(sequences, tags_list, w2i, tag2index,
                                                                                      max_len)
    print("训练样本数量为:%d" % len(sequences_ids))
    print(sequences_ids[0])
    print(padding_sequences_ids[0])
    print(len(padding_sequences_ids[0]))
    for id in sequences_ids[0]:
        print(i2w[id], end=" ")

    print(np.array(padding_sequences_ids).shape, np.array(padding_tags_ids).shape)
    bert_sequence_ids, bert_datatype_ids, bert_label_ids = get_data_array_bert(padding_sequences_ids, padding_tags_ids, tag2index)
    print(np.array(bert_sequence_ids).shape)
    print(np.array(bert_datatype_ids).shape)
    print(np.array(bert_label_ids).shape)
